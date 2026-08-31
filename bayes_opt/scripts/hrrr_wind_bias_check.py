"""Do the leg-level CH4 background biases (§2.2/§28) correlate with HRRR's
10m wind speed/direction -- the meteorology STILT's transport is actually
run on -- and, specifically, does wind direction explain §28.4's still-open
question: why does the land/water contrast in the leg offset flip sign
flight to flight?

HRRR 10m wind (`UGRD`/`VGRD`, `10m_above_ground` level, same hourly
`anl.zarr` archive as hrrr_pblh_vs_mlh_check.py) is grid-relative on HRRR's
native Lambert Conformal grid (standard parallel 38.5N, origin -97.5E per
`hrrrzarr/grid/projparams.json`), not earth-relative -- rotated to true
north/east here before computing speed or direction; skipping that step
would bias wind *direction* by ~15 degrees over the NYC domain (not speed,
which is rotation-invariant).

Two checks:

1. Leg-level regression (script structure parallel to
   mlh_bias_regression_check.py): leg background offset and posterior
   residual against wind speed and against the (u, v) wind components
   directly (avoids circularity problems with regressing against raw
   compass degrees), pooled across flights, flight-demeaned.
2. A direct test of §28.4/§28.5's open question: does the sign of the
   land/water leg-offset contrast (recorded in land_water_background_check.py's
   per-flight results) match an onshore (wind blowing off the water, onto
   land) vs. offshore (blowing off the continent, out to sea) wind
   direction that day?

Run with:
    python3 scripts/hrrr_wind_bias_check.py
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np
import s3fs
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, ".")

from goe.config import Config
from halo_oe.background import detect_legs, _load_receptor_time
from halo_oe.io_bundle import load_inversion
from leg_offset_drift_check import BUNDLE, FLIGHT_DATA_DIR, true_takeoff_hour

REP_LAT, REP_LON = 40.75, -73.95
LAT0, LON0 = 38.5, -97.5   # HRRR Lambert Conformal standard parallel / origin
CONE = np.sin(np.radians(LAT0))

# §28.4's per-flight land/water leg-offset contrast (water - land, ppm), for reference
LAND_WATER_SIGN = {
    "20230726_1": -0.0034, "20230726_2": +0.0043, "20230728_1": -0.0067,
    "20230728_2": -0.0169, "20230805": +0.0041, "20230809": +0.0146,
}


def grid_to_earth_wind(u_grid, v_grid, lon):
    """Rotate HRRR grid-relative wind to earth-relative (Lambert Conformal)."""
    alpha = np.radians(CONE * (lon - LON0))
    u = v_grid * np.sin(alpha) + u_grid * np.cos(alpha)
    v = v_grid * np.cos(alpha) - u_grid * np.sin(alpha)
    return u, v


def wind_dir_from(u, v):
    """Meteorological direction wind is blowing FROM, degrees, 0=N/360, 90=E."""
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def hrrr_wind(fs, date_str: str, hour: int, iy: int, ix: int):
    u_path = (f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/"
              "10m_above_ground/UGRD/10m_above_ground")
    v_path = (f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/"
              "10m_above_ground/VGRD/10m_above_ground")
    u_store = s3fs.S3Map(root=u_path, s3=fs, check=False)
    v_store = s3fs.S3Map(root=v_path, s3=fs, check=False)
    u_grid = float(np.asarray(xr.open_zarr(u_store, consolidated=False)["UGRD"])[iy, ix])
    v_grid = float(np.asarray(xr.open_zarr(v_store, consolidated=False)["VGRD"])[iy, ix])
    return grid_to_earth_wind(u_grid, v_grid, REP_LON)


def nearest_hrrr_index():
    with h5py.File("scratch_hrrr/HRRR_latlon.h5", "r") as f:
        lat, lon = f["latitude"][:], f["longitude"][:]
    d2 = (lat - REP_LAT) ** 2 + (lon - REP_LON) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    return iy, ix


def per_flight_wind_hours(fs, iy, ix, fid: str):
    date = fid.split("_")[0]
    date_hdr, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    full = sorted(glob.glob(os.path.join(FLIGHT_DATA_DIR_FULL, f"*{date}*_F{fnum}_full.h5")))[0]
    with h5py.File(full, "r") as f:
        t = f["Nav_Data/gps_time"][:, 0]
    tc0 = true_takeoff_hour(fid)
    with h5py.File(sorted(glob.glob(os.path.join(FLIGHT_DATA_DIR, f"*{date}*_F{fnum}_full_1000m.h5")))[0], "r") as f:
        tc = f["time"][:]
    hours = sorted(set(range(int(np.floor(tc.min())), int(np.ceil(tc.max())) + 1)))
    hrrr_t, u_list, v_list = [], [], []
    for h in hours:
        if not (0 <= h <= 23):
            continue
        u, v = hrrr_wind(fs, date, h, iy, ix)
        hrrr_t.append(h); u_list.append(u); v_list.append(v)
    return np.array(hrrr_t, dtype=float), np.array(u_list), np.array(v_list)


FLIGHT_DATA_DIR_FULL = "../flight_data"


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    fs = s3fs.S3FileSystem(anon=True)
    iy, ix = nearest_hrrr_index()

    print(f"{'flight':<12}{'mean speed (m/s)':>17}{'mean dir (deg from)':>20}"
          f"{'onshore(SE-ish)?':>18}{'l/w sign (§28.4)':>18}")
    all_rows = []
    per_flight_speed = {}
    for fid in inv.flight_ids:
        hrrr_t, u, v = per_flight_wind_hours(fs, iy, ix, fid)
        speed = np.hypot(u, v)
        mean_u, mean_v = u.mean(), v.mean()
        mean_speed = speed.mean()
        mean_dir = wind_dir_from(mean_u, mean_v)
        onshore = "onshore (S/SE/SW)" if 90 <= mean_dir <= 270 else "offshore (N/NW/NE)"
        lw = LAND_WATER_SIGN[fid]
        print(f"{fid:<12}{mean_speed:>17.2f}{mean_dir:>20.0f}{onshore:>18}{lw:>+18.4f}")
        per_flight_speed[fid] = mean_speed

        # --- leg-level table for the regression ---
        R = inv.receptors
        fi = inv.flight_ids.index(fid)
        sel = R["receptor_flight"].astype(int) == fi
        lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
        bg_offset = R["receptor_background_offset"][sel]
        post_resid = R["enhancement"][sel] - R["modeled"][sel]
        clock_hours = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
        clock_h = clock_hours / 3600.0
        leg_id = detect_legs(
            lat, lon, clock_hours,
            gap_seconds=cfg.get_float("background", "leg_gap_seconds", default=8.0),
            min_leg_size=cfg.get_int("background", "leg_min_size", default=10),
            axis_deg=cfg.get_float("background", "leg_axis_deg", default=45.0),
        )
        for leg in np.unique(leg_id):
            m = leg_id == leg
            leg_clock = clock_h[m].mean()
            k = int(np.argmin(np.abs(hrrr_t - leg_clock)))
            all_rows.append(dict(
                fid=fid, leg=leg, u=u[k], v=v[k], speed=speed[k],
                offset=bg_offset[m].mean(), post_resid=post_resid[m].mean(),
            ))

    fids = sorted(set(r["fid"] for r in all_rows))
    offset_mean = {f: np.mean([r["offset"] for r in all_rows if r["fid"] == f]) for f in fids}
    resid_mean = {f: np.mean([r["post_resid"] for r in all_rows if r["fid"] == f]) for f in fids}
    for r in all_rows:
        r["offset_c"] = r["offset"] - offset_mean[r["fid"]]
        r["resid_c"] = r["post_resid"] - resid_mean[r["fid"]]

    speed = np.array([r["speed"] for r in all_rows])
    u_arr = np.array([r["u"] for r in all_rows])
    v_arr = np.array([r["v"] for r in all_rows])
    offset_c = np.array([r["offset_c"] for r in all_rows])
    resid_c = np.array([r["resid_c"] for r in all_rows])
    n = len(all_rows)
    print(f"\npooled, flight-demeaned legs: n={n}\n")

    def report(y, ylabel):
        r_s, p_s = stats.pearsonr(speed, y)
        print(f"  {ylabel} vs. wind speed:      r={r_s:+.3f}  p={p_s:.3g}")
        X = np.column_stack([np.ones(n), u_arr, v_arr])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ b) ** 2)
        tss = np.sum((y - y.mean()) ** 2)
        r2 = 1 - rss / tss
        # F-test for the pair (u, v) jointly vs. intercept-only
        rss0 = np.sum((y - y.mean()) ** 2)
        f_stat = ((rss0 - rss) / 2) / (rss / (n - 3))
        f_p = 1 - stats.f.cdf(f_stat, 2, n - 3)
        print(f"  {ylabel} vs. (u,v) jointly:    b_u={b[1]:+.5f} b_v={b[2]:+.5f}  "
              f"R²={r2:.4f}  F={f_stat:.2f}  p={f_p:.3g}")

    print("target: leg background offset")
    report(offset_c, "offset")
    print("target: posterior residual")
    report(resid_c, "resid")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fids)))
    fid_color = {f: colors[i] for i, f in enumerate(fids)}
    for ax, y, ylabel in [(axes[0], offset_c, "leg background offset"),
                          (axes[1], resid_c, "posterior residual")]:
        for r, yy in zip(all_rows, y):
            ax.scatter(r["speed"], yy, color=fid_color[r["fid"]])
        ax.set_xlabel("HRRR 10m wind speed (m/s)")
        ax.set_ylabel(f"{ylabel}, flight-demeaned (ppm)")
        ax.axhline(0, color="gray", lw=0.5)
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=fid_color[f], label=f) for f in fids]
    axes[0].legend(handles=handles, fontsize=7, ncol=2)
    plt.savefig("figures/hrrr_wind_bias_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/hrrr_wind_bias_check.png")


if __name__ == "__main__":
    main()
