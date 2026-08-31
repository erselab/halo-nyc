"""Per-leg (not per-flight, not single-point) HRRR 10m wind, focused on
leg-to-leg and flight-to-flight *changes* in wind direction: if HRRR's
modeled wind is rotating during a survey (or two flights see different
modeled wind), but the real wind didn't rotate the same way (or did, and
HRRR missed the timing), STILT's footprints for legs flown right around
that transition are built on the wrong flow direction -- a mismatch that
would plausibly show up as elevated |residual| specifically on legs
adjacent to a modeled wind shift, not a fixed-sign bias.

Improves on hrrr_wind_bias_check.py in two ways:
1. Wind is sampled at each **leg's own centroid** (mean receptor lat/lon),
   not one fixed domain point -- so it captures whatever real spatial wind
   gradient HRRR itself represents across the ~250km domain, not just one
   cell's time series repeated for every leg.
2. Each leg's own HRRR hour (nearest, not one hour per flight) is used, so
   legs late in a flight get a different meteorological hour than legs
   early in it whenever the flight crosses an HRRR hour boundary.

Grid-relative UGRD/VGRD are rotated to earth-relative per grid cell (each
leg's own cell has a slightly different rotation angle on HRRR's Lambert
Conformal grid) before computing speed/direction, same convention as
hrrr_wind_bias_check.py.

To keep remote reads cheap, each unique (date, hour) grid is fetched once
(full ~1059x1799 array, cached in memory) and reused for every leg that
falls in it, rather than one open-zarr call per leg.

Run with:
    python3 scripts/hrrr_wind_leg_change_check.py
"""

from __future__ import annotations

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
from leg_offset_drift_check import BUNDLE, FLIGHT_DATA_DIR

LAT0, LON0 = 38.5, -97.5
CONE = np.sin(np.radians(LAT0))
_GRID_LAT = _GRID_LON = None
_HOUR_CACHE: dict[tuple[str, int], tuple[np.ndarray, np.ndarray]] = {}


def load_grid():
    global _GRID_LAT, _GRID_LON
    if _GRID_LAT is None:
        with h5py.File("scratch_hrrr/HRRR_latlon.h5", "r") as f:
            _GRID_LAT = f["latitude"][:]
            _GRID_LON = f["longitude"][:]
    return _GRID_LAT, _GRID_LON


def nearest_cell(lat, lon):
    glat, glon = load_grid()
    d2 = (glat - lat) ** 2 + (glon - lon) ** 2
    return np.unravel_index(np.argmin(d2), d2.shape)


def fetch_hour_grid(fs, date_str: str, hour: int):
    key = (date_str, hour)
    if key not in _HOUR_CACHE:
        u_path = (f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/"
                  "10m_above_ground/UGRD/10m_above_ground")
        v_path = (f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/"
                  "10m_above_ground/VGRD/10m_above_ground")
        u_store = s3fs.S3Map(root=u_path, s3=fs, check=False)
        v_store = s3fs.S3Map(root=v_path, s3=fs, check=False)
        u = np.asarray(xr.open_zarr(u_store, consolidated=False)["UGRD"])
        v = np.asarray(xr.open_zarr(v_store, consolidated=False)["VGRD"])
        _HOUR_CACHE[key] = (u, v)
    return _HOUR_CACHE[key]


def grid_to_earth_wind(u_grid, v_grid, lon):
    alpha = np.radians(CONE * (lon - LON0))
    u = v_grid * np.sin(alpha) + u_grid * np.cos(alpha)
    v = v_grid * np.cos(alpha) - u_grid * np.sin(alpha)
    return u, v


def wind_dir_from(u, v):
    return np.degrees(np.arctan2(-u, -v)) % 360.0


def circ_diff(a, b):
    """Signed shortest angular difference a - b, wrapped to [-180, 180]."""
    d = (a - b + 180) % 360 - 180
    return d


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    fs = s3fs.S3FileSystem(anon=True)

    all_legs = []
    print(f"{'flight':<12}{'leg':>4}{'lat':>8}{'lon':>9}{'hour(z)':>8}{'speed':>8}{'dir':>6}")
    for fid in inv.flight_ids:
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
        date = fid.split("_")[0]

        legs = np.unique(leg_id)
        rows = []
        for leg in legs:
            m = leg_id == leg
            leg_lat, leg_lon = lat[m].mean(), lon[m].mean()
            leg_t = clock_h[m].mean()
            hour = int(round(leg_t)) % 24
            iy, ix = nearest_cell(leg_lat, leg_lon)
            u_grid_arr, v_grid_arr = fetch_hour_grid(fs, date, hour)
            u_g, v_g = float(u_grid_arr[iy, ix]), float(v_grid_arr[iy, ix])
            u, v = grid_to_earth_wind(u_g, v_g, leg_lon)
            speed = float(np.hypot(u, v))
            wdir = float(wind_dir_from(u, v))
            print(f"{fid:<12}{leg:>4}{leg_lat:>8.2f}{leg_lon:>9.2f}{hour:>8}{speed:>8.2f}{wdir:>6.0f}")
            rows.append(dict(fid=fid, leg=leg, t=leg_t, u=u, v=v, speed=speed, wdir=wdir,
                              offset=bg_offset[m].mean(), post_resid=post_resid[m].mean()))
        rows.sort(key=lambda r: r["t"])
        for k in range(1, len(rows)):
            rows[k]["dspeed"] = rows[k]["speed"] - rows[k - 1]["speed"]
            rows[k]["ddir"] = circ_diff(rows[k]["wdir"], rows[k - 1]["wdir"])
        rows[0]["dspeed"] = np.nan
        rows[0]["ddir"] = np.nan
        all_legs.extend(rows)

    fids = sorted(set(r["fid"] for r in all_legs))
    offset_mean = {f: np.mean([r["offset"] for r in all_legs if r["fid"] == f]) for f in fids}
    resid_mean = {f: np.mean([r["post_resid"] for r in all_legs if r["fid"] == f]) for f in fids}
    for r in all_legs:
        r["offset_c"] = r["offset"] - offset_mean[r["fid"]]
        r["resid_c"] = r["post_resid"] - resid_mean[r["fid"]]

    have_delta = [r for r in all_legs if np.isfinite(r["ddir"])]
    print(f"\nlegs with a defined leg-to-leg delta (excludes first leg of each flight): n={len(have_delta)}")

    abs_ddir = np.array([abs(r["ddir"]) for r in have_delta])
    abs_dspeed = np.array([abs(r["dspeed"]) for r in have_delta])
    abs_offset = np.array([abs(r["offset_c"]) for r in have_delta])
    abs_resid = np.array([abs(r["resid_c"]) for r in have_delta])

    print("\n=== does a bigger leg-to-leg wind SHIFT coincide with a bigger |bias| on that leg? ===")
    for name, y in [("|leg background offset|", abs_offset), ("|posterior residual|", abs_resid)]:
        r_d, p_d = stats.pearsonr(abs_ddir, y)
        r_s, p_s = stats.pearsonr(abs_dspeed, y)
        print(f"  {name} vs |Δwind direction|: r={r_d:+.3f} p={p_d:.3g}")
        print(f"  {name} vs |Δwind speed|:     r={r_s:+.3f} p={p_s:.3g}")

    # signed leg-level regression at this finer (per-leg-location, per-leg-hour) resolution
    u_arr = np.array([r["u"] for r in all_legs])
    v_arr = np.array([r["v"] for r in all_legs])
    speed_arr = np.array([r["speed"] for r in all_legs])
    offset_c = np.array([r["offset_c"] for r in all_legs])
    resid_c = np.array([r["resid_c"] for r in all_legs])
    n = len(all_legs)
    print(f"\n=== signed leg-level regression, per-leg location+hour (n={n} legs) ===")
    for name, y in [("offset", offset_c), ("posterior residual", resid_c)]:
        r_s, p_s = stats.pearsonr(speed_arr, y)
        X = np.column_stack([np.ones(n), u_arr, v_arr])
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        rss = np.sum((y - X @ b) ** 2)
        rss0 = np.sum((y - y.mean()) ** 2)
        f_stat = ((rss0 - rss) / 2) / (rss / (n - 3))
        f_p = 1 - stats.f.cdf(f_stat, 2, n - 3)
        r2 = 1 - rss / rss0
        print(f"  {name} vs speed: r={r_s:+.3f} p={p_s:.3g}  |  vs (u,v): R²={r2:.4f} F={f_stat:.2f} p={f_p:.3g}")

    # per-flight wind-direction sequence + Δdir/|bias| scatter
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fids)))
    fid_color = {f: colors[i] for i, f in enumerate(fids)}
    for f in fids:
        rows = sorted([r for r in all_legs if r["fid"] == f], key=lambda r: r["t"])
        axes[0].plot([r["t"] for r in rows], [r["wdir"] for r in rows], "o-", color=fid_color[f], label=f)
    axes[0].set_xlabel("clock time (UTC hr)")
    axes[0].set_ylabel("HRRR 10m wind direction, per leg (deg from)")
    axes[0].set_title("Per-leg modeled wind direction across each survey")
    axes[0].legend(fontsize=7, ncol=2)

    for r in have_delta:
        axes[1].scatter(abs(r["ddir"]), abs(r["resid_c"]), color=fid_color[r["fid"]])
    axes[1].set_xlabel("|Δ wind direction| from previous leg (deg)")
    axes[1].set_ylabel("|posterior residual|, flight-demeaned (ppm)")
    axes[1].set_title("Leg-to-leg wind shift vs. |bias| on that leg")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=fid_color[f], label=f) for f in fids]
    axes[1].legend(handles=handles, fontsize=7, ncol=2)

    plt.savefig("figures/hrrr_wind_leg_change_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/hrrr_wind_leg_change_check.png")


if __name__ == "__main__":
    main()
