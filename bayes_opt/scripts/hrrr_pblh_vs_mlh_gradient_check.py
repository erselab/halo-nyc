"""Beyond §30.3's binary land/water split: is there a continuous spatial
gradient in the HRRR HPBL - HALO MLH mismatch across the domain -- with
latitude, longitude, distance from the coast (approximated by terrain
elevation, per §28.3's precedent of using DEM_altitude as a coastal-
proximity proxy), or urban/rural character?

Extends hrrr_pblh_vs_mlh_receptor_check.py's per-receptor matching
(each receptor to its own nearest HRRR grid cell/hour) by also keeping each
receptor's lat/lon and DEM elevation, instead of collapsing straight to a
per-flight or land/water-binary summary.

Run with:
    python3 scripts/hrrr_pblh_vs_mlh_gradient_check.py
"""

from __future__ import annotations

import os
import sys

import h5py
import numpy as np
import s3fs
import xarray as xr
from scipy.spatial import cKDTree
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from goe.config import Config
from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion
from leg_offset_drift_check import BUNDLE, FLIGHT_DATA_DIR, is_water, _load_receptor_surface_alt
from mlh_bias_regression_check import _load_receptor_mlh

_HOUR_CACHE: dict[tuple[str, int], np.ndarray] = {}


def build_tree():
    with h5py.File("scratch_hrrr/HRRR_latlon.h5", "r") as f:
        lat, lon = f["latitude"][:], f["longitude"][:]
    pts = np.column_stack([lat.ravel(), lon.ravel()])
    return cKDTree(pts), lat.shape


def fetch_hpbl_hour(fs, date_str: str, hour: int) -> np.ndarray:
    key = (date_str, hour)
    if key not in _HOUR_CACHE:
        path = f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/surface/HPBL/surface"
        store = s3fs.S3Map(root=path, s3=fs, check=False)
        _HOUR_CACHE[key] = np.asarray(xr.open_zarr(store, consolidated=False)["HPBL"])
    return _HOUR_CACHE[key]


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    fs = s3fs.S3FileSystem(anon=True)
    tree, grid_shape = build_tree()

    rows = []
    for fid in inv.flight_ids:
        R = inv.receptors
        fi = inv.flight_ids.index(fid)
        sel = R["receptor_flight"].astype(int) == fi
        lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
        clock_hours = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
        clock_h = clock_hours / 3600.0
        surface_m = _load_receptor_surface_alt(fid, clock_h)
        water = is_water(surface_m)
        halo_mlh = _load_receptor_mlh(fid, clock_h)
        ok = np.isfinite(halo_mlh)

        _, idx = tree.query(np.column_stack([lat, lon]))
        iy, ix = np.unravel_index(idx, grid_shape)
        hours = np.clip(np.round(clock_h).astype(int), 0, 23)
        date = fid.split("_")[0]

        hrrr_vals = np.full(lat.shape, np.nan)
        for h in np.unique(hours[ok]):
            grid = fetch_hpbl_hour(fs, date, int(h))
            m = ok & (hours == h)
            hrrr_vals[m] = grid[iy[m], ix[m]]

        keep = ok & np.isfinite(hrrr_vals)
        for la, lo, s, w, h_, r_ in zip(lat[keep], lon[keep], surface_m[keep], water[keep],
                                          halo_mlh[keep], hrrr_vals[keep]):
            rows.append(dict(fid=fid, lat=la, lon=lo, surface=s, water=bool(w), halo=h_, hrrr=r_, bias=r_ - h_))
        print(f"{fid}: {keep.sum()} receptors matched")

    lat_a = np.array([r["lat"] for r in rows])
    lon_a = np.array([r["lon"] for r in rows])
    surf_a = np.array([r["surface"] for r in rows])
    water_a = np.array([r["water"] for r in rows])
    bias_a = np.array([r["bias"] for r in rows])
    n = len(rows)
    print(f"\ntotal n={n}")

    print("\n=== pooled linear gradients (all receptors) ===")
    for name, x in [("latitude", lat_a), ("longitude", lon_a)]:
        r_, p_ = stats.pearsonr(x, bias_a)
        slope, intercept, *_ = stats.linregress(x, bias_a)
        print(f"  bias vs {name:10s}: r={r_:+.3f} p={p_:.3g}  slope={slope:+.1f} m/deg")

    print("\n=== land-only: bias vs. terrain elevation (coastal-proximity proxy) ===")
    land = ~water_a
    surf_land = surf_a[land]
    bias_land = bias_a[land]
    r_e, p_e = stats.pearsonr(surf_land, bias_land)
    slope_e, intercept_e, *_ = stats.linregress(surf_land, bias_land)
    print(f"  bias vs elevation: r={r_e:+.3f} p={p_e:.3g}  slope={slope_e:+.3f} m_bias/m_elev  n={land.sum()}")
    # quadratic, since a coastal fringe effect (bias worst very close to sea level, fading inland)
    # would be nonlinear
    X2 = np.column_stack([np.ones(land.sum()), surf_land, surf_land ** 2])
    b2, *_ = np.linalg.lstsq(X2, bias_land, rcond=None)
    rss2 = np.sum((bias_land - X2 @ b2) ** 2)
    X1 = np.column_stack([np.ones(land.sum()), surf_land])
    b1, *_ = np.linalg.lstsq(X1, bias_land, rcond=None)
    rss1 = np.sum((bias_land - X1 @ b1) ** 2)
    dfree = land.sum() - 3
    f_stat = ((rss1 - rss2) / 1) / (rss2 / dfree)
    f_p = 1 - stats.f.cdf(f_stat, 1, dfree)
    print(f"  quadratic-in-elevation term: p={f_p:.3g} (tests whether the elevation relationship is curved)")

    print("\n=== land-only: bias vs. lat/lon separately ===")
    for name, x in [("latitude", lat_a[land]), ("longitude", lon_a[land])]:
        r_, p_ = stats.pearsonr(x, bias_land)
        slope, *_ = stats.linregress(x, bias_land)
        print(f"  land bias vs {name:10s}: r={r_:+.3f} p={p_:.3g}  slope={slope:+.1f} m/deg")

    # spatial map: mean bias per ~0.1deg grid cell, pooled over all flights
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)

    sc = axes[0].scatter(lon_a, lat_a, c=bias_a, cmap="RdBu_r", vmin=-500, vmax=500, s=4, alpha=0.5)
    plt.colorbar(sc, ax=axes[0], label="HRRR - HALO bias (m)")
    axes[0].set_xlabel("longitude"); axes[0].set_ylabel("latitude")
    axes[0].set_title(f"Per-receptor bias, all 6 flights (n={n})")

    axes[1].scatter(surf_land, bias_land, s=3, alpha=0.15, color="tab:brown")
    xx = np.linspace(surf_land.min(), surf_land.max(), 100)
    axes[1].plot(xx, b1[0] + b1[1] * xx, "k--", lw=1.5, label="linear")
    axes[1].plot(xx, b2[0] + b2[1] * xx + b2[2] * xx ** 2, "r-", lw=1.5, label=f"quadratic (p={f_p:.2g})")
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_xlabel("terrain elevation under receptor (m)")
    axes[1].set_ylabel("HRRR - HALO bias (m)")
    axes[1].set_title("Land-only: bias vs. elevation")
    axes[1].legend(fontsize=8)

    binned_bias_map(axes[2], lon_a, lat_a, bias_a, bin_deg=0.15, min_count=10,
                     title="Binned mean bias (0.15deg cells, pooled)")

    plt.savefig("figures/hrrr_pblh_vs_mlh_gradient_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/hrrr_pblh_vs_mlh_gradient_check.png")

    # --- faceted by flight (not by day -- 726_1/726_2 and 728_1/728_2 kept separate) ---
    fig, axes = plt.subplots(2, 3, figsize=(16, 10), constrained_layout=True)
    for ax, fid in zip(axes.ravel(), inv.flight_ids):
        m = np.array([r["fid"] == fid for r in rows])
        binned_bias_map(ax, lon_a[m], lat_a[m], bias_a[m], bin_deg=0.2, min_count=5,
                         title=f"{fid} (n={m.sum()})")
    plt.savefig("figures/hrrr_pblh_vs_mlh_gradient_by_flight.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/hrrr_pblh_vs_mlh_gradient_by_flight.png")

    # --- morning vs. afternoon (per §29.2's grouping) ---
    MORNING = {"20230726_1", "20230728_1", "20230805"}
    AFTERNOON = {"20230726_2", "20230728_2", "20230809"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), constrained_layout=True)
    for ax, group, label in [(axes[0], MORNING, "Morning (726_1, 728_1, 805)"),
                              (axes[1], AFTERNOON, "Afternoon (726_2, 728_2, 809)")]:
        m = np.array([r["fid"] in group for r in rows])
        binned_bias_map(ax, lon_a[m], lat_a[m], bias_a[m], bin_deg=0.15, min_count=10,
                         title=f"{label} (n={m.sum()})")
    plt.savefig("figures/hrrr_pblh_vs_mlh_gradient_morning_afternoon.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/hrrr_pblh_vs_mlh_gradient_morning_afternoon.png")

    print("\n=== per-flight lat/lon gradient (for comparison across flights) ===")
    for fid in inv.flight_ids:
        m = np.array([r["fid"] == fid for r in rows])
        r_lat, p_lat = stats.pearsonr(lat_a[m], bias_a[m])
        r_lon, p_lon = stats.pearsonr(lon_a[m], bias_a[m])
        print(f"  {fid:<12} vs lat: r={r_lat:+.3f} p={p_lat:.3g}   vs lon: r={r_lon:+.3f} p={p_lon:.3g}")


def binned_bias_map(ax, lon, lat, bias, bin_deg, min_count, title, vmin=-300, vmax=300):
    if len(lon) == 0:
        ax.set_title(f"{title} -- no data")
        return
    lat_bins = np.arange(lat.min(), lat.max() + bin_deg, bin_deg)
    lon_bins = np.arange(lon.min(), lon.max() + bin_deg, bin_deg)
    H_sum, xedges, yedges = np.histogram2d(lon, lat, bins=[lon_bins, lat_bins], weights=bias)
    H_cnt, *_ = np.histogram2d(lon, lat, bins=[lon_bins, lat_bins])
    with np.errstate(invalid="ignore"):
        H_mean = np.where(H_cnt >= min_count, H_sum / H_cnt, np.nan)
    im = ax.pcolormesh(xedges, yedges, H_mean.T, cmap="RdBu_r", vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label="mean bias (m)")
    ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    ax.set_title(title)


if __name__ == "__main__":
    main()
