"""Receptor-resolution version of hrrr_pblh_vs_mlh_check.py: instead of one
representative domain point standing in for a whole flight, match **every
receptor** to its own nearest HRRR grid cell (via a KDTree over the static
~1059x1799 HRRR lat/lon grid) and its own nearest HRRR analysis hour, then
compare against that receptor's own HALO lidar MLH (nearest-GPS-time
matched to the unclipped file, same convention as mlh_bias_regression_check.py
and hrrr_wind_leg_change_check.py).

This directly answers §30.2's caveat in RESIDUAL_INVESTIGATION.md: a single
point cannot represent real spatial HPBL structure across the ~250km domain,
including the land/water split that mattered for HALO's own MLH (§29.1).
With every receptor matched individually, that split can be tested directly
for HRRR too, and the flight-level bias number becomes a receptor-resolution
mean instead of a one-point sample.

To keep remote reads cheap, each unique (date, hour) HPBL grid is fetched
once (full array, cached in memory) and reused for every receptor that
falls in that hour, regardless of flight.

Run with:
    python3 scripts/hrrr_pblh_vs_mlh_receptor_check.py
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

# §20 Phase 1 footprint self-similarity decay length, for the correlation re-check
DECAY_LENGTH_KM = {
    "20230726_1": 7.9, "20230728_1": 8.6, "20230728_2": 10.0,
    "20230805": 12.1, "20230726_2": 12.9, "20230809": 19.6,
}

_HOUR_CACHE: dict[tuple[str, int], np.ndarray] = {}


def build_tree():
    with h5py.File("scratch_hrrr/HRRR_latlon.h5", "r") as f:
        lat, lon = f["latitude"][:], f["longitude"][:]
    pts = np.column_stack([lat.ravel(), lon.ravel()])
    tree = cKDTree(pts)
    return tree, lat.shape


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

    all_rows = []
    print(f"{'flight':<12}{'n_receptors':>12}{'mean HALO':>11}{'mean HRRR':>11}{'ratio':>8}{'bias':>8}")
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

        # nearest HRRR grid cell for every receptor at once
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
        halo_k, hrrr_k, water_k = halo_mlh[keep], hrrr_vals[keep], water[keep]
        mean_halo, mean_hrrr = halo_k.mean(), hrrr_k.mean()
        print(f"{fid:<12}{keep.sum():>12}{mean_halo:>11.0f}{mean_hrrr:>11.0f}"
              f"{mean_hrrr / mean_halo:>8.2f}{mean_hrrr - mean_halo:>+8.0f}")

        for h, r_, w in zip(halo_k, hrrr_k, water_k):
            all_rows.append(dict(fid=fid, halo=h, hrrr=r_, water=bool(w)))

    print(f"\ntotal receptor-level pairs: n={len(all_rows)}")

    halo_all = np.array([r["halo"] for r in all_rows])
    hrrr_all = np.array([r["hrrr"] for r in all_rows])
    water_all = np.array([r["water"] for r in all_rows])
    bias_all = hrrr_all - halo_all

    print(f"\npooled bias (all receptors, all flights): mean={bias_all.mean():+.0f}m, "
          f"median={np.median(bias_all):+.0f}m, std={bias_all.std():.0f}m")
    r_all, p_all = stats.pearsonr(halo_all, hrrr_all)
    print(f"pooled correlation HALO vs HRRR (raw values, all receptors): r={r_all:+.3f} p={p_all:.3g}")

    print("\n=== land vs water, receptor-resolution HRRR-HALO bias ===")
    for label, mask in [("land", ~water_all), ("water", water_all)]:
        b = bias_all[mask]
        print(f"  {label:6s}: n={mask.sum():6d}  mean bias={b.mean():+7.0f}m  "
              f"mean HALO={halo_all[mask].mean():7.0f}m  mean HRRR={hrrr_all[mask].mean():7.0f}m")
    t_lw, p_lw = stats.ttest_ind(bias_all[~water_all], bias_all[water_all], equal_var=False)
    print(f"  land vs water bias difference: t={t_lw:.2f} p={p_lw:.3g}")

    # per-flight receptor-resolution means, re-checked against §20 decay length
    fids = list(DECAY_LENGTH_KM.keys())
    mean_hrrr_per_flight = {f: np.mean([r["hrrr"] for r in all_rows if r["fid"] == f]) for f in fids}
    mean_halo_per_flight = {f: np.mean([r["halo"] for r in all_rows if r["fid"] == f]) for f in fids}
    decay = np.array([DECAY_LENGTH_KM[f] for f in fids])
    hrrr_x = np.array([mean_hrrr_per_flight[f] for f in fids])
    halo_x = np.array([mean_halo_per_flight[f] for f in fids])
    r_h, p_h = stats.pearsonr(hrrr_x, decay)
    r_a, p_a = stats.pearsonr(halo_x, decay)
    print(f"\nreceptor-resolution mean HRRR HPBL vs §20 decay length: r={r_h:+.3f} p={p_h:.3g}")
    print(f"receptor-resolution mean HALO MLH  vs §20 decay length: r={r_a:+.3f} p={p_a:.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(inv.flight_ids)))
    fid_color = {f: colors[i] for i, f in enumerate(inv.flight_ids)}
    for r in all_rows:
        axes[0].scatter(r["halo"], r["hrrr"], color=fid_color[r["fid"]], s=4, alpha=0.4)
    lims = [0, max(halo_all.max(), hrrr_all.max())]
    axes[0].plot(lims, lims, "k--", lw=1, label="1:1")
    axes[0].set_xlabel("HALO lidar MLH, per receptor (m)")
    axes[0].set_ylabel("HRRR HPBL, per receptor (m)")
    axes[0].set_title(f"n={len(all_rows)} receptors, all 6 flights")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=fid_color[f], label=f)
               for f in inv.flight_ids]
    axes[0].legend(handles=handles, fontsize=7, ncol=2)

    axes[1].boxplot([bias_all[~water_all], bias_all[water_all]], tick_labels=["land", "water"], showmeans=True)
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_ylabel("HRRR - HALO bias, per receptor (m)")
    axes[1].set_title(f"Land vs. water bias (t={t_lw:.2f}, p={p_lw:.3g})")

    plt.savefig("figures/hrrr_pblh_vs_mlh_receptor_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/hrrr_pblh_vs_mlh_receptor_check.png")


if __name__ == "__main__":
    main()
