"""Is the bias we see (leg background offset, §2.2/§28; posterior residual;
and the HRRR-HPBL-vs-HALO-MLH mismatch, §30) connected to whether a
receptor's *footprint* -- its upwind STILT sensitivity, not just the
aircraft's own position -- falls mostly over land or water?

This is a different, more mechanistically direct question than §28.4's land/
water split, which classified each receptor by the terrain directly under
the aircraft (`DEM_altitude` at the flight track point). A receptor flying
over land can have a footprint that's mostly upwind over water (or vice
versa) depending on wind direction that day -- if the CH4/HPBL biases are
really about the air mass sampled, footprint composition should be the
more relevant, and possibly cleaner, predictor.

Method:
1. Full-grid land/water mask, built from HRRR's static `LAND` field
   (`hrrrzarr/sfc/.../surface/LAND/surface`, same source/resolution used for
   HPBL and wind throughout this investigation -- consistency over a finer
   local product), nearest-matched via KDTree to every one of the Jacobian's
   ~1666x1666 (~1km) footprint grid cells.
2. `JacobianFile.receptor_column_sums(weights={"water": water_mask})` --
   already-built streaming infrastructure (one read per flight, ~10s for a
   12.6GB file) -- gives each receptor's total footprint-weighted sensitivity
   and its water-weighted sensitivity; the ratio is the fraction of that
   receptor's footprint sensitivity sitting over water, continuous in
   [0, 1], not a binary label.
3. Regress leg-level background offset and posterior residual (flight-
   demeaned, same convention as mlh_bias_regression_check.py) against
   leg-mean footprint water fraction; separately regress the receptor-level
   HRRR-HPBL-vs-HALO-MLH bias (§30.3's construction, recomputed here)
   against the same quantity.

Run with:
    python3 scripts/footprint_land_water_check.py
"""

from __future__ import annotations

import csv
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

from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.background import detect_legs, _load_receptor_time
from halo_oe.io_bundle import load_inversion
from leg_offset_drift_check import BUNDLE, FLIGHT_DATA_DIR
from mlh_bias_regression_check import _load_receptor_mlh

JAC_DIR = "../stilt/harvard_jacobians"
LAND_DATE_HOUR = ("20230726", 14)   # any date/hour -- LAND is a static terrain mask


def fetch_land_mask(fs):
    date_str, hour = LAND_DATE_HOUR
    path = f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/surface/LAND/surface"
    store = s3fs.S3Map(root=path, s3=fs, check=False)
    return np.asarray(xr.open_zarr(store, consolidated=False)["LAND"])


def build_hrrr_tree():
    with h5py.File("scratch_hrrr/HRRR_latlon.h5", "r") as f:
        lat, lon = f["latitude"][:], f["longitude"][:]
    pts = np.column_stack([lat.ravel(), lon.ravel()])
    return cKDTree(pts), lat.shape


def water_mask_on_jacobian_grid(jf: JacobianFile, tree, hrrr_shape, land_field) -> np.ndarray:
    glat, glon = jf.grid.cell_centers()   # length n_cells, lat-major -- matches Jacobian flatten
    _, idx = tree.query(np.column_stack([glat, glon]), workers=-1)
    iy, ix = np.unravel_index(idx, hrrr_shape)
    land = land_field[iy, ix]
    return 1.0 - land   # water = 1, land = 0


def fetch_hpbl_hour(fs, date_str: str, hour: int, cache: dict) -> np.ndarray:
    key = (date_str, hour)
    if key not in cache:
        path = f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/surface/HPBL/surface"
        store = s3fs.S3Map(root=path, s3=fs, check=False)
        cache[key] = np.asarray(xr.open_zarr(store, consolidated=False)["HPBL"])
    return cache[key]


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    fs = s3fs.S3FileSystem(anon=True)
    tree, hrrr_shape = build_hrrr_tree()

    print("fetching static HRRR land mask...")
    land_field = fetch_land_mask(fs)
    hpbl_cache: dict = {}

    all_legs = []
    all_receptors = []
    for fid in inv.flight_ids:
        print(f"\n=== {fid} ===")
        jf = JacobianFile(os.path.join(JAC_DIR, f"{fid}.nc"))
        wmask = water_mask_on_jacobian_grid(jf, tree, hrrr_shape, land_field)
        sums = jf.receptor_column_sums(active=np.arange(jf.n_cells), weights={"water": wmask})
        footprint_water_frac = sums["water"]["total"] / sums["uniform"]["total"]
        jf.close()
        print(f"  footprint water fraction: mean={footprint_water_frac.mean():.3f}, "
              f"range=[{footprint_water_frac.min():.3f}, {footprint_water_frac.max():.3f}]")

        R = inv.receptors
        fi = inv.flight_ids.index(fid)
        sel = R["receptor_flight"].astype(int) == fi
        lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
        bg_offset = R["receptor_background_offset"][sel]
        post_resid = R["enhancement"][sel] - R["modeled"][sel]
        clock_hours = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
        clock_h = clock_hours / 3600.0
        halo_mlh = _load_receptor_mlh(fid, clock_h)

        hours = np.clip(np.round(clock_h).astype(int), 0, 23)
        date = fid.split("_")[0]
        _, idx = tree.query(np.column_stack([lat, lon]), workers=-1)
        iy, ix = np.unravel_index(idx, hrrr_shape)
        hrrr_hpbl = np.full(lat.shape, np.nan)
        for h in np.unique(hours):
            grid = fetch_hpbl_hour(fs, date, int(h), hpbl_cache)
            m = hours == h
            hrrr_hpbl[m] = grid[iy[m], ix[m]]

        ok = np.isfinite(halo_mlh) & np.isfinite(hrrr_hpbl)
        for i in np.flatnonzero(ok):
            all_receptors.append(dict(
                fid=fid, receptor_idx=int(i), lat=float(lat[i]), lon=float(lon[i]),
                water_frac=float(footprint_water_frac[i]), land_frac=float(1.0 - footprint_water_frac[i]),
                bg_offset=float(bg_offset[i]), post_resid=float(post_resid[i]),
                hpbl_bias=float(hrrr_hpbl[i] - halo_mlh[i]),
            ))
        # receptors dropped only because HALO MLH or matched HRRR HPBL was unavailable
        # that hour still have a valid footprint water/land fraction -- record those too,
        # with hpbl_bias left blank, so the saved dataset covers every receptor in every flight
        for i in np.flatnonzero(~ok):
            all_receptors.append(dict(
                fid=fid, receptor_idx=int(i), lat=float(lat[i]), lon=float(lon[i]),
                water_frac=float(footprint_water_frac[i]), land_frac=float(1.0 - footprint_water_frac[i]),
                bg_offset=float(bg_offset[i]), post_resid=float(post_resid[i]),
                hpbl_bias="",
            ))

        leg_id = detect_legs(
            lat, lon, clock_hours,
            gap_seconds=cfg.get_float("background", "leg_gap_seconds", default=8.0),
            min_leg_size=cfg.get_int("background", "leg_min_size", default=10),
            axis_deg=cfg.get_float("background", "leg_axis_deg", default=45.0),
        )
        for leg in np.unique(leg_id):
            m = leg_id == leg
            all_legs.append(dict(
                fid=fid, leg=leg, water_frac=footprint_water_frac[m].mean(),
                offset=bg_offset[m].mean(), post_resid=post_resid[m].mean(),
            ))

    # --- leg-level: footprint water fraction vs. CH4 background bias ---
    fids = sorted(set(r["fid"] for r in all_legs))
    offset_mean = {f: np.mean([r["offset"] for r in all_legs if r["fid"] == f]) for f in fids}
    resid_mean = {f: np.mean([r["post_resid"] for r in all_legs if r["fid"] == f]) for f in fids}
    for r in all_legs:
        r["offset_c"] = r["offset"] - offset_mean[r["fid"]]
        r["resid_c"] = r["post_resid"] - resid_mean[r["fid"]]

    wf_leg = np.array([r["water_frac"] for r in all_legs])
    offset_c = np.array([r["offset_c"] for r in all_legs])
    resid_c = np.array([r["resid_c"] for r in all_legs])
    n_leg = len(all_legs)

    print(f"\n\n=== leg-level (n={n_leg}, flight-demeaned): footprint water fraction vs. CH4 bias ===")
    r_o, p_o = stats.pearsonr(wf_leg, offset_c)
    slope_o, *_ = stats.linregress(wf_leg, offset_c)
    print(f"  leg background offset:  r={r_o:+.3f} p={p_o:.3g}  slope={slope_o:+.4f} ppm per unit water frac")
    r_r, p_r = stats.pearsonr(wf_leg, resid_c)
    slope_r, *_ = stats.linregress(wf_leg, resid_c)
    print(f"  posterior residual:     r={r_r:+.3f} p={p_r:.3g}  slope={slope_r:+.4f} ppm per unit water frac")

    # --- save the full per-receptor land/water footprint dataset ---
    csv_path = "figures/footprint_land_water_fractions.csv"
    fieldnames = ["fid", "receptor_idx", "lat", "lon", "water_frac", "land_frac",
                  "bg_offset", "post_resid", "hpbl_bias"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(all_receptors)
    print(f"\nsaved per-receptor land/water footprint fractions -> {csv_path} ({len(all_receptors)} rows)")

    has_hpbl = np.array([r["hpbl_bias"] != "" for r in all_receptors])
    with_hpbl = [r for r in all_receptors if r["hpbl_bias"] != ""]

    # --- receptor-level: footprint water fraction vs. HRRR-HALO HPBL bias ---
    wf_rec = np.array([r["water_frac"] for r in with_hpbl])
    hpbl_bias = np.array([r["hpbl_bias"] for r in with_hpbl])
    n_rec = len(with_hpbl)
    r_h, p_h = stats.pearsonr(wf_rec, hpbl_bias)
    slope_h, *_ = stats.linregress(wf_rec, hpbl_bias)
    print(f"\n=== receptor-level (n={n_rec}): footprint water fraction vs. HRRR-HALO HPBL bias ===")
    print(f"  r={r_h:+.3f} p={p_h:.3g}  slope={slope_h:+.1f} m per unit water frac")

    # --- receptor-level: footprint water fraction vs. the actual POST-FIT posterior
    # residual (z - Hx_hat), continuous regression AND a majority-water-vs-majority-land
    # split (parallel to §28.4's binary receptor-position test, but using footprint
    # composition and the true post-fit residual instead of the pre-fit background one) ---
    wf_all = np.array([r["water_frac"] for r in all_receptors])
    resid_all = np.array([r["post_resid"] for r in all_receptors])
    fid_all = np.array([r["fid"] for r in all_receptors])
    fids_u = sorted(set(fid_all))
    resid_c_all = resid_all.copy()
    for f in fids_u:
        m = fid_all == f
        resid_c_all[m] = resid_all[m] - resid_all[m].mean()

    r_pr, p_pr = stats.pearsonr(wf_all, resid_c_all)
    print(f"\n=== receptor-level (n={len(all_receptors)}, flight-demeaned): "
          f"footprint water fraction vs. POST-FIT posterior residual ===")
    print(f"  continuous: r={r_pr:+.3f} p={p_pr:.3g}")

    print(f"  majority-water (frac>0.5) vs. majority-land (frac<0.5), per flight:")
    all_maj_water, all_maj_land = [], []
    for f in fids_u:
        m = fid_all == f
        maj_water = resid_c_all[m][wf_all[m] > 0.5]
        maj_land = resid_c_all[m][wf_all[m] <= 0.5]
        if len(maj_water) < 5 or len(maj_land) < 5:
            print(f"    {f:<12} -- too few majority-water receptors (n={len(maj_water)}) --")
            continue
        t, p = stats.ttest_ind(maj_water, maj_land, equal_var=False)
        print(f"    {f:<12} n_water={len(maj_water):5d} n_land={len(maj_land):5d}  "
              f"mean_water={maj_water.mean():+.4f}  mean_land={maj_land.mean():+.4f}  "
              f"diff={maj_water.mean()-maj_land.mean():+.4f}  t={t:+.2f}  p={p:.3g}")
        all_maj_water.append(maj_water); all_maj_land.append(maj_land)
    if all_maj_water:
        aw, al = np.concatenate(all_maj_water), np.concatenate(all_maj_land)
        t, p = stats.ttest_ind(aw, al, equal_var=False)
        print(f"    {'POOLED':<12} n_water={len(aw):5d} n_land={len(al):5d}  "
              f"mean_water={aw.mean():+.4f}  mean_land={al.mean():+.4f}  "
              f"diff={aw.mean()-al.mean():+.4f}  t={t:+.2f}  p={p:.3g}")

    # per-flight breakdown for both
    print("\n=== per-flight breakdown ===")
    for f in fids:
        legs_f = [r for r in all_legs if r["fid"] == f]
        wf_f = np.array([r["water_frac"] for r in legs_f])
        off_f = np.array([r["offset_c"] for r in legs_f])
        r_f, p_f = stats.pearsonr(wf_f, off_f) if len(wf_f) > 2 else (np.nan, np.nan)
        print(f"  {f:<12} mean footprint water frac (legs)={wf_f.mean():.3f}  "
              f"offset vs water_frac: r={r_f:+.3f} p={p_f:.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fids)))
    fid_color = {f: colors[i] for i, f in enumerate(fids)}
    for r in all_legs:
        axes[0].scatter(r["water_frac"], r["offset_c"], color=fid_color[r["fid"]], s=25)
    axes[0].axhline(0, color="gray", lw=0.5)
    axes[0].set_xlabel("leg-mean footprint water fraction")
    axes[0].set_ylabel("leg background offset, flight-demeaned (ppm)")
    axes[0].set_title(f"n={n_leg} legs; r={r_o:+.2f}, p={p_o:.2g}")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=fid_color[f], label=f) for f in fids]
    axes[0].legend(handles=handles, fontsize=7, ncol=2)

    axes[1].scatter(wf_rec, hpbl_bias, s=3, alpha=0.15, color="tab:purple")
    xx = np.linspace(0, 1, 50)
    b = np.polyfit(wf_rec, hpbl_bias, 1)
    axes[1].plot(xx, np.polyval(b, xx), "k--", lw=1.5)
    axes[1].axhline(0, color="gray", lw=0.5)
    axes[1].set_xlabel("receptor footprint water fraction")
    axes[1].set_ylabel("HRRR - HALO HPBL bias (m)")
    axes[1].set_title(f"n={n_rec} receptors; r={r_h:+.2f}, p={p_h:.2g}")

    plt.savefig("figures/footprint_land_water_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/footprint_land_water_check.png")


if __name__ == "__main__":
    main()
