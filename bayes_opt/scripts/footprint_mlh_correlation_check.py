"""Does the per-flight footprint self-similarity decay length (§20 Phase 1 --
built purely from STILT footprint cosine similarity, no meteorology involved)
correlate with per-flight boundary-layer depth (mixed_layer_height_check.py's
land-only MLH, built purely from the lidar backscatter product, no footprint
involved)? Two independently-derived per-flight quantities agreeing would be
much stronger evidence than either alone that boundary-layer depth is a real
driver of §20's footprint-under-resolution finding, rather than an unrelated
coincidence that happens to also flag `809`.

Only 6 data points (one per flight) -- inherently low-powered -- so every
correlation below is reported with an explicit leave-one-out check dropping
`809` (the extreme point in both variables) to distinguish "a real trend
across all 6 flights" from "809 is unusual in two unrelated ways."

MLH summary statistics recomputed directly here (not imported) using the
same land-only, nearest-GPS-time-matched convention as
mixed_layer_height_check.py and mlh_bias_regression_check.py.

Run with:
    python3 scripts/footprint_mlh_correlation_check.py
"""

from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FULL_DIR = "../flight_data"
CLIP_DIR = "scratch_flight_data_1000m"
WATER_SENTINEL = -0.5

# §20 Phase 1 (footprint_similarity_space_time.py, extended to all 6 flights)
DECAY_LENGTH_KM = {
    "20230726_1": 7.9, "20230728_1": 8.6, "20230728_2": 10.0,
    "20230805": 12.1, "20230726_2": 12.9, "20230809": 19.6,
}


def mlh_summary(fid: str) -> dict:
    date, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    full = sorted(glob.glob(os.path.join(FULL_DIR, f"*{date}*_F{fnum}_full.h5")))[0]
    clip = sorted(glob.glob(os.path.join(CLIP_DIR, f"*{date}*_F{fnum}_full_1000m.h5")))[0]
    with h5py.File(full, "r") as f:
        t = f["Nav_Data/gps_time"][:, 0]
        mlh = f["CH4DataProducts/MixedLayerHeight"][:, 0]
        dem = f["UserInput/DEM_altitude"][:, 0]
    tc = h5py.File(clip, "r")["time"][:]
    tc0, tc1 = tc.min(), tc.max()
    land = dem > WATER_SENTINEL
    water = ~land
    in_window = (t >= tc0) & (t <= tc1) & np.isfinite(mlh)
    land_mlh = mlh[in_window & land]
    water_mlh = mlh[in_window & water]

    nbins = max(3, int((tc1 - tc0) * 6))
    edges = np.linspace(tc0, tc1, nbins + 1)
    tw, mw = t[in_window & land], mlh[in_window & land]
    idx = np.clip(np.digitize(tw, edges) - 1, 0, nbins - 1)
    bin_t = 0.5 * (edges[:-1] + edges[1:])
    bin_med = np.array([np.median(mw[idx == k]) if np.any(idx == k) else np.nan for k in range(nbins)])
    ok = np.isfinite(bin_med)
    slope, *_ = stats.linregress(bin_t[ok], bin_med[ok])

    return dict(
        mean_land_mlh=np.mean(land_mlh), std_land_mlh=np.std(land_mlh),
        max_land_mlh=np.max(land_mlh), range_land_mlh=np.nanmax(bin_med) - np.nanmin(bin_med),
        slope_land_mlh=slope, land_water_gap=np.mean(land_mlh) - np.mean(water_mlh),
    )


def main():
    fids = list(DECAY_LENGTH_KM.keys())
    rows = {fid: mlh_summary(fid) for fid in fids}
    decay = np.array([DECAY_LENGTH_KM[f] for f in fids])

    print(f"{'flight':<12}{'decay(km)':>10}{'mean_land':>11}{'std_land':>10}{'max_land':>10}"
          f"{'slope':>9}{'l-w gap':>9}")
    for f in fids:
        r = rows[f]
        print(f"{f:<12}{DECAY_LENGTH_KM[f]:>10.1f}{r['mean_land_mlh']:>11.0f}{r['std_land_mlh']:>10.0f}"
              f"{r['max_land_mlh']:>10.0f}{r['slope_land_mlh']:>+9.1f}{r['land_water_gap']:>9.0f}")

    exclude = fids.index("20230809")
    keep = [i for i in range(len(fids)) if i != exclude]
    print(f"\n{'metric':<18}{'r (all 6)':>12}{'p (all 6)':>12}{'r (excl. 809)':>16}{'p (excl. 809)':>16}"
          f"{'spearman rho':>15}{'spearman p':>12}")
    for key in ["mean_land_mlh", "std_land_mlh", "max_land_mlh", "range_land_mlh",
                "slope_land_mlh", "land_water_gap"]:
        x = np.array([rows[f][key] for f in fids])
        r_all, p_all = stats.pearsonr(x, decay)
        r_lo, p_lo = stats.pearsonr(x[keep], decay[keep])
        rho, p_rho = stats.spearmanr(x, decay)
        print(f"{key:<18}{r_all:>+12.3f}{p_all:>12.3g}{r_lo:>+16.3f}{p_lo:>16.3g}{rho:>+15.3f}{p_rho:>12.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fids)))
    for ax, key, xlabel in [
        (axes[0], "mean_land_mlh", "mean land-only MLH during survey (m)"),
        (axes[1], "max_land_mlh", "max land-only MLH during survey (m)"),
    ]:
        x = np.array([rows[f][key] for f in fids])
        r, p = stats.pearsonr(x, decay)
        for i, f in enumerate(fids):
            ax.scatter(x[i], decay[i], color=colors[i], s=70, label=f)
        b = np.polyfit(x, decay, 1)
        xx = np.linspace(x.min(), x.max(), 50)
        ax.plot(xx, np.polyval(b, xx), "k--", lw=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("footprint self-similarity half-max decay length (km, §20)")
        ax.set_title(f"r={r:+.2f}, p={p:.3g}")
        ax.legend(fontsize=8)
    plt.savefig("figures/footprint_mlh_correlation.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/footprint_mlh_correlation.png")


if __name__ == "__main__":
    main()
