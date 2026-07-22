"""Footprint cosine similarity as a joint function of (space, time) separation,
across all 6 flights.

Extends §7.3/§7.4's one-off footprint-cosine comparisons (a single hand-picked
peak/dip pair; a single leg-interior baseline number) into the full picture:
for every receptor pair in a flight, compute cosine similarity between their
full (native-grid, not core-restricted -- matching §7.4's convention, since
the whole footprint including its low-sensitivity tail matters for shape
comparison) footprint rows, together with their great-circle distance and
elapsed-time gap, then bin into a 2D (distance, time-gap) grid.

This directly answers "how similar are two receptors' footprints, as a
function of how far apart and how long apart they are" -- separating the
trivial "nearby points look similar" effect from the real question (does the
footprint at a fixed location change between passes at different times),
which is the prerequisite for testing a shared rotation-parameter correction
(§17/§18 in the investigation doc).

Phase 1 of the plan to attack hypothesis (a) -- systematic footprint-shape
errors beyond rotation: §13 found `805` (and weakly `809`) uniquely shows
`Hx̂` decorrelating more slowly along-track than `z`, at every lag, while the
other 4 flights don't. Since `Hx̂` is a linear functional of the footprint
against a smooth density field, a footprint that itself stays similar over a
*longer* along-track distance would mechanically produce exactly that
signature. This script now runs all 6 flights (not just 726_1/805) and
extracts one number per flight -- the along-track half-max decay length,
pooled over all time gaps since §17 already found time gap barely matters at
fixed distance -- so `805`/`809` can be directly compared against the other
four rather than eyeballed pairwise.

Cost note: for each flight, materializes the full (n_receptors x n_cells)
footprint matrix (~15GB at float32 for a ~1400-receptor flight, freed before
moving to the next flight) and one matrix multiply for the full pairwise
cosine similarity -- a few minutes per flight, not cheap like the rest of
this investigation's diagnostics, but a bounded, one-time cost (~6x the
original 2-flight run).

Run with the `analysis` conda env from the bayes_opt directory:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 footprint_similarity_space_time.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km
from adapters.jacobian_operator import JacobianFile
from halo_oe.background import _load_receptor_time

JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
FLIGHTS = ["20230726_1", "20230726_2", "20230728_1", "20230728_2", "20230805", "20230809"]

DIST_BIN_KM = 5.0
DIST_MAX_KM = 150.0
TIME_BIN_S = 300.0        # 5 min
MIN_PAIRS_PER_BIN = 5     # bins with fewer pairs than this are masked as unreliable


def load_footprints(fid: str):
    """Full (n_receptors, n_cells) native-grid footprint matrix, float32, plus
    each receptor's lat/lon and elapsed time (sorted by time)."""
    jf = JacobianFile(os.path.join(JAC_DIR, f"{fid}.nc"))
    lat, lon = jf.receptor_lat, jf.receptor_lon
    t = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
    order = np.argsort(t)
    lat, lon, t = lat[order], lon[order], t[order]

    t0 = time.time()
    active_all = np.arange(jf.n_cells)
    H = jf._materialize(active_all, row_chunk=16, dtype=np.float32)[order]
    jf.close()
    print(f"  {fid}: materialized {H.shape} footprint matrix "
          f"({H.nbytes / 1e9:.1f} GB) in {time.time() - t0:.0f}s")
    return H, lat, lon, t


def pairwise_cosine(H: np.ndarray) -> np.ndarray:
    """Full (n, n) cosine similarity matrix via one matrix multiply."""
    norms = np.linalg.norm(H, axis=1)
    norms[norms == 0] = np.nan   # zero-norm rows -> similarity undefined (NaN), not 0
    t0 = time.time()
    Hn = H / norms[:, None]
    sim = Hn @ Hn.T
    print(f"    pairwise cosine similarity ({sim.shape}) in {time.time() - t0:.0f}s")
    return sim.astype(np.float64)


def analyze_flight(fid: str):
    H, lat, lon, t = load_footprints(fid)
    n = H.shape[0]
    sim = pairwise_cosine(H)
    del H

    dist = _haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    dt = np.abs(t[:, None] - t[None, :])

    iu = np.triu_indices(n, k=1)   # upper triangle, excludes self-pairs and double-counting
    sim_flat, dist_flat, dt_flat = sim[iu], dist[iu], dt[iu]
    finite = np.isfinite(sim_flat)
    sim_flat, dist_flat, dt_flat = sim_flat[finite], dist_flat[finite], dt_flat[finite]

    dist_bins = np.arange(0, DIST_MAX_KM + DIST_BIN_KM, DIST_BIN_KM)
    time_max = dt_flat.max()
    time_bins = np.arange(0, time_max + TIME_BIN_S, TIME_BIN_S)

    di = np.clip(np.digitize(dist_flat, dist_bins) - 1, 0, len(dist_bins) - 2)
    ti = np.clip(np.digitize(dt_flat, time_bins) - 1, 0, len(time_bins) - 2)

    grid_mean = np.full((len(dist_bins) - 1, len(time_bins) - 1), np.nan)
    grid_n = np.zeros_like(grid_mean, dtype=int)
    flat_idx = di * (len(time_bins) - 1) + ti
    sums = np.bincount(flat_idx, weights=sim_flat, minlength=grid_mean.size)
    counts = np.bincount(flat_idx, minlength=grid_mean.size)
    grid_n = counts.reshape(grid_mean.shape)
    with np.errstate(invalid="ignore", divide="ignore"):
        grid_mean = (sums / np.maximum(counts, 1)).reshape(grid_mean.shape)
    grid_mean[grid_n < MIN_PAIRS_PER_BIN] = np.nan

    # 1D distance-only curve, pooled over all time gaps (§17 found time gap
    # barely matters at fixed distance) -- the number Phase 1 actually needs,
    # comparable across flights without reading 6 separate 2D heatmaps.
    sums_1d = np.bincount(di, weights=sim_flat, minlength=len(dist_bins) - 1)
    counts_1d = np.bincount(di, minlength=len(dist_bins) - 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sim_1d = sums_1d / np.maximum(counts_1d, 1)
    sim_1d[counts_1d < MIN_PAIRS_PER_BIN] = np.nan

    print(f"  {fid}: n={n} receptors, {len(sim_flat)} pairs, "
          f"max time gap {time_max:.0f}s, max dist {dist_flat.max():.0f}km")
    return dict(fid=fid, dist_bins=dist_bins, time_bins=time_bins,
                grid_mean=grid_mean, grid_n=grid_n, sim_1d=sim_1d,
                n_receptors=n, n_pairs=len(sim_flat),
                max_time_gap_s=float(time_max), max_dist_km=float(dist_flat.max()))


def half_max_decay_length(dist_bins, sim_1d):
    """Distance at which the along-track similarity curve first crosses half
    of its own short-distance value (linear interpolation between bin
    centers) -- same half-max convention as §14b's empirical MDM decay
    estimate, for direct comparability. NaN if the curve never has a finite
    starting value or never decays to half of it within the tested range."""
    dist_centers = 0.5 * (dist_bins[:-1] + dist_bins[1:])
    valid = np.isfinite(sim_1d)
    if not valid.any():
        return float("nan")
    d, s = dist_centers[valid], sim_1d[valid]
    if s[0] <= 0:
        return float("nan")
    half = 0.5 * s[0]
    below = np.flatnonzero(s <= half)
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(d[0])
    d0, d1, s0, s1 = d[i - 1], d[i], s[i - 1], s[i]
    frac = (s0 - half) / (s0 - s1) if s0 != s1 else 0.0
    return float(d0 + frac * (d1 - d0))


def _grid_shape(n):
    ncols = 3 if n > 2 else n
    nrows = int(np.ceil(n / ncols))
    return nrows, ncols


def main():
    results = []
    for fid in FLIGHTS:
        print(f"\n=== {fid} ===")
        results.append(analyze_flight(fid))

    nrows, ncols = _grid_shape(len(results))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 5.5 * nrows),
                              constrained_layout=True, squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        im = ax.pcolormesh(r["time_bins"] / 60, r["dist_bins"], r["grid_mean"],
                           cmap="viridis", vmin=0, vmax=1, shading="flat")
        ax.set_xlabel("time gap (min)")
        ax.set_ylabel("distance (km)")
        ax.set_title(r["fid"])
        plt.colorbar(im, ax=ax, label="mean footprint cosine similarity")
    for ax in axes.ravel()[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Footprint similarity vs. (distance, time gap) between receptor pairs")
    plt.savefig("runs/footprint_similarity_space_time.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> runs/footprint_similarity_space_time.png")

    # a couple of useful 1D slices per flight, for direct comparison
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows),
                              constrained_layout=True, squeeze=False)
    for ax, r in zip(axes.ravel(), results):
        time_centers = 0.5 * (r["time_bins"][:-1] + r["time_bins"][1:]) / 60
        dist_centers = 0.5 * (r["dist_bins"][:-1] + r["dist_bins"][1:])
        # "within-leg-like" slice: shortest distance bin, similarity vs time
        ax.plot(time_centers, r["grid_mean"][0, :], "o-", label=f"dist ~{dist_centers[0]:.0f}km")
        if len(dist_centers) > 3:
            ax.plot(time_centers, r["grid_mean"][3, :], "s-", label=f"dist ~{dist_centers[3]:.0f}km")
        if len(dist_centers) > 8:
            ax.plot(time_centers, r["grid_mean"][8, :], "^-", label=f"dist ~{dist_centers[8]:.0f}km")
        ax.set_xlabel("time gap (min)")
        ax.set_ylabel("mean footprint cosine similarity")
        ax.set_title(r["fid"])
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
    for ax in axes.ravel()[len(results):]:
        ax.set_visible(False)
    fig.suptitle("Similarity vs. time gap, at fixed distance slices")
    plt.savefig("runs/footprint_similarity_time_slices.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> runs/footprint_similarity_time_slices.png")

    # --- Phase 1 deliverable: along-track (distance-only, time-pooled) decay
    # curves overlaid, and the half-max decay length compared across flights ---
    decay_lengths = {r["fid"]: half_max_decay_length(r["dist_bins"], r["sim_1d"]) for r in results}
    print("\nalong-track footprint half-max decay length (pooled over all time gaps):")
    for fid, dl in decay_lengths.items():
        flag = "  <-- §13 ACF-gap flight" if fid in ("20230805", "20230809") else ""
        print(f"  {fid:>14}: {dl:6.1f} km{flag}")

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in results:
        dist_centers = 0.5 * (r["dist_bins"][:-1] + r["dist_bins"][1:])
        style = "-o" if r["fid"] in ("20230805", "20230809") else "--."
        ax.plot(dist_centers, r["sim_1d"], style, label=r["fid"], lw=2 if r["fid"] in
                ("20230805", "20230809") else 1, ms=4)
    ax.set_xlabel("distance (km)"); ax.set_ylabel("mean footprint cosine similarity")
    ax.set_title("Along-track footprint decay, pooled over all time gaps\n"
                 "(solid = the two flights §13 flagged with the ACF-gap signature)")
    ax.legend(fontsize=8)
    plt.savefig("runs/footprint_similarity_decay_1d.png", bbox_inches="tight", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    fids = list(decay_lengths)
    vals = [decay_lengths[f] for f in fids]
    colors = ["tab:red" if f in ("20230805", "20230809") else "tab:blue" for f in fids]
    ax.bar(range(len(fids)), vals, color=colors)
    ax.set_xticks(range(len(fids))); ax.set_xticklabels(fids, rotation=30, ha="right")
    ax.set_ylabel("half-max decay length (km)")
    ax.set_title("Phase 1: is 805/809's footprint intrinsically broader than the other flights?\n"
                 "(red = §13's ACF-gap flights)")
    plt.savefig("runs/footprint_similarity_decay_length.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plots -> runs/footprint_similarity_decay_1d.png, "
          "runs/footprint_similarity_decay_length.png")

    # --- guaranteed-clean text artifact, independent of how stdout is captured
    # when this runs asynchronously (sbatch, nohup, ...) -- everything needed to
    # analyze the Phase 1 result without re-running the (expensive) job ---
    summary_path = "runs/footprint_similarity_phase1_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Phase 1 (attack hypothesis (a)): per-flight footprint decay comparison\n")
        f.write("=" * 78 + "\n\n")
        f.write(f"{'flight':<14} {'n_receptors':>11} {'n_pairs':>10} "
                f"{'max_time_gap_s':>14} {'max_dist_km':>11}\n")
        for r in results:
            f.write(f"{r['fid']:<14} {r['n_receptors']:>11d} {r['n_pairs']:>10d} "
                    f"{r['max_time_gap_s']:>14.0f} {r['max_dist_km']:>11.0f}\n")
        f.write("\nalong-track footprint half-max decay length (pooled over all time gaps):\n")
        for fid, dl in decay_lengths.items():
            flag = "  <-- Sec13 ACF-gap flight" if fid in ("20230805", "20230809") else ""
            f.write(f"  {fid:>14}: {dl:6.1f} km{flag}\n")
        f.write("\nPrediction being tested: 20230805 and 20230809 should show a measurably\n"
                "longer decay length than the other four flights. If they don't, footprint\n"
                "breadth isn't what differentiates them, and hypothesis (a) should be\n"
                "deprioritized relative to (b)/(c) rather than pursued further.\n")
        f.write("\nRaw 1D decay curve per flight (distance_km, mean_cosine_similarity), "
                "for re-plotting or re-deriving a different decay metric without re-running:\n")
        for r in results:
            dist_centers = 0.5 * (r["dist_bins"][:-1] + r["dist_bins"][1:])
            f.write(f"\n{r['fid']}:\n")
            for d, s in zip(dist_centers, r["sim_1d"]):
                f.write(f"  {d:7.1f}  {s:8.4f}\n" if np.isfinite(s) else f"  {d:7.1f}       nan\n")
    print(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
