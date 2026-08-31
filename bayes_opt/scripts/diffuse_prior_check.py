"""Check whether the still-open residual clusters (RESIDUAL_INVESTIGATION.md
§9: 728_2, 805, 809) sit over cells whose prior emission is thinly split across
multiple categories -- which, in `category_fields` mode, gives those cells an
artificially TIGHT absolute-flux prior uncertainty relative to their total
density (see the conversation: Sa is block-diagonal across categories, each
with its own relative uncertainty rel_k applied to ITS OWN density, so
Var(total flux) = sum_k rel_k^2 e_k^2 shrinks by up to 1/sqrt(N) when a fixed
total is split evenly over N categories). If the unexplained clusters
preferentially sit over "diffuse" (low-concentration-ratio) cells, that's a
plausible, distinct mechanism (separate from the zero-prior-mass cases in
§5/§9, and separate from the buffer/domain-truncation check already ruled
out) for why the inversion can't correct an underestimate there.

No Jacobian read -- pure bundle read (group_fields), same cost class as
buffer_bias_check.py's Phase 0.

Run with the `analysis` conda env from the bayes_opt directory:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 scripts/diffuse_prior_check.py
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other"]

# (label, lat, lon, radius_km, note) -- §9 clusters plus two §5/§7 reference
# points: 726_1's peak (explained: a real natural_gas sub-cluster
# misallocation -- expect HIGH concentration) and dip (unexplained, but
# footprint-shape is already ruled out as the cause).
LOCATIONS = [
    ("726_1 peak (explained, natural_gas)", 40.896213, -73.199949, 10, "positive control"),
    ("726_1 dip (unexplained)", 41.022026, -72.834690, 10, "footprint-shape ruled out"),
    ("728_2 cluster0 (-0.096ppm, ~64km)", 40.770, -73.218, 32, "EXCEEDS corr-length reach"),
    ("728_2 cluster1 (+0.054ppm, ~46km)", 40.507, -74.141, 23, "zero prior mass (§9)"),
    ("805 cluster0 (-0.039ppm, ~30km)", 40.692, -74.320, 15, "EXCEEDS corr-length reach"),
    ("805 cluster2 (+0.036ppm, ~15km)", 40.800, -74.365, 8, "near 'other' mass"),
    ("809 cluster0 (-0.046ppm, ~15km)", 40.719, -74.486, 8, "EXCEEDS corr-length reach"),
    ("809 cluster1 (-0.056ppm, ~10km)", 40.583, -74.636, 5, "zero prior mass (§9)"),
]


def main():
    inv = load_inversion(BUNDLE)
    E = np.stack([inv.group_fields[c] for c in CATEGORIES])   # (4, n_active)
    e_total = E.sum(axis=0)
    sum_sq = (E ** 2).sum(axis=0)                               # rel_k = 1.0 for all -> Var = sum_sq
    nz = e_total > 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        conc = np.where(nz, sum_sq / e_total ** 2, np.nan)        # 1.0 = concentrated, 1/N = split N ways
        abs_sd_ratio = np.where(nz, np.sqrt(sum_sq) / e_total, np.nan)  # abs stddev vs concentrated-equiv

    print(f"domain-wide baseline (n={nz.sum()} nonzero cells):")
    print(f"  concentration ratio  p10/p50/p90 = "
          f"{np.nanpercentile(conc[nz], [10,50,90])}")
    print(f"  abs-sd ratio         p10/p50/p90 = "
          f"{np.nanpercentile(abs_sd_ratio[nz], [10,50,90])}")

    def report(radius_override=None):
        label_col = "location" if radius_override is None else f"location (tight {radius_override}km)"
        print(f"\n{label_col:<38} {'n_nz':>5} {'sum(e)':>9} {'conc_mean':>10} "
              f"{'conc_min':>9} {'sdratio_mean':>13} {'sdratio_min':>12}")
        print("-" * 100)
        out = []
        for label, clat, clon, radius, note in LOCATIONS:
            r = radius_override if radius_override is not None else radius
            d = _haversine_km(inv.core.active_lat, inv.core.active_lon, clat, clon)
            near = (d <= r) & nz
            n = near.sum()
            if n == 0:
                print(f"{label:<38} {'--':>5}  no nonzero cells within {r}km  [{note}]")
                out.append((label, clat, clon, r, note, n, 0, np.nan, np.nan, np.nan, np.nan))
                continue
            se = e_total[near].sum()
            cmean, cmin = np.nanmean(conc[near]), np.nanmin(conc[near])
            smean, smin = np.nanmean(abs_sd_ratio[near]), np.nanmin(abs_sd_ratio[near])
            print(f"{label:<38} {n:>5} {se:>9.3g} {cmean:>10.3f} {cmin:>9.3f} "
                  f"{smean:>13.3f} {smin:>12.3f}   [{note}]")
            out.append((label, clat, clon, r, note, n, se, cmean, cmin, smean, smin))
        return out

    report(radius_override=5)   # apples-to-apples tight radius, exact-centroid comparison
    results = report()          # cluster-scale radius (original, per-location)

    # overview map
    fig, ax = plt.subplots(figsize=(9, 8))
    sc = ax.scatter(inv.core.active_lon[nz], inv.core.active_lat[nz], c=conc[nz],
                     cmap="viridis_r", s=3, vmin=0.25, vmax=1.0)
    plt.colorbar(sc, ax=ax, label="concentration ratio (1=one category, 0.25=split 4 ways)")
    for label, clat, clon, radius, note, *_ in results:
        ax.scatter([clon], [clat], marker="x", color="red", s=60)
        ax.annotate(label.split(" (")[0], (clon, clat), fontsize=7)
    ax.set_title("Prior concentration ratio, with §5/§9 reference locations")
    plt.savefig("figures/diffuse_prior_overview.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/diffuse_prior_overview.png")


if __name__ == "__main__":
    main()
