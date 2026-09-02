"""Bin the per-receptor footprint land/water dataset
(figures/footprint_land_water_fractions.csv, from footprint_land_water_check.py)
by footprint water percentage, instead of a single majority/minority split
or a linear-only regression -- does the relationship look like a smooth
dose-response (steadily more negative/positive as water fraction rises), a
threshold effect (flat until some percentage, then a jump), or something
non-monotonic?

Reads the already-saved CSV (no Jacobian re-read, no S3 fetch -- this is
purely a re-analysis of footprint_land_water_check.py's saved output) and
bins three targets against footprint water fraction, both as fixed-width
0-10%/10-20%/...-100% bins (shows where the data actually is -- most
receptors are low-water-fraction, so high bins are sparse) and as ~8
equal-count quantile bins (trades bin-edge interpretability for even
statistical power across the range):

1. Post-fit posterior residual (flight-demeaned)
2. Leg background offset (flight-demeaned)
3. HRRR-HALO HPBL bias (not demeaned -- already a difference)

Run with:
    python3 scripts/footprint_land_water_bins_check.py
"""

from __future__ import annotations

import csv

import numpy as np
from scipy import stats
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV_PATH = "figures/footprint_land_water_fractions.csv"


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            rows.append(dict(
                fid=r["fid"], water_frac=float(r["water_frac"]),
                bg_offset=float(r["bg_offset"]), post_resid=float(r["post_resid"]),
                hpbl_bias=float(r["hpbl_bias"]) if r["hpbl_bias"] != "" else np.nan,
            ))
    return rows


def flight_demean(rows, key):
    fids = sorted(set(r["fid"] for r in rows))
    means = {f: np.mean([r[key] for r in rows if r["fid"] == f]) for f in fids}
    return np.array([r[key] - means[r["fid"]] for r in rows])


def bin_stats(x, y, edges):
    """Mean, SE, n per bin defined by `edges` (len n_bins+1)."""
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (x >= lo) & (x < hi) & np.isfinite(y)
        n = m.sum()
        if n == 0:
            out.append((lo, hi, n, np.nan, np.nan))
            continue
        vals = y[m]
        out.append((lo, hi, n, vals.mean(), vals.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan))
    return out


def print_table(title, rows):
    print(f"\n{title}")
    print(f"  {'range':<14}{'n':>7}{'mean':>12}{'SE':>10}")
    for lo, hi, n, mean, se in rows:
        mean_s = f"{mean:+.5f}" if np.isfinite(mean) else "  --  "
        se_s = f"{se:.5f}" if np.isfinite(se) else "  --  "
        print(f"  {lo*100:5.1f}-{hi*100:5.1f}%  {n:7d}{mean_s:>12}{se_s:>10}")


def main():
    rows = load()
    n = len(rows)
    water_frac = np.array([r["water_frac"] for r in rows])
    post_resid_c = flight_demean(rows, "post_resid")
    bg_offset_c = flight_demean(rows, "bg_offset")
    hpbl_bias = np.array([r["hpbl_bias"] for r in rows])
    print(f"loaded {n} receptors from {CSV_PATH}")
    print(f"water fraction distribution: mean={water_frac.mean():.3f}, "
          f"median={np.median(water_frac):.3f}, "
          f"p90={np.percentile(water_frac, 90):.3f}, max={water_frac.max():.3f}")

    # --- fixed-width 10% bins ---
    fixed_edges = np.arange(0, 1.01, 0.10)
    for label, y in [("post-fit residual (flight-demeaned, ppm)", post_resid_c),
                      ("leg background offset (flight-demeaned, ppm)", bg_offset_c),
                      ("HRRR - HALO HPBL bias (m)", hpbl_bias)]:
        print_table(f"=== fixed 10%-wide bins: {label} ===", bin_stats(water_frac, y, fixed_edges))

    # --- ~8 equal-count quantile bins ---
    n_qbins = 8
    q_edges = np.quantile(water_frac, np.linspace(0, 1, n_qbins + 1))
    q_edges[-1] += 1e-9   # make the top edge inclusive
    q_edges = np.unique(q_edges)
    results = {}
    for label, key, y in [("post-fit residual", "post_resid_c", post_resid_c),
                           ("leg background offset", "bg_offset_c", bg_offset_c),
                           ("HRRR - HALO HPBL bias", "hpbl_bias", hpbl_bias)]:
        rs = bin_stats(water_frac, y, q_edges)
        print_table(f"=== ~equal-count quantile bins: {label} ===", rs)
        results[key] = rs

    # trend test across quantile bins (linear regression on bin midpoints, weighted by n)
    print("\n=== linear trend across quantile-bin means (weighted by n per bin) ===")
    for label, key in [("post-fit residual", "post_resid_c"),
                        ("leg background offset", "bg_offset_c"),
                        ("HRRR - HALO HPBL bias", "hpbl_bias")]:
        rs = results[key]
        mids = np.array([(lo + hi) / 2 for lo, hi, n, m, se in rs if np.isfinite(m)])
        means = np.array([m for lo, hi, n, m, se in rs if np.isfinite(m)])
        weights = np.array([n for lo, hi, n, m, se in rs if np.isfinite(m)])
        if len(mids) > 2:
            b = np.polyfit(mids, means, 1, w=np.sqrt(weights))
            print(f"  {label:<28} slope~{b[0]:+.4f} per unit water frac (bin-mean fit, informal)")

    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for ax, label, key in [(axes[0], "post-fit residual (ppm)", "post_resid_c"),
                            (axes[1], "leg background offset (ppm)", "bg_offset_c"),
                            (axes[2], "HRRR - HALO HPBL bias (m)", "hpbl_bias")]:
        rs = results[key]
        mids = [(lo + hi) / 2 * 100 for lo, hi, n, m, se in rs]
        means = [m for lo, hi, n, m, se in rs]
        ses = [se if np.isfinite(se) else 0 for lo, hi, n, m, se in rs]
        ns = [n for lo, hi, n, m, se in rs]
        ax.errorbar(mids, means, yerr=[1.96 * s for s in ses], fmt="o-", capsize=4, color="tab:blue")
        for x, y, nn in zip(mids, means, ns):
            if np.isfinite(y):
                ax.annotate(f"n={nn}", (x, y), fontsize=7, textcoords="offset points", xytext=(0, 8))
        ax.axhline(0, color="gray", lw=0.7)
        ax.set_xlabel("footprint water fraction (%), quantile bin midpoint")
        ax.set_ylabel(label)
        ax.set_title(f"{label} vs. footprint water %\n(equal-count bins, 95% CI)")

    plt.savefig("figures/footprint_land_water_bins_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/footprint_land_water_bins_check.png")


if __name__ == "__main__":
    main()
