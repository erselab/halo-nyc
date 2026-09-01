"""Follow-up to `leg_offset_drift_check.py`: is the surface-elevation signal
found there actually a land/water split, and does it survive a within-leg
test (controlling for each leg's own already-fit background offset), not
just the coarser per-leg-average regression?

`UserInput/DEM_altitude` in the unclipped flight files is a clean bimodal
quantity over the NYC domain: land pixels are the real GLOBE elevation
(~10 m and up), water pixels (harbor, rivers, Long Island Sound) are a flat
-1.0 sentinel (no GLOBE/GEBCO value assigned), with essentially nothing in
between -- so a simple threshold gives an unambiguous land/water flag per
receptor (see `is_water` / `WATER_SENTINEL` in `leg_offset_drift_check.py`).

Two tests, in increasing rigor:

1. Leg-level: does each leg's water *fraction* predict its fitted offset,
   pooled across flights (flight-demeaned, as in the drift check)? Because
   the six flights fly their legs in a similar geographic order, water
   fraction by leg index tracks a similar shape to elapsed time across
   flights -- so this alone is run both on its own and jointly with elapsed
   time, to see whether it adds anything elapsed time doesn't already
   explain.
2. Receptor-level, within-leg: for every receptor, take the *pre-leg-offset*
   residual (`obs - plane_background`, the same quantity `fit_leg_offsets`
   uses internally) and subtract that leg's own mean -- this removes each
   leg's overall fitted level entirely, asking only: within a leg, do the
   water receptors and the land receptors disagree systematically? This is
   a much sharper test than (1): it doesn't need cross-flight pooling or
   comparison, isn't confounded by elapsed time or geography *between*
   legs, and uses every receptor rather than one aggregated number per leg.

Run with:
    python3 scripts/land_water_background_check.py
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, ".")

from goe.config import Config
from halo_oe.background import detect_legs, _load_receptor_time
from halo_oe.io_bundle import load_inversion
from leg_offset_drift_check import (
    BUNDLE, FLIGHT_DATA_DIR, per_leg_table, is_water, _load_receptor_surface_alt,
)


def receptor_level_test(inv, cfg, fid: str) -> dict:
    """Within-leg (land, water) residual samples for one flight."""
    R = inv.receptors
    fi = inv.flight_ids.index(fid)
    sel = R["receptor_flight"].astype(int) == fi
    lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
    obs = R["receptor_obs"][sel]
    bg_full = R["receptor_background"][sel]
    bg_offset = R["receptor_background_offset"][sel]
    plane_bg = bg_full - bg_offset
    resid = obs - plane_bg   # the pre-leg-smoothing residual fit_leg_offsets operates on

    clock_hours = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
    clock_h = clock_hours / 3600.0
    surface_m = _load_receptor_surface_alt(fid, clock_h)
    water = is_water(surface_m)

    leg_id = detect_legs(
        lat, lon, clock_hours,
        gap_seconds=cfg.get_float("background", "leg_gap_seconds", default=8.0),
        min_leg_size=cfg.get_int("background", "leg_min_size", default=10),
        axis_deg=cfg.get_float("background", "leg_axis_deg", default=45.0),
    )

    within = np.full(resid.shape, np.nan)
    for leg in np.unique(leg_id):
        m = leg_id == leg
        # need both land and water present to make a within-leg contrast meaningful
        if water[m].sum() == 0 or (~water[m]).sum() == 0:
            continue
        within[m] = resid[m] - resid[m].mean()

    keep = np.isfinite(within)
    return dict(within=within[keep], water=water[keep], leg_id=leg_id[keep])


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))

    # --- (1) leg-level water fraction ------------------------------------
    tables = {fid: per_leg_table(inv, cfg, fid) for fid in inv.flight_ids}
    pooled_elapsed, pooled_wf, pooled_centered = [], [], []
    print("=== (1) leg-level: fitted offset vs. leg water fraction ===")
    print(f"{'flight':<14}{'n_legs':>7}{'  vs water frac (ppm)':>24}{'      r':>8}{'       p':>10}")
    for fid, t in tables.items():
        centered = t["offset"] - t["offset"].mean()
        slope, intercept, r, p, se = stats.linregress(t["water_frac"], t["offset"])
        print(f"{fid:<14}{len(t['offset']):>7}{slope:>24.4f}{r:>8.3f}{p:>10.3g}")
        pooled_elapsed.append(t["elapsed_min"] / 60.0)
        pooled_wf.append(t["water_frac"])
        pooled_centered.append(centered)

    pooled_elapsed = np.concatenate(pooled_elapsed)
    pooled_wf = np.concatenate(pooled_wf)
    pooled_centered = np.concatenate(pooled_centered)

    slope, intercept, r, p, se = stats.linregress(pooled_wf, pooled_centered)
    print(f"\npooled (flight-demeaned) offset vs water fraction (alone): "
          f"slope={slope:+.4f} ppm, r={r:+.3f}, p={p:.3g}, n={pooled_wf.size} legs")
    r_ew = np.corrcoef(pooled_elapsed, pooled_wf)[0, 1]
    print(f"corr(elapsed time, water fraction) pooled: r={r_ew:+.3f}  "
          f"(legs are flown in a similar geographic order across flights)")

    X = np.column_stack([np.ones_like(pooled_elapsed), pooled_elapsed, pooled_wf])
    beta, *_ = np.linalg.lstsq(X, pooled_centered, rcond=None)
    resid_mv = pooled_centered - X @ beta
    n, k = pooled_centered.size, 2
    sigma2 = (resid_mv ** 2).sum() / (n - k - 1)
    se_mv = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tstat = beta / se_mv
    pvals = 2 * (1 - stats.t.cdf(np.abs(tstat), n - k - 1))
    print(f"joint regression offset ~ elapsed_time + water_frac:")
    print(f"  elapsed_time:  beta={beta[1]:+.4f} ppm/hr, p={pvals[1]:.3g}")
    print(f"  water_frac:    beta={beta[2]:+.4f} ppm,    p={pvals[2]:.3g}")

    # --- (2) receptor-level, within-leg -----------------------------------
    print("\n=== (2) receptor-level: within-leg residual, water vs. land ===")
    print(f"{'flight':<14}{'n_water':>9}{'n_land':>9}{'mean water':>12}{'mean land':>12}"
          f"{'diff (ppm)':>12}{'t':>8}{'p':>10}")
    all_within, all_water = [], []
    per_flight_diff, per_flight_se, per_flight_fid = [], [], []
    for fid in inv.flight_ids:
        d = receptor_level_test(inv, cfg, fid)
        w, l = d["within"][d["water"]], d["within"][~d["water"]]
        if len(w) < 5 or len(l) < 5:
            print(f"{fid:<14} -- too few water/land receptors with both classes present in a leg --")
            continue
        t, p = stats.ttest_ind(w, l, equal_var=False)
        diff = w.mean() - l.mean()
        se_diff = np.sqrt(w.var(ddof=1) / len(w) + l.var(ddof=1) / len(l))
        print(f"{fid:<14}{len(w):>9}{len(l):>9}{w.mean():>12.4f}{l.mean():>12.4f}"
              f"{diff:>12.4f}{t:>8.2f}{p:>10.3g}")
        all_within.append(d["within"])
        all_water.append(d["water"])
        per_flight_diff.append(diff)
        per_flight_se.append(se_diff)
        per_flight_fid.append(fid)

    all_within = np.concatenate(all_within)
    all_water = np.concatenate(all_water)
    w, l = all_within[all_water], all_within[~all_water]
    t, p = stats.ttest_ind(w, l, equal_var=False)
    print(f"\npooled (all flights, all legs, leg-demeaned): "
          f"n_water={len(w)}, n_land={len(l)}, "
          f"mean_water={w.mean():+.4f} ppm, mean_land={l.mean():+.4f} ppm, "
          f"diff={w.mean() - l.mean():+.4f} ppm, t={t:.2f}, p={p:.3g}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].boxplot([l, w], tick_labels=["land", "water"], showmeans=True)
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].set_ylabel("within-leg residual, obs - plane background (ppm)")
    axes[0].set_title(f"Pooled across all flights (n={len(all_within)} receptors)\n"
                       f"diff={w.mean() - l.mean():+.4f} ppm, t={t:.2f}, p={p:.3g} "
                       f"-- signs cancel, see right panel")

    x = np.arange(len(per_flight_fid))
    axes[1].bar(x, per_flight_diff, yerr=[1.96 * s for s in per_flight_se],
                color="tab:blue", capsize=4)
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(per_flight_fid, rotation=45, ha="right")
    axes[1].set_ylabel("water - land within-leg residual (ppm), 95% CI")
    axes[1].set_title("Per-flight: real, highly significant, but sign flips by flight")

    plt.tight_layout()
    plt.savefig("figures/land_water_background_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/land_water_background_check.png")


if __name__ == "__main__":
    main()
