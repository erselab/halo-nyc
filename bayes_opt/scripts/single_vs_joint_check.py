"""Does fitting all 6 flights jointly (one shared set of per-cell category
scale factors) meaningfully change a given flight's posterior fluxes and
residuals, compared to inverting that flight alone?

The joint inversion used throughout this investigation
(``runs/legtest_legoffset_6flight``) assumes the same underlying emission
field explains all 6 days at once. If real emissions vary day to day, that
shared fit is a compromise that could systematically mismatch every
individual flight -- indistinguishable, from inside a single joint solve,
from a missing source or a bad background. This has never been tested: every
other diagnostic in this investigation used the joint solve. This script
compares it against 6 single-flight solves (run separately by
``run_single_flight_inversions.sh``, same config otherwise) for the same
flight, receptor by receptor and category by category.

No re-solve here -- reads both sets of already-saved bundles.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from halo_oe.io_bundle import load_inversion

JOINT_BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHTS = ["20230726_1", "20230726_2", "20230728_1", "20230728_2", "20230805", "20230809"]
SINGLE_BUNDLE = lambda fid: f"runs/single_{fid}"


def joint_flight_receptors(inv, fid):
    """This flight's receptors as recorded in the joint bundle, in the joint
    bundle's own concatenation order."""
    R = inv.receptors
    fi = inv.flight_ids.index(fid)
    sel = R["receptor_flight"].astype(int) == fi
    return dict(lat=R["receptor_lat"][sel], lon=R["receptor_lon"][sel],
                z=R["enhancement"][sel], modeled=R["modeled"][sel],
                prior_modeled=R.get("prior_modeled", np.full(sel.sum(), np.nan))[sel],
                flag=R.get("outlier_flag", np.zeros_like(sel)).astype(bool)[sel])


def single_flight_receptors(inv):
    R = inv.receptors
    return dict(lat=R["receptor_lat"], lon=R["receptor_lon"], z=R["enhancement"],
                modeled=R["modeled"], prior_modeled=R.get("prior_modeled", np.full(R["enhancement"].size, np.nan)),
                flag=R.get("outlier_flag", np.zeros_like(R["enhancement"])).astype(bool))


def match_order(joint, single):
    """Confirm the joint bundle's per-flight receptor slice and the single-
    flight bundle's receptors are the same set in the same order (both trace
    back to the same Jacobian file for this flight) before comparing
    element-wise. Returns True if an exact match, else False (and prints why)."""
    if joint["lat"].size != single["lat"].size:
        print(f"    MISMATCH: joint has {joint['lat'].size} receptors, "
              f"single has {single['lat'].size} -- skipping receptor-level comparison")
        return False
    if not (np.allclose(joint["lat"], single["lat"]) and np.allclose(joint["lon"], single["lon"])):
        print("    MISMATCH: same receptor count but coordinates don't line up in order -- "
              "skipping receptor-level comparison")
        return False
    return True


def report_totals(inv):
    rep = inv.report
    return {name: (p, s) for name, p, s in zip(rep["names"], rep["posterior"], rep["posterior_stddev"])}


def main():
    joint = load_inversion(JOINT_BUNDLE)
    joint_prior_totals = {name: p for name, p in zip(joint.report["names"], joint.report["prior"])}

    summary_rows = []
    fig_scatter, ax_scatter = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    fig_map, ax_map = plt.subplots(len(FLIGHTS), 2, figsize=(11, 4.2 * len(FLIGHTS)),
                                    constrained_layout=True, squeeze=False)
    fig_flux, ax_flux = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

    per_flight_rel = {}   # for the bar chart: {category: {fid: (joint_post, single_post)}}

    for fi, fid in enumerate(FLIGHTS):
        print(f"\n=== {fid} ===")
        sd = SINGLE_BUNDLE(fid)
        if not os.path.exists(os.path.join(sd, "layout.json")):
            print(f"    no single-flight bundle at {sd} yet -- run run_single_flight_inversions.sh first")
            continue
        single = load_inversion(sd)

        jr = joint_flight_receptors(joint, fid)
        sr = single_flight_receptors(single)

        jr_good = ~jr["flag"]
        sr_good = ~sr["flag"]
        j_resid = jr["z"] - jr["modeled"]
        s_resid = sr["z"] - sr["modeled"]
        j_rms = np.sqrt(np.nanmean(j_resid[jr_good] ** 2))
        s_rms = np.sqrt(np.nanmean(s_resid[sr_good] ** 2))
        j_bias = np.nanmean(j_resid[jr_good])
        s_bias = np.nanmean(s_resid[sr_good])
        print(f"    joint:  n={jr_good.sum():4d}  bias={j_bias:+.4f}  rms={j_rms:.4f} ppm")
        print(f"    single: n={sr_good.sum():4d}  bias={s_bias:+.4f}  rms={s_rms:.4f} ppm")
        summary_rows.append((fid, jr_good.sum(), j_bias, j_rms, sr_good.sum(), s_bias, s_rms))

        # per-receptor comparison, only if the two bundles' receptor lists line up
        ax_s = ax_scatter.ravel()[fi]
        if match_order(jr, sr):
            fin = jr_good & sr_good
            corr = np.corrcoef(j_resid[fin], s_resid[fin])[0, 1] if fin.sum() > 2 else np.nan
            ax_s.scatter(j_resid[fin], s_resid[fin], s=10, alpha=0.5)
            lim = float(np.nanmax(np.abs(np.concatenate([j_resid[fin], s_resid[fin]])))) * 1.05
            ax_s.plot([-lim, lim], [-lim, lim], "k--", lw=1)
            ax_s.set_xlim(-lim, lim); ax_s.set_ylim(-lim, lim)
            ax_s.text(0.05, 0.92, f"r={corr:.3f}", transform=ax_s.transAxes, fontsize=9)

            vlim = float(np.nanmax(np.abs(np.concatenate([j_resid[fin], s_resid[fin]]))))
            # jr/sr coordinates are identical here (match_order() confirmed it), so
            # either can be used for both panels -- kept explicit for clarity anyway
            panels = [("joint (6-flight)", j_resid, jr["lon"], jr["lat"]),
                      ("single-flight", s_resid, sr["lon"], sr["lat"])]
            for a, (title, resid, lon_, lat_) in zip(ax_map[fi], panels):
                m = a.scatter(lon_[fin], lat_[fin], c=resid[fin], s=16, cmap="RdBu_r",
                              vmin=-vlim, vmax=vlim)
                a.set_title(f"{fid}: {title} residual"); a.set_xlabel("lon")
                fig_map.colorbar(m, ax=a, shrink=0.85)
            ax_map[fi][0].set_ylabel("lat")
        else:
            ax_map[fi][0].set_title(f"{fid}: joint/single receptor mismatch, no map")
        ax_s.set_xlabel("joint residual (ppm)"); ax_s.set_ylabel("single-flight residual (ppm)")
        ax_s.set_title(fid)

        # per-category posterior flux totals: prior, joint posterior, single posterior
        single_totals = report_totals(single)
        joint_totals = report_totals(joint)
        ax_f = ax_flux.ravel()[fi]
        cats = list(single.report["names"])   # per-category names, then "total"
        x = np.arange(len(cats)); w = 0.25
        prior_vals = [joint_prior_totals.get(c, np.nan) for c in cats]
        joint_vals = [joint_totals.get(c, (np.nan, np.nan))[0] for c in cats]
        joint_errs = [joint_totals.get(c, (np.nan, np.nan))[1] for c in cats]
        single_vals = [single_totals.get(c, (np.nan, np.nan))[0] for c in cats]
        single_errs = [single_totals.get(c, (np.nan, np.nan))[1] for c in cats]
        ax_f.bar(x - w, prior_vals, width=w, label="prior")
        ax_f.bar(x, joint_vals, width=w, yerr=joint_errs, capsize=3, label="joint posterior")
        ax_f.bar(x + w, single_vals, width=w, yerr=single_errs, capsize=3, label="single-flight posterior")
        ax_f.set_xticks(x); ax_f.set_xticklabels(cats, rotation=30, ha="right", fontsize=8)
        ax_f.set_title(fid); ax_f.legend(fontsize=7)
        for c, jv, sv in zip(cats, joint_vals, single_vals):
            per_flight_rel.setdefault(c, {})[fid] = (jv, sv)

    print(f"\n{'flight':>14}  {'n_joint':>7}  {'bias_joint':>10}  {'rms_joint':>9}  "
          f"{'n_single':>8}  {'bias_single':>11}  {'rms_single':>10}")
    for row in summary_rows:
        fid, nj, bj, rj, ns, bs, rs = row
        print(f"{fid:>14}  {nj:>7d}  {bj:>+10.4f}  {rj:>9.4f}  {ns:>8d}  {bs:>+11.4f}  {rs:>10.4f}")

    fig_scatter.suptitle("Per-receptor residual: joint 6-flight solve vs. that flight's own single-flight solve\n"
                          "(on the diagonal -> joint fitting changed nothing for this flight)")
    fig_scatter.savefig("figures/single_vs_joint_residual_scatter.png", bbox_inches="tight", dpi=110)
    plt.close(fig_scatter)
    fig_map.suptitle("Posterior residual maps: joint vs. single-flight solve, same color scale per flight")
    fig_map.savefig("figures/single_vs_joint_residual_maps.png", bbox_inches="tight", dpi=110)
    plt.close(fig_map)
    fig_flux.suptitle("Per-category posterior flux: prior vs. joint (6-flight) vs. single-flight solve")
    fig_flux.savefig("figures/single_vs_joint_flux_totals.png", bbox_inches="tight", dpi=110)
    plt.close(fig_flux)
    print("\nplots -> figures/single_vs_joint_residual_scatter.png, "
          "figures/single_vs_joint_residual_maps.png, figures/single_vs_joint_flux_totals.png")


if __name__ == "__main__":
    main()
