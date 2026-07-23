"""Of the leg-level biases found by leg_level_outlier_check.py (pooled,
correlation-aware z-scores on the POSTERIOR residual z-Hx_hat), which ones
were already there before the flux model got a chance to fit anything?

Same pooled test as leg_level_outlier_check.py, computed at both stages for
every leg:

* prior_leg_z  -- using the innovation z - H·xa (prior_modeled, saved
  directly in the bundle, independent of R by construction)
* post_leg_z   -- using the posterior residual z - H·x_hat (modeled)

against the SAME leg covariance R_leg both times (R itself doesn't change
between stages, only the mean residual does), so the two numbers are
directly comparable. A leg whose |post_leg_z| stays close to its
|prior_leg_z| is bias the flux model had no ability to explain at all; a
leg where it drops toward zero is bias the fit actually found a real
mechanism to absorb.

No re-solve; reads the joint 6-flight bundle only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from goe.config import Config
from halo_oe.background import detect_legs, _load_receptor_time
from halo_oe.io_bundle import load_inversion

import joint_correlation_sweep as J

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
SIG_THRESHOLD = 3.0


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    mdm_stddev = cfg.get_float("observations", "mdm_stddev", default=0.025)
    meas_stddev = cfg.get_float("observations", "measurement_stddev", default=0.01)
    mdm_len_km = cfg.get_float("observations", "mdm_correlation_length_km", default=1.5)

    R_all = inv.receptors
    flight_index = R_all["receptor_flight"].astype(int)

    rows = []
    for fid in inv.flight_ids:
        fi = inv.flight_ids.index(fid)
        sel = flight_index == fi
        lat_f, lon_f = R_all["receptor_lat"][sel], R_all["receptor_lon"][sel]
        z_f = R_all["enhancement"][sel]
        modeled_f = R_all["modeled"][sel]
        prior_modeled_f = R_all["prior_modeled"][sel]
        flag_f = R_all.get("outlier_flag", np.zeros_like(z_f)).astype(bool)[sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = (~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f) & np.isfinite(prior_modeled_f))
        lat, lon, z, modeled, prior_modeled, t = (
            a[good] for a in (lat_f, lon_f, z_f, modeled_f, prior_modeled_f, t_f))
        order = np.argsort(t)
        lat, lon, z, modeled, prior_modeled, t = (
            a[order] for a in (lat, lon, z, modeled, prior_modeled, t))

        legs = detect_legs(lat, lon, t)
        prior_resid = z - prior_modeled
        post_resid = z - modeled

        for lid in np.unique(legs):
            m = legs == lid
            n = int(m.sum())
            if n < 3:
                continue
            rec_dist = J.pairwise_haversine(lat[m], lon[m])
            R_leg = J.spatial_kernel(rec_dist, mdm_stddev, mdm_len_km) + meas_stddev**2 * np.eye(n)
            ones = np.ones(n)
            var_mean = float(ones @ R_leg @ ones) / n**2
            sd = np.sqrt(var_mean)

            prior_mean = float(np.mean(prior_resid[m]))
            post_mean = float(np.mean(post_resid[m]))
            prior_z = prior_mean / sd
            post_z = post_mean / sd
            pct_explained = 100 * (1 - abs(post_z) / abs(prior_z)) if prior_z != 0 else float("nan")

            rows.append(dict(fid=fid, leg=int(lid), n=n, prior_mean=prior_mean, post_mean=post_mean,
                              prior_z=prior_z, post_z=post_z, pct_explained=pct_explained))

    rows.sort(key=lambda r: -abs(r["prior_z"]))
    print(f"{'flight':>14} {'leg':>4} {'n':>5} {'prior_mean':>11} {'prior_z':>8} "
          f"{'post_mean':>10} {'post_z':>8} {'%explained':>11}")
    print("-" * 90)
    for r in rows:
        print(f"{r['fid']:>14} {r['leg']:>4} {r['n']:>5} {r['prior_mean']:>+11.4f} {r['prior_z']:>+8.2f} "
              f"{r['post_mean']:>+10.4f} {r['post_z']:>+8.2f} {r['pct_explained']:>10.1f}%")

    unexplained = [r for r in rows if abs(r["prior_z"]) > SIG_THRESHOLD and abs(r["post_z"]) > SIG_THRESHOLD]
    explained = [r for r in rows if abs(r["prior_z"]) > SIG_THRESHOLD and abs(r["post_z"]) <= SIG_THRESHOLD]
    worsened = [r for r in rows if abs(r["prior_z"]) <= SIG_THRESHOLD and abs(r["post_z"]) > SIG_THRESHOLD]

    print(f"\n{len(unexplained)} legs significant at BOTH stages (fit found nothing):")
    for r in unexplained:
        print(f"  {r['fid']} leg {r['leg']}  n={r['n']}  prior_z={r['prior_z']:+.2f} -> "
              f"post_z={r['post_z']:+.2f}  ({r['pct_explained']:.0f}% reduction)")

    print(f"\n{len(explained)} legs significant at the prior stage but resolved by the fit:")
    for r in explained:
        print(f"  {r['fid']} leg {r['leg']}  n={r['n']}  prior_z={r['prior_z']:+.2f} -> "
              f"post_z={r['post_z']:+.2f}  ({r['pct_explained']:.0f}% reduction)")

    if worsened:
        print(f"\n{len(worsened)} legs NOT significant at the prior stage but significant after fitting "
              f"(fit made it worse):")
        for r in worsened:
            print(f"  {r['fid']} leg {r['leg']}  n={r['n']}  prior_z={r['prior_z']:+.2f} -> "
                  f"post_z={r['post_z']:+.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 7))
    colors = {"20230726_1": "tab:blue", "20230726_2": "tab:orange", "20230728_1": "tab:green",
             "20230728_2": "tab:red", "20230805": "tab:purple", "20230809": "tab:brown"}
    for r in rows:
        ax.scatter(r["prior_z"], r["post_z"], c=colors.get(r["fid"], "gray"), s=40)
    lim = max(abs(r["prior_z"]) for r in rows) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="post = prior (fit explained nothing)")
    ax.axhline(SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axhline(-SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axvline(SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axvline(-SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.set_xlabel("prior-stage leg z-score (innovation, z - H·xa)")
    ax.set_ylabel("posterior-stage leg z-score (z - H·x_hat)")
    ax.set_title("Leg-level bias: prior vs. posterior\npoints near the dashed line = the fit found nothing")
    for fid, c in colors.items():
        ax.scatter([], [], c=c, label=fid)
    ax.legend(fontsize=7)
    plt.savefig("runs/leg_level_prior_vs_posterior_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> runs/leg_level_prior_vs_posterior_check.png")

    for r in rows:
        ax.scatter(abs(r["prior_z"]), abs(r["post_z"]), c=colors.get(r["fid"], "gray"), s=40)
    lim = max(abs(r["prior_z"]) for r in rows) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="post = prior (fit explained nothing)")
    ax.axhline(SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axhline(-SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axvline(SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.axvline(-SIG_THRESHOLD, color="gray", ls=":", lw=1)
    ax.set_xlabel("prior-stage leg z-score (innovation, z - H·xa)")
    ax.set_ylabel("posterior-stage leg z-score (z - H·x_hat)")
    ax.set_title("Leg-level bias: prior vs. posterior\npoints near the dashed line = the fit found nothing")
    for fid, c in colors.items():
        ax.scatter([], [], c=c, label=fid)
    ax.legend(fontsize=7)
    plt.savefig("runs/leg_level_abs_val_prior_vs_posterior_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> runs/leg_level_abs_val_prior_vs_posterior_check.png")

if __name__ == "__main__":
    main()
