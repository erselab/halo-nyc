"""Why doesn't the production outlier filter flag legs that visibly stand
out in the posterior residual map?

Because it's a strictly per-receptor test (goe/outliers.py: each receptor's
normalized residual is compared only to its own expected uncertainty, never
pooled with its leg-mates). A leg that's coherently offset by a moderate
amount -- every receptor maybe 1-2 sigma individually -- never crosses a 3
sigma per-point threshold, even though that same moderate offset repeated
across an entire leg is a far stronger signal in aggregate (pooling N
receptors shrinks the effective uncertainty by ~sqrt(N) for independent
noise, though real along-track MDM correlation reduces that effective N --
which is exactly why the pooling has to use the real R, not just average
naively).

This builds the aggregate test itself: for every detected leg, in every
flight, test whether the leg's MEAN posterior residual (z - Hx_hat, the
actual quantity in every residual map this investigation has looked at) is
different from zero given the real along-track-correlated R -- and reports
it next to each leg's own worst individual per-receptor normalized residual,
so the contrast is direct: legs that are decisively significant in
aggregate but have no individual receptor anywhere near the production
filter's 3-sigma cutoff.

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
INDIVIDUAL_THRESHOLD = 3.0   # same convention as the production outlier filter


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    mdm_stddev = cfg.get_float("observations", "mdm_stddev", default=0.025)
    meas_stddev = cfg.get_float("observations", "measurement_stddev", default=0.01)
    mdm_len_km = cfg.get_float("observations", "mdm_correlation_length_km", default=1.5)
    diag_R = mdm_stddev**2 + meas_stddev**2   # constant across receptors (kernel diagonal)

    R_all = inv.receptors
    flight_index = R_all["receptor_flight"].astype(int)

    rows = []
    for fid in inv.flight_ids:
        fi = inv.flight_ids.index(fid)
        sel = flight_index == fi
        lat_f, lon_f = R_all["receptor_lat"][sel], R_all["receptor_lon"][sel]
        z_f, modeled_f = R_all["enhancement"][sel], R_all["modeled"][sel]
        flag_f = R_all.get("outlier_flag", np.zeros_like(z_f)).astype(bool)[sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        lat, lon, z, modeled, t = (a[good] for a in (lat_f, lon_f, z_f, modeled_f, t_f))
        order = np.argsort(t)
        lat, lon, z, modeled, t = (a[order] for a in (lat, lon, z, modeled, t))

        legs = detect_legs(lat, lon, t)
        resid = z - modeled
        # per-receptor normalized residual, same "posterior kind" convention as
        # goe.outliers (z - Hx_hat)/sqrt(diag R); diag(R) is a constant here
        nr_individual = resid / np.sqrt(diag_R)

        for lid in np.unique(legs):
            m = legs == lid
            n = int(m.sum())
            if n < 3:
                continue
            r_leg = resid[m]
            rec_dist = J.pairwise_haversine(lat[m], lon[m])
            R_leg = J.spatial_kernel(rec_dist, mdm_stddev, mdm_len_km) + meas_stddev**2 * np.eye(n)

            mean_r = float(np.mean(r_leg))
            ones = np.ones(n)
            var_mean = float(ones @ R_leg @ ones) / n**2
            var_mean_independent = diag_R / n   # naive, ignoring correlation -- for contrast
            leg_z = mean_r / np.sqrt(var_mean)

            max_individual = float(np.max(np.abs(nr_individual[m])))
            n_individually_flagged = int(np.sum(np.abs(nr_individual[m]) > INDIVIDUAL_THRESHOLD))

            rows.append(dict(fid=fid, leg=int(lid), n=n, mean_resid=mean_r,
                              leg_z=leg_z, max_individual_z=max_individual,
                              n_flagged=n_individually_flagged,
                              correlation_penalty=np.sqrt(var_mean / var_mean_independent)))

    rows.sort(key=lambda r: -abs(r["leg_z"]))
    print(f"{'flight':>14} {'leg':>4} {'n':>5} {'mean_resid':>11} {'leg_z':>8} "
          f"{'max_indiv_z':>12} {'n_flagged(3sig)':>16} {'corr_penalty':>13}")
    print("-" * 100)
    for r in rows:
        print(f"{r['fid']:>14} {r['leg']:>4} {r['n']:>5} {r['mean_resid']:>+11.4f} "
              f"{r['leg_z']:>+8.2f} {r['max_individual_z']:>12.2f} {r['n_flagged']:>16d} "
              f"{r['correlation_penalty']:>13.2f}")

    striking = [r for r in rows if abs(r["leg_z"]) > 3 and r["max_individual_z"] < 3]
    print(f"\n{len(striking)} legs are decisively significant in aggregate (|leg_z|>3) "
          f"while NO individual receptor in them crosses the production filter's "
          f"{INDIVIDUAL_THRESHOLD}-sigma threshold:")
    for r in striking:
        print(f"  {r['fid']}  leg {r['leg']}  n={r['n']}  mean_resid={r['mean_resid']:+.4f} ppm  "
              f"leg_z={r['leg_z']:+.2f}  (worst individual point only {r['max_individual_z']:.2f}sigma)")

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["tab:red" if abs(r["leg_z"]) > 3 and r["max_individual_z"] < INDIVIDUAL_THRESHOLD
             else "tab:blue" for r in rows]
    ax.scatter([r["max_individual_z"] for r in rows], [abs(r["leg_z"]) for r in rows], c=colors, s=40)
    ax.axhline(3, color="k", ls="--", lw=1, label="leg_z = 3")
    ax.axvline(INDIVIDUAL_THRESHOLD, color="k", ls=":", lw=1, label=f"per-receptor threshold = {INDIVIDUAL_THRESHOLD}")
    ax.set_xlabel("worst individual per-receptor |normalized residual| in the leg")
    ax.set_ylabel("|leg-mean z-score| (pooled, correlation-aware)")
    ax.set_title("Legs significant in aggregate (red) vs. individually (right of dotted line)\n"
                 "top-left = exactly the blind spot: obvious in the posterior, invisible to the filter")
    ax.legend(fontsize=8)
    plt.savefig("runs/leg_level_outlier_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> runs/leg_level_outlier_check.png")


if __name__ == "__main__":
    main()
