"""Check whether along_track_scale_check.py's ACF/structure-function results
are sensitive to a handful of large point-source excursions in z, per the
question: a few large amplitude outliers could (a) inflate the ACF's
per-series variance normalization, distorting the whole curve, not just
near the outlier, and (b) dominate individual structure-function lag bins
via the squared-difference term, especially at longer lags where a bin has
fewer eligible pairs.

Reports, per flight: how extreme the top |z| values are (robust z-score),
the min/median pair-count per lag bin (bins with few pairs are the ones a
single extreme pair could dominate), and a robustness re-check -- same ACF
computed after dropping the most extreme few receptors -- for 805, 726_2
(the two flights that showed the strong, similar-looking signal) and 809
(weak signal), to see whether that comparison survives.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")

from goe.config import Config
from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion
from halo_oe.plotting import _time_binned_autocorr

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
MAX_LAG_S = 180.0
BIN_WIDTH_S = 5.0
CHECK_FLIGHTS = ["20230805", "20230726_2", "20230809"]
N_DROP = 5   # how many of the most extreme |z| points to drop for the robustness re-check


def counts_per_bin(t, max_lag_s=MAX_LAG_S, bin_width_s=BIN_WIDTH_S):
    n = len(t)
    nbins = max(1, int(np.ceil(max_lag_s / bin_width_s)))
    counts = np.zeros(nbins, dtype=int)
    for i in range(n):
        j = i + 1
        while j < n and t[j] - t[i] <= max_lag_s:
            b = min(int((t[j] - t[i]) / bin_width_s), nbins - 1)
            counts[b] += 1
            j += 1
    return counts


def main():
    inv = load_inversion(BUNDLE)
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)

    for fid in CHECK_FLIGHTS:
        i = inv.flight_ids.index(fid)
        flight_sel = flight_index == i
        lat_f, lon_f = R["receptor_lat"][flight_sel], R["receptor_lon"][flight_sel]
        z_f, modeled_f, flag_f = z_all[flight_sel], modeled_all[flight_sel], flag_all[flight_sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        z, modeled, t = z_f[good], modeled_f[good], t_f[good]
        order = np.argsort(t)
        z, modeled, t = z[order], modeled[order], t[order]

        med = np.median(z)
        mad = 1.4826 * np.median(np.abs(z - med))
        robust_z = (z - med) / mad
        top = np.argsort(-np.abs(robust_z))[:N_DROP]

        print(f"\n=== {fid} (n={z.size}) ===")
        print(f"z: median={med:.4f}  MAD-sigma={mad:.4f}  max|z|={np.max(np.abs(z)):.4f}  "
              f"n(|robust_z|>4)={np.sum(np.abs(robust_z) > 4)}  n(|robust_z|>6)={np.sum(np.abs(robust_z) > 6)}")
        print(f"top {N_DROP} most extreme z values (robust z-score): " +
              ", ".join(f"{z[k]:.3f}(rz={robust_z[k]:.1f})" for k in top))

        counts = counts_per_bin(t)
        nz_counts = counts[counts > 0]
        print(f"pairs per lag bin: min={counts.min()} median={int(np.median(counts))} "
              f"max={counts.max()}  (bins with <10 pairs: {(counts[:36] < 10).sum()}/36 in first 180s)")

        # original ACF
        z_std = (z - z.mean()) / z.std()
        m_std = (modeled - modeled.mean()) / (modeled.std() + 1e-12)
        lag, ac_z = _time_binned_autocorr(z_std, t, MAX_LAG_S, BIN_WIDTH_S)
        _, ac_m = _time_binned_autocorr(m_std, t, MAX_LAG_S, BIN_WIDTH_S)

        # robustness re-check: drop the N_DROP most extreme |z| receptors and recompute
        keep = np.ones(z.size, dtype=bool)
        keep[top] = False
        z2, modeled2, t2 = z[keep], modeled[keep], t[keep]
        z2_std = (z2 - z2.mean()) / z2.std()
        m2_std = (modeled2 - modeled2.mean()) / (modeled2.std() + 1e-12)
        lag2, ac_z2 = _time_binned_autocorr(z2_std, t2, MAX_LAG_S, BIN_WIDTH_S)
        _, ac_m2 = _time_binned_autocorr(m2_std, t2, MAX_LAG_S, BIN_WIDTH_S)

        keep_bins = np.isfinite(ac_z) & np.isfinite(ac_m) & np.isfinite(ac_z2) & np.isfinite(ac_m2) & (lag > 0)
        gap_orig = (ac_m - ac_z)[keep_bins]
        gap_drop = (ac_m2 - ac_z2)[keep_bins]
        lag_k = lag[keep_bins]
        short = lag_k <= 30
        print(f"gap (ac_modeled - ac_z), ORIGINAL:      mean<=30s={np.mean(gap_orig[short]):+.3f}  "
              f"mean>30s={np.mean(gap_orig[~short]):+.3f}")
        print(f"gap (ac_modeled - ac_z), {N_DROP} DROPPED:  mean<=30s={np.mean(gap_drop[short]):+.3f}  "
              f"mean>30s={np.mean(gap_drop[~short]):+.3f}")
        print(f"z std: original={z.std():.4f}  after drop={z2.std():.4f}  "
              f"(ratio {z2.std()/z.std():.3f})")


if __name__ == "__main__":
    main()
