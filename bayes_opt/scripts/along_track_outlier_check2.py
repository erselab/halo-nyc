"""Extends along_track_outlier_check.py per two follow-up questions:

1. Full outlier count at several sigma thresholds, not an arbitrary top-5.
2. Outliers defined as LOCAL excursions, not deviations from the flight-wide
   mean/median. A point sitting on a slowly-varying but otherwise unremarkable
   stretch of background can be several sigma from the flight's *global*
   mean without being an excursion at all -- what matters is how far it sits
   from ITS OWN neighborhood. For each point, the local baseline is the
   median of same-leg samples in a window around it, EXCLUDING a small buffer
   immediately adjacent (so a multi-sample plume doesn't get folded into its
   own baseline estimate) -- then normalized by the typical local-residual
   size across the flight (robust, MAD-based).
3. A minimum length scale distinguishes ISOLATED single-sample excursions
   (likely measurement artifacts) from CLUSTERED multi-sample ones (a real
   point source is physically extended and should imprint on more than one
   consecutive sample): an extreme point counts as clustered only if a
   same-sign neighbor within a short time window also shows a local
   excursion above a lower bar.

For clustered (likely-real) events, also reports the model's Hx̂ at the same
point -- if a real event exists and the model stays flat, that IS the
transport-under-resolution signature this whole check is about, not
something to exclude; only isolated points are candidates for dropping.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")

from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion
from halo_oe.plotting import _time_binned_autocorr

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
MAX_LAG_S = 180.0
BIN_WIDTH_S = 5.0

LOCAL_WINDOW_S = 45.0     # how far out to look for the local baseline
LOCAL_BUFFER_S = 8.0      # excluded zone immediately around the point (~1-2 samples)
EXTREME_THRESH = 3.0      # local-excursion sigma defining a "candidate" extreme point
NEIGHBOR_THRESH = 1.5     # sigma a same-sign neighbor must clear to count as "clustered"
NEIGHBOR_WINDOW_S = 15.0  # how far away (elapsed time) to look for that neighbor


def local_excursion_z(z, t, window_s=LOCAL_WINDOW_S, buffer_s=LOCAL_BUFFER_S):
    """Deviation of each point from the median of its own (nearby, non-adjacent)
    same-leg neighborhood, normalized by the robust spread of those deviations
    across the whole flight. A turn gap (much larger than window_s) naturally
    keeps the window from crossing into a different leg."""
    n = len(z)
    local_resid = np.full(n, np.nan)
    for i in range(n):
        dt = np.abs(t - t[i])
        ring = (dt <= window_s) & (dt > buffer_s)
        if ring.sum() >= 3:
            local_resid[i] = z[i] - np.median(z[ring])
    mad = 1.4826 * np.nanmedian(np.abs(local_resid - np.nanmedian(local_resid)))
    return local_resid / mad, local_resid


def classify_extremes(local_z, t, thresh=EXTREME_THRESH,
                       neighbor_thresh=NEIGHBOR_THRESH, window_s=NEIGHBOR_WINDOW_S):
    idx = np.flatnonzero(np.abs(local_z) > thresh)
    clustered, isolated = [], []
    for i in idx:
        sign = np.sign(local_z[i])
        near = np.flatnonzero((np.abs(t - t[i]) <= window_s) & (np.abs(t - t[i]) > 0))
        has_partner = any(np.isfinite(local_z[j]) and np.sign(local_z[j]) == sign
                           and abs(local_z[j]) > neighbor_thresh for j in near)
        (clustered if has_partner else isolated).append(i)
    return np.array(clustered, dtype=int), np.array(isolated, dtype=int)


def main():
    inv = load_inversion(BUNDLE)
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)

    for fid in inv.flight_ids:
        i = inv.flight_ids.index(fid)
        flight_sel = flight_index == i
        lat_f, lon_f = R["receptor_lat"][flight_sel], R["receptor_lon"][flight_sel]
        z_f, modeled_f, flag_f = z_all[flight_sel], modeled_all[flight_sel], flag_all[flight_sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        z, modeled, t = z_f[good], modeled_f[good], t_f[good]
        order = np.argsort(t)
        z, modeled, t = z[order], modeled[order], t[order]

        # global (flight-wide) robust z-score, for comparison against the local one
        med, mad_g = np.median(z), 1.4826 * np.median(np.abs(z - np.median(z)))
        global_z = (z - med) / mad_g

        local_z, local_resid = local_excursion_z(z, t)

        print(f"\n=== {fid} (n={z.size}) ===")
        print(f"counts exceeding threshold -- GLOBAL (median/MAD) vs LOCAL (excursion from neighborhood):")
        print(f"{'threshold':>10} {'global':>8} {'local':>8}")
        for thr in (3, 4, 5, 6, 8):
            print(f"{thr:>10}σ {np.sum(np.abs(global_z) > thr):>8d} "
                  f"{np.nansum(np.abs(local_z) > thr):>8d}")

        clustered, isolated = classify_extremes(local_z, t)
        n_extreme = len(clustered) + len(isolated)
        print(f"of {n_extreme} points with local excursion >{EXTREME_THRESH}sigma: "
              f"{len(clustered)} CLUSTERED (real-point-source-like), "
              f"{len(isolated)} ISOLATED (single-sample, artifact-like)")
        if len(clustered):
            print("  clustered events -- z, local_z, modeled at same point:")
            for k in clustered:
                print(f"    t={t[k]:8.1f}s  z={z[k]:+.4f} (local_z={local_z[k]:+.1f}, "
                      f"global_z={global_z[k]:+.1f})  modeled={modeled[k]:+.4f}")
        if len(isolated):
            print("  isolated points -- z, local_z, global_z:")
            for k in isolated[:8]:
                print(f"    t={t[k]:8.1f}s  z={z[k]:+.4f} (local_z={local_z[k]:+.1f}, "
                      f"global_z={global_z[k]:+.1f})")

        if len(isolated) == 0:
            print("  no isolated points to drop -- skipping robustness re-check")
            continue
        keep = np.ones(z.size, dtype=bool)
        keep[isolated] = False
        z2, modeled2, t2 = z[keep], modeled[keep], t[keep]

        z_std = (z - z.mean()) / z.std()
        m_std = (modeled - modeled.mean()) / (modeled.std() + 1e-12)
        lag, ac_z = _time_binned_autocorr(z_std, t, MAX_LAG_S, BIN_WIDTH_S)
        _, ac_m = _time_binned_autocorr(m_std, t, MAX_LAG_S, BIN_WIDTH_S)

        z2_std = (z2 - z2.mean()) / z2.std()
        m2_std = (modeled2 - modeled2.mean()) / (modeled2.std() + 1e-12)
        lag2, ac_z2 = _time_binned_autocorr(z2_std, t2, MAX_LAG_S, BIN_WIDTH_S)
        _, ac_m2 = _time_binned_autocorr(m2_std, t2, MAX_LAG_S, BIN_WIDTH_S)

        keep_bins = np.isfinite(ac_z) & np.isfinite(ac_m) & np.isfinite(ac_z2) & np.isfinite(ac_m2) & (lag > 0)
        gap_orig = (ac_m - ac_z)[keep_bins]
        gap_drop = (ac_m2 - ac_z2)[keep_bins]
        lag_k = lag[keep_bins]
        short = lag_k <= 30
        print(f"  gap, ORIGINAL:              mean<=30s={np.mean(gap_orig[short]):+.3f}  "
              f"mean>30s={np.mean(gap_orig[~short]):+.3f}")
        print(f"  gap, {len(isolated)} ISOLATED DROPPED: mean<=30s={np.mean(gap_drop[short]):+.3f}  "
              f"mean>30s={np.mean(gap_drop[~short]):+.3f}")
        print(f"  z std: original={z.std():.4f}  after drop={z2.std():.4f}  "
              f"(ratio {z2.std()/z.std():.3f})")


if __name__ == "__main__":
    main()
