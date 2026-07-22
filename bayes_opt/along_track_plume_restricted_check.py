"""Phase 2 of the plan to attack hypothesis (a): restrict §13's along-track
ACF-gap check to plume-affected receptors only, instead of pooling over the
whole flight.

§13 (along_track_acf_quant.py) found `805` alone shows a clean, positive,
lag-consistent gap (ac_modeled - ac_z > 0 at every lag, meaning Hx̂
decorrelates more slowly than z) -- `809` weakly, the other four flights not
at all. §13 explicitly flagged its own biggest limitation: that gap is
computed pooled over the ENTIRE flight, most of which is far from any real
point-source encounter -- diluting real localized under-resolution against a
long track of near-zero modeled signal. This was proposed as the natural
follow-up but never built.

Reuses, rather than reimplements:
- `along_track_outlier_check2.py`'s clustered-excursion classification
  (`local_excursion_z`, `classify_extremes`) -- every real clustered
  excursion, not the landfill/WWTP-dominance-filtered subset used elsewhere.
- `joint_correlation_sweep.group_into_events` -- merges nearby clustered
  points into events (same EVENT_MERGE_GAP_S convention).
- `halo_oe.plotting._time_binned_autocorr` -- the same ACF used everywhere
  else in this investigation.

For each flight, builds the "plume-affected" receptor mask as the union of
each clustered event's members plus a time padding window (same convention
as joint_correlation_sweep's ELEVATED_PAD_S), then recomputes the same
ac_modeled - ac_z gap table as along_track_acf_quant.py on that restricted
subset, printed directly alongside the original (whole-flight, pooled)
numbers for comparison.

No re-solve, no Jacobian read at all -- everything needed (z, modeled,
elapsed time) is already in the saved bundle / flight_data files.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")

from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion
from halo_oe.plotting import _time_binned_autocorr

from along_track_outlier_check2 import classify_extremes, local_excursion_z
from joint_correlation_sweep import group_into_events

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
MAX_LAG_S = 180.0
BIN_WIDTH_S = 5.0
SHORT_LAG_CUTOFF_S = 30.0
EVENT_MERGE_GAP_S = 20.0   # same convention as joint_correlation_sweep.py
ELEVATED_PAD_S = 30.0      # padding either side of an event's own extent, to
                           # capture the rise/fall around the extreme points,
                           # not just the 1-3 points that crossed threshold


def acf_gap_table(z, modeled, t):
    """(lag, gap=ac_modeled-ac_z) for one receptor set, same convention as
    along_track_acf_quant.py -- returns None if too few points for a stddev."""
    if len(z) < 10 or z.std() == 0 or modeled.std() == 0:
        return None
    z_std = (z - z.mean()) / z.std()
    m_std = (modeled - modeled.mean()) / (modeled.std() + 1e-12)
    lag, ac_z = _time_binned_autocorr(z_std, t, MAX_LAG_S, BIN_WIDTH_S)
    _, ac_m = _time_binned_autocorr(m_std, t, MAX_LAG_S, BIN_WIDTH_S)
    keep = np.isfinite(ac_z) & np.isfinite(ac_m) & (lag > 0)
    return lag[keep], (ac_m - ac_z)[keep]


def plume_mask(z, t):
    """Union of every clustered event's own extent, padded, over this
    flight's (already good/sorted) receptors."""
    local_z, _ = local_excursion_z(z, t)
    clustered, _ = classify_extremes(local_z, t)
    mask = np.zeros(len(z), dtype=bool)
    if clustered.size == 0:
        return mask, 0
    events = group_into_events(np.sort(clustered), t, gap_s=EVENT_MERGE_GAP_S)
    for members in events:
        t0, t1 = t[members].min(), t[members].max()
        mask |= (t >= t0 - ELEVATED_PAD_S) & (t <= t1 + ELEVATED_PAD_S)
    return mask, len(events)


def main():
    inv = load_inversion(BUNDLE)
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)

    print(f"{'flight':<14} {'n_total':>7} {'n_plume':>7} {'n_events':>8}  "
          f"{'pooled<=30s':>11} {'pooled>30s':>10}  {'plume<=30s':>10} {'plume>30s':>9}  "
          f"(gap = ac_modeled - ac_z; + = model smoother/slower-decorrelating)")
    print("-" * 110)

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

        pooled = acf_gap_table(z, modeled, t)
        mask, n_events = plume_mask(z, t)

        plume = acf_gap_table(z[mask], modeled[mask], t[mask]) if mask.sum() >= 10 else None

        def fmt(tbl, short):
            if tbl is None:
                return "     n/a"
            lag_k, gap = tbl
            sel = (lag_k <= SHORT_LAG_CUTOFF_S) if short else (lag_k > SHORT_LAG_CUTOFF_S)
            return f"{np.mean(gap[sel]):>+8.3f}" if sel.any() else "     n/a"

        print(f"{fid:<14} {z.size:>7d} {mask.sum():>7d} {n_events:>8d}  "
              f"{fmt(pooled, True):>11} {fmt(pooled, False):>10}  "
              f"{fmt(plume, True):>10} {fmt(plume, False):>9}")

    print("\n(805/809 are §13's ACF-gap flights; if under-resolution is real and "
          "localized, their plume-restricted gap should get clearer/larger, not "
          "smaller, and the other four should not develop a new positive gap "
          "they didn't have pooled.)")


if __name__ == "__main__":
    main()
