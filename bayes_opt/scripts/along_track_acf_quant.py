"""Quantitative follow-up to along_track_scale_check.py's ACF plot: the
visual read ("805/809/728_1 show the model decorrelating more slowly")
needs a direct numeric check per the investigation's own §14.6 rule -- a
plot impression, especially one with curves crossing, isn't trustworthy on
sight alone.

Recomputes the same per-flight along-track ACF (obs z vs modeled Hx̂) and
reports, per flight: the sign and magnitude of (ac_modeled - ac_z) at each
lag bin, where it crosses zero, and separate short-lag (<=30s, closest to
the fine/resolved-scale question) vs long-lag (>30s) mean gaps -- instead of
a single qualitative "which curve is on top" impression.
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, ".")

from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.background import domain_insensitive_mask, _load_receptor_time
from halo_oe.io_bundle import load_inversion
from halo_oe.plotting import _time_binned_autocorr

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
MAX_LAG_S = 180.0
BIN_WIDTH_S = 5.0
SHORT_LAG_CUTOFF_S = 30.0


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)

    print(f"{'flight':<14} {'n_crossings':>11} {'first_crossing_s':>17} "
          f"{'mean_gap<=30s':>14} {'mean_gap>30s':>13}  (gap = ac_modeled - ac_z; + = model smoother)")
    print("-" * 90)

    for i, fid in enumerate(inv.flight_ids):
        flight_sel = flight_index == i
        lat_f = R["receptor_lat"][flight_sel]
        lon_f = R["receptor_lon"][flight_sel]
        z_f = z_all[flight_sel]
        modeled_f = modeled_all[flight_sel]
        flag_f = flag_all[flight_sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        z, modeled, t = z_f[good], modeled_f[good], t_f[good]
        order = np.argsort(t)
        z, modeled, t = z[order], modeled[order], t[order]

        z_std = (z - z.mean()) / z.std()
        m_std = (modeled - modeled.mean()) / (modeled.std() + 1e-12)
        lag, ac_z = _time_binned_autocorr(z_std, t, MAX_LAG_S, BIN_WIDTH_S)
        _, ac_m = _time_binned_autocorr(m_std, t, MAX_LAG_S, BIN_WIDTH_S)

        keep = np.isfinite(ac_z) & np.isfinite(ac_m) & (lag > 0)   # drop the synthetic lag=0 point
        lag_k, gap = lag[keep], (ac_m - ac_z)[keep]

        sign = np.sign(gap)
        crossings = np.sum(sign[1:] != sign[:-1])
        first_cross = lag_k[1:][sign[1:] != sign[:-1]]
        first_cross_s = f"{first_cross[0]:.0f}" if len(first_cross) else "none"

        short = lag_k <= SHORT_LAG_CUTOFF_S
        long = ~short
        print(f"{fid:<14} {crossings:>11d} {first_cross_s:>17} "
              f"{np.mean(gap[short]):>14.3f} {np.mean(gap[long]):>13.3f}")

        # full per-lag printout for this flight
        print(f"    lag(s): " + " ".join(f"{v:6.0f}" for v in lag_k))
        print(f"    gap   : " + " ".join(f"{v:+6.2f}" for v in gap))


if __name__ == "__main__":
    main()
