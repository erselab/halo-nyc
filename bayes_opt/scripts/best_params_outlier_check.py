"""Compare the 12 single-flight bundles from run_single_flight_best_params.sh
(leg offsets on, mdm_correlation_length_km=2.25, outlier filtering off vs on)
against each other and against the earlier single-flight baseline
(runs/single_<fid>, mdm_correlation_length_km=1.5, no outlier filter) --
so the "best params" change and the outlier on/off change can each be
attributed separately, not conflated.

No re-solve; reads the saved bundles only.
"""

from __future__ import annotations

import sys

import numpy as np

sys.path.insert(0, ".")

from halo_oe.io_bundle import load_inversion

FLIGHTS = ["20230726_1", "20230726_2", "20230728_1", "20230728_2", "20230805", "20230809"]
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other", "total"]


def residual_stats(inv):
    R = inv.receptors
    z, modeled = R["enhancement"], R["modeled"]
    flag = R.get("outlier_flag", np.zeros_like(z)).astype(bool)
    good = ~flag & np.isfinite(z) & np.isfinite(modeled)
    resid = (z - modeled)[good]
    return dict(n=int(good.sum()), n_flagged=int(flag.sum()), n_total=int(z.size),
                bias=float(np.mean(resid)), rms=float(np.sqrt(np.mean(resid**2))))


def flux_scales(inv):
    rep = inv.report
    prior = dict(zip(rep["names"], rep["prior"]))
    post = dict(zip(rep["names"], rep["posterior"]))
    return {c: (post[c] / prior[c] if c in prior and prior[c] else float("nan")) for c in CATEGORIES if c in prior}


def main():
    print(f"{'flight':>14}  {'n_flag(on)':>10}  "
          f"{'bias_off':>9} {'rms_off':>8}  {'bias_on':>9} {'rms_on':>8}  "
          f"{'bias_old':>9} {'rms_old':>8}   (old = mdm_len=1.5km, no outlier filter)")
    print("-" * 110)

    scale_rows = {"off": {}, "on": {}, "old": {}}
    for fid in FLIGHTS:
        inv_off = load_inversion(f"runs/single_{fid}_bestparams_outlier_off")
        inv_on = load_inversion(f"runs/single_{fid}_bestparams_outlier_on")
        inv_old = load_inversion(f"runs/single_{fid}")

        s_off = residual_stats(inv_off)
        s_on = residual_stats(inv_on)
        s_old = residual_stats(inv_old)

        print(f"{fid:>14}  {s_on['n_flagged']:>10d}  "
              f"{s_off['bias']:>+9.4f} {s_off['rms']:>8.4f}  "
              f"{s_on['bias']:>+9.4f} {s_on['rms']:>8.4f}  "
              f"{s_old['bias']:>+9.4f} {s_old['rms']:>8.4f}")

        scale_rows["off"][fid] = flux_scales(inv_off)
        scale_rows["on"][fid] = flux_scales(inv_on)
        scale_rows["old"][fid] = flux_scales(inv_old)

    for cat in CATEGORIES:
        print(f"\n=== category scale factor (posterior/prior): {cat} ===")
        print(f"{'flight':>14}  {'off (2.25km)':>13}  {'on (2.25km)':>13}  {'old (1.5km)':>13}")
        for fid in FLIGHTS:
            vals = [scale_rows[v][fid].get(cat, float("nan")) for v in ("off", "on", "old")]
            print(f"{fid:>14}  {vals[0]:>13.3f}  {vals[1]:>13.3f}  {vals[2]:>13.3f}")


if __name__ == "__main__":
    main()
