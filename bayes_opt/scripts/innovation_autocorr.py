#!/usr/bin/env python3
"""Along-track autocorrelation of the prior innovation, per flight.

The innovation ``d_b = z − H xa`` (observed enhancement minus prior-modeled
enhancement) is **independent of the observation-error model R** — unlike the
posterior residual ``z − H x̂``, whose structure depends on R through the gain
``K = Sa Hᵀ (H Sa Hᵀ + R)⁻¹``. Its along-track autocorrelation therefore probes
the coherence length of the model-data mismatch (representation + transport
error, plus any prior-flux signal the inversion has not yet removed) without the
circularity of tuning R against a residual R itself shaped.

Two views per flight, because receptor spacing is not perfectly uniform (turns,
altitude legs, dropouts break the index↔distance proportionality):

  * autocorrelation vs **sample-index lag** — how many receptors apart;
  * autocorrelation vs **along-track distance (km)** — the physically comparable
    axis across flights, and a proxy for elapsed observation time at roughly
    constant ground speed.

Caveat: ``d_b`` mixes error with real flux signal the prior missed, and both are
smooth along-track, so the estimated length is an *upper bound* on the pure
error correlation length. Read it comparatively (e.g. morning vs afternoon), not
as an absolute number.

This reads saved inversion bundles (``run_halo.py --save``), which now store the
prior-modeled enhancement ``H xa`` needed for the innovation. Regenerate any
bundle written before that field existed.

Usage (from bayes_opt/):
    python3 scripts/innovation_autocorr.py runs/20230726_1 runs/20230726_2
    python3 scripts/innovation_autocorr.py runs/*                 # every bundle under runs/
    python3 scripts/innovation_autocorr.py runs/* --out acf.png --max-dist 40 --nbins 40
    python3 scripts/innovation_autocorr.py runs/* --detrend       # remove along-track linear trend
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# make the halo_oe package importable regardless of the working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halo_oe  # noqa: F401,E402  (side effect: wires goe/adapters onto sys.path)
from halo_oe.io_bundle import load_inversion  # noqa: E402

_EARTH_KM = 6371.0088
_trapezoid = getattr(np, "trapezoid", getattr(np, "trapz", None))  # np 2.x renamed trapz


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between paired points (vectorized, degrees)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * _EARTH_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _along_track_km(lat, lon):
    """Cumulative along-track distance (km) from the first receptor (0 at start).

    Sums consecutive great-circle steps in storage order, which for these
    coarsened flight tracks is time/along-track order.
    """
    if lat.size < 2:
        return np.zeros_like(lat)
    step = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    return np.concatenate([[0.0], np.cumsum(step)])


def _detrend_along_track(x, s):
    """Remove a linear trend in the along-track coordinate ``s`` from ``x``."""
    A = np.vstack([s, np.ones_like(s)]).T
    coef, *_ = np.linalg.lstsq(A, x, rcond=None)
    return x - A @ coef


def _acf_index(x, max_lag):
    """Biased, variance-normalized autocorrelation vs integer sample lag."""
    x = x - x.mean()
    n = x.size
    denom = float(x @ x)
    lags = np.arange(0, min(max_lag, n - 1) + 1)
    acf = np.array([float(x[: n - k] @ x[k:]) / denom for k in lags])
    return lags, acf


def _correlogram_distance(x, s, edges):
    """Autocorrelation vs along-track separation, binned by ``edges`` (km).

    All O(n²) receptor pairs are binned by |s_i − s_j|; each bin reports the
    mean normalized product (correlation) and its pair count.
    """
    x = (x - x.mean()).astype(np.float64)
    var = float(x @ x) / x.size
    i, j = np.triu_indices(x.size, k=1)          # unordered pairs, exclude self
    sep = np.abs(s[i] - s[j])
    prod = (x[i] * x[j]) / var
    which = np.digitize(sep, edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    corr = np.full(centers.size, np.nan)
    counts = np.zeros(centers.size, dtype=int)
    for b in range(1, edges.size):
        m = which == b
        counts[b - 1] = int(m.sum())
        if counts[b - 1]:
            corr[b - 1] = float(prod[m].mean())
    return centers, corr, counts


def _efolding_km(centers, corr):
    """Distance where the correlogram first falls to 1/e (linear interp), or nan."""
    thr = 1.0 / np.e
    ok = ~np.isnan(corr)
    c, r = centers[ok], corr[ok]
    below = np.where(r < thr)[0]
    if below.size == 0 or below[0] == 0:
        return np.nan
    k = below[0]
    (x0, y0), (x1, y1) = (c[k - 1], r[k - 1]), (c[k], r[k])
    return x0 + (thr - y0) * (x1 - x0) / (y1 - y0) if y1 != y0 else x0


def _integral_length_km(centers, corr):
    """Integral scale ∫C(d)dd from 0 to the first zero crossing (trapezoid)."""
    ok = ~np.isnan(corr)
    c, r = np.concatenate([[0.0], centers[ok]]), np.concatenate([[1.0], corr[ok]])
    neg = np.where(r <= 0)[0]
    cut = neg[0] + 1 if neg.size else c.size
    return float(_trapezoid(r[:cut], c[:cut]))


def _iter_flights(bundle_dir):
    """Yield (flight_id, innovation, lat, lon) for each flight in a bundle."""
    inv = load_inversion(bundle_dir)
    rec = inv.receptors
    for needed in ("enhancement", "prior_modeled", "receptor_lat", "receptor_lon"):
        if needed not in rec:
            raise KeyError(
                f"{bundle_dir}: bundle is missing '{needed}'. Regenerate it with the "
                f"updated run_halo.py (e.g. python run_halo.py config.ini "
                f"--flights <id> --save <id>) so H·xa is stored.")
    d_b = np.asarray(rec["enhancement"]) - np.asarray(rec["prior_modeled"])
    lat, lon = np.asarray(rec["receptor_lat"]), np.asarray(rec["receptor_lon"])
    fidx = np.asarray(rec.get("receptor_flight", np.zeros(d_b.size, int)))
    ids = inv.flight_ids or [os.path.basename(os.path.normpath(bundle_dir))]
    for k in np.unique(fidx):
        m = fidx == k
        valid = m & np.isfinite(d_b)          # drop outlier receptors (NaN innovation)
        label = ids[int(k)] if int(k) < len(ids) else f"{os.path.basename(bundle_dir)}[{k}]"
        yield label, d_b[valid], lat[valid], lon[valid]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("bundles", nargs="+", help="Inversion bundle directories (run_halo --save).")
    p.add_argument("--out", default="innovation_autocorr.png", help="Output PNG path.")
    p.add_argument("--max-dist", type=float, default=None,
                   help="Max along-track separation to plot (km); default = median track length.")
    p.add_argument("--nbins", type=int, default=40, help="Distance bins for the correlogram.")
    p.add_argument("--max-lag", type=int, default=60, help="Max sample-index lag.")
    p.add_argument("--detrend", action="store_true",
                   help="Remove a linear along-track trend from each flight's innovation "
                        "(suppresses very-long-wavelength prior bias inflating the length).")
    args = p.parse_args()

    flights = []
    for b in args.bundles:
        for label, d_b, lat, lon in _iter_flights(b):
            if d_b.size < 8:
                print(f"  (skipping {label}: only {d_b.size} valid receptors)")
                continue
            s = _along_track_km(lat, lon)
            x = _detrend_along_track(d_b, s) if args.detrend else d_b
            flights.append({"label": label, "x": x, "s": s})

    if not flights:
        sys.exit("No usable flights found in the given bundles.")

    track_len = np.median([f["s"][-1] for f in flights])
    max_dist = args.max_dist if args.max_dist else 0.6 * track_len
    edges = np.linspace(0.0, max_dist, args.nbins + 1)

    # --- compute per flight -------------------------------------------------
    for f in flights:
        f["lags"], f["acf"] = _acf_index(f["x"], args.max_lag)
        f["centers"], f["corr"], f["counts"] = _correlogram_distance(f["x"], f["s"], edges)
        f["efold"] = _efolding_km(f["centers"], f["corr"])
        f["integral"] = _integral_length_km(f["centers"], f["corr"])
        f["spacing"] = f["s"][-1] / max(f["x"].size - 1, 1)

    # --- report -------------------------------------------------------------
    print(f"\nInnovation (z − H·xa) along-track autocorrelation "
          f"— {len(flights)} flight(s)\n")
    print(f"{'flight':<14} {'n':>6} {'track_km':>9} {'d_recep_km':>11} "
          f"{'e-fold_km':>10} {'integral_km':>12}")
    print("-" * 66)
    for f in sorted(flights, key=lambda g: g["label"]):
        ef = f"{f['efold']:.2f}" if np.isfinite(f["efold"]) else "  >range"
        print(f"{f['label']:<14} {f['x'].size:>6d} {f['s'][-1]:>9.1f} "
              f"{f['spacing']:>11.3f} {ef:>10} {f['integral']:>12.2f}")
    print("\ne-fold / integral length are proxies for the mismatch coherence length; a\n"
          "longer value in the afternoon flight(s) supports the increased-length-scale\n"
          "hypothesis. Both are upper bounds (innovation includes real flux signal).")

    # --- plot ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    order = sorted(range(len(flights)), key=lambda k: flights[k]["label"])
    for rank, k in enumerate(order):
        f = flights[k]
        c = cmap(rank / max(len(flights) - 1, 1))
        axL.plot(f["lags"], f["acf"], color=c, lw=1.6, label=f["label"])
        axR.plot(f["centers"], f["corr"], color=c, lw=1.6, marker="o", ms=3, label=f["label"])

    for ax in (axL, axR):
        ax.axhline(0.0, color="0.6", lw=0.8)
        ax.axhline(1.0 / np.e, color="0.6", lw=0.8, ls=":")
        ax.set_ylabel("autocorrelation")
        ax.set_ylim(-0.35, 1.02)
        ax.legend(fontsize=8, ncol=2)
    axL.set_title("vs sample-index lag")
    axL.set_xlabel("lag (receptors apart)")
    axR.set_title("vs along-track distance  (∝ observation time)")
    axR.set_xlabel("along-track separation (km)")
    axR.set_xlim(0, max_dist)
    fig.suptitle("Prior innovation autocorrelation (z − H·xa; independent of R)")
    fig.savefig(args.out, dpi=140)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
