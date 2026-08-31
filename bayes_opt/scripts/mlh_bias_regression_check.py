"""Regress leg-level bias against mixed-layer height (MLH), allowing for
curvature since the boundary layer evolves nonlinearly over a flight
(grows, plateaus, sometimes declines -- see mixed_layer_height_check.py),
and cross-check against the leg-index/elapsed-time proxy already used in
leg_offset_drift_check.py.

Two "bias" targets, tested separately since they mean different things:

1. The already-fit **leg background offset** (`receptor_background_offset`,
   §2.2/§28) -- the per-leg additive background level, upstream of flux
   fitting. This is the cleaner target: it isn't contaminated by whatever
   the flux solve does in-domain.
2. The **posterior residual** (`enhancement - modeled`, i.e. `z - Hx̂`) --
   the quantity every residual map and "is this flight explained" claim in
   RESIDUAL_INVESTIGATION.md is actually about. Noisier and, in-domain,
   partly explainable by real flux signal rather than background/transport
   error -- reported as a secondary check, not the primary one.

MLH is computed per leg from **land-only** receptors only (mean of
`CH4DataProducts/MixedLayerHeight` over receptors with `DEM_altitude > -0.5`,
matched to each clipped receptor by nearest GPS time in the unclipped
file) -- mixing in water receptors would reintroduce the ~500m land/water
step found in mixed_layer_height_check.py, which is a different, already-
documented effect (§28) and would swamp any real MLH-magnitude relationship.

For each target and each regressor (leg MLH, leg elapsed time since
takeoff), fits both a linear and a quadratic (degree-2) model, pooled across
all 6 flights after flight-demeaning (same convention as §28), and reports
whether the quadratic term is a statistically meaningful improvement (F-test
on the residual sum of squares) -- not just a lower RMS, since 2 extra
parameters will always fit training data at least as well.

Run with:
    python3 scripts/mlh_bias_regression_check.py
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
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
    BUNDLE, FLIGHT_DATA_DIR, FULL_FLIGHT_DATA_DIR, true_takeoff_hour,
    is_water, _load_receptor_surface_alt,
)


def _load_receptor_mlh(fid: str, clock_h: np.ndarray) -> np.ndarray:
    """MLH (m AGL) at each receptor, nearest-GPS-time matched, from the unclipped file."""
    date, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    matches = sorted(glob.glob(os.path.join(FULL_FLIGHT_DATA_DIR, f"*{date}*_F{fnum}_full.h5")))
    with h5py.File(matches[0], "r") as f:
        t_full = f["Nav_Data/gps_time"][:, 0]
        mlh = f["CH4DataProducts/MixedLayerHeight"][:, 0]
    idx = np.clip(np.searchsorted(t_full, clock_h), 0, len(t_full) - 1)
    return mlh[idx]


def per_leg_bias_table(inv, cfg, fid: str, min_land_pts: int = 5) -> dict:
    R = inv.receptors
    fi = inv.flight_ids.index(fid)
    sel = R["receptor_flight"].astype(int) == fi
    lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
    bg_offset = R["receptor_background_offset"][sel]
    post_resid = R["enhancement"][sel] - R["modeled"][sel]

    clock_hours = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)
    clock_h = clock_hours / 3600.0
    elapsed_min = (clock_h - true_takeoff_hour(fid)) * 60.0
    surface_m = _load_receptor_surface_alt(fid, clock_h)
    land = ~is_water(surface_m)
    mlh = _load_receptor_mlh(fid, clock_h)

    leg_id = detect_legs(
        lat, lon, clock_hours,
        gap_seconds=cfg.get_float("background", "leg_gap_seconds", default=8.0),
        min_leg_size=cfg.get_int("background", "leg_min_size", default=10),
        axis_deg=cfg.get_float("background", "leg_axis_deg", default=45.0),
    )

    legs = np.unique(leg_id)
    rows = []
    for leg in legs:
        m = leg_id == leg
        land_mlh = mlh[m & land]
        land_mlh = land_mlh[np.isfinite(land_mlh)]
        if len(land_mlh) < min_land_pts:
            continue
        rows.append(dict(
            leg=leg, elapsed_min=elapsed_min[m].mean(), mlh=np.median(land_mlh),
            n_land=len(land_mlh), offset=bg_offset[m].mean(), post_resid=post_resid[m].mean(),
        ))
    return rows


def poly_fit_compare(x: np.ndarray, y: np.ndarray, label: str):
    """Linear vs. quadratic fit; F-test for whether the quadratic term earns its keep."""
    n = len(x)
    X1 = np.column_stack([np.ones(n), x])
    X2 = np.column_stack([np.ones(n), x, x ** 2])

    b1, *_ = np.linalg.lstsq(X1, y, rcond=None)
    b2, *_ = np.linalg.lstsq(X2, y, rcond=None)
    rss1 = np.sum((y - X1 @ b1) ** 2)
    rss2 = np.sum((y - X2 @ b2) ** 2)
    tss = np.sum((y - y.mean()) ** 2)
    r2_1 = 1 - rss1 / tss
    r2_2 = 1 - rss2 / tss

    df1, df2 = n - 2, n - 3
    if rss1 > rss2 and df2 > 0:
        f_stat = ((rss1 - rss2) / 1) / (rss2 / df2)
        f_p = 1 - stats.f.cdf(f_stat, 1, df2)
    else:
        f_stat, f_p = 0.0, 1.0

    slope1, r1, p1 = stats.linregress(x, y)[:3]
    print(f"  {label}:")
    print(f"    linear:    slope={b1[1]:+.5f}  R²={r2_1:.4f}  p(slope)={stats.linregress(x, y).pvalue:.3g}")
    print(f"    quadratic: b1={b2[1]:+.5f} b2={b2[2]:+.7f}  R²={r2_2:.4f}  "
          f"(ΔR²={r2_2 - r2_1:+.4f})  F={f_stat:.2f}  p(quad term)={f_p:.3g}")
    return dict(b1=b1, b2=b2, r2_1=r2_1, r2_2=r2_2, f_p=f_p)


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))

    all_rows = []
    print(f"{'flight':<12}{'n_legs (land-qualified)':>26}")
    for fid in inv.flight_ids:
        rows = per_leg_bias_table(inv, cfg, fid)
        print(f"{fid:<12}{len(rows):>26}")
        for r in rows:
            r["fid"] = fid
        all_rows.extend(rows)

    fids = sorted(set(r["fid"] for r in all_rows))
    offset_mean = {f: np.mean([r["offset"] for r in all_rows if r["fid"] == f]) for f in fids}
    resid_mean = {f: np.mean([r["post_resid"] for r in all_rows if r["fid"] == f]) for f in fids}
    for r in all_rows:
        r["offset_c"] = r["offset"] - offset_mean[r["fid"]]
        r["resid_c"] = r["post_resid"] - resid_mean[r["fid"]]

    mlh = np.array([r["mlh"] for r in all_rows])
    elapsed_h = np.array([r["elapsed_min"] for r in all_rows]) / 60.0
    offset_c = np.array([r["offset_c"] for r in all_rows])
    resid_c = np.array([r["resid_c"] for r in all_rows])
    n = len(all_rows)
    print(f"\npooled, flight-demeaned, land-qualified legs: n={n}\n")

    print("=== target: leg background offset (pre-flux-fit background bias) ===")
    fit_o_mlh = poly_fit_compare(mlh, offset_c, "vs. land-only MLH (m)")
    fit_o_t = poly_fit_compare(elapsed_h, offset_c, "vs. elapsed time since takeoff (hr)")

    print("\n=== target: posterior residual z - Hx̂ (secondary -- contaminated by real flux signal in-domain) ===")
    fit_r_mlh = poly_fit_compare(mlh, resid_c, "vs. land-only MLH (m)")
    fit_r_t = poly_fit_compare(elapsed_h, resid_c, "vs. elapsed time since takeoff (hr)")

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, 6))
    fid_color = {f: colors[i] for i, f in enumerate(inv.flight_ids)}

    def scatter_with_fit(ax, x, y, fit, xlabel, ylabel, title):
        for f in fids:
            xs = [r for r in all_rows if r["fid"] == f]
            ax.scatter([r["mlh"] if "MLH" in xlabel else r["elapsed_min"] / 60.0 for r in xs],
                       [r["offset_c"] if "offset" in ylabel else r["resid_c"] for r in xs],
                       color=fid_color[f], label=f, s=28)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, fit["b1"][0] + fit["b1"][1] * xx, "k--", lw=1, label="linear")
        ax.plot(xx, fit["b2"][0] + fit["b2"][1] * xx + fit["b2"][2] * xx ** 2, "r-", lw=1.5,
                label=f"quadratic (p={fit['f_p']:.2g})")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
        ax.legend(fontsize=7, ncol=2)

    scatter_with_fit(axes[0, 0], mlh, offset_c, fit_o_mlh, "land-only MLH (m)",
                      "leg background offset, flight-demeaned (ppm)", "Background offset vs. MLH")
    scatter_with_fit(axes[0, 1], elapsed_h, offset_c, fit_o_t, "elapsed time (hr)",
                      "leg background offset, flight-demeaned (ppm)", "Background offset vs. elapsed time")
    scatter_with_fit(axes[1, 0], mlh, resid_c, fit_r_mlh, "land-only MLH (m)",
                      "posterior residual, flight-demeaned (ppm)", "Posterior residual vs. MLH")
    scatter_with_fit(axes[1, 1], elapsed_h, resid_c, fit_r_t, "elapsed time (hr)",
                      "posterior residual, flight-demeaned (ppm)", "Posterior residual vs. elapsed time")

    plt.savefig("figures/mlh_bias_regression_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/mlh_bias_regression_check.png")


if __name__ == "__main__":
    main()
