"""Does continuous (per-receptor, spatially-kriged) background correction --
rather than the production per-leg-*constant* offset -- remove the ~50km
monotonic within-leg gradient found near `728_2`'s Long Island landfills
(landfill_gap_sensitivity_check.py)?

This matters specifically because `fit_leg_offsets` (already on in every
bundle this investigation uses -- confirmed directly, `use_leg_offsets=true`
in the saved config) fits exactly ONE constant per leg, GP-smoothed across
*legs* in elapsed time. It has no mechanism to represent a value that varies
continuously along a single leg's track -- a monotonic gradient spanning an
entire leg is structurally invisible to it, regardless of correlation-time
tuning. §4.2 tested a genuinely continuous alternative (a per-receptor GP
over space, not one value per leg) for `728_1` and found a real but
plateauing ~11-12% RMS improvement -- never re-run for `728_2`, which is
what this script does.

Method: reuses the exact `plane -> per-leg-offset` decomposition already in
the saved bundle (`receptor_background` = plane+offset combined,
`receptor_background_offset` = the offset alone, so the plane-only value is
just their difference -- no re-fit needed). Fits a standard GP regression
(exponential spatial kernel over great-circle distance, sweeping the
correlation length) to the plane-only residual at the domain-insensitive
(`fit_mask`) receptors -- the same "background-representative" receptor set
`fit_leg_offsets` itself uses -- and predicts a continuous background offset
at every receptor, including the ones near the landfills that are NOT
necessarily domain-insensitive. Compares the resulting corrected residual,
in the Long Island region specifically, against the current per-leg-constant
result.

No re-solve; reads the saved joint bundle and one streamed Jacobian pass
(domain sensitivity, for the fit_mask -- same cheap call used throughout
this investigation).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km
from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.background import domain_insensitive_mask
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FID = "20230728_2"

# candidate correlation lengths (km) -- spans well below and above the
# ~50km gradient's own scale, to see whether/where RMS actually plateaus
LENGTHS_KM = [5, 10, 20, 30, 50, 75, 100, 150]
PRIOR_STDDEV = 0.05      # same magnitude convention as leg_offset_stddev
NOISE_STDDEV = 0.02      # per-receptor scatter, same convention as leg_offset_noise_stddev

LANDFILL_SITES = {
    "Blydenburgh Road Landfill": (40.815385, -73.211658),
    "Town of Smithtown MSF": (40.867708, -73.251183),
}
LANDFILL_RADIUS_KM = 20.0


def gp_predict(train_lat, train_lon, train_y, query_lat, query_lon, length_km,
               prior_stddev=PRIOR_STDDEV, noise_stddev=NOISE_STDDEV):
    """Standard GP regression, exponential kernel over great-circle distance."""
    n = len(train_lat)
    Dtt = np.zeros((n, n))
    for i in range(n):
        Dtt[i, :] = _haversine_km(train_lat[i], train_lon[i], train_lat, train_lon)
    Ktt = prior_stddev**2 * np.exp(-Dtt / length_km) + noise_stddev**2 * np.eye(n)

    m = len(query_lat)
    Dqt = np.zeros((m, n))
    for i in range(m):
        Dqt[i, :] = _haversine_km(query_lat[i], query_lon[i], train_lat, train_lon)
    Kqt = prior_stddev**2 * np.exp(-Dqt / length_km)

    alpha = np.linalg.solve(Ktt, train_y)
    return Kqt @ alpha


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    ds_quantile = cfg.get_float("background", "domain_sensitivity_quantile", default=1.0)

    R = inv.receptors
    fi = inv.flight_ids.index(FID)
    sel = R["receptor_flight"].astype(int) == fi
    lat = R["receptor_lat"][sel]
    lon = R["receptor_lon"][sel]
    obs = R["receptor_obs"][sel]
    bg_full = R["receptor_background"][sel]           # plane + leg offset
    leg_offset = R["receptor_background_offset"][sel]  # leg offset alone
    plane_only = bg_full - leg_offset
    z_current = R["enhancement"][sel]                  # obs - (plane + leg offset)
    modeled = R["modeled"][sel]
    flag = R.get("outlier_flag", np.zeros_like(obs)).astype(bool)[sel]

    jf = JacobianFile(os.path.join(JAC_DIR, f"{FID}.nc"))
    domain_sensitivity = jf.receptor_column_sums(inv.core.active, row_chunk=16)["uniform"]["inside"]
    jf.close()
    fit_mask = domain_insensitive_mask(domain_sensitivity, ds_quantile) & ~flag

    good = ~flag & np.isfinite(obs) & np.isfinite(plane_only)
    print(f"{FID}: n={good.sum()} good receptors, {fit_mask.sum()} in fit_mask "
          f"(domain_sensitivity_quantile={ds_quantile})")

    resid_after_plane = (obs - plane_only)[good]
    lat_g, lon_g = lat[good], lon[good]
    train = fit_mask[good]

    resid_current = (z_current - modeled)[good]
    rms_current = np.sqrt(np.mean(resid_current**2))
    print(f"\ncurrent (per-leg-constant) RMS over full flight: {rms_current:.4f} ppm")

    print(f"\n{'L(km)':>7} {'RMS_full':>10} {'RMS_full_%chg':>14} "
          f"{'RMS_landfill':>13} {'RMS_landfill_%chg':>18}")

    landfill_near = np.zeros(good.sum(), dtype=bool)
    for _, (clat, clon) in LANDFILL_SITES.items():
        landfill_near |= _haversine_km(lat_g, lon_g, clat, clon) <= LANDFILL_RADIUS_KM
    rms_landfill_current = np.sqrt(np.mean(resid_current[landfill_near]**2))
    print(f"{'--':>7} {rms_current:>10.4f} {'--':>14} {rms_landfill_current:>13.4f} {'--':>18}")

    results = {}
    for L in LENGTHS_KM:
        krige = gp_predict(lat_g[train], lon_g[train], resid_after_plane[train],
                           lat_g, lon_g, length_km=L)
        z_new = obs[good] - plane_only[good] - krige
        resid_new = z_new - modeled[good]
        rms_new = np.sqrt(np.mean(resid_new**2))
        rms_new_landfill = np.sqrt(np.mean(resid_new[landfill_near]**2))
        pct_full = 100 * (rms_new - rms_current) / rms_current
        pct_landfill = 100 * (rms_new_landfill - rms_landfill_current) / rms_landfill_current
        print(f"{L:>7} {rms_new:>10.4f} {pct_full:>+13.1f}% {rms_new_landfill:>13.4f} {pct_landfill:>+17.1f}%")
        results[L] = dict(krige=krige, resid_new=resid_new, z_new=z_new)

    # map comparison at the correlation length that helps most in the landfill region
    best_L = min(LENGTHS_KM, key=lambda L: np.sqrt(np.mean(results[L]["resid_new"][landfill_near]**2)))
    print(f"\nbest-performing length in the landfill region: {best_L}km")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    vmax = np.abs(resid_current[landfill_near]).max()
    for ax, (title, vals) in zip(axes, [
            ("current (per-leg-constant)", resid_current),
            (f"continuous krige, L={best_L}km", results[best_L]["resid_new"]),
            (f"continuous krige, L={LENGTHS_KM[0]}km (tightest)", results[LENGTHS_KM[0]]["resid_new"])]):
        sc = ax.scatter(lon_g[landfill_near], lat_g[landfill_near], c=vals[landfill_near],
                        s=25, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for name, (clat, clon) in LANDFILL_SITES.items():
            ax.scatter([clon], [clat], marker="*", s=300, c="k")
        ax.set_title(f"{title}\nresidual (z-modeled) near Long Island landfills")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        plt.colorbar(sc, ax=ax, label="ppm")
    plt.savefig("runs/continuous_kriging_check_728_2.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> runs/continuous_kriging_check_728_2.png")


if __name__ == "__main__":
    main()
