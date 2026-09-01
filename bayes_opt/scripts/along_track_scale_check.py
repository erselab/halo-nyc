"""Compare the along-track variability of observed enhancement (z) to modeled
enhancement (Hx̂), per flight, to directly estimate the along-track scale
below which the model stops resolving real structure. A systematic version of
RESIDUAL_INVESTIGATION.md §7's one-off gradient-sharpness comparison (done
for a single 726_1 hotspot/dip pair only).

Two along-track statistics, both binned by real elapsed-time lag (not sample
index -- reuses/mirrors halo_oe.plotting._time_binned_autocorr's binning,
since sample spacing is ~constant within a leg but turn gaps are minutes):

* autocorrelation (normalized, shape of decay)
* structure function D(tau) = <(x(t+tau)-x(t))^2> (absolute ppm^2 units, so
  z and modeled are directly comparable in magnitude, not just shape)

Includes a noise-floor control: the same structure function computed on z
restricted to the domain-insensitive (fit_mask) receptors, where modeled
enhancement is ~0. Per RESIDUAL_INVESTIGATION.md §12, background-fitting
itself leaves a real leg-to-leg residual there -- so some of z's fine-scale
variability is background noise, not under-resolved transport signal, and
that floor must be subtracted from the interpretation, not attributed to a
transport-resolution gap.

One cheap Jacobian column-sum stream per flight (fit_mask, same call used in
leg_offset_check.py/buffer_bias_check.py) plus one small flight_data/*.h5
read per flight (elapsed time) -- no Jacobian materialize, no re-solve.

Run with the `analysis` conda env from the bayes_opt directory:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 scripts/along_track_scale_check.py
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
from halo_oe.background import domain_insensitive_mask, _load_receptor_time
from halo_oe.io_bundle import load_inversion
from halo_oe.plotting import _time_binned_autocorr

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"
MAX_LAG_S = 180.0
BIN_WIDTH_S = 5.0


def time_binned_structure(x, t, max_lag_s=MAX_LAG_S, bin_width_s=BIN_WIDTH_S):
    """D(tau) = <(x(t+tau)-x(t))^2>, binned by real elapsed-time lag."""
    n = len(x)
    nbins = max(1, int(np.ceil(max_lag_s / bin_width_s)))
    edges = np.linspace(0.0, max_lag_s, nbins + 1)
    sums = np.zeros(nbins)
    counts = np.zeros(nbins, dtype=int)
    for i in range(n):
        j = i + 1
        while j < n and t[j] - t[i] <= max_lag_s:
            b = min(int((t[j] - t[i]) / bin_width_s), nbins - 1)
            sums[b] += (x[j] - x[i]) ** 2
            counts[b] += 1
            j += 1
    D = np.divide(sums, counts, out=np.full(nbins, np.nan), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, D


def median_speed_km_per_s(lat, lon, t):
    """Empirical along-track ground speed, from consecutive same-leg pairs
    (dt < 6s excludes turn gaps), for converting a time lag axis to km."""
    dt = np.diff(t)
    d = _haversine_km(lat[:-1], lon[:-1], lat[1:], lon[1:])
    keep = (dt > 0) & (dt < 6.0)
    return float(np.median(d[keep] / dt[keep]))


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)
    ds_quantile = cfg.get_float("background", "domain_sensitivity_quantile", default=1.0)

    pooled = {"lag": [], "D_z": [], "D_m": [], "D_zbg": []}

    n_flights = len(inv.flight_ids)
    fig_ac, ax_ac = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True, squeeze=False)
    fig_sf, ax_sf = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True, squeeze=False)

    for i, fid in enumerate(inv.flight_ids):
        # per-flight mask over the global concatenated arrays -- _load_receptor_time
        # and receptor_column_sums both need per-flight-length, per-flight-order
        # arrays (validated against the raw h5 / Jacobian file), not the global ones
        flight_sel = flight_index == i
        lat_f = R["receptor_lat"][flight_sel]
        lon_f = R["receptor_lon"][flight_sel]
        z_f = z_all[flight_sel]
        modeled_f = modeled_all[flight_sel]
        flag_f = flag_all[flight_sel]

        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        jf = JacobianFile(os.path.join(JAC_DIR, f"{fid}.nc"))
        domain_sensitivity_f = jf.receptor_column_sums(inv.core.active, row_chunk=16)["uniform"]["inside"]
        jf.close()
        # bundle's per-flight receptor order == Jacobian file's own receptor order
        # (concatenated directly from jf.receptor_lat/lon in io_bundle.py, exact-match
        # verified in the earlier buffer/domain-diagnostic check) -- no reorder needed

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        lat, lon, z, modeled, t, domain_sensitivity = (
            a[good] for a in (lat_f, lon_f, z_f, modeled_f, t_f, domain_sensitivity_f))

        order = np.argsort(t)
        lat, lon, z, modeled, t, domain_sensitivity = (
            a[order] for a in (lat, lon, z, modeled, t, domain_sensitivity))

        fit_mask = domain_insensitive_mask(domain_sensitivity, ds_quantile)

        speed = median_speed_km_per_s(lat, lon, t)
        print(f"{fid}: n={z.size}  fit_mask={fit_mask.sum()}  "
              f"median speed={speed*1000:.0f} m/s  (180s lag ~ {speed*180:.0f}km)")

        z_std = (z - z.mean()) / z.std()
        m_std = (modeled - modeled.mean()) / (modeled.std() + 1e-12)
        lag_ac, ac_z = _time_binned_autocorr(z_std, t, MAX_LAG_S, BIN_WIDTH_S)
        _, ac_m = _time_binned_autocorr(m_std, t, MAX_LAG_S, BIN_WIDTH_S)

        lag_sf, D_z = time_binned_structure(z, t, MAX_LAG_S, BIN_WIDTH_S)
        _, D_m = time_binned_structure(modeled, t, MAX_LAG_S, BIN_WIDTH_S)
        D_zbg = np.full_like(D_z, np.nan)
        if fit_mask.sum() > 5:
            _, D_zbg = time_binned_structure(z[fit_mask], t[fit_mask], MAX_LAG_S, BIN_WIDTH_S)

        pooled["lag"].append(lag_sf)
        pooled["D_z"].append(D_z)
        pooled["D_m"].append(D_m)
        pooled["D_zbg"].append(D_zbg)

        a = ax_ac.flat[i]
        keep = np.isfinite(ac_z) & np.isfinite(ac_m)
        a.plot(lag_ac[keep], ac_z[keep], "o-", label="observed z", ms=4)
        a.plot(lag_ac[keep], ac_m[keep], "s-", label="modeled Hx̂", ms=4)
        a.axhline(0, color="gray", lw=0.8)
        a.set_title(fid); a.set_xlabel("lag (s)"); a.set_ylabel("autocorr")
        if i == 0:
            a.legend(fontsize=8)

        a = ax_sf.flat[i]
        keep = np.isfinite(D_z) & np.isfinite(D_m)
        a.plot(lag_sf[keep], D_z[keep], "o-", label="observed z", ms=4)
        a.plot(lag_sf[keep], D_m[keep], "s-", label="modeled Hx̂", ms=4)
        if np.isfinite(D_zbg).any():
            a.plot(lag_sf, D_zbg, "d--", color="gray", alpha=0.7, label="z, fit_mask only (noise floor)", ms=3)
        a.set_title(f"{fid}  (~{speed*1000:.0f} m/s)")
        a.set_xlabel("lag (s)"); a.set_ylabel("D(tau) (ppm²)")
        if i == 0:
            a.legend(fontsize=7)

    fig_ac.suptitle("Along-track autocorrelation: observed vs modeled enhancement")
    fig_ac.savefig("figures/along_track_autocorr.png", bbox_inches="tight", dpi=110)
    plt.close(fig_ac)
    fig_sf.suptitle("Along-track structure function: observed vs modeled enhancement")
    fig_sf.savefig("figures/along_track_structure.png", bbox_inches="tight", dpi=110)
    plt.close(fig_sf)
    print("\nplots -> figures/along_track_autocorr.png, figures/along_track_structure.png")

    # pooled across all 6 flights (common lag grid already shared: same edges)
    lag = pooled["lag"][0]
    D_z_pool = np.nanmean(np.stack(pooled["D_z"]), axis=0)
    D_m_pool = np.nanmean(np.stack(pooled["D_m"]), axis=0)
    D_zbg_pool = np.nanmean(np.stack(pooled["D_zbg"]), axis=0)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(lag, D_z_pool, "o-", label="observed z (all 6 flights)")
    ax.plot(lag, D_m_pool, "s-", label="modeled Hx̂ (all 6 flights)")
    ax.plot(lag, D_zbg_pool, "d--", color="gray", label="z, fit_mask only (noise floor)")
    ax.set_xlabel("lag (s)"); ax.set_ylabel("D(tau) (ppm²)")
    ax.set_title("Pooled along-track structure function, all 6 flights")
    ax.legend()
    plt.savefig("figures/along_track_structure_pooled.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/along_track_structure_pooled.png")

    print("\nlag(s)  D_obs    D_modeled  D_obs-fitmask(noise floor)")
    for k in range(len(lag)):
        print(f"{lag[k]:6.1f}  {D_z_pool[k]:.5f}  {D_m_pool[k]:.5f}   {D_zbg_pool[k]:.5f}")


if __name__ == "__main__":
    main()
