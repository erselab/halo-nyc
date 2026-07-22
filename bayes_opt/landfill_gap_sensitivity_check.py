"""Does adding back positive emissions at the two real, currently-zero
landfills near `728_2`'s Long Island residual (found by §21's point-source
audit: Blydenburgh Road Landfill, 5.1km from the cluster; Town of Smithtown
Municipal Services Facility, 11.2km) actually explain the observed gap
there, at physically plausible magnitudes?

Both facilities stopped GHGRP-reporting years ago (2014 and 2012
respectively) with `reporting_status = "STOPPED_REPORTING_VALID_REASON"`,
so M3T's non-reporter carry-forward never picks them up (§21) -- they carry
exactly zero density in the prior. But they DID report real emissions while
active, which gives an actual physical magnitude to test against, rather
than an arbitrary unit density: for each facility, its own last-reported
year is used as the reference source strength, under both the `reported`
and `generation_first` GHGRP methods (§16 already found these can differ by
30x+ for facilities with active gas-capture systems, so both bound the
plausible range rather than picking one).

Method: same regularized single-parameter Bayesian update used throughout
this investigation (§14a/§16's amplification tests, §18's rotation test) --
but built around the FULL real receptor set within reach of each facility
(not hand-picked "elevated" receptors), so the fit's own weighting (via the
real Jacobian sensitivity at the facility's cell, not geographic distance)
is what decides which receptors matter. A quick look at the raw data first
(gap = z - modeled vs. distance to Blydenburgh Road) shows why this
matters: there's a positive-gap cluster within ~5km and a *negative*-gap
cluster only slightly further out (~4-8km) -- not the simple "surrounded by
extra positive signal" pattern a straightforward missing point source would
naively predict, so the fit needs the real sensitivity pattern, not a
distance cutoff, to make sense of it.

Prior: amplification x ~ N(0, 1) -- skeptical (x=0 means "still emits
nothing," matching the facility's current GHGRP status), x=1 means "emits
exactly at its own last-reported historical rate." Posterior x significantly
above 0 is the test; x near 1 (not needing an implausible multiple) is what
would make the historical-rate explanation credible.

No re-solve; reads the joint 6-flight bundle and streams individual
Jacobian rows for the receptors in reach of each facility.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km
from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.io_bundle import load_inversion

import joint_correlation_sweep as J

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FID = "20230728_2"
M3T_DATA = "/scratch/scrowel3_lab/M3T/python/src/m3t/data"
GHGRP_FACILITY_DATA = "/scratch/scrowel3_lab/M3T_Processed/M3T_Processed/GHGRP/facility_data.csv"

_MT_TO_MOL_S = 1e6 / (16.043 * 365 * 24 * 3600)   # metric tonnes CH4/yr -> mol/s
KM_PER_DEG_LAT = 111.32

SITES = {
    "Blydenburgh Road Landfill": dict(facility_id=1002125, lat=40.815385, lon=-73.211658),
    "Town of Smithtown MSF": dict(facility_id=1004637, lat=40.867708, lon=-73.251183),
}
SEARCH_RADIUS_KM = 20.0   # generous vs. the ~15-20km footprint decay length (§7.3)
PRIOR_SIGMA = 1.0         # x ~ N(0, PRIOR_SIGMA^2); relative-uncertainty convention used throughout


def cell_area_m2(lat_deg):
    """Spherical-cap band area (m^2) of one ~0.01deg native-grid cell at this latitude."""
    res_deg = 0.009  # matches the Jacobian's own native grid spacing (checked directly elsewhere)
    dlat = np.radians(res_deg)
    dlon = np.radians(res_deg)
    lat0 = np.radians(lat_deg)
    R = 6.371e6
    return R * R * dlon * (np.sin(lat0 + dlat / 2) - np.sin(lat0 - dlat / 2))


def reference_density(facility_id, lat, methods=("ghg_quantity", "generation_first_HH6")):
    """This facility's own last-reported-year emissions, in mol/s, per method column."""
    em = pd.read_parquet(f"{M3T_DATA}/GHGRP_landfills.parquet")
    rows = em[em["facility_id"] == facility_id].sort_values("year")
    last = rows.iloc[-1]
    area = cell_area_m2(lat)
    out = {}
    for col in methods:
        mt_per_yr = last[col]
        mol_s = mt_per_yr * _MT_TO_MOL_S
        flux_nmol_m2_s = mol_s * 1e9 / area
        out[col] = dict(year=int(last["year"]), mt_per_yr=float(mt_per_yr),
                        density_umol_m2_s=flux_nmol_m2_s * 1e-3)
    return out


def bayesian_amplification(A, b, R, sigma_prior=PRIOR_SIGMA):
    """Closed-form posterior of a single scalar amplification x, b = A*x + eps,
    eps ~ N(0, R), x ~ N(0, sigma_prior^2)."""
    Rinv_A = np.linalg.solve(R, A)
    Rinv_b = np.linalg.solve(R, b)
    precision = 1.0 / sigma_prior**2 + A @ Rinv_A
    var = 1.0 / precision
    mean = var * (A @ Rinv_b)
    return float(mean), float(np.sqrt(var))


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    mdm_stddev = cfg.get_float("observations", "mdm_stddev", default=0.025)
    meas_stddev = cfg.get_float("observations", "measurement_stddev", default=0.01)
    mdm_len_km = cfg.get_float("observations", "mdm_correlation_length_km", default=1.5)

    R_all = inv.receptors
    fi = inv.flight_ids.index(FID)
    sel = R_all["receptor_flight"].astype(int) == fi
    lat_f = R_all["receptor_lat"][sel]
    lon_f = R_all["receptor_lon"][sel]
    z_f = R_all["enhancement"][sel]
    modeled_f = R_all["modeled"][sel]
    flag_f = R_all.get("outlier_flag", np.zeros_like(z_f)).astype(bool)[sel]
    good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)

    jf = JacobianFile(os.path.join(JAC_DIR, f"{FID}.nc"))
    jac = jf._ds.variables[jf._jac_var]
    grid = jf.grid

    fig, axes = plt.subplots(1, len(SITES), figsize=(8 * len(SITES), 6), constrained_layout=True)
    site_results = {}

    for ax, (name, site) in zip(np.atleast_1d(axes), SITES.items()):
        d = _haversine_km(lat_f, lon_f, site["lat"], site["lon"])
        near = good & (d <= SEARCH_RADIUS_KM)
        idx = np.flatnonzero(near)
        print(f"\n=== {name} ({site['lat']},{site['lon']}): "
              f"{idx.size} receptors within {SEARCH_RADIUS_KM}km ===")

        # nearest native-grid cell to the facility
        ci = np.abs(grid.lat - site["lat"]).argmin()
        cj = np.abs(grid.lon - site["lon"]).argmin()

        H = np.empty(idx.size)
        for k, ri in enumerate(idx):
            H[k] = np.asarray(jac[int(ri), :, :])[ci, cj]
        b = (z_f - modeled_f)[idx]
        lat_i, lon_i = lat_f[idx], lon_f[idx]

        rec_dist = J.pairwise_haversine(lat_i, lon_i)
        Rmat = J.spatial_kernel(rec_dist, mdm_stddev, mdm_len_km) + meas_stddev**2 * np.eye(idx.size)

        refs = reference_density(site["facility_id"], site["lat"])
        site_results[name] = dict(H=H, b=b, R=Rmat, idx=idx, refs=refs)

        print(f"  sensitivity H at facility cell: min={H.min():.3e} max={H.max():.3e} "
              f"mean|H|={np.abs(H).mean():.3e}  ({np.sum(np.abs(H) > 1e-6)} of {idx.size} receptors "
              f"have non-negligible sensitivity)")

        for method, r in refs.items():
            A = H * r["density_umol_m2_s"]
            mean, sd = bayesian_amplification(A, b, Rmat)
            implied_mt_yr = mean * r["mt_per_yr"]
            print(f"  [{method}, last active {r['year']}, ref={r['mt_per_yr']:.1f} MT/yr]: "
                  f"posterior amplification x = {mean:+.3f} +/- {sd:.3f}  "
                  f"({mean/sd:+.2f} sigma from zero)  "
                  f"-> implied current rate {implied_mt_yr:+.1f} MT/yr")

        sc = ax.scatter(lon_i, lat_i, c=b, s=25, cmap="RdBu_r",
                        vmin=-np.abs(b).max(), vmax=np.abs(b).max())
        ax.scatter([site["lon"]], [site["lat"]], marker="*", s=300, c="k", label=name)
        ax.set_title(f"{name}: gap (z-modeled) near facility")
        ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend(fontsize=8)
        plt.colorbar(sc, ax=ax, label="z - modeled (ppm)")

    # joint 2-parameter fit: both facilities together, over the union of their receptor sets,
    # to check whether the two are separable or degenerate (nearby, could double-count)
    names = list(SITES)
    union_idx = np.unique(np.concatenate([site_results[n]["idx"] for n in names]))
    pos = {ri: k for k, ri in enumerate(union_idx)}
    A2 = np.zeros((union_idx.size, 2))
    for j, name in enumerate(names):
        r = site_results[name]
        for k, ri in enumerate(r["idx"]):
            A2[pos[ri], j] = r["H"][k] * r["refs"]["ghg_quantity"]["density_umol_m2_s"]
    b2 = np.zeros(union_idx.size)
    lat2 = lat_f[union_idx]; lon2 = lon_f[union_idx]
    for name in names:
        r = site_results[name]
        for k, ri in enumerate(r["idx"]):
            b2[pos[ri]] = r["b"][k]
    rec_dist2 = J.pairwise_haversine(lat2, lon2)
    R2 = J.spatial_kernel(rec_dist2, mdm_stddev, mdm_len_km) + meas_stddev**2 * np.eye(union_idx.size)
    Sa2 = (PRIOR_SIGMA**2) * np.eye(2)
    precision2 = np.linalg.inv(Sa2) + A2.T @ np.linalg.solve(R2, A2)
    cov2 = np.linalg.inv(precision2)
    mean2 = cov2 @ (A2.T @ np.linalg.solve(R2, b2))
    corr2 = cov2[0, 1] / np.sqrt(cov2[0, 0] * cov2[1, 1])
    print(f"\n=== joint 2-parameter fit ('reported' method for both) ===")
    for j, name in enumerate(names):
        print(f"  {name}: x = {mean2[j]:+.3f} +/- {np.sqrt(cov2[j,j]):.3f}  "
              f"({mean2[j]/np.sqrt(cov2[j,j]):+.2f} sigma)")
    print(f"  posterior correlation between the two amplitudes: {corr2:+.3f}")

    jf.close()
    plt.savefig("runs/landfill_gap_sensitivity_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> runs/landfill_gap_sensitivity_check.png")


if __name__ == "__main__":
    main()
