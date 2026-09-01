"""Does letting `natural_gas`'s prior spatial correlation length grow beyond
its configured 5km let the inversion's actual flux-state mechanism explain
`728_2`'s Long Island gradient -- the one combination of category x
mechanism this investigation hadn't tried yet (point sources: §21/
landfill_gap_sensitivity_check.py; category-blind background: continuous_
kriging_check.py; this: the real, already-nonzero-density `natural_gas`
category, at its own actual mechanism).

Same regularized marginal-likelihood machinery as §15's landfill/wastewater
sweep (joint_correlation_sweep.py) -- Sa = prior spatial covariance over
candidate cells, R = observation error covariance (held fixed at its
current configured value; only the PRIOR side is swept here, since that's
specifically what was asked), Sigma = A Sa A^T + R, marginal log-likelihood
of b = z - Hx_hat -- but scoped to natural_gas cells and 728_2's Long Island
receptors instead of point-source events, and with a much larger
neighborhood radius (cells and receptors both), since testing correlation
lengths up to 100+km needs room for that to matter.

Beyond the log-likelihood number itself (§15's own lesson: a sweep that
keeps improving without a peak needs a second check, not just a bigger
number) -- also reconstructs the actual posterior cell-level correction at
the best-supported length and checks whether it visibly closes the gap in
the residual map, not just the marginal-likelihood score.

No re-solve; reads the joint 6-flight bundle and streams real Jacobian rows
for the receptors in the test region.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.decomposition import relative_uncertainties
from halo_oe.io_bundle import load_inversion

import joint_correlation_sweep as J

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FID = "20230728_2"
CATEGORY = "natural_gas"

CENTER = (40.80, -73.25)   # roughly between the Long Island landfills / on the gradient
# cell/receptor radii checked directly for tractable matrix sizes before running: 40km
# gives ~3900 candidate cells (Sa ~3900x3900, dense marginal-likelihood matrices are
# still fast at this size); 100km would have given ~20,000 cells -- intractable for the
# dense Sigma = A Sa A^T + R this script forms at every sweep point.
CELL_RADIUS_KM = 40.0
RECEPTOR_RADIUS_KM = 60.0

LENGTHS_KM = [5, 10, 20, 30, 50, 75, 100, 150]   # 5 = current config value (baseline)


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    mdm_stddev = cfg.get_float("observations", "mdm_stddev", default=0.025)
    meas_stddev = cfg.get_float("observations", "measurement_stddev", default=0.01)
    mdm_len_km = cfg.get_float("observations", "mdm_correlation_length_km", default=1.5)
    rel_unc = relative_uncertainties([CATEGORY], cfg)
    sigma_prior = rel_unc[CATEGORY]

    R_all = inv.receptors
    fi = inv.flight_ids.index(FID)
    sel = R_all["receptor_flight"].astype(int) == fi
    lat_f, lon_f = R_all["receptor_lat"][sel], R_all["receptor_lon"][sel]
    z_f, modeled_f = R_all["enhancement"][sel], R_all["modeled"][sel]
    flag_f = R_all.get("outlier_flag", np.zeros_like(z_f)).astype(bool)[sel]

    d_rec = J._haversine_km(lat_f, lon_f, CENTER[0], CENTER[1])
    rgood = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f) & (d_rec <= RECEPTOR_RADIUS_KM)
    idx = np.flatnonzero(rgood)
    lat_r, lon_r = lat_f[idx], lon_f[idx]
    b = (z_f - modeled_f)[idx]
    print(f"{FID}: {idx.size} receptors within {RECEPTOR_RADIUS_KM}km of {CENTER}")

    dens = inv.group_fields[CATEGORY]
    d_cell = J._haversine_km(inv.core.active_lat, inv.core.active_lon, CENTER[0], CENTER[1])
    cmask = (dens > 0) & (d_cell <= CELL_RADIUS_KM)
    c_lat, c_lon = inv.core.active_lat[cmask], inv.core.active_lon[cmask]
    e_c = dens[cmask]
    flat_idx = inv.core.active[cmask]
    print(f"{cmask.sum()} candidate {CATEGORY} cells within {CELL_RADIUS_KM}km (nonzero density)")

    jf = JacobianFile(os.path.join(JAC_DIR, f"{FID}.nc"))
    jac = jf._ds.variables[jf._jac_var]
    H = np.empty((idx.size, cmask.sum()))
    for k, ri in enumerate(idx):
        row = np.asarray(jac[int(ri), :, :]).reshape(-1)
        H[k, :] = row[flat_idx]
    jf.close()
    A0 = H * e_c[None, :]

    rec_dist = J.pairwise_haversine(lat_r, lon_r)
    R_fixed = J.spatial_kernel(rec_dist, mdm_stddev, mdm_len_km) + meas_stddev**2 * np.eye(idx.size)
    cell_dist = J.pairwise_haversine(c_lat, c_lon)

    print(f"\n{'L(km)':>7} {'Delta_loglik':>13} {'gap_reduction_%':>16} {'max|dx|':>9}")
    results = {}
    for L in LENGTHS_KM:
        Sa = J.spatial_kernel(cell_dist, sigma_prior, L)
        Sigma = A0 @ Sa @ A0.T + R_fixed
        sign, logdet = np.linalg.slogdet(Sigma)
        sol = np.linalg.solve(Sigma, b)
        loglik = -0.5 * (b @ sol + logdet + len(b) * np.log(2 * np.pi))

        dx = Sa @ A0.T @ sol                    # posterior mean cell-level correction
        pred_correction = A0 @ dx               # implied change at each receptor
        resid_after = b - pred_correction
        gap_reduction = 100 * (1 - np.sqrt(np.mean(resid_after**2)) / np.sqrt(np.mean(b**2)))

        results[L] = dict(loglik=loglik, dx=dx, pred_correction=pred_correction, resid_after=resid_after)
        print(f"{L:>7} {loglik - results[LENGTHS_KM[0]]['loglik'] if L != LENGTHS_KM[0] else 0.0:>+13.2f} "
              f"{gap_reduction:>15.1f}% {np.abs(dx).max():>9.3f}")

    best_L = max(LENGTHS_KM, key=lambda L: results[L]["loglik"])
    print(f"\nbest (max marginal log-lik): L={best_L}km")

    fig, axes = plt.subplots(1, 3, figsize=(21, 6), constrained_layout=True)
    vmax = np.abs(b).max()
    baseline_L = LENGTHS_KM[0]
    for ax, (title, vals) in zip(axes, [
            (f"current (L={baseline_L}km)", results[baseline_L]["resid_after"]),
            (f"best-fit L={best_L}km", results[best_L]["resid_after"]),
            ("original gap (no correction)", b)]):
        sc = ax.scatter(lon_r, lat_r, c=vals, s=25, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for name, (clat, clon) in [("Blydenburgh Road", (40.815385, -73.211658)),
                                    ("Smithtown", (40.867708, -73.251183))]:
            ax.scatter([clon], [clat], marker="*", s=200, c="k")
        ax.set_title(f"{title}\nresidual after natural_gas correction")
        ax.set_xlabel("lon"); ax.set_ylabel("lat")
        plt.colorbar(sc, ax=ax, label="ppm")
    plt.savefig("figures/natural_gas_correlation_length_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/natural_gas_correlation_length_check.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(c_lon, c_lat, c=results[best_L]["dx"], s=15, cmap="RdBu_r",
                    vmin=-np.abs(results[best_L]["dx"]).max(), vmax=np.abs(results[best_L]["dx"]).max())
    ax.set_title(f"posterior natural_gas scale-factor correction, L={best_L}km")
    ax.set_xlabel("lon"); ax.set_ylabel("lat")
    plt.colorbar(sc, ax=ax, label="dx (scale-factor units)")
    plt.savefig("figures/natural_gas_correlation_length_dx_map.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/natural_gas_correlation_length_dx_map.png")


if __name__ == "__main__":
    main()
