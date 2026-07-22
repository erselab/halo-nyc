"""Does rotating the footprint about the receptor -- a proxy for a systematic
wind-direction/transport-angle error -- improve the fit at the already-
identified landfill/WWTP point-source events, more than the current
(unrotated) footprint does?

Motivated by the footprint_similarity_space_time.py finding: at fixed
distance, footprint cosine similarity is flat vs. time gap (no detectable
within-flight drift), which argues against a *time-varying* correction but
is exactly the condition under which a single *shared, fixed* rotation
parameter is worth testing. That test could only show self-consistency,
though -- it can't tell us whether a fixed bias exists, since it never
compares a footprint to the actual data. This script does that comparison.

Method: reuse joint_correlation_sweep.collect_events() (§14b/§15's 15
landfill/WWTP-dominated events across the 6 flights) and its candidate-cell
neighborhoods. For each event, for each elevated receptor, read its raw
Jacobian row and build a local interpolator over a patch big enough to
contain every candidate cell at every trial rotation (rotation preserves
distance from the receptor, so the patch only needs to be as wide as the
candidate-cell search radius). For a grid of trial angles theta, resample
each candidate cell's Jacobian value at its *pre-rotation* position (rotating
the query point by -theta about the receptor), rebuild A0_rot = H_rot *
density, and evaluate the same regularized marginal-likelihood used in
joint_correlation_sweep.py (prior spatial length and MDM correlation length
held at the current config baseline -- only theta varies), summed across
events (and per flight, and per event) to see whether the data support a
nonzero rotation, and whether any such rotation is at all consistent
flight-to-flight (a real transport bias) or scattered (an overfitting
artifact of one free parameter per event).

theta=0 must reproduce joint_correlation_sweep's own baseline (L_prior=0,
L_obs=1.5km) log-likelihood almost exactly (up to interpolation-vs-direct-
indexing floating point) -- printed as a sanity check.

No re-solve; reuses the cached-row-read / closed-form-Gaussian pattern
already established by point_source_amplification_check.py and
joint_correlation_sweep.py.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import matplotlib
from scipy.interpolate import RegularGridInterpolator

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.jacobian_operator import JacobianFile
from goe.config import Config
from halo_oe.decomposition import relative_uncertainties
from halo_oe.io_bundle import load_inversion

import joint_correlation_sweep as J

THETA_GRID_DEG = np.arange(-30.0, 30.1, 2.0)
PATCH_RADIUS_KM = 20.0   # must exceed SEARCH_RADIUS_KM (10km); rotation preserves radius from receptor
KM_PER_DEG_LAT = 111.32


def local_patch(row2d, grid_lat, grid_lon, center_lat, center_lon, radius_km=PATCH_RADIUS_KM):
    dlat = radius_km / KM_PER_DEG_LAT
    dlon = radius_km / (KM_PER_DEG_LAT * max(np.cos(np.radians(center_lat)), 0.1))
    lat_idx = np.flatnonzero((grid_lat >= center_lat - dlat) & (grid_lat <= center_lat + dlat))
    lon_idx = np.flatnonzero((grid_lon >= center_lon - dlon) & (grid_lon <= center_lon + dlon))
    return grid_lat[lat_idx], grid_lon[lon_idx], row2d[np.ix_(lat_idx, lon_idx)]


def offsets_km(lat, lon, ref_lat, ref_lon):
    dx = (lon - ref_lon) * KM_PER_DEG_LAT * np.cos(np.radians(ref_lat))
    dy = (lat - ref_lat) * KM_PER_DEG_LAT
    return dx, dy


def rotate_back_to_lonlat(dx, dy, ref_lat, ref_lon, theta_deg):
    """Where a point currently at offset (dx, dy) from the receptor would have
    been *before* rotating the whole footprint by +theta -- i.e. the position
    to sample the unrotated row at, to get the rotated row's value here."""
    th = np.radians(-theta_deg)
    c, s = np.cos(th), np.sin(th)
    dx2 = dx * c - dy * s
    dy2 = dx * s + dy * c
    lat = ref_lat + dy2 / KM_PER_DEG_LAT
    lon = ref_lon + dx2 / (KM_PER_DEG_LAT * np.cos(np.radians(ref_lat)))
    return lat, lon


def main():
    inv = load_inversion(J.BUNDLE)
    cfg = Config(os.path.join(J.BUNDLE, "config.ini"))
    mdm_stddev = cfg.get_float("observations", "mdm_stddev", default=0.025)
    meas_stddev = cfg.get_float("observations", "measurement_stddev", default=0.01)
    L_obs = cfg.get_float("observations", "mdm_correlation_length_km", default=1.5)
    L_prior = cfg.get_float("category_spatial", "default", default=0.0)
    rel_unc = relative_uncertainties(J.CATEGORIES, cfg)
    print(f"baseline config: mdm_stddev={mdm_stddev} measurement_stddev={meas_stddev} "
          f"L_obs={L_obs}km L_prior={L_prior}km (landfill/wastewater default)")

    events = J.collect_events(inv)
    print(f"{len(events)} qualifying landfill/WWTP-dominated events across all 6 flights")

    jf_cache = {}

    def get_jf(fid):
        if fid not in jf_cache:
            jf_cache[fid] = JacobianFile(os.path.join(J.JAC_DIR, f"{fid}.nc"))
        return jf_cache[fid]

    # --- prepare, per event: candidate cells + per-receptor local interpolators
    # (all independent of theta -- the sweep itself only resamples + re-solves) ---
    prepared = []
    for ev in events:
        d_active = J._haversine_km(inv.core.active_lat, inv.core.active_lon, ev["elat"], ev["elon"])
        near = d_active <= J.NEIGHBORHOOD_RADIUS_KM
        dens = inv.group_fields[ev["dom_cat"]]
        cell_mask = near & (dens > 0)
        if cell_mask.sum() == 0:
            continue
        c_lat = inv.core.active_lat[cell_mask]
        c_lon = inv.core.active_lon[cell_mask]
        e_c = dens[cell_mask]

        jf = get_jf(ev["fid"])
        jac = jf._ds.variables[jf._jac_var]
        interps, H0 = [], np.empty((len(ev["r_raw_idx"]), len(e_c)))
        for m, ri in enumerate(ev["r_raw_idx"]):
            row2d = np.asarray(jac[int(ri), :, :])
            r_lat, r_lon = ev["r_lat"][m], ev["r_lon"][m]
            plat, plon, patch = local_patch(row2d, jf.grid.lat, jf.grid.lon, r_lat, r_lon)
            interp = RegularGridInterpolator((plat, plon), patch, method="linear",
                                              bounds_error=False, fill_value=0.0)
            interps.append(interp)
            H0[m, :] = interp((c_lat, c_lon))   # theta=0 sanity value, direct from patch

        b = ev["z"] - ev["modeled"]
        cell_dist = J.pairwise_haversine(c_lat, c_lon)
        rec_dist = J.pairwise_haversine(ev["r_lat"], ev["r_lon"])
        sigma_prior = rel_unc[ev["dom_cat"]]
        Sa = J.spatial_kernel(cell_dist, sigma_prior, L_prior)
        R = J.spatial_kernel(rec_dist, mdm_stddev, L_obs) + meas_stddev ** 2 * np.eye(len(b))

        prepared.append(dict(
            fid=ev["fid"], dom_cat=ev["dom_cat"], b=b, e_c=e_c, c_lat=c_lat, c_lon=c_lon,
            r_lat=ev["r_lat"], r_lon=ev["r_lon"], interps=interps, H0=H0, Sa=Sa, R=R,
        ))

    print(f"{len(prepared)} events have a nonempty candidate-cell neighborhood")

    def event_loglik(p, theta):
        if theta == 0.0:
            H = p["H0"]
        else:
            H = np.empty_like(p["H0"])
            for m, interp in enumerate(p["interps"]):
                dx, dy = offsets_km(p["c_lat"], p["c_lon"], p["r_lat"][m], p["r_lon"][m])
                qlat, qlon = rotate_back_to_lonlat(dx, dy, p["r_lat"][m], p["r_lon"][m], theta)
                H[m, :] = interp((qlat, qlon))
        A0 = H * p["e_c"][None, :]
        Sigma = A0 @ p["Sa"] @ A0.T + p["R"]
        sign, logdet = np.linalg.slogdet(Sigma)
        sol = np.linalg.solve(Sigma, p["b"])
        return -0.5 * (p["b"] @ sol + logdet + len(p["b"]) * np.log(2 * np.pi))

    # sanity check: theta=0 direct-index vs. bilinear-interpolated-at-grid-points
    # should reproduce joint_correlation_sweep's own (0km, 1.5km) baseline closely
    baseline_total = sum(event_loglik(p, 0.0) for p in prepared)
    print(f"\nsanity check -- theta=0 total log-lik: {baseline_total:.3f} "
          f"(compare to joint_correlation_sweep.py's L_prior=0,L_obs=1.5 baseline)")

    fids = sorted(set(p["fid"] for p in prepared))
    per_flight = {fid: [] for fid in fids}
    total_curve = []
    for theta in THETA_GRID_DEG:
        per_event_ll = [event_loglik(p, theta) for p in prepared]
        total_curve.append(sum(per_event_ll))
        for fid in fids:
            per_flight[fid].append(sum(ll for p, ll in zip(prepared, per_event_ll) if p["fid"] == fid))

    total_curve = np.array(total_curve)
    dtotal = total_curve - total_curve[np.argmin(np.abs(THETA_GRID_DEG))]
    best_i = np.argmax(total_curve)
    print(f"\npooled: best theta = {THETA_GRID_DEG[best_i]:+.0f} deg "
          f"(Delta log-lik = {dtotal[best_i]:+.2f} over {len(prepared)} events)")

    print(f"\n{'theta(deg)':>10}  {'Delta_loglik_pooled':>19}  " +
          "  ".join(f"{fid:>14}" for fid in fids))
    for i, theta in enumerate(THETA_GRID_DEG):
        row = "  ".join(f"{per_flight[fid][i] - per_flight[fid][np.argmin(np.abs(THETA_GRID_DEG))]:>14.2f}"
                        for fid in fids)
        print(f"{theta:>10.0f}  {dtotal[i]:>19.2f}  {row}")

    # per-event best theta -- is any preferred rotation consistent across events,
    # or scattered (i.e. one free parameter per event just fitting noise)?
    per_event_best = []
    for p in prepared:
        lls = np.array([event_loglik(p, th) for th in THETA_GRID_DEG])
        per_event_best.append(THETA_GRID_DEG[np.argmax(lls)])
    per_event_best = np.array(per_event_best)
    print(f"\nper-event best theta: median={np.median(per_event_best):+.1f}  "
          f"IQR=[{np.percentile(per_event_best,25):+.1f}, {np.percentile(per_event_best,75):+.1f}]  "
          f"values={sorted(per_event_best)}")

    for fc in jf_cache.values():
        fc.close()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    ax.plot(THETA_GRID_DEG, dtotal, "o-", color="k", label="pooled (all events)")
    for fid in fids:
        curve = np.array(per_flight[fid]) - per_flight[fid][np.argmin(np.abs(THETA_GRID_DEG))]
        ax.plot(THETA_GRID_DEG, curve, ".--", alpha=0.6, label=fid)
    ax.axvline(0, color="gray", lw=1)
    ax.set_xlabel("trial rotation theta (deg)")
    ax.set_ylabel("Delta log marginal likelihood vs. theta=0")
    ax.set_title("Does rotating footprints about the receptor\nimprove the fit at point-source events?")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.hist(per_event_best, bins=np.arange(-31, 32, 2))
    ax.set_xlabel("per-event best-fit theta (deg)")
    ax.set_ylabel("number of events")
    ax.set_title("Spread of per-event optimal rotation\n(tight -> real bias; scattered -> overfitting)")

    plt.savefig("runs/rotation_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> runs/rotation_check.png")


if __name__ == "__main__":
    main()
