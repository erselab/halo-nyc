"""Joint sensitivity sweep: treat the prior spatial correlation length for
landfill/WWTP cells and the observation-error (MDM) correlation length as
fitting terms together, and see which combination the data actually
support -- via marginal likelihood, not just point estimates.

For each landfill/WWTP-dominated event (from landfill_wwtp_coherence_check.py):
build a small local linear-Gaussian model over its nearby candidate-category
cells and its elevated receptors --

    prior:      dx ~ N(0, Sa),      Sa[c,c']    = sigma_prior^2 exp(-d_cc'/L_prior)
    likelihood: b = A dx + eps,     R[i,i']     = mdm_stddev^2 exp(-d_ii'/L_obs) + meas_stddev^2 I
                A[i,c] = H[i,c] * density_c,     b_i = z_i - modeled_i

marginalizing out dx gives b ~ N(0, Sigma),  Sigma = A Sa A^T + R -- the
marginal log-likelihood of the observed local gap under each (L_prior,
L_obs) pair, summed across all qualifying events (independent), is what
determines which combination the data support: the log|Sigma| term
penalizes complexity automatically (Occam's razor), so this isn't just
"more freedom always fits better."

L_prior=0 (current [category_spatial] default for landfill/wastewater) and
L_obs=1.5km (current [observations] mdm_correlation_length_km) is the
baseline the sweep is measured against.

Still no re-solve -- everything here is a small (n_cells x n_cells,
n_receptors x n_receptors) closed-form Gaussian update per event, reusing
the same cached Jacobian-row-read pattern as point_source_amplification_
check.py.
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
from halo_oe.background import _load_receptor_time
from halo_oe.decomposition import relative_uncertainties
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"

LOCAL_WINDOW_S = 45.0
LOCAL_BUFFER_S = 8.0
EXTREME_THRESH = 3.0
NEIGHBOR_THRESH = 1.5
NEIGHBOR_WINDOW_S = 15.0
EVENT_MERGE_GAP_S = 20.0
ELEVATED_PAD_S = 20.0
ELEVATED_THRESH = 1.0
SEARCH_RADIUS_KM = 10.0
NEIGHBORHOOD_RADIUS_KM = 6.0   # candidate-cell neighborhood for the spatial prior kernel
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other"]

L_PRIOR_GRID_KM = [0.0, 2.0]   # confirmed inert in the first pass; kept only as a sanity check
L_OBS_GRID_KM = [1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]


def local_excursion_z(z, t, window_s=LOCAL_WINDOW_S, buffer_s=LOCAL_BUFFER_S):
    n = len(z)
    local_resid = np.full(n, np.nan)
    for i in range(n):
        dt = np.abs(t - t[i])
        ring = (dt <= window_s) & (dt > buffer_s)
        if ring.sum() >= 3:
            local_resid[i] = z[i] - np.median(z[ring])
    mad = 1.4826 * np.nanmedian(np.abs(local_resid - np.nanmedian(local_resid)))
    return local_resid / mad


def classify_extremes(local_z, t, thresh=EXTREME_THRESH,
                       neighbor_thresh=NEIGHBOR_THRESH, window_s=NEIGHBOR_WINDOW_S):
    idx = np.flatnonzero(np.abs(local_z) > thresh)
    clustered = []
    for i in idx:
        sign = np.sign(local_z[i])
        near = np.flatnonzero((np.abs(t - t[i]) <= window_s) & (np.abs(t - t[i]) > 0))
        if any(np.isfinite(local_z[j]) and np.sign(local_z[j]) == sign
               and abs(local_z[j]) > neighbor_thresh for j in near):
            clustered.append(i)
    return np.array(sorted(clustered), dtype=int)


def group_into_events(idx, t, gap_s=EVENT_MERGE_GAP_S):
    if len(idx) == 0:
        return []
    events, cur = [], [idx[0]]
    for a, b in zip(idx[:-1], idx[1:]):
        if t[b] - t[a] <= gap_s:
            cur.append(b)
        else:
            events.append(cur)
            cur = [b]
    events.append(cur)
    return events


def pairwise_haversine(lat, lon):
    n = len(lat)
    D = np.zeros((n, n))
    for i in range(n):
        D[i, :] = _haversine_km(lat[i], lon[i], lat, lon)
    return D


def spatial_kernel(dist, sigma, length_km):
    if length_km <= 0:
        return sigma ** 2 * np.eye(dist.shape[0])
    return sigma ** 2 * np.exp(-dist / length_km)


def collect_events(inv):
    """Same detection/filtering as landfill_wwtp_coherence_check.py, but
    keeping full per-event receptor/cell info instead of just the decay curve."""
    events_out = []
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)

    for fid in inv.flight_ids:
        fi = inv.flight_ids.index(fid)
        flight_sel = flight_index == fi
        lat_f, lon_f = R["receptor_lat"][flight_sel], R["receptor_lon"][flight_sel]
        z_f, modeled_f = R["enhancement"][flight_sel], R["modeled"][flight_sel]
        flag_all = R.get("outlier_flag", np.zeros_like(R["enhancement"])).astype(bool)
        flag_f = flag_all[flight_sel]
        t_f = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat_f, lon_f)

        good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
        raw_idx_all = np.arange(z_f.size)
        lat, lon, z, modeled, t, raw_idx = (
            a[good] for a in (lat_f, lon_f, z_f, modeled_f, t_f, raw_idx_all))
        order = np.argsort(t)
        lat, lon, z, modeled, t, raw_idx = (a[order] for a in (lat, lon, z, modeled, t, raw_idx))

        local_z = local_excursion_z(z, t)
        clustered_idx = classify_extremes(local_z, t)
        events = group_into_events(clustered_idx, t)

        for members in events:
            members = np.array(members)
            elat, elon = lat[members].mean(), lon[members].mean()
            d_active = _haversine_km(inv.core.active_lat, inv.core.active_lon, elat, elon)
            near = d_active <= SEARCH_RADIUS_KM
            mass = {c: inv.group_fields[c][near].sum() for c in CATEGORIES}
            ps_mass = mass["landfill"] + mass["wastewater"]
            other_mass = mass["natural_gas"] + mass["other"]
            if ps_mass <= 0 or ps_mass <= other_mass:
                continue
            dom_cat = "landfill" if mass["landfill"] >= mass["wastewater"] else "wastewater"

            t0, t1 = t[members].min(), t[members].max()
            sign = np.sign(np.mean(z[members]))
            window = (t >= t0 - ELEVATED_PAD_S) & (t <= t1 + ELEVATED_PAD_S)
            elevated = window & (np.sign(local_z) == sign) & (np.abs(local_z) > ELEVATED_THRESH)
            elevated_idx = np.flatnonzero(elevated)
            if elevated_idx.size < 2:
                continue

            events_out.append(dict(
                fid=fid, dom_cat=dom_cat, elat=elat, elon=elon,
                r_lat=lat[elevated_idx], r_lon=lon[elevated_idx],
                r_raw_idx=raw_idx[elevated_idx],
                z=z[elevated_idx], modeled=modeled[elevated_idx],
            ))
    return events_out


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    mdm_stddev = cfg.get_float("observations", "mdm_stddev", default=0.025)
    meas_stddev = cfg.get_float("observations", "measurement_stddev", default=0.01)
    rel_unc = relative_uncertainties(CATEGORIES, cfg)
    print(f"mdm_stddev={mdm_stddev}  measurement_stddev={meas_stddev}  rel_unc={rel_unc}")

    events = collect_events(inv)
    print(f"{len(events)} qualifying landfill/WWTP-dominated events across all 6 flights")

    # cache Jacobian files/rows per flight
    jf_cache, row_cache = {}, {}

    def get_row(fid, raw_idx):
        key = (fid, int(raw_idx))
        if key not in row_cache:
            if fid not in jf_cache:
                jf_cache[fid] = JacobianFile(os.path.join(JAC_DIR, f"{fid}.nc"))
            jf = jf_cache[fid]
            jac = jf._ds.variables[jf._jac_var]
            row_cache[key] = np.asarray(jac[int(raw_idx), :, :]).reshape(-1)
        return row_cache[key]

    # precompute, per event: A0 (H*density over neighborhood cells), b, cell
    # distances, receptor distances -- everything that doesn't depend on
    # (L_prior, L_obs) so the sweep itself is cheap
    prepared = []
    for ev in events:
        d_active = _haversine_km(inv.core.active_lat, inv.core.active_lon, ev["elat"], ev["elon"])
        near = d_active <= NEIGHBORHOOD_RADIUS_KM
        dens = inv.group_fields[ev["dom_cat"]]
        cell_mask = near & (dens > 0)
        if cell_mask.sum() == 0:
            continue
        c_lat = inv.core.active_lat[cell_mask]
        c_lon = inv.core.active_lon[cell_mask]
        e_c = dens[cell_mask]
        flat_idx = inv.core.active[cell_mask]

        rows = np.stack([get_row(ev["fid"], ri) for ri in ev["r_raw_idx"]])   # (m, n_grid_cells)
        H_ic = rows[:, flat_idx]                                              # (m, n_near_cells)
        A0 = H_ic * e_c[None, :]                                              # (m, n_near_cells)
        b = ev["z"] - ev["modeled"]

        cell_dist = pairwise_haversine(c_lat, c_lon)
        rec_dist = pairwise_haversine(ev["r_lat"], ev["r_lon"])

        prepared.append(dict(dom_cat=ev["dom_cat"], A0=A0, b=b,
                              cell_dist=cell_dist, rec_dist=rec_dist, n_cells=len(e_c)))

    print(f"{len(prepared)} events have a nonempty candidate-cell neighborhood "
          f"(median n_cells={np.median([p['n_cells'] for p in prepared]):.0f})")

    loglik = np.zeros((len(L_PRIOR_GRID_KM), len(L_OBS_GRID_KM)))
    for pi, L_prior in enumerate(L_PRIOR_GRID_KM):
        for oi, L_obs in enumerate(L_OBS_GRID_KM):
            total = 0.0
            for p in prepared:
                sigma_prior = rel_unc[p["dom_cat"]]
                Sa = spatial_kernel(p["cell_dist"], sigma_prior, L_prior)
                R = spatial_kernel(p["rec_dist"], mdm_stddev, L_obs) + meas_stddev ** 2 * np.eye(len(p["b"]))
                Sigma = p["A0"] @ Sa @ p["A0"].T + R
                b = p["b"]
                sign, logdet = np.linalg.slogdet(Sigma)
                sol = np.linalg.solve(Sigma, b)
                ll = -0.5 * (b @ sol + logdet + len(b) * np.log(2 * np.pi))
                total += ll
            loglik[pi, oi] = total

    baseline = loglik[L_PRIOR_GRID_KM.index(0.0), L_OBS_GRID_KM.index(1.5)]
    dloglik = loglik - baseline

    print(f"\nlog-likelihood relative to baseline (L_prior=0km, L_obs=1.5km):")
    header = "L_prior\\L_obs " + " ".join(f"{l:>8.1f}" for l in L_OBS_GRID_KM)
    print(header)
    for pi, L_prior in enumerate(L_PRIOR_GRID_KM):
        row = " ".join(f"{dloglik[pi, oi]:>8.2f}" for oi in range(len(L_OBS_GRID_KM)))
        print(f"{L_prior:>13.1f} {row}")

    best = np.unravel_index(np.argmax(loglik), loglik.shape)
    print(f"\nbest combination: L_prior={L_PRIOR_GRID_KM[best[0]]}km, "
          f"L_obs={L_OBS_GRID_KM[best[1]]}km  (Delta log-lik = {dloglik[best]:+.2f} "
          f"over {len(prepared)} events, {sum(p['A0'].shape[0] for p in prepared)} total receptor-obs)")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(dloglik, origin="lower", aspect="auto", cmap="RdBu_r",
                    vmin=-np.abs(dloglik).max(), vmax=np.abs(dloglik).max())
    ax.set_xticks(range(len(L_OBS_GRID_KM))); ax.set_xticklabels(L_OBS_GRID_KM)
    ax.set_yticks(range(len(L_PRIOR_GRID_KM))); ax.set_yticklabels(L_PRIOR_GRID_KM)
    ax.set_xlabel("observation-error (MDM) correlation length (km)")
    ax.set_ylabel("prior spatial correlation length, landfill/WWTP (km)")
    ax.set_title("Delta log marginal likelihood vs. baseline (0km, 1.5km)")
    for pi in range(len(L_PRIOR_GRID_KM)):
        for oi in range(len(L_OBS_GRID_KM)):
            ax.text(oi, pi, f"{dloglik[pi, oi]:+.1f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax, label="Delta log-likelihood")
    plt.savefig("figures/joint_correlation_sweep.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/joint_correlation_sweep.png")

    for fc in jf_cache.values():
        fc.close()


if __name__ == "__main__":
    main()
