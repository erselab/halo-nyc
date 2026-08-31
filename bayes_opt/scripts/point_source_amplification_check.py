"""Step 2 of the point-source amplification idea (follows
point_source_intersection_check.py): for each clustered-excursion event that
has nearby prior mass, estimate the source amplification needed to close the
gap between observed and modeled -- using a receptor-AVERAGED least-squares
fit across all "elevated" receptors near the event, not a single picked one.

For candidate source cell c in category k, the sensitivity of receptor i's
modeled enhancement to a change in c's scale factor x_{k,c} is
    A_i = H[i, c] * e_{k,c}          (e = that category's prior density at c)
and the unexplained gap is
    b_i = z_i - modeled_i            (already-optimized model's residual)
The least-squares amplification (minimizing sum_i (A_i * dx - b_i)^2) is
    dx = sum(A_i * b_i) / sum(A_i^2)
averaging over every elevated receptor in the event rather than solving from
one receptor's row alone, which is far noisier (measurement/footprint noise
in a single row, vs. an average sensitivity-weighted estimate here).

"Elevated" receptors for an event are its own flagged (>3sigma local
excursion) members PLUS same-sign neighbors within a small time pad that
clear a lower bar (>1sigma) -- capturing the actual ramp of the plume
crossing, not just the one or two samples that happened to cross the
arbitrary 3sigma threshold used to first flag the event.

Needs one targeted Jacobian row read per elevated receptor (jac[i, :, :],
the same cheap single-row-read pattern used earlier in this investigation
for flag_leg_edge_discontinuities) -- reads are cached so a receptor used in
more than one event is only read once. No full-file stream, no re-solve.
"""

from __future__ import annotations

import os
import sys

import numpy as np

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
FLIGHT = "20230805"

LOCAL_WINDOW_S = 45.0
LOCAL_BUFFER_S = 8.0
EXTREME_THRESH = 3.0
NEIGHBOR_THRESH = 1.5
NEIGHBOR_WINDOW_S = 15.0
EVENT_MERGE_GAP_S = 20.0
ELEVATED_PAD_S = 20.0
ELEVATED_THRESH = 1.0
SEARCH_RADIUS_KM = 10.0
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other"]


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


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    sigma_obs = cfg.get_float("observations", "error_stddev", default=0.02)
    rel_unc = relative_uncertainties(CATEGORIES, cfg)
    print(f"sigma_obs={sigma_obs} (from [observations] error_stddev); "
          f"prior relative uncertainty per category: {rel_unc}")
    R = inv.receptors
    flight_index = R["receptor_flight"].astype(int)
    z_all = R["enhancement"]
    modeled_all = R["modeled"]
    flag_all = R.get("outlier_flag", np.zeros_like(z_all)).astype(bool)

    fi = inv.flight_ids.index(FLIGHT)
    flight_sel = flight_index == fi
    lat_f = R["receptor_lat"][flight_sel]
    lon_f = R["receptor_lon"][flight_sel]
    z_f, modeled_f, flag_f = z_all[flight_sel], modeled_all[flight_sel], flag_all[flight_sel]
    t_f = _load_receptor_time(FLIGHT, FLIGHT_DATA_DIR, lat_f, lon_f)

    good = ~flag_f & np.isfinite(z_f) & np.isfinite(modeled_f)
    raw_idx_all = np.arange(z_f.size)
    lat, lon, z, modeled, t, raw_idx = (
        a[good] for a in (lat_f, lon_f, z_f, modeled_f, t_f, raw_idx_all))
    order = np.argsort(t)
    lat, lon, z, modeled, t, raw_idx = (
        a[order] for a in (lat, lon, z, modeled, t, raw_idx))
    # raw_idx[k] is now the row index into the flight's own Jacobian file for
    # the filtered/sorted position k -- needed for targeted single-row reads

    local_z = local_excursion_z(z, t)
    clustered_idx = classify_extremes(local_z, t)
    events = group_into_events(clustered_idx, t)
    print(f"{FLIGHT}: {len(events)} discrete excursion events")

    jf = JacobianFile(os.path.join(JAC_DIR, f"{FLIGHT}.nc"))
    jac = jf._ds.variables[jf._jac_var]
    row_cache = {}

    def get_row(k):
        ri = int(raw_idx[k])
        if ri not in row_cache:
            row_cache[ri] = np.asarray(jac[ri, :, :]).reshape(-1)
        return row_cache[ri]

    print(f"\n{'event':>5} {'dom_cat':>12} {'cell_dist_km':>12} {'n_elev':>6} "
          f"{'x_hat':>7} {'dx (+-sd)':>13} {'new_x':>7}  significance")
    print("-" * 95)

    for ei, members in enumerate(events):
        members = np.array(members)
        t0, t1 = t[members].min(), t[members].max()
        sign = np.sign(np.mean(z[members]))
        window = (t >= t0 - ELEVATED_PAD_S) & (t <= t1 + ELEVATED_PAD_S)
        elevated = window & (np.sign(local_z) == sign) & (np.abs(local_z) > ELEVATED_THRESH)
        elevated_idx = np.flatnonzero(elevated)
        if elevated_idx.size < 2:
            print(f"{ei:>5}  -- fewer than 2 elevated receptors, skipping --")
            continue

        elat, elon = lat[members].mean(), lon[members].mean()
        d_active = _haversine_km(inv.core.active_lat, inv.core.active_lon, elat, elon)
        near = d_active <= SEARCH_RADIUS_KM
        mass = {c: inv.group_fields[c][near].sum() for c in CATEGORIES}
        if all(v <= 0 for v in mass.values()):
            print(f"{ei:>5}  -- no prior mass nearby, skipping --")
            continue

        # peak-amplitude receptor's row picks the candidate cell: among active
        # cells within the search radius, the one maximizing sensitivity x
        # density -- NOT density alone, which can pick a cell the footprint
        # barely touches (found by inspection: H there ~1e-5 vs a ~1e-3 peak
        # elsewhere in the same row, making the least-squares fit divide by
        # ~0 and blow up to nonsense amplification factors).
        peak_k = members[np.argmax(np.abs(z[members] - modeled[members]))]
        peak_row = get_row(peak_k)
        best = None
        for c in CATEGORIES:
            dens_c = inv.group_fields[c]
            weight = np.where(near, peak_row[inv.core.active] * dens_c, -np.inf)
            j = int(np.argmax(weight))
            if best is None or weight[j] > best[0]:
                best = (weight[j], c, j)
        _, dom_cat, c_idx = best
        c_lat, c_lon = inv.core.active_lat[c_idx], inv.core.active_lon[c_idx]
        cell_dist = _haversine_km(elat, elon, c_lat, c_lon)
        e_kc = inv.group_fields[dom_cat][c_idx]
        x_hat_kc = inv.block(dom_cat)[c_idx]
        flat_cell_idx = inv.core.active[c_idx]

        A, b = [], []
        for k in elevated_idx:
            row = get_row(k)
            H_ic = row[flat_cell_idx]
            A.append(H_ic * e_kc)
            b.append(z[k] - modeled[k])
        A, b = np.array(A), np.array(b)
        if np.sum(A ** 2) == 0:
            print(f"{ei:>5}  -- candidate cell has zero sensitivity to these receptors, skipping --")
            continue

        # regularized (Bayesian, not unconstrained-LSQ) 1-parameter update:
        # prior dx ~ N(0, sigma_prior^2) with sigma_prior = this category's
        # relative uncertainty (same prior the real inversion uses at this
        # cell); likelihood b_i = A_i*dx + eps_i, eps_i ~ N(0, sigma_obs^2).
        # Naturally shrinks toward "no change" when the elevated receptors
        # don't share consistent sensitivity to this one cell, instead of
        # the unconstrained-LSQ blowups seen before adding this.
        sigma_prior = rel_unc[dom_cat]
        precision = 1.0 / sigma_prior ** 2 + np.sum(A ** 2) / sigma_obs ** 2
        dx = (np.sum(A * b) / sigma_obs ** 2) / precision
        dx_sd = np.sqrt(1.0 / precision)
        new_x = x_hat_kc + dx
        sig = abs(dx) / dx_sd if dx_sd > 0 else 0.0

        note = "well-constrained" if sig > 2 else "prior-dominated (weak evidence)"
        print(f"{ei:>5} {dom_cat:>12} {cell_dist:>12.2f} {len(elevated_idx):>6} "
              f"{x_hat_kc:>7.2f} {dx:>+7.2f}±{dx_sd:<5.2f} {new_x:>7.2f}  "
              f"{sig:>5.1f}sigma  {note}")

    jf.close()


if __name__ == "__main__":
    main()
