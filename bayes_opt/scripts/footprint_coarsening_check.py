"""Phase 3 of the plan to attack hypothesis (a): does artificially coarsening
real footprints shrink modeled amplitude at real clustered events, more for
`805`/`809` than for an already-explained control flight?

§7's original motivation: STILT/HRRR transport resolution is coarser than
the observation binning, so the model may be structurally unable to
reproduce sharp real enhancements. Phases 1-2 found `805`/`809` (not the
other 4 flights) show the along-track signature this hypothesis predicts,
and that signature gets *clearer*, not weaker, once isolated to real
plume-affected receptors (`along_track_plume_restricted_check.py`). This
phase tests the mechanism directly and causally: deliberately smooth each
event receptor's raw Jacobian row with a small Gaussian kernel (sigma = 0
[baseline], 1, 2, 3, 5 km -- conserves total footprint mass), recompute the
modeled enhancement, and see whether/how fast it shrinks toward zero as
sigma increases. If under-resolution is what differentiates 805/809, that
shrinkage should be *faster* for their events than for a control flight's.

Reuses:
- `along_track_outlier_check2.py`'s clustered-event classification
  (`local_excursion_z`, `classify_extremes`) and `joint_correlation_sweep.
  group_into_events` -- same event definition as Phase 2, but every
  clustered event (no landfill/WWTP-dominance filter -- this test is about
  any sharp real feature, not point sources specifically).
- Each flight's own single-flight bundle (`runs/single_<fid>`, from
  `run_single_flight_inversions.sh`) -- that flight's own best-fit density,
  not the shared joint one, is the right density for this test.
- `halo_oe.plotting._flux_block_prior_density` -- the same prior-density
  lookup used by the production flux-map plot.

For each event's elevated receptors, only the CORE contribution to Hx̂ (the
part depending on the fine-scale footprint) is recomputed under coarsening;
the buffer/background-offset contribution is added back unchanged (buffer
is already coarse; background-offset is a per-receptor scalar unrelated to
footprint shape) via a delta against the bundle's own saved `modeled` value
-- same delta trick already validated in rotation_check.py.

No re-solve; reads existing single-flight bundles and streams a small local
patch of each event receptor's Jacobian row (no full-matrix materialization).
"""

from __future__ import annotations

import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, ".")

from adapters.jacobian_operator import JacobianFile
from halo_oe.background import _load_receptor_time
from halo_oe.io_bundle import load_inversion
from halo_oe.plotting import _flux_block_prior_density

from along_track_outlier_check2 import classify_extremes, local_excursion_z
from joint_correlation_sweep import group_into_events

JAC_DIR = "/scratch/scrowel3_lab/halo-nyc/stilt/harvard_jacobians"
FLIGHT_DATA_DIR = "/scratch/scrowel3_lab/halo-nyc/flight_data"

TEST_FLIGHTS = ["20230805", "20230809", "20230726_2"]   # last one is the negative control
CONTROL_FLIGHTS = {"20230726_2"}

EVENT_MERGE_GAP_S = 20.0
ELEVATED_PAD_S = 20.0
ELEVATED_THRESH = 1.0
PATCH_RADIUS_KM = 30.0     # must clear footprint decay (~15-20km, §7.3) + largest sigma
SIGMAS_KM = [0.0, 1.0, 2.0, 3.0, 5.0]
KM_PER_DEG_LAT = 111.32


def collect_all_events(z, modeled, t):
    """Every clustered excursion event (no category filter), with the
    'elevated' receptor window around each -- same convention as
    joint_correlation_sweep.collect_events, minus the landfill/WWTP mass
    dominance test."""
    local_z, _ = local_excursion_z(z, t)
    clustered, _ = classify_extremes(local_z, t)
    if clustered.size == 0:
        return []
    events = group_into_events(np.sort(clustered), t, gap_s=EVENT_MERGE_GAP_S)
    out = []
    for members in events:
        members = np.array(members)
        t0, t1 = t[members].min(), t[members].max()
        sign = np.sign(np.mean(z[members]))
        window = (t >= t0 - ELEVATED_PAD_S) & (t <= t1 + ELEVATED_PAD_S)
        elevated = window & (np.sign(local_z) == sign) & (np.abs(local_z) > ELEVATED_THRESH)
        idx = np.flatnonzero(elevated)
        if idx.size >= 2:
            out.append(idx)
    return out


def local_patch_bounds(grid_lat, grid_lon, center_lat, center_lon, radius_km=PATCH_RADIUS_KM):
    dlat = radius_km / KM_PER_DEG_LAT
    dlon = radius_km / (KM_PER_DEG_LAT * max(np.cos(np.radians(center_lat)), 0.1))
    lat_idx = np.flatnonzero((grid_lat >= center_lat - dlat) & (grid_lat <= center_lat + dlat))
    lon_idx = np.flatnonzero((grid_lon >= center_lon - dlon) & (grid_lon <= center_lon + dlon))
    return lat_idx, lon_idx


def core_contribution(row2d, lat_idx, lon_idx, core_row, core_col, total_density, sigma_px):
    """H (optionally Gaussian-smoothed) dotted against the total posterior
    density, restricted to core cells that fall inside this patch."""
    r0, c0 = lat_idx.min(), lon_idx.min()
    patch = row2d[np.ix_(lat_idx, lon_idx)]
    if sigma_px > 0:
        patch = gaussian_filter(patch, sigma=sigma_px, mode="constant", cval=0.0)
    in_patch = ((core_row >= lat_idx.min()) & (core_row <= lat_idx.max()) &
                (core_col >= lon_idx.min()) & (core_col <= lon_idx.max()))
    pr, pc = core_row[in_patch] - r0, core_col[in_patch] - c0
    return float(np.sum(patch[pr, pc] * total_density[in_patch]))


def main():
    grid_res_km = None
    results = {}

    for fid in TEST_FLIGHTS:
        bundle_dir = f"runs/single_{fid}"
        if not os.path.exists(os.path.join(bundle_dir, "layout.json")):
            print(f"{fid}: no single-flight bundle at {bundle_dir} -- skipping")
            continue
        inv = load_inversion(bundle_dir)
        R = inv.receptors
        z, modeled = R["enhancement"], R["modeled"]
        flag = R.get("outlier_flag", np.zeros_like(z)).astype(bool)
        lat, lon = R["receptor_lat"], R["receptor_lon"]
        t = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)

        good = ~flag & np.isfinite(z) & np.isfinite(modeled)
        raw_idx_all = np.arange(z.size)
        lat_g, lon_g, z_g, modeled_g, t_g, raw_idx = (
            a[good] for a in (lat, lon, z, modeled, t, raw_idx_all))
        order = np.argsort(t_g)
        lat_g, lon_g, z_g, modeled_g, t_g, raw_idx = (
            a[order] for a in (lat_g, lon_g, z_g, modeled_g, t_g, raw_idx))

        events = collect_all_events(z_g, modeled_g, t_g)
        print(f"\n=== {fid} ({'CONTROL' if fid in CONTROL_FLIGHTS else 'test'}): "
              f"{len(events)} clustered events ===")
        if not events:
            continue

        cats = [b.name for b in inv.state.blocks if b.name not in ("bc", "buffer")]
        total_density = sum(inv.block(name) * _flux_block_prior_density(inv, name) for name in cats)

        jf = JacobianFile(os.path.join(JAC_DIR, f"{fid}.nc"))
        jac = jf._ds.variables[jf._jac_var]
        if grid_res_km is None:
            grid_res_km = abs(jf.grid.lat[1] - jf.grid.lat[0]) * KM_PER_DEG_LAT
        core_row, core_col = np.unravel_index(inv.core.active, jf.grid.shape)

        # per-sigma, per-event mean |modeled_coarsened - modeled_original| across
        # its elevated receptors (the delta this test is actually about)
        shrink_by_sigma = {s: [] for s in SIGMAS_KM}
        for ev_idx in events:
            for ri in ev_idx:
                row2d = np.asarray(jac[int(raw_idx[ri]), :, :])
                r_lat, r_lon = lat_g[ri], lon_g[ri]
                lat_idx, lon_idx = local_patch_bounds(jf.grid.lat, jf.grid.lon, r_lat, r_lon)
                base = core_contribution(row2d, lat_idx, lon_idx, core_row, core_col,
                                          total_density, sigma_px=0.0)
                for s_km in SIGMAS_KM:
                    if s_km == 0.0:
                        delta = 0.0
                    else:
                        smoothed = core_contribution(row2d, lat_idx, lon_idx, core_row, core_col,
                                                      total_density, sigma_px=s_km / grid_res_km)
                        delta = smoothed - base
                    coarsened_modeled = modeled_g[ri] + delta
                    shrink_by_sigma[s_km].append(coarsened_modeled - modeled_g[ri])
        jf.close()

        print(f"  sigma(km)  mean_delta_modeled  n_receptors")
        for s_km in SIGMAS_KM:
            vals = np.array(shrink_by_sigma[s_km])
            print(f"  {s_km:>7.1f}  {vals.mean():>+18.5f}  {vals.size:>11d}")
        results[fid] = shrink_by_sigma

    print("\n(if resolution differentiates 805/809 from the control, the magnitude of "
          "mean_delta_modeled should grow faster with sigma for 805/809 than for 726_2 -- "
          "compare the sigma=5.0 row across flights, not just whether it's nonzero.)")

    for fid, sbs in results.items():
        rate = np.mean([abs(v) for v in sbs[5.0]]) if sbs[5.0] else float("nan")
        print(f"  {fid}: mean |shrinkage| at sigma=5km = {rate:.5f}")


if __name__ == "__main__":
    main()
