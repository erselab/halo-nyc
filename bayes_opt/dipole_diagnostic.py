"""Extend the RESIDUAL_INVESTIGATION.md section-5 prior-shape/dipole diagnostic
from 20230726_1 / 20230728_1 to the three flights that were never worked
through: 20230728_2, 20230805, 20230809.

Method (same as the two flights already done by hand):
  1. Bin each flight's residuals (z - Hx̂, outliers excluded) onto a coarse
     grid and flag spatially-coherent, same-sign clusters (not just noisy
     single bins).
  2. For each cluster's centroid, check how much prior mass each category
     (natural_gas, landfill, wastewater, other) has within that category's
     configured [category_spatial] correlation length -- this sets the hard
     ceiling on how far the flux-state posterior can "reach" to cover the
     feature, independent of the prior's shape.
  3. Look at the posterior-minus-prior difference in natural_gas (the only
     category here with a nonzero correlation length, so the only one that
     can produce a dipole rather than an independent point nudge) near the
     cluster.
  4. Save an overlay plot per cluster (residual scatter + all 4 category
     priors, zoomed to the cluster) for visual inspection -- a numeric
     summary alone is not trusted here (see investigation doc §9.6).

Run from the bayes_opt directory with the `analysis` conda env:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 dipole_diagnostic.py
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, ".")

from adapters.gridded_state import _haversine_km
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
TARGET_FLIGHTS = ["20230728_2", "20230805", "20230809"]
CATEGORIES = ["natural_gas", "landfill", "wastewater", "other"]
CORR_LEN_KM = {"natural_gas": 5.0, "landfill": 0.0, "wastewater": 0.0, "other": 0.0}
BIN_DEG = 0.045          # ~5km, matches natural_gas correlation length
MIN_BIN_COUNT = 3        # receptors needed for a bin's mean to count
CLUSTER_Z = 1.5          # bin |mean| / flight-MAD threshold to flag a bin
MIN_CLUSTER_BINS = 3      # connected same-sign bins needed to call it a cluster


def flights_present(flight_ids, flight_index):
    out = []
    for i, fid in enumerate(flight_ids):
        sel = flight_index == i
        if sel.any():
            out.append((fid, sel))
    return out


def bin_residuals(rlat, rlon, resid):
    lat_edges = np.arange(rlat.min() - BIN_DEG, rlat.max() + BIN_DEG, BIN_DEG)
    lon_edges = np.arange(rlon.min() - BIN_DEG, rlon.max() + BIN_DEG, BIN_DEG)
    ilat = np.digitize(rlat, lat_edges) - 1
    ilon = np.digitize(rlon, lon_edges) - 1
    n_lat, n_lon = len(lat_edges) - 1, len(lon_edges) - 1
    mean = np.full((n_lat, n_lon), np.nan)
    count = np.zeros((n_lat, n_lon), dtype=int)
    for i in range(n_lat):
        for j in range(n_lon):
            m = (ilat == i) & (ilon == j)
            c = m.sum()
            count[i, j] = c
            if c >= MIN_BIN_COUNT:
                mean[i, j] = resid[m].mean()
    lat_c = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_c = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    return mean, count, lat_c, lon_c


def find_clusters(mean, lat_c, lon_c, mad):
    """Flood-fill connected (rook-adjacent) same-sign bins with |mean| > CLUSTER_Z*mad."""
    n_lat, n_lon = mean.shape
    flagged = np.abs(mean) > CLUSTER_Z * mad
    visited = np.zeros_like(flagged, dtype=bool)
    clusters = []
    for i in range(n_lat):
        for j in range(n_lon):
            if not flagged[i, j] or visited[i, j]:
                continue
            sign = np.sign(mean[i, j])
            stack, comp = [(i, j)], []
            visited[i, j] = True
            while stack:
                ci, cj = stack.pop()
                comp.append((ci, cj))
                # 8-connectivity: legs fly a fixed SW/NE diagonal axis, so
                # consecutive same-leg bins are diagonal neighbors, not just
                # rook-adjacent -- 4-connectivity would fragment a real
                # diagonal streak into many spurious tiny "clusters".
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1),
                               (1, 1), (1, -1), (-1, 1), (-1, -1)):
                    ni, nj = ci + di, cj + dj
                    if (0 <= ni < n_lat and 0 <= nj < n_lon and not visited[ni, nj]
                            and flagged[ni, nj] and np.sign(mean[ni, nj]) == sign):
                        visited[ni, nj] = True
                        stack.append((ni, nj))
            if len(comp) >= MIN_CLUSTER_BINS:
                lats = np.array([lat_c[ci] for ci, cj in comp])
                lons = np.array([lon_c[cj] for ci, cj in comp])
                vals = np.array([mean[ci, cj] for ci, cj in comp])
                clusters.append({
                    "sign": sign, "n_bins": len(comp),
                    "centroid_lat": lats.mean(), "centroid_lon": lons.mean(),
                    "peak": vals[np.argmax(np.abs(vals))],
                    "lat_span_km": (lats.max() - lats.min()) * 111.0,
                    "lon_span_km": (lons.max() - lons.min()) * 111.0 * np.cos(np.radians(lats.mean())),
                })
    clusters.sort(key=lambda c: -c["n_bins"])
    return clusters


def prior_mass_near(inv, clat, clon, radius_km):
    d = _haversine_km(inv.core.active_lat, inv.core.active_lon, clat, clon)
    near = d <= radius_km
    return {cat: float(inv.group_fields[cat][near].sum()) for cat in CATEGORIES}, near


def plot_cluster(inv, fid, cluster, rlat, rlon, resid, out_path):
    clat, clon = cluster["centroid_lat"], cluster["centroid_lon"]
    pad = 0.3
    latm = (inv.core.active_lat > clat - pad) & (inv.core.active_lat < clat + pad)
    lonm = (inv.core.active_lon > clon - pad) & (inv.core.active_lon < clon + pad)
    cellm = latm & lonm
    rsel = ((rlat > clat - pad) & (rlat < clat + pad)
            & (rlon > clon - pad) & (rlon < clon + pad))

    fig, ax = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    a = ax[0, 0]
    sc = a.scatter(rlon[rsel], rlat[rsel], c=resid[rsel], cmap="RdBu_r",
                    vmin=-np.abs(resid[rsel]).max(), vmax=np.abs(resid[rsel]).max(), s=8)
    a.scatter([clon], [clat], marker="x", color="k", s=80)
    plt.colorbar(sc, ax=a, label="residual (ppm)")
    a.set_title(f"{fid} residual  (x = cluster centroid)")

    # the natural_gas state block is a *multiplicative scale factor* (xa == 1.0
    # everywhere); group_fields is the separate prior flux-density map. The
    # actual posterior flux perturbation is (scale - 1) * prior_density, not
    # a raw block-minus-groupprior difference (those are different quantities
    # and don't subtract meaningfully).
    scale = inv.block("natural_gas")
    flux_diff = (scale - 1.0) * inv.group_fields["natural_gas"]
    a = ax[0, 1]
    vmax = np.abs(flux_diff[cellm]).max() if cellm.any() and np.any(flux_diff[cellm]) else 1
    sc = a.scatter(inv.core.active_lon[cellm], inv.core.active_lat[cellm], c=flux_diff[cellm],
                    cmap="PuOr_r", s=10, vmin=-vmax, vmax=vmax)
    a.scatter([clon], [clat], marker="x", color="k", s=80)
    plt.colorbar(sc, ax=a, label="(scale-1) x prior (natural_gas)")
    a.set_title("natural_gas: posterior flux perturbation")

    for k, cat in enumerate(CATEGORIES):
        a = ax.flat[2 + k]
        vals = inv.group_fields[cat][cellm]
        sc = a.scatter(inv.core.active_lon[cellm], inv.core.active_lat[cellm], c=vals,
                        cmap="viridis", s=10)
        a.scatter([clon], [clat], marker="x", color="k", s=80)
        plt.colorbar(sc, ax=a, label=f"prior {cat}")
        a.set_title(f"prior: {cat}")

    fig.suptitle(f"{fid} cluster @ ({clat:.3f}, {clon:.3f}), "
                 f"sign={'+' if cluster['sign'] > 0 else '-'}, peak={cluster['peak']:.3f} ppm, "
                 f"~{max(cluster['lat_span_km'], cluster['lon_span_km']):.0f}km")
    plt.savefig(out_path, bbox_inches="tight", dpi=110)
    plt.close(fig)


def main():
    inv = load_inversion(BUNDLE)
    R = inv.receptors
    rlat, rlon = R["receptor_lat"], R["receptor_lon"]
    z, modeled = R["enhancement"], R["modeled"]
    resid = z - modeled
    flag = R.get("outlier_flag", np.zeros_like(z)).astype(bool)
    flight_index = R["receptor_flight"].astype(int)
    sels = dict(flights_present(inv.flight_ids, flight_index))

    for fid in TARGET_FLIGHTS:
        sel = sels[fid] & ~flag & np.isfinite(resid)
        rl, ro, rs = rlat[sel], rlon[sel], resid[sel]
        mad = 1.4826 * np.median(np.abs(rs - np.median(rs)))
        mean, count, lat_c, lon_c = bin_residuals(rl, ro, rs)
        clusters = find_clusters(mean, lat_c, lon_c, mad)

        print(f"\n=== {fid} ===  n_obs={sel.sum()}  MAD={mad:.4f} ppm  "
              f"n_clusters={len(clusters)}")
        for ci, c in enumerate(clusters[:4]):
            radius = max(CORR_LEN_KM["natural_gas"], 1.0)
            mass, near = prior_mass_near(inv, c["centroid_lat"], c["centroid_lon"], radius)
            zero_mass = all(v == 0.0 for v in mass.values())
            span = max(c["lat_span_km"], c["lon_span_km"])
            reach = "within natural_gas 5km reach" if span <= 2 * CORR_LEN_KM["natural_gas"] \
                else "EXCEEDS natural_gas 5km reach"
            print(f"  cluster {ci}: n_bins={c['n_bins']} sign={'+' if c['sign']>0 else '-'} "
                  f"peak={c['peak']:.3f} ppm centroid=({c['centroid_lat']:.3f},{c['centroid_lon']:.3f}) "
                  f"span~{span:.0f}km  {reach}")
            print(f"    prior mass within {radius:.0f}km: " +
                  ", ".join(f"{k}={v:.3g}" for k, v in mass.items()) +
                  ("   [ZERO PRIOR MASS]" if zero_mass else ""))
            out_path = f"runs/dipole_diagnostic_{fid}_cluster{ci}.png"
            plot_cluster(inv, fid, c, rlat, rlon, resid, out_path)
            print(f"    plot -> {out_path}")


if __name__ == "__main__":
    main()
