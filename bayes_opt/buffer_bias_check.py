"""Check whether the buffer region is contributing to the unexplained residual
bias found in RESIDUAL_INVESTIGATION.md §9 (728_2's broad clusters, 805/809's
leg-banding).

Phase 0 (this file, `phase0_tile_check`): using only the already-saved bundle
plus a cheap re-read of the inventory file (NOT the Jacobian), compare the
buffer's posterior to its prior per super-cell, and check whether any §9
cluster sits near enough to a buffer super-cell for a coarse-tile smoothing
error to plausibly explain it directly. Buffer super-cells only exist outside
the core mask, so this also reports the distance from each cluster to the
nearest buffer cell -- if that distance is large, a co-located tile-edge
artifact cannot be the mechanism (a footprint-reach argument, tested
separately in Phase 1, remains possible).

Run with the `analysis` conda env from the bayes_opt directory:
    /gpfs/fs1/home/scrowel3/miniforge3/envs/analysis/bin/python3 buffer_bias_check.py
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
from goe.config import Config
from halo_oe.buffer import build_buffer
from halo_oe.emissions import category_priors_on_grid
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"

# §9 cluster centroids (flight, lat, lon, note)
CLUSTERS = [
    ("20230728_2", 40.770, -73.218, "cluster0, -0.096ppm, ~64km, EXCEEDS reach"),
    ("20230728_2", 40.507, -74.141, "cluster1, +0.054ppm, ~46km, zero prior mass"),
    ("20230805", 40.692, -74.320, "cluster0, -0.039ppm, ~30km, EXCEEDS reach"),
    ("20230805", 40.800, -74.365, "cluster2, +0.036ppm, ~15km, near 'other' mass"),
    ("20230809", 40.719, -74.486, "cluster0, -0.046ppm, ~15km, EXCEEDS reach"),
    ("20230809", 40.583, -74.636, "cluster1, -0.056ppm, ~10km, zero prior mass"),
]


def build_prior_and_diff(inv, cfg):
    # config.ini stores emissions path relative to the bayes_opt dir; run this
    # script with that as cwd.
    emissions_path = cfg.get("emissions", "path")
    priors = category_priors_on_grid(emissions_path, inv.grid, sources=(inv.inventory,))
    buf = build_buffer(inv.grid, inv.core, cfg)

    # sanity check: rebuilt buffer membership must match the one baked into the
    # saved Jacobian-derived buffer operator (same config, same grid -> deterministic)
    saved_membership = inv.buffer["membership"] if inv.buffer and \
        "membership" in inv.buffer else None
    if saved_membership is not None:
        match = np.array_equal(buf.membership.reshape(inv.grid.shape), saved_membership)
        print(f"rebuilt buffer membership matches saved bundle: {match}")

    sd = cfg.get_float("buffer", "stddev", default=1.0)
    floor = cfg.get_float("buffer", "stddev_floor", default=0.0)
    xa, sigma = buf.prior_moments(priors[inv.inventory], sd, floor)
    post = inv.buffer["value"]
    diff = post - xa
    return buf, xa, sigma, post, diff


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))
    buf, xa, sigma, post, diff = build_prior_and_diff(inv, cfg)

    print(f"\nbuffer: {buf.n_super} super-cells, "
          f"{(xa != 0).sum()} ({(xa != 0).mean():.1%}) with nonzero prior")
    print(f"posterior-prior diff: min={diff.min():.4g} max={diff.max():.4g} "
          f"mean|diff|={np.abs(diff).mean():.4g}")
    print(f"|diff|/sigma: p50={np.median(np.abs(diff)/sigma):.3f} "
          f"p90={np.percentile(np.abs(diff)/sigma, 90):.3f} "
          f"max={np.max(np.abs(diff)/sigma):.3f}")

    fig, ax = plt.subplots(figsize=(9, 8))
    vmax = np.abs(diff).max()
    sc = ax.scatter(buf.center_lon, buf.center_lat, c=diff, cmap="PuOr_r",
                     s=8, vmin=-vmax, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="buffer posterior - prior")
    bbox = cfg.get_literal("domain", "bbox", default=None)
    if bbox:
        lat0, lat1, lon0, lon1 = bbox
        ax.plot([lon0, lon1, lon1, lon0, lon0], [lat0, lat0, lat1, lat1, lat0],
                "k--", lw=1, label="core bbox")
    for fid, clat, clon, note in CLUSTERS:
        ax.scatter([clon], [clat], marker="x", color="k", s=60)
        ax.annotate(fid, (clon, clat), fontsize=7)
    ax.legend()
    ax.set_title("Buffer posterior-prior, all super-cells, with §9 cluster locations")
    plt.savefig("runs/buffer_overview.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> runs/buffer_overview.png")

    print("\ndistance from each §9 cluster to the nearest buffer super-cell:")
    for fid, clat, clon, note in CLUSTERS:
        d = _haversine_km(buf.center_lat, buf.center_lon, clat, clon)
        j = np.argmin(d)
        in_core = inv.core.mask[np.argmin(np.abs(inv.grid.lat - clat)),
                                 np.argmin(np.abs(inv.grid.lon - clon))]
        print(f"  {fid} ({clat:.3f},{clon:.3f}) [{note}]: "
              f"nearest buffer cell {d[j]:.0f}km away "
              f"(diff={diff[j]:+.4g}, |diff|/sigma={abs(diff[j])/sigma[j]:.2f}), "
              f"cluster itself is {'INSIDE' if in_core else 'outside'} the core mask")
    return inv


DOMAIN_DIAG = "runs/legtest_legoffset_6flight/runs/domain_diag_6flight/domain_diag.nc"


def phase1_correlate(inv):
    """Join the --diagnose-domain per-receptor out-of-core fractions onto the
    bundle's residuals (matched by flight + lat/lon) and check whether receptors
    with more out-of-core sensitivity have larger residual bias."""
    import netCDF4

    with netCDF4.Dataset(DOMAIN_DIAG) as ds:
        d_flight_ids = ds["receptor_flight"].flight_ids.split(", ")
        d_flight = np.asarray(ds["receptor_flight"][:])
        d_lat = np.asarray(ds["receptor_lat"][:])
        d_lon = np.asarray(ds["receptor_lon"][:])
        frac_u = np.asarray(ds["fraction_outside_uniform"][:])
        frac_e = np.asarray(ds["fraction_outside_emission"][:])

    R = inv.receptors
    rlat, rlon = R["receptor_lat"], R["receptor_lon"]
    resid = R["enhancement"] - R["modeled"]
    flag = R.get("outlier_flag", np.zeros_like(resid)).astype(bool)
    b_flight = R["receptor_flight"].astype(int)

    assert list(d_flight_ids) == list(inv.flight_ids), \
        f"flight order mismatch: {d_flight_ids} vs {inv.flight_ids}"

    frac_e_matched = np.full(rlat.shape, np.nan)
    frac_u_matched = np.full(rlat.shape, np.nan)
    for fi, fid in enumerate(inv.flight_ids):
        bsel = b_flight == fi
        dsel = d_flight == fi
        n_b, n_d = bsel.sum(), dsel.sum()
        print(f"{fid}: bundle n={n_b}  domain_diag n={n_d}  "
              f"{'OK' if n_b == n_d else 'MISMATCH'}")
        if n_b != n_d:
            continue
        # both come from the same Jacobian file's receptor order -> positional
        # match is valid once counts agree; verify with a coordinate check too
        assert np.allclose(rlat[bsel], d_lat[dsel], atol=1e-6), f"{fid} lat mismatch"
        assert np.allclose(rlon[bsel], d_lon[dsel], atol=1e-6), f"{fid} lon mismatch"
        frac_e_matched[bsel] = frac_e[dsel]
        frac_u_matched[bsel] = frac_u[dsel]

    good = ~flag & np.isfinite(resid) & np.isfinite(frac_e_matched)

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for i, fid in enumerate(inv.flight_ids):
        sel = good & (b_flight == i)
        r = np.corrcoef(frac_e_matched[sel], np.abs(resid[sel]))[0, 1] if sel.sum() > 2 else np.nan
        a = ax.flat[i]
        a.scatter(frac_e_matched[sel], np.abs(resid[sel]), s=4, alpha=0.4)
        a.set_title(f"{fid}  (r={r:.2f}, n={sel.sum()})")
        a.set_xlabel("fraction of explained enhancement from outside core")
        a.set_ylabel("|residual| (ppm)")
    plt.savefig("runs/buffer_bias_scatter.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> runs/buffer_bias_scatter.png")

    fig, ax = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for i, fid in enumerate(inv.flight_ids):
        sel = good & (b_flight == i)
        a = ax.flat[i]
        sc = a.scatter(rlon[sel], rlat[sel], c=frac_e_matched[sel], cmap="magma_r", s=6,
                        vmin=0, vmax=np.nanpercentile(frac_e_matched[good], 95))
        plt.colorbar(sc, ax=a, label="frac outside core (emission-wtd)")
        a.set_title(fid)
    fig.suptitle("Per-receptor out-of-core emission-weighted fraction")
    plt.savefig("runs/buffer_bias_map.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> runs/buffer_bias_map.png")

    print("\nPearson corr(frac_outside_emission, |residual|) per flight:")
    for i, fid in enumerate(inv.flight_ids):
        sel = good & (b_flight == i)
        r = np.corrcoef(frac_e_matched[sel], np.abs(resid[sel]))[0, 1] if sel.sum() > 2 else np.nan
        ru = np.corrcoef(frac_u_matched[sel], np.abs(resid[sel]))[0, 1] if sel.sum() > 2 else np.nan
        print(f"  {fid}: r(emission-wtd)={r:+.3f}  r(uniform)={ru:+.3f}  n={sel.sum()}")


if __name__ == "__main__":
    inv = main()
    phase1_correlate(inv)
