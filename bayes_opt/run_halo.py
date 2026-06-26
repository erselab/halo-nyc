#!/usr/bin/env python3
"""Driver for the HALO CH4 flux inversion.

Composes the generic goe-inversion framework with the HALO-specific inputs
(regridded inventory prior, per-flight planar background) into a single inversion
and writes the posterior. The heavy lifting (forward operator, background,
observations) lives in :mod:`halo_oe.pipeline`; this file is just the CLI.

EDGAR, EPA, and Pittsburgh are three *alternative* inventories of the same NYC
emissions, so a normal run uses exactly one as the prior (``[emissions] inventory``).
Use ``--compare`` to invert each separately and tabulate the posteriors, which
shows how prior-dependent the flux estimate is — the Jacobian is read only once
and reused across the three solves.

Run (from the bayes_opt directory):
    python run_halo.py config.ini                 # primary inventory
    python run_halo.py config.ini --inventory epa # override
    python run_halo.py config.ini --compare       # all three
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# this script lives at the bayes_opt top level; put that directory on the path so
# the `halo_oe` package imports regardless of the current working directory.
# Importing the package then wires goe/adapters onto sys.path (see halo_oe/__init__.py).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import halo_oe  # noqa: F401,E402  (side effect: makes goe/adapters importable)
from adapters.gridded_state import GriddedState  # noqa: E402
from adapters.io import write_posterior  # noqa: E402
from adapters.jacobian_operator import JacobianFile  # noqa: E402
from goe import desroziers_diagnostics, tune_variance_scales  # noqa: E402
from goe.config import Config  # noqa: E402
from halo_oe.diagnostics import out_of_core_sensitivity, summarize_out_of_core  # noqa: E402
from halo_oe.emissions import category_priors_on_grid  # noqa: E402
from halo_oe.io_bundle import load_inversion, save_inversion  # noqa: E402
from halo_oe.pipeline import flight_paths, invert, load_context  # noqa: E402
from halo_oe.plotting import plot_buffer_regions, plot_posterior, plot_residuals  # noqa: E402


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


def _load_cfg(config_path, overrides=None):
    """Load the config and apply any ``section.key=value`` CLI overrides."""
    return Config(config_path).apply_overrides(overrides)


def _run_dir(config_path: str, cfg, save=None) -> str:
    """Directory that receives all of this run's outputs (created if needed).

    The base is ``[output] dir`` (default ``runs``); a relative base is resolved
    against the config file's own directory so outputs land next to the config
    regardless of the working directory. ``--save NAME`` selects the run
    subdirectory ``<base>/NAME`` (``NAME`` is used literally, so nested names like
    ``jul/26`` are kept); an absolute ``NAME`` is used as-is. Without ``--save``
    the base directory itself is used.
    """
    base = cfg.get("output", "dir", default="runs")
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(os.path.abspath(config_path)), base)
    d = base if not save else (save if os.path.isabs(save) else os.path.join(base, save))
    os.makedirs(d, exist_ok=True)
    return d


def _write_receptor_diagnostics(out_path, ctx, res):
    """Append per-receptor variables (coords, obs, flight, outlier flag) to output.

    Concatenated across all assimilated flights, in the same order as the stacked
    observations, with a per-receptor flight index for grouping.
    """
    import netCDF4
    import numpy as np

    cat = lambda attr: np.concatenate([np.asarray(getattr(jf, attr)) for jf in ctx.jfs])
    n = sum(jf.n_receptors for jf in ctx.jfs)
    with netCDF4.Dataset(out_path, "a") as ds:
        if "receptor" not in ds.dimensions:
            ds.createDimension("receptor", n)
        for name, data in (("receptor_lat", cat("receptor_lat")),
                           ("receptor_lon", cat("receptor_lon")),
                           ("receptor_obs", cat("receptor_obs")),
                           ("receptor_background", np.asarray(ctx.background))):
            if name not in ds.variables:
                ds.createVariable(name, "f8", ("receptor",))[:] = data
        if ctx.flight_index is not None and "receptor_flight" not in ds.variables:
            v = ds.createVariable("receptor_flight", "i4", ("receptor",))
            v.flight_ids = ", ".join(ctx.flight_ids)
            v[:] = np.asarray(ctx.flight_index)
        if res.outlier_mask is not None and "outlier_flag" not in ds.variables:
            v = ds.createVariable("outlier_flag", "i1", ("receptor",))
            v.long_name = "1 = observation flagged as outlier and excluded from the fit"
            v[:] = res.outlier_mask.astype("i1")


def _report_tuning(cfg, res):
    """Report model-data-mismatch diagnostics and max-likelihood error scales.

    Non-destructive: prints the Desroziers consistency check and the
    marginal-likelihood-optimal variance multipliers, plus the config changes
    that would apply them. Tune choices come from the ``[tuning]`` section.
    """
    d = desroziers_diagnostics(res.problem, res.posterior)
    print("\n--- error tuning ---")
    print(f"  reduced chi-square (current): {d['reduced_chi_square']:.3f}")
    print(f"  Desroziers R consistency r_scale: {d['r_scale']:.3f}  "
          f"(>1 => assumed R too small)")

    tune_R = cfg.get_bool("tuning", "tune_R", default=True)
    tune_Sa = cfg.get_bool("tuning", "tune_Sa", default=False)
    vr = tune_variance_scales(res.problem, tune_Sa=tune_Sa, tune_R=tune_R)
    print(f"  max-likelihood scales: alpha_R={vr.alpha_R:.3f}  alpha_Sa={vr.alpha_Sa:.3f}"
          f"  (log-likelihood {vr.log_likelihood:.2f})")
    if tune_R:
        print(f"  -> set [observations] error_inflation = {vr.alpha_R:.3f}  "
              f"(or scale mdm/measurement stddev by {vr.alpha_R**0.5:.3f})")
    if tune_Sa:
        print(f"  -> scale [prior] scalar_stddev by {vr.alpha_Sa**0.5:.3f}")


def run(config_path: str, inventory: str | None = None, tune: bool = False,
        flights=None, save=None, overrides=None) -> str:
    """Run a single inversion with the primary (or overridden) inventory."""
    cfg = _load_cfg(config_path, overrides)
    inv = inventory or cfg.get("emissions", "inventory", default="pitt")

    decompose = cfg.get_bool("decomposition", "enabled", default=False)
    method = cfg.get("decomposition", "method", default="partition")

    ctx = load_context(cfg, inventories=[inv], flights=flights)
    print(f"Flights ({ctx.n_flights}): {', '.join(ctx.flight_ids)}  "
          f"-> {ctx.obs.n_obs} observations")
    print(f"Active core cells: {ctx.core.n_active} of {ctx.grid.n_cells}")
    print(f"Inventory (prior): {inv}")

    res = invert(ctx, inv, decompose=decompose, method=method)
    print(f"Problem: {res.problem.n_obs} obs x {res.problem.n_state} state; "
          f"solved via {res.posterior.strategy}-space form.  mode={res.mode}")
    if res.assignment is not None:
        groups = {}
        for label, g in res.assignment.items():
            groups.setdefault(g, []).append(label)
        print("Category grouping:")
        for g, labs in groups.items():
            print(f"  {g}: {len(labs)} sub-categories")
    for k, v in res.diagnostics.items():
        print(f"  {k}: {v:.4g}")
    print("\n" + res.report.as_table() + "\n")

    if tune:
        _report_tuning(cfg, res)

    diag = dict(res.diagnostics)
    for i, name in enumerate(res.report.names):
        diag[f"flux_prior_{name}"] = res.report.prior[i]
        diag[f"flux_posterior_{name}"] = res.report.posterior[i]
        diag[f"flux_posterior_stddev_{name}"] = res.report.posterior_stddev[i]

    xa = res.state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0)
                                for b in res.state.blocks})
    run_dir = _run_dir(config_path, cfg, save)
    out_path = os.path.join(run_dir, "posterior.nc")
    write_posterior(out_path, res.state, res.posterior, prior_mean=xa, diagnostics=diag)
    _write_receptor_diagnostics(out_path, ctx, res)   # coords, obs, outlier_flag
    print(f"Wrote {out_path}")
    if res.diagnostics.get("n_outliers", 0):
        print(f"  flagged {int(res.diagnostics['n_outliers'])} outlier receptors "
              f"(saved as 'outlier_flag')")

    if save:   # --save also drops a reusable bundle alongside posterior.nc
        save_inversion(run_dir, ctx, res)
        print(f"Saved reusable inversion bundle to {run_dir}/  "
              f"(reload with halo_oe.io_bundle.load_inversion)")
    for jf in ctx.jfs:
        jf.close()
    return out_path


def run_compare(config_path: str, flights=None, overrides=None, save=None) -> None:
    """Invert each inventory separately (one Jacobian read) and tabulate results."""
    cfg = _load_cfg(config_path, overrides)
    inventories = _split(cfg.get("emissions", "compare", default="edgar,epa,pitt"))

    ctx = load_context(cfg, inventories=inventories, flights=flights)
    print(f"Flights ({ctx.n_flights}): {', '.join(ctx.flight_ids)}  "
          f"-> {ctx.obs.n_obs} observations")
    print(f"Active core cells: {ctx.core.n_active} of {ctx.grid.n_cells}")
    print(f"Comparing inventories as alternative priors: {inventories}\n")

    run_dir = _run_dir(config_path, cfg, save)

    rows = []
    for inv in inventories:
        res = invert(ctx, inv)
        r = res.report
        rows.append((inv, r.prior[0], r.posterior[0], r.posterior_stddev[0],
                     r.scale_factor[0], res.diagnostics["reduced_chi_square"]))
        xa = res.state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0)
                                    for b in res.state.blocks})
        write_posterior(os.path.join(run_dir, f"posterior_{inv}.nc"), res.state,
                        res.posterior, prior_mean=xa, diagnostics=res.diagnostics)

    label = cfg.get("flux", "unit_label", default="prior-units x m^2 (native)")
    print(f"{'inventory':<10} {'prior':>14} {'posterior':>14} {'± 1σ':>12} "
          f"{'scale':>8} {'χ²ᵣ':>8}")
    print("-" * 70)
    for inv, pr, po, sd, sc, chi in rows:
        print(f"{inv:<10} {pr:>14.4g} {po:>14.4g} {sd:>12.4g} {sc:>8.3f} {chi:>8.3f}")
    print(f"\nunits: {label}")
    print("Note: rows are independent inversions with different priors — do NOT sum them.")
    for jf in ctx.jfs:
        jf.close()


def diagnose_domain(config_path: str, flights=None, overrides=None, save=None) -> None:
    """Report how much receptor sensitivity falls OUTSIDE the core mask.

    Streams each flight's full Jacobian once and prints the fraction of column
    sensitivity (raw and emission-weighted) outside the core — the data-driven
    test for whether a buffer region (or a larger core) is needed. Writes a
    per-receptor netCDF for mapping which receptors see outside the domain.
    """
    import netCDF4
    cfg = _load_cfg(config_path, overrides)
    inv = cfg.get("emissions", "inventory", default="pitt")
    bbox = cfg.get_literal("domain", "bbox", default=None)
    row_chunk = cfg.get_int("jacobian", "row_chunk", default=16)

    grid = core = prior = None
    rows, per_receptor = [], []
    for fid, path in flight_paths(cfg, flights):
        jf = JacobianFile(path)
        if grid is None:
            grid = jf.grid
            mask = grid.bbox_mask(*bbox) if bbox is not None else None
            core = GriddedState(grid, mask, name="core")
            prior = category_priors_on_grid(cfg.get("emissions", "path"), grid, sources=(inv,))[inv]
        res = out_of_core_sensitivity(jf, core, prior_field=prior, row_chunk=row_chunk)
        rows.append((fid, summarize_out_of_core(res)))
        per_receptor.append((fid, np.asarray(jf.receptor_lat), np.asarray(jf.receptor_lon), res))
        jf.close()

    print(f"Out-of-core sensitivity  (core bbox {bbox}, inventory {inv}, "
          f"{core.n_active} of {grid.n_cells} cells active)")
    print(f"{'flight':<12} {'weighting':<10} {'integrated':>11} {'p50':>7} {'p75':>7} {'p90':>7}")
    print("-" * 60)
    for fid, summ in rows:
        for name, s in summ.items():
            print(f"{fid:<12} {name:<10} {s['integrated_fraction_outside']:>11.3f} "
                  f"{s['receptor_fraction_p50']:>7.3f} {s['receptor_fraction_p75']:>7.3f} "
                  f"{s['receptor_fraction_p90']:>7.3f}")
    print("\nThe emission-weighted 'integrated' value is the headline: the fraction of the\n"
          "explained enhancement originating outside the core. If it is sizeable, add a\n"
          "buffer region (or enlarge the core) until it becomes small.")

    run_dir = _run_dir(config_path, cfg, save)
    diag_path = os.path.join(run_dir, "domain_diag.nc")
    lat = np.concatenate([p[1] for p in per_receptor])
    lon = np.concatenate([p[2] for p in per_receptor])
    flight_idx = np.concatenate([np.full(p[1].size, i) for i, p in enumerate(per_receptor)])
    with netCDF4.Dataset(diag_path, "w") as ds:
        ds.createDimension("receptor", lat.size)
        ds.createVariable("receptor_lat", "f8", ("receptor",))[:] = lat
        ds.createVariable("receptor_lon", "f8", ("receptor",))[:] = lon
        v = ds.createVariable("receptor_flight", "i4", ("receptor",))
        v.flight_ids = ", ".join(p[0] for p in per_receptor); v[:] = flight_idx
        for name in per_receptor[0][3]:
            frac = np.concatenate([p[3][name]["fraction_outside"] for p in per_receptor])
            ds.createVariable(f"fraction_outside_{name}", "f8", ("receptor",))[:] = frac
    print(f"Wrote per-receptor diagnostic to {diag_path}")


def size_core(config_path: str, flights=None, overrides=None, save=None,
              fractions=(0.8, 0.9, 0.95, 0.99)) -> None:
    """Suggest a core bounding box from where the data actually constrain flux.

    Streams each flight's Jacobian once (no solve), builds the per-cell
    emission-weighted sensitivity (explained enhancement), and reports, for each
    target share of that signal, the smallest bounding box capturing it, the
    number of grid cells inside (the would-be state size), and the share the box
    actually captures. Pick the smallest box that captures most of the signal and
    let the buffer absorb the rest. Writes the sensitivity field to a netCDF.
    """
    import netCDF4
    from halo_oe.diagnostics import core_sizing

    cfg = _load_cfg(config_path, overrides)
    inv = cfg.get("emissions", "inventory", default="pitt")
    bbox = cfg.get_literal("domain", "bbox", default=None)
    row_chunk = cfg.get_int("jacobian", "row_chunk", default=16)

    grid = prior = None
    jfs = []
    for fid, path in flight_paths(cfg, flights):
        jf = JacobianFile(path)
        if grid is None:
            grid = jf.grid
            prior = category_priors_on_grid(cfg.get("emissions", "path"), grid, sources=(inv,))[inv]
        elif jf.grid.shape != grid.shape:
            raise ValueError(f"flight {fid!r} has grid {jf.grid.shape}, expected {grid.shape}; "
                             "all flights must share one grid for core sizing")
        jfs.append(jf)
    res = core_sizing(jfs, grid, prior, fractions=fractions, row_chunk=row_chunk)
    for jf in jfs:
        jf.close()

    cur = (f"{grid.bbox_mask(*bbox).sum()} cells" if bbox is not None else "whole grid")
    print(f"Core sizing  (inventory {inv}, {len(jfs)} flight(s); "
          f"current [domain] bbox {bbox} -> {cur})")
    print(f"effective constrained cells (participation ratio): {res['participation_ratio']:.1f}")
    print(f"\n{'capture':>8} {'n_active':>9} {'actual':>8}   suggested bbox "
          f"[lat_min, lat_max, lon_min, lon_max]")
    print("-" * 78)
    for r in res["rows"]:
        b = r["bbox"]
        print(f"{r['fraction_target']*100:>6.0f}% {r['n_active']:>9d} "
              f"{r['captured_weighted']*100:>6.1f}%   "
              f"[{b[0]:.3f}, {b[1]:.3f}, {b[2]:.3f}, {b[3]:.3f}]")
    print("\nPick the smallest box that captures most of the signal; the buffer absorbs the\n"
          "rest. Apply one with:  --set domain.bbox=\"[lat_min, lat_max, lon_min, lon_max]\"")

    run_dir = _run_dir(config_path, cfg, save)
    out_nc = os.path.join(run_dir, "core_sizing.nc")
    with netCDF4.Dataset(out_nc, "w") as ds:
        ds.createDimension("lat", grid.n_lat); ds.createDimension("lon", grid.n_lon)
        ds.createVariable("lat", "f8", ("lat",))[:] = grid.lat
        ds.createVariable("lon", "f8", ("lon",))[:] = grid.lon
        ds.createVariable("sensitivity", "f4", ("lat", "lon"))[:] = \
            res["sensitivity"].reshape(grid.shape)
        ds.createVariable("explained_enhancement", "f4", ("lat", "lon"))[:] = \
            res["weighted"].reshape(grid.shape)
        ds.participation_ratio = res["participation_ratio"]
        for r in res["rows"]:
            setattr(ds, f"bbox_{int(r['fraction_target']*100)}pct", r["bbox"])
    print(f"Wrote per-cell sensitivity field to {out_nc}")

    try:
        from halo_oe.plotting import plot_core_sizing
        png = os.path.join(run_dir, "core_sizing.png")
        plot_core_sizing(out_nc, out_path=png, current_bbox=bbox)
        print(f"Wrote core-sizing map to {png}")
    except Exception as exc:   # plotting is best-effort; the table/netCDF are the result
        print(f"(skipped core-sizing plot: {exc})")


def plot_buffer_regions(config_path: str, flights=None, out_path: str | None = None,
                        overrides=None, save=None) -> None:
    """Map the core and buffer regions with their prior mean and diagonal σ.

    A prior-only diagnostic: builds the grid, core mask, buffer super-cells and the
    regridded inventory prior from the Jacobian *metadata* alone (no large-array
    read, no solve), then draws three maps over the core∪buffer window:

    1. prior mean flux density — core cells (inventory density) and buffer
       super-cells (their area-weighted mean) on one color scale;
    2. prior 1σ (diagonal) flux density — core ``scalar_stddev × density`` and
       buffer ``[buffer] stddev × density`` (same moments used to build the block);
    3. relative prior σ (σ / |mean|) — shows where the prior is loose vs tight and
       makes the core-vs-buffer freedom obvious at a glance.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from halo_oe.buffer import build_buffer
    from matplotlib.patches import Rectangle

    cfg = _load_cfg(config_path, overrides)
    if not cfg.get_bool("buffer", "enabled", default=False):
        print("[buffer] enabled = false — nothing to plot. Enable the buffer first.")
        return
    inv = cfg.get("emissions", "inventory", default="pitt")
    bbox = cfg.get_literal("domain", "bbox", default=None)

    fid, path = flight_paths(cfg, flights)[0]      # geometry only; any flight works
    jf = JacobianFile(path)
    grid = jf.grid
    mask = grid.bbox_mask(*bbox) if bbox is not None else None
    core = GriddedState(grid, mask, name="core")
    jf.close()

    buf = build_buffer(grid, core, cfg)
    if buf is None:
        print("Buffer is empty for this configuration (check outer_bbox / mode).")
        return
    prior = category_priors_on_grid(cfg.get("emissions", "path"), grid, sources=(inv,))[inv]

    scalar_sd = cfg.get_float("prior", "scalar_stddev", default=0.5)
    buf_sd = cfg.get_float("buffer", "stddev", default=1.0)
    buf_floor = cfg.get_float("buffer", "stddev_floor", default=0.0)
    b_mean, b_sigma = buf.prior_moments(prior, buf_sd, buf_floor)

    shape = grid.shape
    pflat = np.asarray(prior, dtype=float).reshape(shape)

    mean_f = np.full(shape, np.nan); sig_f = np.full(shape, np.nan)
    mean_f[core.mask] = pflat[core.mask]
    sig_f[core.mask] = scalar_sd * np.abs(pflat[core.mask])
    bm = buf.to_field(b_mean); bs = buf.to_field(b_sigma)
    inbuf = ~np.isnan(bm)
    mean_f[inbuf] = bm[inbuf]; sig_f[inbuf] = bs[inbuf]
    denom = np.abs(mean_f)
    rel_f = np.full(shape, np.nan)
    ok = denom > 0
    rel_f[ok] = sig_f[ok] / denom[ok]

    # crop to the core∪buffer window (+margin) so the maps are legible
    region = core.mask | inbuf
    ii, jj = np.where(region)
    m = 3
    i0, i1 = max(ii.min() - m, 0), min(ii.max() + m + 1, shape[0])
    j0, j1 = max(jj.min() - m, 0), min(jj.max() + m + 1, shape[1])
    latc, lonc = grid.lat[i0:i1], grid.lon[j0:j1]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    panels = [("prior mean flux density", mean_f, "viridis", None),
              ("prior 1σ (diagonal)", sig_f, "magma", None),
              ("relative prior σ (σ/|mean|)", rel_f, "cividis", (0, max(scalar_sd, buf_sd) * 1.5))]
    for ax, (title, fld, cmap, vlim) in zip(axes, panels):
        sub = fld[i0:i1, j0:j1]
        kw = {} if vlim is None else {"vmin": vlim[0], "vmax": vlim[1]}
        pm = ax.pcolormesh(lonc, latc, sub, cmap=cmap, shading="nearest", **kw)
        fig.colorbar(pm, ax=ax, shrink=0.85)
        if bbox is not None:
            ax.add_patch(Rectangle((bbox[2], bbox[0]), bbox[3] - bbox[2], bbox[1] - bbox[0],
                                   fill=False, ec="red", lw=1.5, label="core"))
        # buffer super-cell centers (within the window)
        sel = ((buf.center_lat >= latc[0]) & (buf.center_lat <= latc[-1]) &
               (buf.center_lon >= lonc[0]) & (buf.center_lon <= lonc[-1]))
        ax.scatter(buf.center_lon[sel], buf.center_lat[sel], s=4, c="white",
                   edgecolors="k", linewidths=0.2, alpha=0.6)
        ax.set_title(title); ax.set_xlabel("lon"); ax.set_ylabel("lat")
    label = cfg.get("flux", "unit_label", default="prior-units (native)")
    fig.suptitle(f"Core (red box, {core.n_active} cells) + buffer "
                 f"({buf.n_super} super-cells, mode={cfg.get('buffer', 'mode', default='coarse')}) "
                 f"— inventory {inv}  [{label}]")

    run_dir = _run_dir(config_path, cfg, save)
    if out_path is None:
        out_path = os.path.join(run_dir, "buffer_regions.png")
    elif not os.path.isabs(out_path):
        out_path = os.path.join(run_dir, out_path)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"Core: {core.n_active} cells;  buffer: {buf.n_super} super-cells over "
          f"{int((buf.membership >= 0).sum())} native cells.")
    print(f"Buffer prior mean range [{b_mean.min():.3g}, {b_mean.max():.3g}], "
          f"σ range [{b_sigma.min():.3g}, {b_sigma.max():.3g}]  (relative σ = {buf_sd:g}).")
    print(f"Wrote buffer-region map to {out_path}")


def main():
    p = argparse.ArgumentParser(description="Run the HALO CH4 flux inversion.")
    p.add_argument("config", help="Path to the HALO inversion config (INI) file.")
    p.add_argument("--inventory", default=None,
                   help="Override the primary inventory (e.g. edgar, epa, pitt).")
    p.add_argument("--compare", action="store_true",
                   help="Invert each inventory in [emissions] compare and tabulate.")
    p.add_argument("--tune", action="store_true",
                   help="Report model-data-mismatch diagnostics and max-likelihood "
                        "error-variance scales (non-destructive).")
    p.add_argument("--flights", default=None,
                   help="Comma-separated flight ids to assimilate jointly (overrides "
                        "[jacobian] flights), e.g. 20230726_1,20230726_2.")
    p.add_argument("--save", default='runs/', metavar="NAME",
                   help="Run subdirectory under [output] dir (e.g. --save 20230726_1 -> "
                        "runs/20230726_1/); NAME is used literally, an absolute path is "
                        "used as-is. All outputs (posterior.nc, diagnostics) go there, and "
                        "a reusable inversion bundle is written alongside for post-hoc "
                        "analysis. Without --save, outputs go directly in [output] dir.")
    p.add_argument("--diagnose-domain", action="store_true",
                   help="Report the fraction of receptor sensitivity outside the core "
                        "mask (whether a buffer region is needed); does not invert.")
    p.add_argument("--size-core", action="store_true",
                   help="Suggest core bounding boxes from where the data constrain flux "
                        "(emission-weighted sensitivity); does not invert.")
    p.add_argument("--plot-buffer", nargs="?", const=True, default=None, metavar="PNG",
                   help="Map the core and buffer regions with their prior mean and "
                        "diagonal σ (prior-only; no solve). Optional output PNG path.")
    p.add_argument("--plot-diagnostics", default=False, metavar="PNG",
                   help="Plot the posterior fields and observation fitting diagnostics.")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE",
                   help="Override a config value without editing the file; repeatable. "
                        "E.g. --set buffer.enabled=true --set prior.scalar_stddev=0.3")
    args = p.parse_args()
    flights = _split(args.flights) if args.flights else None
    ov = args.overrides
    if args.plot_buffer is not None:
        plot_buffer_regions(args.config, flights=flights, overrides=ov, save=args.save,
                            out_path=None if args.plot_buffer is True else args.plot_buffer)
    elif args.diagnose_domain:
        diagnose_domain(args.config, flights=flights, overrides=ov, save=args.save)
    elif args.size_core:
        size_core(args.config, flights=flights, overrides=ov, save=args.save)
    elif args.compare:
        run_compare(args.config, flights=flights, overrides=ov, save=args.save)
    else:
        run(args.config, inventory=args.inventory, tune=args.tune, flights=flights,
            save=args.save, overrides=ov)
        if args.plot_diagnostics is not None:
            plot_posterior(load_inversion('runs/' + args.save), out_path='runs/' + args.save)
            plot_residuals('runs/' + args.save, out_path='runs/' + args.save)



if __name__ == "__main__":
    main()
