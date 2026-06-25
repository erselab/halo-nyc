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

Run:
    python -m halo_oe.run_halo halo_oe/config.ini                 # primary inventory
    python -m halo_oe.run_halo halo_oe/config.ini --inventory epa # override
    python -m halo_oe.run_halo halo_oe/config.ini --compare       # all three
(from the bayes_opt directory, or with bayes_opt on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import os
import sys

# importing the package wires goe/adapters onto sys.path (see halo_oe/__init__.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402  (side effect: makes goe/adapters importable)

from goe.config import Config  # noqa: E402
from goe import desroziers_diagnostics, tune_variance_scales  # noqa: E402
from adapters.io import write_posterior  # noqa: E402

from halo_oe.pipeline import invert, load_context  # noqa: E402


def _split(s):
    return [x.strip() for x in s.split(",") if x.strip()]


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
        flights=None) -> str:
    """Run a single inversion with the primary (or overridden) inventory."""
    cfg = Config(config_path)
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
    out_path = cfg.get("output", "path", default=f"halo_posterior_{inv}.nc")
    write_posterior(out_path, res.state, res.posterior, prior_mean=xa, diagnostics=diag)
    _write_receptor_diagnostics(out_path, ctx, res)   # coords, obs, outlier_flag
    print(f"Wrote {out_path}")
    if res.diagnostics.get("n_outliers", 0):
        print(f"  flagged {int(res.diagnostics['n_outliers'])} outlier receptors "
              f"(saved as 'outlier_flag')")
    for jf in ctx.jfs:
        jf.close()
    return out_path


def run_compare(config_path: str, flights=None) -> None:
    """Invert each inventory separately (one Jacobian read) and tabulate results."""
    cfg = Config(config_path)
    inventories = _split(cfg.get("emissions", "compare", default="edgar,epa,pitt"))

    ctx = load_context(cfg, inventories=inventories, flights=flights)
    print(f"Flights ({ctx.n_flights}): {', '.join(ctx.flight_ids)}  "
          f"-> {ctx.obs.n_obs} observations")
    print(f"Active core cells: {ctx.core.n_active} of {ctx.grid.n_cells}")
    print(f"Comparing inventories as alternative priors: {inventories}\n")

    out_stem = cfg.get("output", "path", default="halo_posterior.nc")
    stem, ext = os.path.splitext(out_stem)

    rows = []
    for inv in inventories:
        res = invert(ctx, inv)
        r = res.report
        rows.append((inv, r.prior[0], r.posterior[0], r.posterior_stddev[0],
                     r.scale_factor[0], res.diagnostics["reduced_chi_square"]))
        xa = res.state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0)
                                    for b in res.state.blocks})
        write_posterior(f"{stem}_{inv}{ext}", res.state, res.posterior,
                        prior_mean=xa, diagnostics=res.diagnostics)

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
    args = p.parse_args()
    flights = _split(args.flights) if args.flights else None
    if args.compare:
        run_compare(args.config, flights=flights)
    else:
        run(args.config, inventory=args.inventory, tune=args.tune, flights=flights)


if __name__ == "__main__":
    main()
