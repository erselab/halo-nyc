#!/usr/bin/env python3
"""Driver for the HALO CH4 flux inversion.

Composes the generic goe-inversion framework with the HALO-specific inputs
(regridded inventory priors, per-receptor background) into a single inversion and
writes the posterior. Like the framework's own example driver, this file contains
no inverse-theory math — only the wiring particular to HALO:

    Jacobian file              -> forward operator over an NYC-core mask
    inventory (edgar/epa/pitt) -> one per-cell multiplicative-scalar block each
    background                 -> enhancement vector z (+ obs error R)
    background-offset block    -> optional per-flight residual background
    spatial prior covariance   -> compact-support correlation per category
                               -> GaussianLinearProblem -> solve -> netCDF

Run:  python -m halo_oe.run_halo halo_oe/config.ini
(from the bayes_opt directory, or with bayes_opt on PYTHONPATH).
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# importing the package wires goe/adapters onto sys.path (see halo_oe/__init__.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402  (side effect: makes goe/adapters importable)

from goe import (  # noqa: E402
    BlockDiagonalCovariance,
    DiagonalCovariance,
    GaussianLinearProblem,
    StateSpace,
    degrees_of_freedom,
    reduced_chi_square,
    solve,
)
from goe.config import Config  # noqa: E402
from adapters.covariance_builders import build_spatial_covariance  # noqa: E402
from adapters.gridded_state import GriddedState  # noqa: E402
from adapters.io import write_posterior  # noqa: E402
from adapters.jacobian_operator import JacobianFile  # noqa: E402
from adapters.observations import build_observations  # noqa: E402
from adapters.scaling_blocks import category_blocks, offset_block  # noqa: E402

from halo_oe.background import receptor_background  # noqa: E402
from halo_oe.emissions import category_priors_on_grid  # noqa: E402


def run(config_path: str) -> str:
    cfg = Config(config_path)

    # -- forward operator over the NYC core mask ---------------------------
    jac_path = cfg.get("jacobian", "path")
    jf = JacobianFile(jac_path)
    bbox = cfg.get_literal("domain", "bbox", default=None)
    mask = jf.grid.bbox_mask(*bbox) if bbox is not None else None
    core = GriddedState(jf.grid, mask, name="core")
    print(f"Active core cells: {core.n_active} of {jf.grid.n_cells}")

    base = jf.operator(
        active=core.active,
        in_memory=cfg.get_bool("jacobian", "in_memory", default=True),
        row_chunk=cfg.get_int("jacobian", "row_chunk", default=16),
    )

    # -- inventory priors -> per-cell category-scalar blocks ---------------
    emis_h5 = cfg.get("emissions", "path")
    sources = tuple(s.strip() for s in cfg.get("emissions", "sources",
                                               default="edgar,epa,pitt").split(","))
    priors = category_priors_on_grid(emis_h5, jf.grid, sources=sources)
    cat_blocks, cat_ops = category_blocks(core, base, priors)

    blocks = list(cat_blocks)
    operators = dict(cat_ops)

    # -- prior covariance per category -------------------------------------
    scalar_sd = cfg.get_float("prior", "scalar_stddev", default=0.5)
    corr_km = cfg.get_float("prior", "correlation_length_km", default=0.0)
    cov_map: dict[str, object] = {}
    for b in cat_blocks:
        if corr_km > 0:
            cov_map[b.name] = build_spatial_covariance(core, scalar_sd, corr_km)
        else:
            cov_map[b.name] = DiagonalCovariance.isotropic(core.n_active, scalar_sd ** 2)

    # -- observations: enhancement = obs - background ----------------------
    obs_values = jf.receptor_obs
    if obs_values is None:
        raise ValueError("Jacobian file has no stored observations (receptor_xch4).")
    baseline = receptor_background(jf, cfg)          # STUB: constant for now
    obs = build_observations(
        obs_values,
        error_stddev=cfg.get_float("observations", "error_stddev", default=0.02),
        baseline=baseline,
        error_inflation=cfg.get_float("observations", "error_inflation", default=1.0),
    )

    # -- optional per-flight background-offset block -----------------------
    n_offsets = cfg.get_int("offset", "n_groups", default=0)
    if n_offsets > 0:
        # one Jacobian file is one flight here -> a single group by default
        assignments = (np.zeros(obs.n_obs, dtype=int) if n_offsets == 1
                       else np.arange(obs.n_obs) % n_offsets)
        off_blk, off_op = offset_block(assignments, n_offsets, name="bc")
        blocks.append(off_blk)
        operators["bc"] = off_op
        cov_map["bc"] = DiagonalCovariance.isotropic(
            n_offsets, cfg.get_float("offset", "stddev", default=0.05) ** 2)

    # -- assemble and solve ------------------------------------------------
    state = StateSpace(blocks)
    H = state.block_column(operators)
    Sa = BlockDiagonalCovariance([cov_map[b.name] for b in state.blocks])
    xa = state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0)
                            for b in state.blocks})

    problem = GaussianLinearProblem(H=H, z=obs.z, xa=xa, Sa=Sa, R=obs.R)
    print(f"Problem: {problem.n_obs} obs x {problem.n_state} state; solving...")
    posterior = solve(problem)
    print(f"Solved via {posterior.strategy}-space form.")

    diag = {"reduced_chi_square": reduced_chi_square(problem, posterior)}
    if problem.n_state <= 2000:
        diag["degrees_of_freedom"] = degrees_of_freedom(problem, posterior)
    for k, v in diag.items():
        print(f"  {k}: {v:.4g}")

    out_path = cfg.get("output", "path", default="halo_posterior.nc")
    write_posterior(out_path, state, posterior, prior_mean=xa, diagnostics=diag)
    print(f"Wrote {out_path}")
    jf.close()
    return out_path


def main():
    p = argparse.ArgumentParser(description="Run the HALO CH4 flux inversion.")
    p.add_argument("config", help="Path to the HALO inversion config (INI) file.")
    args = p.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
