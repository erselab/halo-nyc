"""Reusable building blocks for the HALO inversion.

The expensive part of an inversion is streaming the multi-gigabyte Jacobian and
forming the masked forward operator. That work — together with the domain mask,
the per-flight background, and the observation vector — is independent of *which*
inventory is used as the prior. This module separates that shared context
(:func:`load_context`, done once) from the per-inventory solve
(:func:`invert_with_inventory`), so a single run and the three-way inventory
comparison can both reuse one Jacobian read.

EDGAR, EPA, and Pittsburgh are alternative complete inventories of the same
emissions, so each inversion uses exactly one of them as the prior — never a sum.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from goe import (
    BlockDiagonalCovariance,
    DiagonalCovariance,
    GaussianLinearProblem,
    StateSpace,
    degrees_of_freedom,
    reduced_chi_square,
    solve,
)
from goe.config import Config
from goe.operators import LinearOperator
from adapters.covariance_builders import build_spatial_covariance
from adapters.gridded_state import Grid, GriddedState
from adapters.jacobian_operator import JacobianFile
from adapters.observations import Observations, build_observations
from adapters.scaling_blocks import category_blocks, offset_block

from .background import receptor_background
from .decomposition import (
    assemble_category_scalar_state,
    category_covariance,
    decompose_solved_categories,
    partition_by_prior_variance,
    relative_uncertainties,
)
from .emissions import category_priors_on_grid, group_priors_on_grid
from .flux import FluxReport, estimate_fluxes
from .groups import keyword_map_from_config

__all__ = ["InversionContext", "InversionResult", "load_context",
           "invert_with_inventory"]


@dataclass
class InversionContext:
    """Inventory-independent inputs, built once per Jacobian read."""

    cfg: Config
    jf: JacobianFile
    grid: Grid
    core: GriddedState
    base: LinearOperator                 # forward operator over active cells
    background: np.ndarray               # per-receptor background
    obs: Observations                    # enhancement vector z + error R
    priors: dict[str, np.ndarray]        # {inventory: prior field on the grid}


@dataclass
class InversionResult:
    """Outputs of a single-inventory inversion."""

    inventory: str
    state: StateSpace
    problem: GaussianLinearProblem
    posterior: object
    report: FluxReport
    diagnostics: dict
    mode: str = "total"                 # "total" | "partition" | "category_scalars"
    assignment: dict | None = None      # {sub_category_label: group}, when decomposing


def load_context(cfg: Config, inventories) -> InversionContext:
    """Build the inventory-independent context (reads the Jacobian once).

    Parameters
    ----------
    cfg:
        Parsed configuration.
    inventories:
        Iterable of inventory names whose prior fields should be regridded and
        cached (e.g. just the primary, or all three for comparison).
    """
    jf = JacobianFile(cfg.get("jacobian", "path"))
    grid = jf.grid

    bbox = cfg.get_literal("domain", "bbox", default=None)
    mask = grid.bbox_mask(*bbox) if bbox is not None else None
    core = GriddedState(grid, mask, name="core")

    base = jf.operator(
        active=core.active,
        in_memory=cfg.get_bool("jacobian", "in_memory", default=True),
        row_chunk=cfg.get_int("jacobian", "row_chunk", default=16),
    )

    priors = category_priors_on_grid(
        cfg.get("emissions", "path"), grid, sources=tuple(inventories))

    background = receptor_background(jf, cfg)
    obs = build_observations(
        jf.receptor_obs,
        error_stddev=cfg.get_float("observations", "error_stddev", default=0.02),
        baseline=background,
        error_inflation=cfg.get_float("observations", "error_inflation", default=1.0),
    )
    return InversionContext(cfg, jf, grid, core, base, background, obs, priors)


def _offset_pieces(cfg, n_obs):
    """Return (block, operator, covariance) for the background-offset block, or None."""
    n_offsets = cfg.get_int("offset", "n_groups", default=0)
    if n_offsets <= 0:
        return None
    assignments = (np.zeros(n_obs, dtype=int) if n_offsets == 1
                   else np.arange(n_obs) % n_offsets)
    blk, op = offset_block(assignments, n_offsets, name="bc")
    cov = DiagonalCovariance.isotropic(n_offsets, cfg.get_float("offset", "stddev", default=0.05) ** 2)
    return blk, op, cov


def _finalize(problem, posterior):
    diagnostics = {"reduced_chi_square": reduced_chi_square(problem, posterior)}
    if problem.n_state <= 2000:
        diagnostics["degrees_of_freedom"] = degrees_of_freedom(problem, posterior)
    return diagnostics


def _group_fields(ctx, inventory):
    """Regrid the inventory's grouped category prior fields (full grid + active)."""
    kwmap = keyword_map_from_config(ctx.cfg)
    fields, assignment = group_priors_on_grid(
        ctx.cfg.get("emissions", "path"), inventory, ctx.grid, keyword_map=kwmap)
    active = {g: ctx.core.from_field(fields[g]) for g in fields}
    return fields, active, assignment


DECOMPOSE_METHODS = ("partition", "category_fields", "category_scalars")


def invert_with_inventory(
    ctx: InversionContext, inventory: str,
    decompose: bool = False, method: str = "partition",
) -> InversionResult:
    """Solve the inversion using one inventory as the prior.

    Modes (no cross-inventory total is ever formed):

    * ``decompose=False`` — per-cell scalar field on the inventory total; report
      the inventory total only.
    * ``decompose=True`` with ``method``:
        - ``"partition"`` — solve the per-cell total, then split the posterior by
          prior category variance (data constrain the total; prior shapes split).
        - ``"category_fields"`` — solve a per-cell scalar field *per category*,
          each with its own covariance (diagonal for point sources, spatial for
          diffuse). Data-informed and spatially resolved; recommended.
        - ``"category_scalars"`` — domain scalar per category + per-cell total
          correction (weakly identified from a single flight; prior-sensitive).
    """
    cfg, core = ctx.cfg, ctx.core
    n_obs = ctx.obs.n_obs
    unit_scale = cfg.get_float("flux", "unit_scale", default=1.0)
    unit_label = cfg.get("flux", "unit_label", default="prior-units x m^2 (native)")
    offset = _offset_pieces(cfg, n_obs)

    def _append_offset(blocks, operators, cov_blocks):
        if offset is not None:
            blk, op, cov = offset
            blocks.append(blk); operators["bc"] = op; cov_blocks.append(cov)

    if decompose and method == "category_scalars":
        _, group_active, assignment = _group_fields(ctx, inventory)
        blocks, operators, cov_blocks, names = assemble_category_scalar_state(
            core, ctx.base, group_active, cfg, n_obs)
        _append_offset(blocks, operators, cov_blocks)
        state = StateSpace(blocks)
        Sa = BlockDiagonalCovariance(cov_blocks)
        xa = state.fill(0.0, **{b.name: (1.0 if b.name == "categories" else 0.0)
                                for b in state.blocks})
        problem = GaussianLinearProblem(H=state.block_column(operators), z=ctx.obs.z,
                                        xa=xa, Sa=Sa, R=ctx.obs.R)
        posterior = solve(problem)
        report = decompose_solved_categories(
            posterior, state, core, group_active, ctx.grid, names,
            unit_scale=unit_scale, unit_label=unit_label)
        return InversionResult(inventory, state, problem, posterior, report,
                               _finalize(problem, posterior), "category_scalars", assignment)

    if decompose and method == "category_fields":
        group_fields, _, assignment = _group_fields(ctx, inventory)
        cat_blocks, cat_ops = category_blocks(core, ctx.base, group_fields)
        names = [b.name for b in cat_blocks]
        blocks = list(cat_blocks)
        operators = dict(cat_ops)
        cov_blocks = category_covariance(core, names, cfg)
        _append_offset(blocks, operators, cov_blocks)
        state = StateSpace(blocks)
        Sa = BlockDiagonalCovariance(cov_blocks)
        xa = state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0) for b in state.blocks})
        problem = GaussianLinearProblem(H=state.block_column(operators), z=ctx.obs.z,
                                        xa=xa, Sa=Sa, R=ctx.obs.R)
        posterior = solve(problem)
        # sub-categories of ONE inventory are additive -> a total row is valid
        report = estimate_fluxes(posterior, state, core, group_fields, ctx.grid,
                                 prior_state=xa, unit_scale=unit_scale,
                                 unit_label=unit_label, include_total=True)
        return InversionResult(inventory, state, problem, posterior, report,
                               _finalize(problem, posterior), "category_fields", assignment)

    # --- per-cell total scalar on the inventory (default + variance partition) ---
    if inventory not in ctx.priors:
        raise KeyError(f"prior field for inventory {inventory!r} not loaded")
    cat_blocks, cat_ops = category_blocks(core, ctx.base, {inventory: ctx.priors[inventory]})
    blocks = list(cat_blocks)
    operators = dict(cat_ops)
    scalar_sd = cfg.get_float("prior", "scalar_stddev", default=0.5)
    corr_km = cfg.get_float("prior", "correlation_length_km", default=0.0)
    cov_blocks = [build_spatial_covariance(core, scalar_sd, corr_km) if corr_km > 0
                  else DiagonalCovariance.isotropic(core.n_active, scalar_sd ** 2)]
    _append_offset(blocks, operators, cov_blocks)

    state = StateSpace(blocks)
    Sa = BlockDiagonalCovariance(cov_blocks)
    xa = state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0) for b in state.blocks})
    problem = GaussianLinearProblem(H=state.block_column(operators), z=ctx.obs.z,
                                    xa=xa, Sa=Sa, R=ctx.obs.R)
    posterior = solve(problem)
    diagnostics = _finalize(problem, posterior)

    if decompose:  # method == "partition"
        _, group_active, assignment = _group_fields(ctx, inventory)
        rel = relative_uncertainties(list(group_active.keys()), cfg)
        report = partition_by_prior_variance(
            posterior, state, core, group_active, rel, ctx.grid, total_block=inventory,
            unit_scale=unit_scale, unit_label=unit_label)
        return InversionResult(inventory, state, problem, posterior, report,
                               diagnostics, "partition", assignment)

    report = estimate_fluxes(
        posterior, state, core, {inventory: ctx.priors[inventory]}, ctx.grid,
        prior_state=xa, unit_scale=unit_scale, unit_label=unit_label)
    return InversionResult(inventory, state, problem, posterior, report, diagnostics, "total")
