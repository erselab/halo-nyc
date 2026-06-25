"""Reusable building blocks for the HALO inversion.

The expensive part of an inversion is streaming the multi-gigabyte Jacobian and
forming the masked forward operator. That work — together with the domain mask,
the per-flight background, and the observation vector — is independent of *which*
inventory is used as the prior. This module separates that shared context
(:func:`load_context`, done once) from the per-inventory solve
(:func:`invert`), so a single run and the three-way inventory
comparison can both reuse one Jacobian read.

EDGAR, EPA, and Pittsburgh are alternative complete inventories of the same
emissions, so each inversion uses exactly one of them as the prior — never a sum.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from goe import (
    BlockDiagonalCovariance,
    BlockRow,
    DiagonalCovariance,
    GaussianLinearProblem,
    StateSpace,
    degrees_of_freedom,
    flag_outliers,
    reduced_chi_square,
    solve,
    subset_observations,
)
from goe.config import Config
from goe.operators import LinearOperator
from adapters.covariance_builders import build_spatial_covariance
from adapters.gridded_state import Grid, GriddedState
from adapters.jacobian_operator import JacobianFile
from adapters.observations import Observations, build_observations
from adapters.scaling_blocks import category_blocks, offset_block

from .obs_error import build_obs_error_covariance

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

__all__ = ["InversionContext", "InversionResult", "load_context", "flight_paths",
           "invert"]


@dataclass
class InversionContext:
    """Inventory-independent inputs, built once per Jacobian read.

    Supports one or several flights: the flux state is shared across flights and
    each flight contributes its own block of observation rows. For a single flight
    these reduce to the obvious quantities.
    """

    cfg: Config
    grid: Grid
    core: GriddedState
    base: LinearOperator                 # forward operator over active cells (rows stacked over flights)
    background: np.ndarray               # per-receptor background (all flights concatenated)
    obs: Observations                    # enhancement vector z + error R (all flights)
    priors: dict[str, np.ndarray]        # {inventory: prior field on the grid}
    jfs: list = field(default_factory=list)            # open JacobianFile per flight
    flight_ids: list = field(default_factory=list)     # flight identifier per flight
    flight_index: np.ndarray | None = None             # flight number per observation

    @property
    def n_flights(self) -> int:
        return len(self.jfs)


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
    outlier_mask: np.ndarray | None = None   # bool per original observation (True = flagged)


def flight_paths(cfg: Config, flights=None):
    """Resolve the list of ``(flight_id, jacobian_path)`` to assimilate.

    Selection order:

    * ``flights`` argument (e.g. from ``--flights``), if given;
    * ``[jacobian] flights`` (comma-separated flight ids) joined with
      ``[jacobian] dir``;
    * ``[jacobian] path`` (a single file) for backward compatibility.

    A flight id is the Jacobian file stem (e.g. ``20230726_1``).
    """
    if flights is None:
        spec = cfg.get("jacobian", "flights", default=None)
        flights = [s.strip() for s in spec.split(",") if s.strip()] if spec else None
    if flights:
        d = cfg.get("jacobian", "dir", default=".")
        return [(f, os.path.join(d, f + ".nc")) for f in flights]
    path = cfg.get("jacobian", "path")
    return [(os.path.splitext(os.path.basename(path))[0], path)]


def load_context(cfg: Config, inventories, flights=None) -> InversionContext:
    """Build the inventory-independent context for one or more flights.

    Each flight's Jacobian is read once and masked to the active cells; the flux
    state is shared, so the per-flight forward operators are stacked by rows
    (:class:`goe.BlockRow`), observations and backgrounds are concatenated, and
    the observation-error covariances form a block-diagonal (errors correlate
    within a flight, not across flights). ``flights`` selects which flights to
    assimilate (see :func:`flight_paths`) — this is how single days and
    combinations are run for experiments.
    """
    paths = flight_paths(cfg, flights)
    in_memory = cfg.get_bool("jacobian", "in_memory", default=True)
    row_chunk = cfg.get_int("jacobian", "row_chunk", default=16)
    error_stddev = cfg.get_float("observations", "error_stddev", default=0.02)
    inflation = cfg.get_float("observations", "error_inflation", default=1.0)
    components = cfg.get("observations", "error_model", default="simple") == "components"

    grid = core = priors = None
    jfs, bases, backgrounds, zs, raws, Rs, flight_ids, flight_index = [], [], [], [], [], [], [], []
    for fi, (fid, path) in enumerate(paths):
        jf = JacobianFile(path)
        if grid is None:
            grid = jf.grid
            bbox = cfg.get_literal("domain", "bbox", default=None)
            mask = grid.bbox_mask(*bbox) if bbox is not None else None
            core = GriddedState(grid, mask, name="core")
            priors = category_priors_on_grid(
                cfg.get("emissions", "path"), grid, sources=tuple(inventories))
        elif jf.grid.shape != grid.shape:
            raise ValueError(f"flight {fid!r} has grid {jf.grid.shape}, expected {grid.shape}")

        base_f = jf.operator(active=core.active, in_memory=in_memory, row_chunk=row_chunk)
        sens = base_f.matvec(np.ones(core.n_active))
        bg_f = receptor_background(jf, cfg, domain_sensitivity=sens)
        obs_f = build_observations(jf.receptor_obs, error_stddev=error_stddev,
                                   baseline=bg_f, error_inflation=inflation)
        R_f = (build_obs_error_covariance(jf.receptor_lat, jf.receptor_lon, cfg)
               if components else obs_f.R)

        jfs.append(jf); bases.append(base_f); backgrounds.append(bg_f)
        zs.append(obs_f.z); raws.append(obs_f.raw); Rs.append(R_f)
        flight_ids.append(fid)
        flight_index.append(np.full(jf.n_receptors, fi, dtype=int))

    base = bases[0] if len(bases) == 1 else BlockRow(bases)
    R = Rs[0] if len(Rs) == 1 else BlockDiagonalCovariance(Rs)
    background = np.concatenate(backgrounds)
    obs = Observations(z=np.concatenate(zs), R=R,
                       raw=np.concatenate(raws), baseline=background)
    return InversionContext(cfg, grid, core, base, background, obs, priors,
                            jfs, flight_ids, np.concatenate(flight_index))


def _offset_pieces(cfg, flight_index):
    """Background-offset block: one optimized offset per flight, or None.

    Each flight gets its own additive background offset (its observations map to
    its flight's offset parameter). For a single flight this is one offset.
    """
    if cfg.get_int("offset", "n_groups", default=0) <= 0:
        return None
    flight_index = np.asarray(flight_index, dtype=int)
    n_flights = int(flight_index.max()) + 1
    blk, op = offset_block(flight_index, n_flights, name="bc")
    cov = DiagonalCovariance.isotropic(n_flights, cfg.get_float("offset", "stddev", default=0.05) ** 2)
    return blk, op, cov


def _finalize(problem, posterior, n_outliers=0):
    diagnostics = {"reduced_chi_square": reduced_chi_square(problem, posterior),
                   "n_obs_used": problem.n_obs, "n_outliers": n_outliers}
    if problem.n_state <= 2000:
        diagnostics["degrees_of_freedom"] = degrees_of_freedom(problem, posterior)
    return diagnostics


def _solve_with_qc(problem, cfg):
    """Solve, optionally rejecting outlier observations and re-solving.

    Controlled by ``[observations]``: ``outlier_threshold`` (0 = off), ``outlier_kind``
    (``innovation`` (default) => normalized by the full expected mismatch
    ``H Sa Hᵀ + R`` — the proper gross-error check; ``posterior`` => residual
    relative to ``R`` only), and ``outlier_iterations``. Returns
    ``(problem, posterior, n_flagged)``; the returned problem is the reduced one,
    so diagnostics reflect the kept data.
    """
    posterior = solve(problem)
    n0 = problem.n_obs
    flagged = np.zeros(n0, dtype=bool)          # over the ORIGINAL observations
    threshold = cfg.get_float("observations", "outlier_threshold", default=0.0)
    if not threshold or threshold <= 0:
        return problem, posterior, flagged
    kind = cfg.get("observations", "outlier_kind", default="innovation")
    max_iter = cfg.get_int("observations", "outlier_iterations", default=2)
    kept = np.arange(n0)                         # original indices still in the problem
    for _ in range(max(1, max_iter)):
        mask, _ = flag_outliers(problem, posterior=posterior, threshold=threshold, kind=kind)
        if not mask.any():
            break
        flagged[kept[mask]] = True
        kept = kept[~mask]
        problem = subset_observations(problem, ~mask)
        posterior = solve(problem)
    return problem, posterior, flagged


def _group_fields(ctx, inventory):
    """Regrid the inventory's grouped category prior fields (full grid + active)."""
    kwmap = keyword_map_from_config(ctx.cfg)
    fields, assignment = group_priors_on_grid(
        ctx.cfg.get("emissions", "path"), inventory, ctx.grid, keyword_map=kwmap)
    active = {g: ctx.core.from_field(fields[g]) for g in fields}
    return fields, active, assignment


DECOMPOSE_METHODS = ("partition", "category_fields", "category_scalars")


def invert(
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
    flight_index = ctx.flight_index if ctx.flight_index is not None else np.zeros(n_obs, dtype=int)
    offset = _offset_pieces(cfg, flight_index)

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
        problem, posterior, flagged = _solve_with_qc(problem, cfg)
        report = decompose_solved_categories(
            posterior, state, core, group_active, ctx.grid, names,
            unit_scale=unit_scale, unit_label=unit_label)
        return InversionResult(inventory, state, problem, posterior, report,
                               _finalize(problem, posterior, int(flagged.sum())),
                               "category_scalars", assignment, flagged)

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
        problem, posterior, flagged = _solve_with_qc(problem, cfg)
        # sub-categories of ONE inventory are additive -> a total row is valid
        report = estimate_fluxes(posterior, state, core, group_fields, ctx.grid,
                                 prior_state=xa, unit_scale=unit_scale,
                                 unit_label=unit_label, include_total=True)
        return InversionResult(inventory, state, problem, posterior, report,
                               _finalize(problem, posterior, int(flagged.sum())),
                               "category_fields", assignment, flagged)

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
    problem, posterior, flagged = _solve_with_qc(problem, cfg)
    diagnostics = _finalize(problem, posterior, int(flagged.sum()))

    if decompose:  # method == "partition"
        _, group_active, assignment = _group_fields(ctx, inventory)
        rel = relative_uncertainties(list(group_active.keys()), cfg)
        report = partition_by_prior_variance(
            posterior, state, core, group_active, rel, ctx.grid, total_block=inventory,
            unit_scale=unit_scale, unit_label=unit_label)
        return InversionResult(inventory, state, problem, posterior, report,
                               diagnostics, "partition", assignment, flagged)

    report = estimate_fluxes(
        posterior, state, core, {inventory: ctx.priors[inventory]}, ctx.grid,
        prior_state=xa, unit_scale=unit_scale, unit_label=unit_label)
    return InversionResult(inventory, state, problem, posterior, report, diagnostics,
                           "total", None, flagged)
