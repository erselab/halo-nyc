"""Save and reload a complete HALO inversion for post-hoc analysis.

Re-reading the multi-gigabyte Jacobians for every new analysis is wasteful, and
unnecessary: once an inversion is solved, **post-hoc aggregation and
disaggregation never need the forward operator again** — they are linear
functionals of the posterior. :func:`save_inversion` writes a self-contained
directory bundle holding everything needed to reconstruct the prior and posterior
distributions and the observation context; :func:`load_inversion` reloads it into
objects you can re-aggregate, re-attribute, or plot without any re-solve.

Bundle layout (a directory):

* ``factors.npz`` — posterior mean + the covariance factors (`Sa`, `W`,
  Cholesky of `G`, or the state-space Cholesky) from :mod:`goe.serialization`,
  plus the prior mean ``xa``. These reproduce ``aᵀx̂`` and ``aᵀŜa`` exactly for
  any functional, with no operator.
* ``fields.nc`` — geometry (grid, mask), per-block posterior fields + stddev +
  prior fields mapped onto the grid, the inventory's super-category prior fields
  (for re-grouping), and per-receptor observations / background / enhancement /
  modeled value / flight / outlier flag.
* ``layout.json`` — state-block layout, posterior factor structure, inventory,
  mode, flight ids, bbox.
* ``report.json`` — the flux report and diagnostics.
* ``config.ini`` — the configuration used (provenance).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np

from goe import (
    GaussianLinearProblem,
    StateSpace,
    posterior_from_flat,
    posterior_to_flat,
)
from goe.state import Block
from adapters.gridded_state import Grid, GriddedState

from .emissions import group_priors_on_grid
from .groups import keyword_map_from_config

__all__ = ["save_inversion", "load_inversion", "SavedInversion"]


def _gridded_block_names(state):
    """Block names that live on the grid (everything except the offset block)."""
    return [b.name for b in state.blocks if b.name != "bc"]


def save_inversion(dirpath: str, ctx, res) -> str:
    """Write a complete inversion bundle to ``dirpath`` (created if needed)."""
    import netCDF4

    os.makedirs(dirpath, exist_ok=True)
    state, post, core, grid = res.state, res.posterior, ctx.core, ctx.grid

    # --- posterior factors + prior mean -> factors.npz -------------------
    structure, arrays = posterior_to_flat(post)
    xa = state.fill(0.0, **{b.name: (0.0 if b.name == "bc" else 1.0) for b in state.blocks})
    arrays = dict(arrays); arrays["xa"] = xa
    np.savez(os.path.join(dirpath, "factors.npz"), **arrays)

    # inventory super-category prior fields, restricted to active cells (compact;
    # full grids are reconstructable from these + the geometry)
    kwmap = keyword_map_from_config(ctx.cfg)
    group_fields, assignment = group_priors_on_grid(
        ctx.cfg.get("emissions", "path"), res.inventory, grid, keyword_map=kwmap)

    # --- geometry, compact fields, receptors -> fields.nc ----------------
    with netCDF4.Dataset(os.path.join(dirpath, "fields.nc"), "w") as ds:
        ds.grid_shape = list(grid.shape)
        ds.createDimension("lat", grid.n_lat)
        ds.createDimension("lon", grid.n_lon)
        ds.createDimension("cell", core.n_active)
        ds.createVariable("lat", "f8", ("lat",))[:] = grid.lat
        ds.createVariable("lon", "f8", ("lon",))[:] = grid.lon
        ds.createVariable("active", "i8", ("cell",))[:] = core.active   # flat (lat-major) indices
        for g in group_fields:                                          # prior on active cells
            ds.createVariable(f"groupprior_{g}", "f8", ("cell",))[:] = core.from_field(group_fields[g])

        if "bc" in state.names:
            parts, stds = state.unpack(post.mean), state.unpack(post.stddev())
            ds.createDimension("bc", state.block("bc").size)
            ds.createVariable("bc", "f8", ("bc",))[:] = parts["bc"]
            ds.createVariable("bc_stddev", "f8", ("bc",))[:] = stds["bc"]

        # per-receptor (all flights concatenated, in observation order)
        n = sum(jf.n_receptors for jf in ctx.jfs)
        ds.createDimension("receptor", n)
        cat = lambda a: np.concatenate([np.asarray(getattr(jf, a)) for jf in ctx.jfs])
        ds.createVariable("receptor_lat", "f8", ("receptor",))[:] = cat("receptor_lat")
        ds.createVariable("receptor_lon", "f8", ("receptor",))[:] = cat("receptor_lon")
        ds.createVariable("receptor_obs", "f8", ("receptor",))[:] = cat("receptor_obs")
        ds.createVariable("receptor_background", "f8", ("receptor",))[:] = np.asarray(ctx.background)
        ds.createVariable("enhancement", "f8", ("receptor",))[:] = ctx.obs.z
        if ctx.flight_index is not None:
            v = ds.createVariable("receptor_flight", "i4", ("receptor",))
            v.flight_ids = ", ".join(ctx.flight_ids)
            v[:] = np.asarray(ctx.flight_index)
        modeled = np.full(n, np.nan)   # modeled enhancement on the obs used (outliers -> NaN)
        kept = ~res.outlier_mask if res.outlier_mask is not None else np.ones(n, bool)
        modeled[kept] = res.problem.H.matvec(post.mean)
        ds.createVariable("modeled", "f8", ("receptor",))[:] = modeled
        if res.outlier_mask is not None:
            ds.createVariable("outlier_flag", "i1", ("receptor",))[:] = res.outlier_mask.astype("i1")

    # --- layout / report / config (JSON + INI) ---------------------------
    layout = {
        "inventory": res.inventory,
        "mode": res.mode,
        "flight_ids": list(ctx.flight_ids),
        "bbox": ctx.cfg.get_literal("domain", "bbox", default=None),
        "blocks": [{"name": b.name, "size": int(b.size)} for b in state.blocks],
        "gridded_blocks": _gridded_block_names(state),
        "assignment": {str(k): v for k, v in assignment.items()},
        "posterior": structure,
    }
    with open(os.path.join(dirpath, "layout.json"), "w") as f:
        json.dump(layout, f, indent=2, default=lambda o: o.tolist())

    report_d = res.report.to_dict()
    report_d["diagnostics"] = {k: float(v) for k, v in res.diagnostics.items()}
    with open(os.path.join(dirpath, "report.json"), "w") as f:
        json.dump(report_d, f, indent=2)

    ctx.cfg.write(os.path.join(dirpath, "config.ini"))
    return dirpath


@dataclass
class SavedInversion:
    """A reloaded inversion; supports post-hoc analysis without the operator.

    Attributes mirror what a fresh solve produces — ``posterior`` (with working
    ``mean``/``cov_matvec``/``variances``), ``state``, ``core``, ``grid``, the
    prior mean ``xa``, the per-block and super-category ``prior_fields`` /
    ``group_fields``, the ``receptors`` table, and ``report``/``diagnostics``.
    Pass these straight to :mod:`halo_oe.flux` / :mod:`halo_oe.decomposition`
    helpers to re-aggregate or re-attribute with any grouping.
    """

    inventory: str
    mode: str
    flight_ids: list
    state: StateSpace
    core: GriddedState
    grid: Grid
    posterior: object
    xa: np.ndarray
    group_fields: dict        # {super-category: prior on active cells}
    receptors: dict
    report: dict
    diagnostics: dict

    def estimate(self, A):
        """Mean and covariance of linear functionals ``A`` of the posterior.

        ``A`` is ``(k, n_state)``. Returns ``(A x̂, A Ŝ Aᵀ)`` — exact, no operator.
        """
        A = np.atleast_2d(np.asarray(A, dtype=float))
        means = A @ self.posterior.mean
        SAt = np.column_stack([self.posterior.cov_matvec(A[i]) for i in range(A.shape[0])])
        cov = A @ SAt
        return means, 0.5 * (cov + cov.T)

    def block(self, name: str) -> np.ndarray:
        """Posterior values of a state block (per active cell, or per group/offset)."""
        return self.state.unpack(self.posterior.mean)[name]

    def field(self, name: str) -> np.ndarray:
        """Posterior of a gridded block scattered onto the full ``(lat, lon)`` grid."""
        return self.core.to_field(self.block(name))


def load_inversion(dirpath: str) -> SavedInversion:
    """Reload a bundle written by :func:`save_inversion`."""
    import netCDF4

    with open(os.path.join(dirpath, "layout.json")) as f:
        layout = json.load(f)
    arrays = dict(np.load(os.path.join(dirpath, "factors.npz")))
    posterior = posterior_from_flat(layout["posterior"], arrays)

    state = StateSpace([Block(b["name"], int(b["size"])) for b in layout["blocks"]])

    with netCDF4.Dataset(os.path.join(dirpath, "fields.nc")) as ds:
        grid = Grid(np.asarray(ds["lat"][:]), np.asarray(ds["lon"][:]))
        active = np.asarray(ds["active"][:])
        mask = np.zeros(grid.shape, dtype=bool)
        mask.reshape(-1)[active] = True
        core = GriddedState(grid, mask, name="core")
        group_fields = {v[len("groupprior_"):]: np.asarray(ds[v][:])
                        for v in ds.variables if v.startswith("groupprior_")}
        rec_vars = [v for v in ds.variables
                    if v.startswith("receptor") or v in ("enhancement", "modeled", "outlier_flag")]
        receptors = {v: np.asarray(ds[v][:]) for v in rec_vars}

    with open(os.path.join(dirpath, "report.json")) as f:
        report = json.load(f)
    diagnostics = report.pop("diagnostics", {})

    return SavedInversion(
        inventory=layout["inventory"], mode=layout["mode"], flight_ids=layout["flight_ids"],
        state=state, core=core, grid=grid, posterior=posterior, xa=arrays["xa"],
        group_fields=group_fields, receptors=receptors, report=report, diagnostics=diagnostics)
