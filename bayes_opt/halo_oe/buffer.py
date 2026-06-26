"""Build a coarse buffer region around the core inversion domain.

The receptors are sensitive to emissions outside the core mask; if that signal has
nowhere to go it aliases into the core edge cells and the background. A **buffer**
gives those out-of-core emissions their own (coarse) flux degrees of freedom: each
buffer *super-cell* is a group of native grid cells outside the core, carrying one
uniform flux unknown. Its forward column is the summed Jacobian over its native
cells (built in one streamed pass; see
:meth:`adapters.jacobian_operator.JacobianFile.operator_with_buffer`).

Two ways to define the super-cells (``[buffer] mode``):

* ``coarse`` — tile the out-of-core region into super-cells by an integer
  ``factor`` (or a target ``resolution_deg``), optionally limited to an
  ``outer_bbox``. The generic "coarser resolution outside" option.
* ``mask`` — an integer **label field** on the grid (``mask_file``); each distinct
  positive label becomes one super-cell. This lets you define arbitrary buffer
  blocks (named regions, sectors) explicitly.

The buffer is a nuisance state: it absorbs out-of-core signal (tightening the core
posterior through cross-covariance) but is not part of the reported core total.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .flux import cell_areas_m2

__all__ = ["Buffer", "build_buffer"]


@dataclass
class Buffer:
    """A partition of out-of-core native cells into coarse super-cells.

    Attributes
    ----------
    membership:
        Length ``n_cells`` (flat, lat-major) array: super-cell index of each native
        cell, or ``-1`` for cells not in the buffer.
    n_super:
        Number of super-cells (the buffer block size).
    center_lat, center_lon:
        Per-super-cell mean coordinates.
    cell_count:
        Number of native cells per super-cell.
    area:
        Total surface area (m^2) per super-cell.
    """

    membership: np.ndarray
    n_super: int
    center_lat: np.ndarray
    center_lon: np.ndarray
    cell_count: np.ndarray
    area: np.ndarray
    grid_shape: tuple

    def prior_mean(self, inventory_field) -> np.ndarray:
        """Area-weighted mean inventory flux density per super-cell.

        The natural prior mean for a super-cell's uniform flux: the area-weighted
        average of the (full-grid) inventory emission over its native cells.
        """
        e = np.asarray(inventory_field, dtype=float).reshape(-1)
        a = cell_areas_m2_flat(self.grid_shape, self._areas_full)
        valid = self.membership >= 0
        num = np.bincount(self.membership[valid], (e * a)[valid], self.n_super)
        return num / self.area

    def prior_moments(self, inventory_field, stddev: float, floor: float = 0.0):
        """Per-super-cell prior ``(mean, sigma)`` for the buffer flux.

        ``mean`` is :meth:`prior_mean` (area-weighted inventory flux density).
        ``sigma`` is ``stddev`` *relative* to that mean, with each super-cell's
        magnitude floored so empty super-cells keep a finite prior: the floor is
        ``floor`` if positive, else an automatic ``1e-3 * max(|mean|)``. This is
        the exact prior used to build the buffer block in
        :func:`halo_oe.pipeline._buffer_pieces`.
        """
        mean = self.prior_mean(inventory_field)
        f = floor if floor and floor > 0 else np.abs(mean).max() * 1e-3
        sigma = float(stddev) * np.maximum(np.abs(mean), f)
        return mean, sigma

    # cached full-grid areas (set in build_buffer)
    _areas_full: np.ndarray = None

    def to_field(self, values, fill=np.nan) -> np.ndarray:
        """Scatter per-super-cell ``values`` back onto the full ``(lat, lon)`` grid."""
        values = np.asarray(values, dtype=float)
        flat = np.full(int(np.prod(self.grid_shape)), fill, dtype=float)
        valid = self.membership >= 0
        flat[valid] = values[self.membership[valid]]
        return flat.reshape(self.grid_shape)


def cell_areas_m2_flat(shape, areas_full):
    return np.asarray(areas_full).reshape(-1)


def _outer_mask(grid, core, cfg):
    """Candidate buffer cells: inside the optional outer bbox, outside the core."""
    outer_bbox = cfg.get_literal("buffer", "outer_bbox", default=None)
    m = grid.bbox_mask(*outer_bbox) if outer_bbox is not None else np.ones(grid.shape, bool)
    return m & ~core.mask


def _coarsen_factor(grid, cfg) -> int:
    res = cfg.get_float("buffer", "resolution_deg", default=None)
    if res:
        native = abs(float(grid.lat[1] - grid.lat[0]))
        return max(1, int(round(res / native)))
    return max(1, cfg.get_int("buffer", "factor", default=10))


def _load_labels(cfg, grid) -> np.ndarray:
    path = cfg.get("buffer", "mask_file", default=None)
    if path is None:
        raise ValueError("[buffer] mode=mask requires mask_file")
    if path.endswith(".npy"):
        labels = np.load(path)
    else:
        import netCDF4
        with netCDF4.Dataset(path) as ds:
            name = next((v for v in ("labels", "buffer_label", "label", "region")
                         if v in ds.variables), None)
            if name is None:
                raise KeyError(f"no label variable found in {path!r}")
            labels = np.asarray(ds[name][:])
    if labels.shape != grid.shape:
        raise ValueError(f"label field has shape {labels.shape}, expected {grid.shape}")
    return labels.astype(int)


def _renumber(tile_index_grid, outer):
    """Assign consecutive super-cell ids to tiles that contain >=1 outer cell."""
    masked = np.where(outer, tile_index_grid, -1)
    present = np.unique(masked[masked >= 0])
    remap = -np.ones(int(masked.max()) + 1, dtype=int) if present.size else np.array([], int)
    for k, t in enumerate(present):
        remap[t] = k
    membership = np.where(masked >= 0, remap[np.clip(masked, 0, None)], -1)
    return membership.reshape(-1), int(present.size)


def build_buffer(grid, core, cfg) -> Buffer | None:
    """Construct the :class:`Buffer` from ``[buffer]`` config, or ``None`` if disabled."""
    if not cfg.get_bool("buffer", "enabled", default=False):
        return None
    outer = _outer_mask(grid, core, cfg)
    if outer.sum() == 0:
        return None
    mode = cfg.get("buffer", "mode", default="coarse")

    if mode == "mask":
        labels = _load_labels(cfg, grid)
        membership, n_super = _renumber(labels, outer & (labels > 0))
    else:  # coarse
        f = _coarsen_factor(grid, cfg)
        ii, jj = np.indices(grid.shape)
        n_tile_lon = (grid.n_lon + f - 1) // f
        tile = (ii // f) * n_tile_lon + (jj // f)
        membership, n_super = _renumber(tile, outer)

    if n_super == 0:
        return None

    areas_full = cell_areas_m2(grid)
    glat, glon = grid.cell_centers()
    a_flat = areas_full.reshape(-1)
    valid = membership >= 0
    counts = np.bincount(membership[valid], minlength=n_super)
    center_lat = np.bincount(membership[valid], glat[valid], n_super) / counts
    center_lon = np.bincount(membership[valid], glon[valid], n_super) / counts
    area = np.bincount(membership[valid], a_flat[valid], n_super)

    buf = Buffer(membership=membership, n_super=n_super, center_lat=center_lat,
                 center_lon=center_lon, cell_count=counts, area=area, grid_shape=grid.shape)
    buf._areas_full = areas_full
    return buf
