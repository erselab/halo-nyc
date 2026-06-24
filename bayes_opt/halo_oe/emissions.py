"""Regrid the NYC CH4 emission inventories onto a Jacobian grid.

The goe-inversion ``category_blocks`` adapter expects one *prior field per
category*, defined on the same grid as the forward operator (the Jacobian's
emission grid). The HALO inventory file ``nyc_ch4_emissions.h5`` stores three
sources — ``edgar``, ``epa``, ``pitt`` — each as a ``(n_subcategory, n_lat,
n_lon)`` array on its own coarser grid, with sub-category labels in the file
attributes.

This module collapses each source over its sub-category axis (giving a total
emission field for that source) and regrids it, by nearest neighbor, onto the
Jacobian grid, zero-filling cells outside the inventory's coverage. The result is
one ``(n_lat, n_lon)`` field per source, ready to become a per-cell
multiplicative-scalar block.

This is HALO/data-specific code and intentionally lives outside the framework. If
the "regrid a field from grid A to grid B" step proves generally useful, promote
that piece to ``goe-inversion/adapters`` later.
"""

from __future__ import annotations

import h5py
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from adapters.gridded_state import Grid

from .groups import DEFAULT_KEYWORD_MAP, group_indices

__all__ = ["DEFAULT_SOURCES", "load_source_totals", "regrid_to_grid",
           "category_priors_on_grid", "load_subcategory_fields",
           "group_priors_on_grid"]

DEFAULT_SOURCES = ("edgar", "epa", "pitt")


def load_source_totals(
    emissions_h5: str, sources=DEFAULT_SOURCES
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Load each source summed over its sub-category axis, plus its lat/lon.

    Returns ``(totals, lat, lon)`` where ``totals[source]`` is a 2-D
    ``(n_lat, n_lon)`` array.
    """
    totals: dict[str, np.ndarray] = {}
    with h5py.File(emissions_h5, "r") as f:
        lat = np.asarray(f["lat"][:], dtype=float)
        lon = np.asarray(f["lon"][:], dtype=float)
        for s in sources:
            arr = np.asarray(f[s][:], dtype=float)
            totals[s] = arr.sum(axis=0) if arr.ndim == 3 else arr
    return totals, lat, lon


def regrid_to_grid(
    field: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
    target: Grid,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Nearest-neighbor regrid a 2-D ``field`` onto ``target`` (a goe ``Grid``).

    Cells of ``target`` outside the source coverage receive ``fill_value``.
    """
    interp = RegularGridInterpolator(
        (src_lat, src_lon), field, method="nearest",
        bounds_error=False, fill_value=fill_value,
    )
    glat, glon = np.meshgrid(target.lat, target.lon, indexing="ij")
    return interp((glat, glon))


def category_priors_on_grid(
    emissions_h5: str,
    target: Grid,
    sources=DEFAULT_SOURCES,
    fill_value: float = 0.0,
) -> dict[str, np.ndarray]:
    """Return ``{source: prior_field}`` regridded onto ``target``.

    Each prior field has shape ``target.shape`` and is the total emission of that
    source (summed over sub-categories) on the Jacobian grid. These are the prior
    fields passed to :func:`adapters.scaling_blocks.category_blocks`.
    """
    totals, slat, slon = load_source_totals(emissions_h5, sources)
    return {
        s: regrid_to_grid(totals[s], slat, slon, target, fill_value=fill_value)
        for s in sources
    }


def load_subcategory_fields(emissions_h5: str, inventory: str):
    """Load one inventory's native sub-category layers and their labels.

    Returns ``(array, labels, lat, lon)`` where ``array`` is ``(n_subcat, n_lat,
    n_lon)`` and ``labels`` is the list of sub-category names from the file
    attribute ``<inventory>_categories``.
    """
    with h5py.File(emissions_h5, "r") as f:
        lat = np.asarray(f["lat"][:], dtype=float)
        lon = np.asarray(f["lon"][:], dtype=float)
        arr = np.asarray(f[inventory][:], dtype=float)
        labels = [s.strip() for s in f.attrs[f"{inventory}_categories"].split(";")]
    if arr.ndim != 3 or arr.shape[0] != len(labels):
        raise ValueError(
            f"inventory {inventory!r}: array shape {arr.shape} inconsistent with "
            f"{len(labels)} sub-category labels")
    return arr, labels, lat, lon


def group_priors_on_grid(
    emissions_h5: str,
    inventory: str,
    target: Grid,
    keyword_map=DEFAULT_KEYWORD_MAP,
    fill_value: float = 0.0,
):
    """Regrid one inventory's sub-categories, grouped into super-categories.

    Sub-categories are assigned to groups by keyword matching (see
    :mod:`halo_oe.groups`), summed within each group, and regridded onto
    ``target``. Returns ``(group_fields, assignment)`` where ``group_fields`` is
    ``{group: field_on_grid}`` (non-empty groups only) and ``assignment`` is
    ``{sub_category_label: group}`` for inspection.
    """
    arr, labels, slat, slon = load_subcategory_fields(emissions_h5, inventory)
    indices, assignment = group_indices(labels, keyword_map)
    group_fields = {
        g: regrid_to_grid(arr[ix].sum(axis=0), slat, slon, target, fill_value=fill_value)
        for g, ix in indices.items()
    }
    return group_fields, assignment
