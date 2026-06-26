"""Domain-truncation diagnostics: do the receptors see outside the core mask?

The inversion solves fluxes only inside the core mask; everything outside is held
at the prior. If the receptors' column footprints extend appreciably beyond the
core, that out-of-core signal has nowhere to go and is forced into the core edge
cells or the background — a bias that a **buffer region** is meant to absorb.

This module quantifies the problem before any buffer is built: for each receptor
it computes the fraction of its column sensitivity (and of its emission-weighted
sensitivity, i.e. of the *explained enhancement*) that lies **outside** the core.
A small fraction means the core contains the footprints and a buffer adds little;
a large fraction means truncation is biasing the solution and a buffer (or a
larger core) is warranted. The emission-weighted, domain-integrated fraction is
the headline number.
"""

from __future__ import annotations

import numpy as np

__all__ = ["out_of_core_sensitivity", "summarize_out_of_core",
           "cell_sensitivity_field", "core_sizing"]


def out_of_core_sensitivity(jf, core, prior_field=None, row_chunk: int = 16):
    """Per-receptor fraction of (weighted) sensitivity outside the core mask.

    Parameters
    ----------
    jf:
        An open :class:`adapters.jacobian_operator.JacobianFile` (one flight).
    core:
        The :class:`adapters.gridded_state.GriddedState` defining the active cells.
    prior_field:
        Optional full-grid prior emission field; if given, an emission-weighted
        result (fraction of explained enhancement) is included alongside the raw
        ``uniform`` one.
    row_chunk:
        Receptors per disk read (streams the full Jacobian once).

    Returns
    -------
    dict
        ``{weight_name: {total, inside, outside, fraction_outside}}`` per receptor.
    """
    weights = None if prior_field is None else {"emission": prior_field}
    sums = jf.receptor_column_sums(core.active, weights=weights, row_chunk=row_chunk)
    out = {}
    for name, d in sums.items():
        total, inside = d["total"], d["inside"]
        frac = 1.0 - np.divide(inside, total, out=np.zeros_like(inside), where=total > 0)
        out[name] = {"total": total, "inside": inside, "outside": total - inside,
                     "fraction_outside": frac}
    return out


def cell_sensitivity_field(jfs, grid, prior_field, row_chunk: int = 16):
    """Per-cell *explained-enhancement* weight summed over flights.

    For each grid cell, ``w = (Σ_receptors H[:, cell]) × prior[cell]`` — how much
    observed enhancement that cell's prior emission explains. Returns the flat
    (lat-major) arrays ``(sensitivity, weighted)``: the raw column-sensitivity and
    the emission-weighted version, accumulated over all flights in ``jfs``.
    """
    prior = np.asarray(prior_field, dtype=float).reshape(-1)
    sens = np.zeros(grid.n_cells, dtype=float)
    for jf in jfs:
        sens += jf.cell_column_sums(row_chunk=row_chunk)
    return sens, sens * prior


def core_sizing(jfs, grid, prior_field, fractions=(0.8, 0.9, 0.95, 0.99),
                row_chunk: int = 16):
    """Suggest core bounding boxes that capture a target share of the signal.

    Ranks cells by emission-weighted sensitivity (:func:`cell_sensitivity_field`)
    and, for each target ``fraction`` of the total explained enhancement, reports
    the **bounding box of the smallest set of cells** reaching it, the number of
    grid cells inside that box (the would-be state size), and the share actually
    captured by the box (≥ the target, since the box also includes interior
    low-weight cells). Use it to pick the core: the smallest box that captures most
    of the signal, leaving the rest to the buffer.

    Returns
    -------
    dict
        ``{"weighted": flat array, "sensitivity": flat array,
           "participation_ratio": float, "rows": [per-fraction dicts]}``.
        ``participation_ratio = (Σw)² / Σw²`` is a cheap, solve-free estimate of
        the *effective* number of cells carrying the signal — a hint at how many
        cells are even worth solving for.
    """
    sens, w = cell_sensitivity_field(jfs, grid, prior_field, row_chunk=row_chunk)
    total_w = w.sum()
    total_s = sens.sum()
    glat, glon = grid.cell_centers()                       # (n_lat, n_lon) each
    flat_lat, flat_lon = glat.reshape(-1), glon.reshape(-1)

    order = np.argsort(w)[::-1]
    cumw = np.cumsum(w[order])
    rows = []
    for frac in fractions:
        k = int(np.searchsorted(cumw, frac * total_w)) + 1 if total_w > 0 else 0
        k = max(1, min(k, w.size))
        cells = order[:k]
        bbox = [float(flat_lat[cells].min()), float(flat_lat[cells].max()),
                float(flat_lon[cells].min()), float(flat_lon[cells].max())]
        mask = grid.bbox_mask(*bbox)
        rows.append({
            "fraction_target": float(frac),
            "bbox": bbox,
            "n_active": int(mask.sum()),
            "captured_weighted": float(w[mask.reshape(-1)].sum() / total_w) if total_w else float("nan"),
            "captured_uniform": float(sens[mask.reshape(-1)].sum() / total_s) if total_s else float("nan"),
        })
    pr = float(total_w ** 2 / np.sum(w ** 2)) if np.any(w) else float("nan")
    return {"weighted": w, "sensitivity": sens, "participation_ratio": pr, "rows": rows}


def summarize_out_of_core(result) -> dict:
    """Summary statistics of an :func:`out_of_core_sensitivity` result.

    For each weighting, returns the **domain-integrated** fraction outside
    (sum of outside / sum of total over all receptors — the headline number) and
    per-receptor percentiles of the fraction outside.
    """
    summary = {}
    for name, d in result.items():
        total, outside, frac = d["total"], d["outside"], d["fraction_outside"]
        integ = float(outside.sum() / total.sum()) if total.sum() > 0 else float("nan")
        summary[name] = {
            "integrated_fraction_outside": integ,
            "receptor_fraction_p50": float(np.nanmedian(frac)),
            "receptor_fraction_p75": float(np.nanpercentile(frac, 75)),
            "receptor_fraction_p90": float(np.nanpercentile(frac, 90)),
            "n_receptors": int(frac.size),
        }
    return summary
