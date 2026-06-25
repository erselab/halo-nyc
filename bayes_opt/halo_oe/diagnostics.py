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

__all__ = ["out_of_core_sensitivity", "summarize_out_of_core"]


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
