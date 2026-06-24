"""Per-receptor background (baseline) for the HALO inversion.

The forward operator predicts an *enhancement* above some background, so each
observation must have a background subtracted before assimilation
(``z = observation - background``). The framework's
:func:`adapters.observations.build_observations` accepts a per-observation
``baseline`` array — this module is responsible for producing it.

STUB: the current implementation returns a single constant background for every
receptor (read from config). This makes the pipeline run end-to-end immediately,
but a constant background is almost certainly too crude for real results — the
background varies in space/time and dominates the enhancement at these column
amounts.

TODO (replace the constant): plausible real sources, in rough order of fidelity —
  1. STILT-WRF column boundary conditions: the ``xwrf_bc_ch4`` values in
     ``bnd/wrf_d01/<flight>_xbnd.h5`` produced by ``halo_column_xgas_bc.py``,
     matched to each receptor by obs id / lat / lon / time.
  2. A clean-air / free-tropospheric baseline estimated per flight from the
     observations themselves (e.g. a low percentile of upwind/edge receptors).
  3. A satellite or model climatology sampled at the receptor locations/times.

Whatever the source, this function must return a 1-D array of length
``n_receptors`` aligned with the Jacobian's receptor axis. The optimized
background-offset block in the driver can absorb a residual *constant* error per
flight, but it cannot fix a spatially wrong baseline — so this is worth getting
right.
"""

from __future__ import annotations

import numpy as np

__all__ = ["constant_background", "receptor_background"]


def constant_background(n_receptors: int, value: float) -> np.ndarray:
    """Return a constant background of ``value`` for every receptor."""
    return np.full(int(n_receptors), float(value))


def receptor_background(jacobian_file, config) -> np.ndarray:
    """Return the per-receptor background array (length ``n_receptors``).

    Parameters
    ----------
    jacobian_file:
        An open :class:`adapters.jacobian_operator.JacobianFile`; provides
        ``n_receptors`` and receptor coordinates for future matching logic.
    config:
        A :class:`goe.config.Config`; the stub reads
        ``[observations] baseline`` as the constant value.

    NOTE: stub implementation — see the module docstring for how to make this
    real. Replace the body below; keep the signature and the return contract
    (1-D, length ``n_receptors``, same units as the observations).
    """
    value = config.get_float("observations", "baseline", default=0.0)
    return constant_background(jacobian_file.n_receptors, value)
