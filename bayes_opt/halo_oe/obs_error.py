"""Build the HALO observation-error covariance R from physical components.

The observation-error (model-data mismatch, MDM) budget is a sum of physically
distinct terms; lumping them into one diagonal number is what makes reduced
chi-square come out far from 1. This module assembles ``R`` from:

* **measurement error** — the HALO retrieval uncertainty, independent per
  receptor (diagonal);
* **representation + transport error** — the part the 1 km column Jacobian cannot
  resolve plus transport-model error. Adjacent 1 km receptors have heavily
  overlapping column footprints, so this term is **spatially correlated
  along-track**, not independent. Treating it as diagonal makes the inversion
  overconfident (too many effectively-independent observations).

So ``R = diag(meas_var) + MDM``, where ``MDM`` has variance ``mdm_stddev**2`` and a
compact-support correlation length over the receptor positions. The result is a
single :class:`goe.SparseCovariance`. The correlation structure is built with the
generic, problem-agnostic :func:`adapters.covariance_from_coordinates`; this
module only supplies the HALO specifics (receptor coordinates, which components
exist, default magnitudes). The overall scale is then a hyperparameter to tune
(see :mod:`goe.tuning`).
"""

from __future__ import annotations

import numpy as np

from adapters.covariance_builders import covariance_from_coordinates
from adapters.gridded_state import _EARTH_RADIUS_KM
from goe.covariance import DiagonalCovariance

__all__ = ["receptors_to_km", "build_obs_error_covariance"]


def receptors_to_km(lat, lon, lat0=None, lon0=None):
    """Project receptor lat/lon to a local east/north plane in km.

    An equirectangular projection about a reference point — accurate for the
    ~100 km spatial extent of a single flight and giving Euclidean distances in km
    suitable for the correlation builder.
    """
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    lat0 = float(np.mean(lat)) if lat0 is None else lat0
    lon0 = float(np.mean(lon)) if lon0 is None else lon0
    x = np.radians(lon - lon0) * np.cos(np.radians(lat0)) * _EARTH_RADIUS_KM
    y = np.radians(lat - lat0) * _EARTH_RADIUS_KM
    return np.column_stack([x, y])


def build_obs_error_covariance(
    lat, lon, config, measurement_variance=None,
):
    """Assemble ``R`` for the HALO observations from its error components.

    Reads the ``[observations]`` config section:

    * ``measurement_stddev`` — per-receptor measurement 1-sigma (used if
      ``measurement_variance`` is not supplied explicitly).
    * ``mdm_stddev`` — representation+transport 1-sigma.
    * ``mdm_correlation_length_km`` — along-track decorrelation length; ``0``
      makes the MDM diagonal (independent).
    * ``error_inflation`` — multiplies the whole ``R`` (>= 1).

    Parameters
    ----------
    lat, lon:
        Receptor coordinates.
    measurement_variance:
        Optional explicit per-receptor measurement variance (e.g. from the HALO
        product). Overrides ``measurement_stddev``.

    Returns
    -------
    goe.Covariance
        ``R``; a :class:`goe.SparseCovariance` when an MDM correlation length is
        set, otherwise a :class:`goe.DiagonalCovariance`.
    """
    lat = np.asarray(lat, dtype=float)
    n = lat.shape[0]

    if measurement_variance is None:
        meas_sd = config.get_float("observations", "measurement_stddev", default=0.0)
        meas_var = np.full(n, meas_sd ** 2)
    else:
        meas_var = np.broadcast_to(np.asarray(measurement_variance, dtype=float), (n,)).astype(float)

    mdm_sd = config.get_float("observations", "mdm_stddev", default=0.0)
    corr_km = config.get_float("observations", "mdm_correlation_length_km", default=0.0)
    inflation = config.get_float("observations", "error_inflation", default=1.0)

    if mdm_sd > 0 and corr_km and corr_km > 0:
        coords = receptors_to_km(lat, lon)
        R = covariance_from_coordinates(
            coords, stddev=mdm_sd, correlation_length=corr_km,
            diagonal_variance=meas_var + mdm_sd ** 2,
        )
        if inflation != 1.0:
            from goe.covariance import ScaledCovariance
            R = ScaledCovariance(R, inflation)
        return R

    # no correlation requested -> plain diagonal (measurement + uncorrelated MDM)
    return DiagonalCovariance(inflation * (meas_var + mdm_sd ** 2))
