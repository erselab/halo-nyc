"""Per-receptor background (baseline) for the HALO inversion.

The forward operator predicts an *enhancement* above some background, so each
observation must have a background subtracted before assimilation
(``z = observation - background``). The framework's
:func:`adapters.observations.build_observations` accepts a per-observation
``baseline`` array — this module produces it.

Method: per-flight, lower-envelope planar fit.
-----------------------------------------------
The inflow / free-tropospheric background of a column XCH4 field varies slowly in
space and from flight to flight (different day, time, air mass), whereas the urban
enhancement is localized and sharp. We exploit that separation by fitting a
**low-order polynomial surface in (lat, lon)** to the **lower envelope** of a
single flight's observed columns:

* fitting per flight lets each flight's overall level and gradient float
  independently — capturing day/time variation as different surfaces;
* fitting to a low quantile of the residuals (not all points) keeps the surface
  riding the *clean* air rather than being pulled up into the plume, which would
  bias fluxes low;
* a low polynomial degree (default 1, a plane) has too few degrees of freedom to
  chase the localized enhancement, so it captures the smooth baseline and leaves
  the signal for the inversion.

The background-offset block in the driver (kept, with its own configurable prior)
can still absorb a residual constant per flight on top of this surface.

This implementation operates on a single flight's receptor arrays. The driver
runs one Jacobian (= one flight) at a time and passes that flight's receptor
coordinates/observations; for multi-flight assimilation, call
:func:`flight_background` per flight and concatenate.

Other background sources (e.g. a model boundary condition convolved with the
column weighting function) can be swapped in behind :func:`receptor_background`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "constant_background",
    "polynomial_design",
    "fit_lower_envelope_surface",
    "flight_background",
    "receptor_background",
]


def constant_background(n_receptors: int, value: float) -> np.ndarray:
    """Return a constant background of ``value`` for every receptor."""
    return np.full(int(n_receptors), float(value))


def polynomial_design(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    """Design matrix of 2-D polynomial terms up to total ``degree``.

    Columns are ordered ``1, x, y, x^2, xy, y^2, ...``. ``x`` and ``y`` should be
    centered (e.g. anomalies from their means) for numerical conditioning.
    """
    cols = []
    for d in range(degree + 1):
        for i in range(d + 1):
            cols.append((x ** (d - i)) * (y ** i))
    return np.column_stack(cols)


def fit_lower_envelope_surface(
    x: np.ndarray,
    y: np.ndarray,
    value: np.ndarray,
    degree: int = 1,
    quantile: float = 0.25,
    n_iter: int = 5,
):
    """Fit a polynomial surface to the lower envelope of ``value``.

    Iteratively refits the surface to the subset of points whose residuals fall
    in the lowest ``quantile`` fraction, so the fit converges onto the clean-air
    floor rather than the mean. Returns ``(coeffs, design_all)`` where
    ``design_all @ coeffs`` evaluates the background at every input point.

    Parameters
    ----------
    x, y:
        Coordinates (will be centered internally).
    value:
        Quantity whose lower envelope is sought (the observed column).
    degree:
        Polynomial degree (1 = plane). Space/time are collinear within a flight,
        so degree 1 in (lat, lon) is the recommended default.
    quantile:
        Fraction of lowest-residual points retained each iteration (0 < q <= 1).
    n_iter:
        Number of refinement iterations.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    value = np.asarray(value, dtype=float)
    xc = x - x.mean()
    yc = y - y.mean()
    design = polynomial_design(xc, yc, degree)
    ncols = design.shape[1]

    keep = np.ones(value.shape[0], dtype=bool)
    coeffs, *_ = np.linalg.lstsq(design[keep], value[keep], rcond=None)
    for _ in range(max(0, n_iter)):
        resid = value - design @ coeffs
        thr = np.quantile(resid, quantile)
        new_keep = resid <= thr
        if new_keep.sum() < ncols + 1:
            break
        keep = new_keep
        coeffs, *_ = np.linalg.lstsq(design[keep], value[keep], rcond=None)

    return coeffs, design


def flight_background(
    lat: np.ndarray,
    lon: np.ndarray,
    value: np.ndarray,
    degree: int = 1,
    quantile: float = 0.25,
    n_iter: int = 5,
) -> np.ndarray:
    """Per-receptor background for one flight via a lower-envelope surface fit.

    Returns the fitted background evaluated at every receptor (length =
    ``len(value)``), in the same units as ``value``.
    """
    coeffs, design = fit_lower_envelope_surface(
        lat, lon, value, degree=degree, quantile=quantile, n_iter=n_iter
    )
    return design @ coeffs


def receptor_background(jacobian_file, config) -> np.ndarray:
    """Return the per-receptor background array (length ``n_receptors``).

    Reads the method and parameters from the ``[background]`` config section:

    * ``method`` = ``planar`` (default) or ``constant``
    * ``degree`` (default 1), ``envelope_quantile`` (default 0.25),
      ``n_iter`` (default 5) for the planar fit
    * ``constant_value`` for the constant fallback (defaults to
      ``[observations] baseline``)

    Falls back to a constant if receptor coordinates are unavailable.
    """
    method = config.get("background", "method", default="planar")
    n = jacobian_file.n_receptors

    if method == "constant":
        value = config.get_float("background", "constant_value", default=None)
        if value is None:
            value = config.get_float("observations", "baseline", default=0.0)
        return constant_background(n, value)

    lat = jacobian_file.receptor_lat
    lon = jacobian_file.receptor_lon
    obs = jacobian_file.receptor_obs
    if lat is None or lon is None or obs is None:
        value = config.get_float("observations", "baseline", default=0.0)
        return constant_background(n, value)

    return flight_background(
        lat, lon, obs,
        degree=config.get_int("background", "degree", default=1),
        quantile=config.get_float("background", "envelope_quantile", default=0.25),
        n_iter=config.get_int("background", "n_iter", default=5),
    )
