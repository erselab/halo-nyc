"""Aggregate a posterior scalar field into integrated emission fluxes.

The inversion solves for dimensionless, per-cell *multiplicative scalars* on a
prior emission field. The quantity actually wanted is the domain-integrated
emission — total NYC CH4, per category and overall, prior vs posterior, with an
uncertainty — which is a *linear functional* of the state:

    total_c = sum_i  s_{c,i} * prior_{c,i} * area_i  =  a_cᵀ x

where ``a_c`` has entries ``prior_{c,i} * area_i`` on category-c cells and zero
elsewhere. Because it is linear, the posterior mean and variance of the total
follow directly:

    E[total]   = a_cᵀ x̂
    Var[total] = a_cᵀ Ŝ a_c

and several totals (per category, plus the overall sum across categories) are
evaluated together as ``A x̂`` and ``A Ŝ Aᵀ``, the latter capturing cross-category
posterior covariance. The overall total correctly accounts for correlations
between category blocks.

Units: the integrated quantity is ``[emission-flux-density] x m^2``. If the prior
fields are emission flux densities (e.g. mol m^-2 s^-1), the totals are emission
rates (e.g. mol s^-1); pass a ``unit_scale``/``unit_label`` to convert to
whatever you report (e.g. kt CH4 yr^-1). The emission units of
``nyc_ch4_emissions.h5`` are not asserted here — set the scale deliberately.

The generic kernel (:func:`linear_estimate`) uses only the public ``Posterior``
API and could be promoted into the framework later; the rest (cell areas, prior x
area aggregation vectors, category grouping) is HALO/flux-specific and stays here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from adapters.gridded_state import Grid, GriddedState, _EARTH_RADIUS_KM
from goe.solver import Posterior
from goe.state import StateSpace

__all__ = ["cell_areas_m2", "linear_estimate", "FluxReport", "estimate_fluxes"]

_EARTH_RADIUS_M = _EARTH_RADIUS_KM * 1000.0


def cell_areas_m2(grid: Grid) -> np.ndarray:
    """Per-cell surface area (m^2) for a regular lat/lon grid.

    Uses the exact spherical-cap band area for each latitude row,
    ``R^2 * dlon * (sin(lat_north) - sin(lat_south))``, with cell edges taken at
    the midpoints between adjacent centers (and half-spacing at the borders).
    Returns a ``(n_lat, n_lon)`` array.
    """
    lat = grid.lat
    lon = grid.lon

    def edges(c):
        c = np.asarray(c, dtype=float)
        mid = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[c[0] - (mid[0] - c[0])], mid, [c[-1] + (c[-1] - mid[-1])]])

    lat_e = np.radians(edges(lat))
    lon_e = np.radians(edges(lon))
    dlon = np.abs(np.diff(lon_e))                      # (n_lon,)
    dsin = np.abs(np.diff(np.sin(lat_e)))              # (n_lat,)
    return _EARTH_RADIUS_M ** 2 * np.outer(dsin, dlon)


def linear_estimate(posterior: Posterior, A: np.ndarray):
    """Evaluate linear functionals of the posterior with uncertainty.

    Parameters
    ----------
    posterior:
        A solved :class:`goe.solver.Posterior`.
    A:
        A ``(k, n_state)`` array whose rows are the functionals ``a``.

    Returns
    -------
    means:
        ``A x̂`` — length ``k``.
    cov:
        ``A Ŝ Aᵀ`` — ``(k, k)`` covariance of the estimated quantities, obtained
        by applying the (matrix-free) posterior covariance to each functional.
    """
    A = np.atleast_2d(np.asarray(A, dtype=float))
    means = A @ posterior.mean
    SAt = np.column_stack([posterior.cov_matvec(A[i]) for i in range(A.shape[0])])
    cov = A @ SAt
    cov = 0.5 * (cov + cov.T)
    return means, cov


@dataclass
class FluxReport:
    """Prior vs posterior integrated fluxes for each category and the total.

    All arrays are aligned with :attr:`names` (the per-category names followed by
    ``"total"``). Values are in the integrated units implied by the prior fields
    and ``unit_scale`` (see :attr:`unit_label`).
    """

    names: list[str]
    prior: np.ndarray            # prior total per name
    posterior: np.ndarray        # posterior total per name
    posterior_stddev: np.ndarray  # 1-sigma uncertainty per name
    scale_factor: np.ndarray     # posterior / prior
    unit_label: str

    def as_table(self) -> str:
        rows = [f"{'category':<12} {'prior':>14} {'posterior':>14} "
                f"{'± 1σ':>12} {'scale':>8}",
                "-" * 64]
        for i, n in enumerate(self.names):
            rows.append(
                f"{n:<12} {self.prior[i]:>14.4g} {self.posterior[i]:>14.4g} "
                f"{self.posterior_stddev[i]:>12.4g} {self.scale_factor[i]:>8.3f}"
            )
        rows.append(f"\nunits: {self.unit_label}")
        return "\n".join(rows)

    def to_dict(self) -> dict:
        return {
            "names": list(self.names),
            "prior": self.prior.tolist(),
            "posterior": self.posterior.tolist(),
            "posterior_stddev": self.posterior_stddev.tolist(),
            "scale_factor": self.scale_factor.tolist(),
            "unit_label": self.unit_label,
        }


def estimate_fluxes(
    posterior: Posterior,
    state: StateSpace,
    core: GriddedState,
    prior_fields: dict[str, np.ndarray],
    grid: Grid,
    prior_state: np.ndarray | None = None,
    unit_scale: float = 1.0,
    unit_label: str = "prior-units x m^2 (native)",
    include_total: bool = True,
) -> FluxReport:
    """Integrate per-category scalar fields into total emissions with uncertainty.

    Parameters
    ----------
    posterior:
        The solved posterior over ``state``.
    state:
        The full state space (category blocks + any offset blocks).
    core:
        The masked grid over which the category scalars are defined; supplies the
        active-cell layout and lets prior fields be restricted to active cells.
    prior_fields:
        ``{category: prior_emission_field}`` on the full ``grid`` (the same
        absolute fields used to build the category-scalar blocks).
    grid:
        The grid the prior fields live on (for cell areas).
    prior_state:
        Optional prior state vector; if given, prior totals use it (its category
        blocks are typically all ones). Defaults to all-ones on category blocks.
    unit_scale, unit_label:
        Multiply integrated values by ``unit_scale`` and label them; use to
        convert ``prior-units x m^2`` into your reporting units.
    include_total:
        Whether to append a summed "total" row. Only valid when the categories
        are *additive* sub-sectors of one inventory; never set this when the
        blocks are independent alternative inventories. A "total" is emitted only
        when there is more than one category anyway.

    Returns
    -------
    FluxReport
        Prior/posterior totals, 1σ uncertainty, and scale factors per category
        and overall.
    """
    areas = core.from_field(cell_areas_m2(grid))           # (n_active,)
    categories = [name for name in prior_fields if name in state.names]

    # A "total" row that sums categories is only meaningful when the categories
    # are *additive* (e.g. sub-sectors of one inventory). It must NOT be used to
    # sum independent alternative inventories, so it is added only when there is
    # more than one category and the caller opts in.
    add_total = len(categories) > 1 and include_total

    # aggregation vectors a_c over the full state (zero outside category-c block)
    n_rows = len(categories) + (1 if add_total else 0)
    A = np.zeros((n_rows, state.size))
    for k, name in enumerate(categories):
        prior_active = core.from_field(prior_fields[name])  # absolute emission
        A[k, state.slice(name)] = prior_active * areas * unit_scale
    if add_total:
        A[-1] = A[:len(categories)].sum(axis=0)            # additive sub-category sum

    # prior totals: a_cᵀ xa  (scalars usually 1)
    xa = prior_state if prior_state is not None else state.fill(
        0.0, **{n: 1.0 for n in categories})
    prior_totals = A @ xa

    post_means, post_cov = linear_estimate(posterior, A)
    post_sd = np.sqrt(np.clip(np.diag(post_cov), 0.0, None))

    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(prior_totals != 0, post_means / prior_totals, np.nan)

    names = categories + (["total"] if add_total else [])
    return FluxReport(
        names=names,
        prior=prior_totals,
        posterior=post_means,
        posterior_stddev=post_sd,
        scale_factor=scale,
        unit_label=unit_label,
    )
