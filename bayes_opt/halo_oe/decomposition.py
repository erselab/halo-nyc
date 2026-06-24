"""Decompose a posterior flux into configurable super-category totals.

Two attribution modes, selected by whether per-category domain scalars are solved
(the optional ``g_k``):

* **Prior-variance partition** (``g_k`` off). We solve only a per-cell scalar
  ``s(x)`` on the inventory total, so the data constrain the total flux but not
  its split among co-located categories. Each cell's posterior total is then
  apportioned to categories in proportion to their prior variance
  ``sigma_k^2(x) = (rel_k E_k(x))^2`` — the optimal-estimation result for the
  components of an observed sum. The split is prior-shaped; the data inform only
  the total. The per-category total is an *affine* functional of ``s``, so its
  posterior uncertainty propagates through the existing covariance.

* **Solved category scalars** (``g_k`` on). We add one domain-wide scalar per
  category (prior mean 1, its own prior sigma) whose forward column is that
  category's spatial fingerprint ``H E_k``. The flux model is
  ``f(x) = sum_k g_k E_k(x) + E_tot(x) c(x)``, fully linear, with the data
  informing the split through the categories' differing spatial patterns. The
  per-category total is the linear functional ``g_k * integral(E_k) +
  integral(E_k c)``.

Both modes return a :class:`halo_oe.flux.FluxReport` whose per-category rows sum
exactly to the inventory total, with cross-category covariance included.
"""

from __future__ import annotations

import numpy as np

from goe.covariance import DiagonalCovariance
from goe.operators import DenseOperator, LinearOperator
from goe.state import Block
from adapters.covariance_builders import build_spatial_covariance
from adapters.gridded_state import GriddedState
from adapters.scaling_blocks import scale_columns

from .flux import FluxReport, cell_areas_m2, linear_estimate

__all__ = [
    "relative_uncertainties",
    "prior_variance_weights",
    "partition_by_prior_variance",
    "category_covariance",
    "assemble_category_scalar_state",
    "decompose_solved_categories",
]


def relative_uncertainties(group_names, cfg) -> dict[str, float]:
    """Per-group relative prior uncertainty from the ``[category_uncertainty]`` section.

    Reads ``default`` and any per-group override (e.g. ``natural_gas = 0.6``).
    """
    default = cfg.get_float("category_uncertainty", "default", default=0.5)
    return {g: cfg.get_float("category_uncertainty", g, default=default) for g in group_names}


def prior_variance_weights(group_active, rel_unc):
    """Per-cell variance weights ``w_k(x) = (rel_k E_k)^2 / sum_j (rel_j E_j)^2``.

    Returns ``(weights, e_total)`` with ``weights`` a dict of length-n_active
    arrays summing to 1 where the total is positive (0 elsewhere).
    """
    names = list(group_active.keys())
    var = {g: (rel_unc[g] * np.asarray(group_active[g], dtype=float)) ** 2 for g in names}
    denom = sum(var.values())
    e_total = sum(np.asarray(group_active[g], dtype=float) for g in names)
    weights = {}
    nz = denom > 0
    for g in names:
        w = np.zeros_like(denom)
        w[nz] = var[g][nz] / denom[nz]
        weights[g] = w
    return weights, e_total


def category_covariance(core, group_names, cfg):
    """Per-category prior covariances reflecting each source's spatial error type.

    For a per-cell scalar field per category, the prior covariance encodes how the
    category's emissions are spatially known:

    * **point sources** (landfills, WWTPs — locations known, only magnitude
      uncertain) -> a **diagonal** covariance: per-cell, spatially independent.
    * **diffuse / spatially-uncertain sources** -> a **spatial** covariance with a
      modest decorrelation length: nearby cells' errors correlate.

    The per-cell standard deviation comes from ``[category_uncertainty]`` and the
    decorrelation length (km) from ``[category_spatial]`` (0 -> diagonal). A
    ``default`` in each section applies to groups without an explicit entry.

    Returns one :class:`goe.Covariance` per name, aligned with ``group_names``.
    """
    rel = relative_uncertainties(group_names, cfg)
    default_corr = cfg.get_float("category_spatial", "default", default=0.0)
    covs = []
    for g in group_names:
        corr_km = cfg.get_float("category_spatial", g, default=default_corr)
        if corr_km and corr_km > 0:
            covs.append(build_spatial_covariance(core, rel[g], corr_km))
        else:
            covs.append(DiagonalCovariance.isotropic(core.n_active, rel[g] ** 2))
    return covs


def _make_report(names, prior, means, cov, unit_label):
    sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(np.asarray(prior) != 0, means / prior, np.nan)
    return FluxReport(names=list(names), prior=np.asarray(prior), posterior=np.asarray(means),
                      posterior_stddev=sd, scale_factor=scale, unit_label=unit_label)


def partition_by_prior_variance(
    posterior, state, core, group_active, rel_unc, grid, total_block,
    unit_scale: float = 1.0, unit_label: str = "prior-units x m^2 (native)",
) -> FluxReport:
    """Partition a per-cell-total posterior into categories by prior variance.

    Parameters
    ----------
    total_block:
        Name of the per-cell scalar block (mean 1) on the inventory total.
    group_active, rel_unc:
        Per-group active-cell prior fields and relative uncertainties.
    """
    areas = core.from_field(cell_areas_m2(grid))
    names = list(group_active.keys())
    weights, e_total = prior_variance_weights(group_active, rel_unc)
    sl = state.slice(total_block)

    # category k total = P_k + m_k^T (s - 1),  m_k(cell) = area * w_k * E_total
    A = np.zeros((len(names) + 1, state.size))
    prior = np.zeros(len(names) + 1)
    b = np.zeros(len(names) + 1)
    for k, g in enumerate(names):
        m_k = areas * weights[g] * e_total * unit_scale
        A[k, sl] = m_k
        prior[k] = (areas * np.asarray(group_active[g]) * unit_scale).sum()  # P_k
        b[k] = prior[k] - m_k.sum()
    A[-1] = A[:-1].sum(axis=0)            # inventory total
    prior[-1] = prior[:-1].sum()
    b[-1] = prior[-1] - A[-1, sl].sum()   # ~0 by construction

    means_lin, cov = linear_estimate(posterior, A)
    means = means_lin + b
    return _make_report(names + ["total"], prior, means, cov, unit_label)


def assemble_category_scalar_state(core, base, group_active, cfg, n_obs, offset_op=None):
    """Build the solved-category state: g_k scalars + per-cell total correction.

    Returns ``(blocks, operators, cov_blocks, group_names)`` ready to assemble a
    StateSpace, forward operator, and block-diagonal prior. The category block is
    a single block of size K (one domain scalar per group, prior mean 1); the
    ``total_corr`` block is a per-cell correction on the inventory total (prior
    mean 0). The optional offset block (bc) is appended if provided.
    """
    names = list(group_active.keys())
    e_total_active = sum(np.asarray(group_active[g], dtype=float) for g in names)

    # category fingerprint block: column k = H E_k
    cols = np.column_stack([base.matvec(np.asarray(group_active[g], dtype=float))
                            for g in names])
    cat_op = DenseOperator(cols)
    cat_block = Block(name="categories", size=len(names),
                      metadata={"kind": "category_scalars", "groups": names})

    # per-cell correction block on the inventory total (mean 0)
    corr_state = GriddedState(core.grid, core.mask, name="total_corr")
    corr_block = corr_state.block()
    corr_op = scale_columns(base, e_total_active)

    rel = relative_uncertainties(names, cfg)
    cat_cov = DiagonalCovariance(np.array([rel[g] ** 2 for g in names]))

    scalar_sd = cfg.get_float("prior", "scalar_stddev", default=0.5)
    corr_km = cfg.get_float("prior", "correlation_length_km", default=0.0)
    corr_cov = (build_spatial_covariance(core, scalar_sd, corr_km) if corr_km > 0
                else DiagonalCovariance.isotropic(core.n_active, scalar_sd ** 2))

    blocks = [cat_block, corr_block]
    operators = {"categories": cat_op, "total_corr": corr_op}
    cov_blocks = [cat_cov, corr_cov]
    return blocks, operators, cov_blocks, names


def decompose_solved_categories(
    posterior, state, core, group_active, grid, group_names,
    cat_block="categories", corr_block="total_corr",
    unit_scale: float = 1.0, unit_label: str = "prior-units x m^2 (native)",
) -> FluxReport:
    """Decompose when category scalars g_k were solved explicitly.

    category k total = g_k * integral(area E_k) + integral(area E_k c).
    """
    areas = core.from_field(cell_areas_m2(grid))
    cat_sl = state.slice(cat_block)
    corr_sl = state.slice(corr_block)

    A = np.zeros((len(group_names) + 1, state.size))
    prior = np.zeros(len(group_names) + 1)
    for k, g in enumerate(group_names):
        ga = np.asarray(group_active[g], dtype=float)
        P_k = (areas * ga * unit_scale).sum()
        A[k, cat_sl.start + k] = P_k          # coefficient on g_k
        A[k, corr_sl] = areas * ga * unit_scale  # coefficient on c
        prior[k] = P_k                         # at g_k=1, c=0
    A[-1] = A[:-1].sum(axis=0)
    prior[-1] = prior[:-1].sum()

    means, cov = linear_estimate(posterior, A)
    return _make_report(group_names + ["total"], prior, means, cov, unit_label)
