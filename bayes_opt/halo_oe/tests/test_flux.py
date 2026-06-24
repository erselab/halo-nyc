"""Tests for the flux-aggregation module (halo_oe.flux).

Validates cell-area computation, the generic linear-functional estimator against
a dense reference, and the end-to-end flux report on a small synthetic problem
built through the framework. No real data files are needed.

Run directly:  python halo_oe/tests/test_flux.py
(from the bayes_opt directory, or with bayes_opt on PYTHONPATH).
"""

from __future__ import annotations

import os
import sys

import numpy as np

# make bayes_opt importable, then halo_oe wires goe/adapters onto the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import halo_oe  # noqa: F401,E402

from goe import (  # noqa: E402
    BlockDiagonalCovariance, DenseOperator, DiagonalCovariance,
    GaussianLinearProblem, StateSpace, solve,
)
from adapters.gridded_state import Grid, GriddedState  # noqa: E402
from adapters.scaling_blocks import category_blocks, offset_block  # noqa: E402
from halo_oe.flux import cell_areas_m2, linear_estimate, estimate_fluxes  # noqa: E402


def test_cell_areas_reasonable():
    # ~0.01 deg cells near 40N: area should be ~1 km^2 = 1e6 m^2, order-of-magnitude
    g = Grid(lat=np.arange(40.0, 40.1, 0.01), lon=np.arange(-74.0, -73.9, 0.01))
    A = cell_areas_m2(g)
    assert A.shape == g.shape
    assert np.all(A > 0)
    # total area of the patch ~ (0.1deg * 111km) * (0.1deg*~85km) ~ 9.4e7 m^2... per cell ~1e6
    assert 5e5 < A.mean() < 2e6
    # global sphere sanity: a full 1-deg band integrates sensibly (monotone in lat handled)
    assert np.allclose(A[0], A[-1], rtol=0.05)  # nearly constant over a tiny lat span


def test_linear_estimate_matches_dense():
    rng = np.random.default_rng(50)
    n_obs, n_state = 30, 8
    H = rng.standard_normal((n_obs, n_state))
    xa = np.zeros(n_state)
    Sa = DiagonalCovariance(rng.uniform(0.5, 2, n_state))
    R = DiagonalCovariance(rng.uniform(0.1, 0.3, n_obs))
    z = rng.standard_normal(n_obs)
    post = solve(GaussianLinearProblem(H=H, z=z, xa=xa, Sa=Sa, R=R))

    A = rng.standard_normal((3, n_state))
    means, cov = linear_estimate(post, A)

    Shat = post.cov_dense()
    assert np.allclose(means, A @ post.mean)
    assert np.allclose(cov, A @ Shat @ A.T, atol=1e-8)
    # variances are non-negative
    assert np.all(np.diag(cov) >= -1e-10)


def _build_synthetic_inversion(seed=51):
    rng = np.random.default_rng(seed)
    grid = Grid(lat=np.linspace(40.0, 40.3, 6), lon=np.linspace(-74.0, -73.7, 6))
    core = GriddedState(grid, name="core")
    n_obs = 60
    base = DenseOperator(rng.uniform(0, 0.02, (n_obs, core.n_active)))
    priors = {
        "edgar": rng.uniform(1, 5, grid.shape),
        "epa": rng.uniform(1, 5, grid.shape),
    }
    cat_blocks, cat_ops = category_blocks(core, base, priors)
    off_blk, off_op = offset_block(np.zeros(n_obs, dtype=int), 1, name="bc")
    state = StateSpace(cat_blocks + [off_blk])
    H = state.block_column({**cat_ops, "bc": off_op})
    Sa = BlockDiagonalCovariance([
        DiagonalCovariance.isotropic(core.n_active, 0.25),
        DiagonalCovariance.isotropic(core.n_active, 0.25),
        DiagonalCovariance.isotropic(1, 0.01),
    ])
    R = DiagonalCovariance.isotropic(n_obs, 1e-3)
    xa = state.fill(0.0, edgar=1.0, epa=1.0, bc=0.0)
    x_true = state.pack({
        "edgar": rng.uniform(0.6, 1.4, core.n_active),
        "epa": rng.uniform(0.6, 1.4, core.n_active),
        "bc": np.array([0.05]),
    })
    z = H.matvec(x_true) + R.half(rng.standard_normal(n_obs))
    post = solve(GaussianLinearProblem(H=H, z=z, xa=xa, Sa=Sa, R=R))
    return post, state, core, priors, grid, xa


def test_single_category_has_no_total():
    """One inventory as prior -> no summed 'total' row (would be redundant)."""
    post, state, core, priors, grid, xa = _build_synthetic_inversion()
    rep = estimate_fluxes(post, state, core, {"edgar": priors["edgar"]}, grid,
                          prior_state=xa)
    assert rep.names == ["edgar"]
    assert rep.posterior.shape == (1,)


def test_additive_total_can_be_disabled():
    """include_total=False suppresses the summed row even for >1 category."""
    post, state, core, priors, grid, xa = _build_synthetic_inversion()
    rep = estimate_fluxes(post, state, core, priors, grid, prior_state=xa,
                          include_total=False)
    assert rep.names == ["edgar", "epa"]


def test_estimate_fluxes_consistency():
    post, state, core, priors, grid, xa = _build_synthetic_inversion()
    rep = estimate_fluxes(post, state, core, priors, grid, prior_state=xa)

    # names: the two (additive, synthetic) categories + total
    assert rep.names == ["edgar", "epa", "total"]
    # total prior/posterior equal the sum of category prior/posterior
    assert np.isclose(rep.prior[-1], rep.prior[:-1].sum())
    assert np.isclose(rep.posterior[-1], rep.posterior[:-1].sum())

    # cross-check prior totals against a direct integral: sum(prior_field*area) on active cells
    areas = core.from_field(cell_areas_m2(grid))
    for i, name in enumerate(["edgar", "epa"]):
        direct = (core.from_field(priors[name]) * areas).sum()
        assert np.isclose(rep.prior[i], direct)

    # posterior total uncertainty must be >= 0 and finite
    assert np.all(np.isfinite(rep.posterior_stddev))
    assert np.all(rep.posterior_stddev >= 0)
    # scale factor = posterior/prior
    assert np.allclose(rep.scale_factor[:-1], rep.posterior[:-1] / rep.prior[:-1])


def test_total_uncertainty_includes_cross_covariance():
    """Var(total) uses the full Ŝ, not the sum of independent category variances."""
    post, state, core, priors, grid, xa = _build_synthetic_inversion()
    rep = estimate_fluxes(post, state, core, priors, grid, prior_state=xa)
    # rebuild the aggregation rows to compare variances directly
    areas = core.from_field(cell_areas_m2(grid))
    A = np.zeros((2, state.size))
    for k, name in enumerate(["edgar", "epa"]):
        A[k, state.slice(name)] = core.from_field(priors[name]) * areas
    a_total = A.sum(0)
    _, cov = linear_estimate(post, np.vstack([A, a_total]))
    var_total = cov[2, 2]
    var_sum_indep = cov[0, 0] + cov[1, 1]
    assert np.isclose(rep.posterior_stddev[-1] ** 2, var_total)
    # total variance differs from the independent-sum (cross term present)
    assert not np.isclose(var_total, var_sum_indep)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
