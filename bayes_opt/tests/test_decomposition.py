"""Tests for category grouping and posterior flux decomposition.

Covers the configurable keyword grouping, the prior-variance partition (category
scalars off), and the solved-category-scalar decomposition (g_k on). All checks
are synthetic; the key invariant is that per-category totals sum exactly to the
inventory total in both modes, with prior totals equal to the direct integral.

Run:  python tests/test_decomposition.py  (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402

from goe import (BlockDiagonalCovariance, DenseOperator, DiagonalCovariance,
                 GaussianLinearProblem, StateSpace, solve)  # noqa: E402
from adapters.gridded_state import Grid, GriddedState  # noqa: E402
from adapters.scaling_blocks import category_blocks  # noqa: E402
from halo_oe.groups import assign_groups, group_indices, DEFAULT_KEYWORD_MAP  # noqa: E402
from halo_oe.flux import cell_areas_m2  # noqa: E402
from goe.covariance import SparseCovariance  # noqa: E402
from halo_oe.decomposition import (  # noqa: E402
    prior_variance_weights, partition_by_prior_variance, category_covariance,
    assemble_category_scalar_state, decompose_solved_categories,
)


class _Cfg:
    def __init__(self, d=None):
        self._d = d or {}
    def get_float(self, s, k, default=None):
        return float(self._d.get((s, k), default)) if self._d.get((s, k), default) is not None else default


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def test_assign_groups_keywords():
    labels = ["Natural Gas Distribution", "Landfills MSW", "Wastewater Treatment Domestic",
              "Enteric Fermentation", "Combustion Mobile", "Something Unmatched"]
    a = assign_groups(labels)
    assert a["Natural Gas Distribution"] == "natural_gas"
    assert a["Landfills MSW"] == "landfill"
    assert a["Wastewater Treatment Domestic"] == "wastewater"
    assert a["Enteric Fermentation"] == "agriculture"
    assert a["Combustion Mobile"] == "combustion"
    assert a["Something Unmatched"] == "other"


def test_group_indices_collects_rows():
    labels = ["NG_distribution", "Landfill", "NG_transmission", "Wastewater_treatment"]
    idx, _ = group_indices(labels)
    assert set(idx["natural_gas"]) == {0, 2}
    assert list(idx["landfill"]) == [1]
    assert list(idx["wastewater"]) == [3]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _setup(seed=70):
    rng = np.random.default_rng(seed)
    grid = Grid(lat=np.linspace(40.0, 40.4, 8), lon=np.linspace(-74.2, -73.8, 8))
    core = GriddedState(grid, name="core")
    n = core.n_active
    # two category fields with different spatial patterns
    glat, glon = np.meshgrid(grid.lat, grid.lon, indexing="ij")
    Ea = (1 + np.exp(-(((glat-40.1)/0.1)**2 + ((glon+74.1)/0.1)**2)))
    Eb = (1 + np.exp(-(((glat-40.3)/0.1)**2 + ((glon+73.9)/0.1)**2)))
    group_active = {"a": core.from_field(Ea), "b": core.from_field(Eb)}
    E_total = group_active["a"] + group_active["b"]
    n_obs = 50
    base = DenseOperator(rng.uniform(0, 0.02, (n_obs, n)))
    return rng, grid, core, group_active, E_total, base, n_obs


# --------------------------------------------------------------------------- #
# prior-variance partition (category scalars off)
# --------------------------------------------------------------------------- #

def test_variance_weights_sum_to_one():
    _, _, _, group_active, _, _, _ = _setup()
    w, e_total = prior_variance_weights(group_active, {"a": 0.5, "b": 0.5})
    s = w["a"] + w["b"]
    assert np.allclose(s, 1.0)
    # equal rel uncertainty + here weights track E_k^2 share
    assert np.all(w["a"] >= 0) and np.all(w["b"] >= 0)


def test_partition_sums_to_total_and_prior():
    rng, grid, core, group_active, E_total, base, n_obs = _setup()
    # per-cell total scalar inversion on the inventory total
    inv_field = core.to_field(E_total)
    cat_blocks, cat_ops = category_blocks(core, base, {"pitt": inv_field})
    state = StateSpace(cat_blocks)
    H = state.block_column(cat_ops)
    Sa = BlockDiagonalCovariance([DiagonalCovariance.isotropic(core.n_active, 0.25)])
    R = DiagonalCovariance.isotropic(n_obs, 1e-3)
    xa = state.fill(1.0)
    z = H.matvec(state.fill(0.0, pitt=rng.uniform(0.7, 1.3, core.n_active))) + 1e-3*rng.standard_normal(n_obs)
    post = solve(GaussianLinearProblem(H=H, z=z, xa=xa, Sa=Sa, R=R))

    rep = partition_by_prior_variance(post, state, core, group_active,
                                      {"a": 0.5, "b": 0.5}, grid, total_block="pitt")
    assert rep.names == ["a", "b", "total"]
    # per-category posterior sums to the total row
    assert np.isclose(rep.posterior[0] + rep.posterior[1], rep.posterior[-1])
    assert np.isclose(rep.prior[0] + rep.prior[1], rep.prior[-1])
    # prior totals equal direct integral of each category field
    areas = core.from_field(cell_areas_m2(grid))
    assert np.isclose(rep.prior[0], (areas * group_active["a"]).sum())
    # total posterior equals integral(area * E_total * s_hat)
    s_hat = state.unpack(post.mean)["pitt"]
    assert np.isclose(rep.posterior[-1], (areas * E_total * s_hat).sum())


# --------------------------------------------------------------------------- #
# solved category scalars (g_k on)
# --------------------------------------------------------------------------- #

def test_solved_category_decomposition_sums():
    rng, grid, core, group_active, E_total, base, n_obs = _setup(seed=71)
    cfg = _Cfg({("category_uncertainty", "default"): 0.5,
                ("prior", "scalar_stddev"): 0.3, ("prior", "correlation_length_km"): 0.0})
    blocks, operators, cov_blocks, names = assemble_category_scalar_state(
        core, base, group_active, cfg, n_obs)
    assert names == ["a", "b"]
    state = StateSpace(blocks)
    H = state.block_column(operators)
    Sa = BlockDiagonalCovariance(cov_blocks)
    R = DiagonalCovariance.isotropic(n_obs, 1e-3)
    xa = state.fill(0.0, **{b.name: (1.0 if b.name == "categories" else 0.0) for b in state.blocks})

    # synthetic truth: g=(1.2,0.8), small c
    x_true = state.pack({"categories": np.array([1.2, 0.8]),
                         "total_corr": 0.05 * rng.standard_normal(core.n_active)})
    z = H.matvec(x_true) + 1e-3 * rng.standard_normal(n_obs)
    post = solve(GaussianLinearProblem(H=H, z=z, xa=xa, Sa=Sa, R=R))

    rep = decompose_solved_categories(post, state, core, group_active, grid, names)
    assert rep.names == ["a", "b", "total"]
    assert np.isclose(rep.posterior[0] + rep.posterior[1], rep.posterior[-1])
    # prior category totals equal direct integrals
    areas = core.from_field(cell_areas_m2(grid))
    assert np.isclose(rep.prior[0], (areas * group_active["a"]).sum())
    assert np.isclose(rep.prior[1], (areas * group_active["b"]).sum())
    # posterior near the (well-constrained) truth in this low-noise setup
    g_hat = state.unpack(post.mean)["categories"]
    assert abs(g_hat[0] - 1.2) < 0.3 and abs(g_hat[1] - 0.8) < 0.3


def test_category_covariance_diagonal_vs_spatial():
    _, _, core, group_active, _, _, _ = _setup()
    # group 'a' -> spatial (corr 5 km, diffuse), group 'b' -> diagonal (point source)
    cfg = _Cfg({("category_uncertainty", "default"): 0.4,
                ("category_spatial", "default"): 0.0,
                ("category_spatial", "a"): 5.0})
    covs = category_covariance(core, ["a", "b"], cfg)
    assert isinstance(covs[0], SparseCovariance)     # diffuse -> spatially correlated
    assert isinstance(covs[1], DiagonalCovariance)   # point source -> diagonal
    # diagonal variance uses the per-cell relative sigma squared
    assert np.allclose(covs[1].variances, 0.4 ** 2)


def test_category_fields_sum_to_total():
    """Per-cell field per category -> additive totals sum to the inventory total."""
    rng, grid, core, group_active, E_total, base, n_obs = _setup(seed=72)
    from adapters.scaling_blocks import category_blocks as _cb
    from halo_oe.flux import estimate_fluxes
    group_fields = {g: core.to_field(group_active[g]) for g in group_active}
    cat_blocks, cat_ops = _cb(core, base, group_fields)
    names = [b.name for b in cat_blocks]
    cfg = _Cfg({("category_uncertainty", "default"): 0.3,
                ("category_spatial", "default"): 0.0, ("category_spatial", "a"): 4.0})
    cov_blocks = category_covariance(core, names, cfg)
    state = StateSpace(cat_blocks)
    H = state.block_column(cat_ops)
    Sa = BlockDiagonalCovariance(cov_blocks)
    R = DiagonalCovariance.isotropic(n_obs, 1e-3)
    xa = state.fill(1.0)
    z = H.matvec(state.fill(1.0, a=rng.uniform(0.8, 1.3, core.n_active))) + 1e-3*rng.standard_normal(n_obs)
    post = solve(GaussianLinearProblem(H=H, z=z, xa=xa, Sa=Sa, R=R))
    rep = estimate_fluxes(post, state, core, group_fields, grid, prior_state=xa, include_total=True)
    assert rep.names[-1] == "total"
    assert np.isclose(rep.posterior[:-1].sum(), rep.posterior[-1])


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
