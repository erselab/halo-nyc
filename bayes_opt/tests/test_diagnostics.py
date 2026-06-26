"""Tests for the out-of-core (domain-truncation) sensitivity diagnostic.

Builds a synthetic Jacobian whose sensitivity is concentrated in a known region,
then checks that the diagnostic correctly reports the fraction of sensitivity
(raw and emission-weighted) falling outside a chosen core mask.

Run:  python tests/test_diagnostics.py   (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402

from adapters.gridded_state import Grid, GriddedState  # noqa: E402
from adapters.jacobian_operator import JacobianFile  # noqa: E402
from halo_oe.diagnostics import (  # noqa: E402
    core_sizing, out_of_core_sensitivity, summarize_out_of_core)

try:
    import netCDF4  # noqa: E402
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

LAT = np.linspace(40.0, 41.0, 12)
LON = np.linspace(-74.6, -73.4, 12)


def _write_jacobian(path, n_rec, H):
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("receptor", n_rec)
    ds.createDimension("emitter_lat", len(LAT)); ds.createDimension("emitter_lon", len(LON))
    ds.createVariable("emissions_lat", "f8", ("emitter_lat",))[:] = LAT
    ds.createVariable("emissions_lon", "f8", ("emitter_lon",))[:] = LON
    ds.createVariable("jacobian", "f8", ("receptor", "emitter_lat", "emitter_lon"))[:] = H
    rng = np.random.default_rng(0)
    ds.createVariable("receptor_xch4", "f8", ("receptor",))[:] = 1.9 + rng.uniform(0, 0.1, n_rec)
    ds.createVariable("receptor_lat", "f8", ("receptor",))[:] = rng.uniform(LAT[0], LAT[-1], n_rec)
    ds.createVariable("receptor_lon", "f8", ("receptor",))[:] = rng.uniform(LON[0], LON[-1], n_rec)
    ds.close()


def test_out_of_core_fraction():
    if not HAVE_DEPS:
        print("  skip test_out_of_core_fraction (deps missing)")
        return
    rng = np.random.default_rng(1)
    n_rec = 25
    g = Grid(LAT, LON)
    # core = lower-left quadrant
    core = GriddedState(g, g.bbox_mask(40.0, 40.5, -74.6, -74.0))
    # build a Jacobian: each receptor puts a known fraction of sensitivity inside
    # the core and the rest outside
    H = np.zeros((n_rec, g.n_lat, g.n_lon))
    inside_field = g.bbox_mask(40.0, 40.5, -74.6, -74.0)
    target_inside = rng.uniform(0.2, 0.8, n_rec)
    for i in range(n_rec):
        ins = rng.uniform(0, 1, (g.n_lat, g.n_lon)) * inside_field
        out = rng.uniform(0, 1, (g.n_lat, g.n_lon)) * (~inside_field)
        ins *= target_inside[i] / ins.sum()
        out *= (1 - target_inside[i]) / out.sum()
        H[i] = ins + out

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.nc")
        _write_jacobian(path, n_rec, H)
        with JacobianFile(path) as jf:
            res = out_of_core_sensitivity(jf, core)        # unweighted
            # per-receptor fraction-outside matches the injected (1 - target_inside)
            assert np.allclose(res["uniform"]["fraction_outside"], 1 - target_inside, atol=1e-9)
            summ = summarize_out_of_core(res)
            assert "integrated_fraction_outside" in summ["uniform"]
            assert 0.0 <= summ["uniform"]["integrated_fraction_outside"] <= 1.0


def test_emission_weighting_changes_fraction():
    if not HAVE_DEPS:
        print("  skip test_emission_weighting_changes_fraction (deps missing)")
        return
    rng = np.random.default_rng(2)
    n_rec = 20
    g = Grid(LAT, LON)
    core = GriddedState(g, g.bbox_mask(40.0, 40.5, -74.6, -74.0))
    H = rng.uniform(0, 0.02, (n_rec, g.n_lat, g.n_lon))
    # emission concentrated OUTSIDE the core -> emission-weighted fraction higher
    prior = np.where(core.mask, 0.1, 1.0)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.nc")
        _write_jacobian(path, n_rec, H)
        with JacobianFile(path) as jf:
            res = out_of_core_sensitivity(jf, core, prior_field=prior)
            assert set(res) == {"uniform", "emission"}
            su = summarize_out_of_core(res)
            assert su["emission"]["integrated_fraction_outside"] > \
                   su["uniform"]["integrated_fraction_outside"]


def test_core_sizing_concentrated_signal():
    if not HAVE_DEPS:
        print("  skip test_core_sizing_concentrated_signal (deps missing)")
        return
    rng = np.random.default_rng(3)
    n_rec = 30
    g = Grid(LAT, LON)
    # sensitivity present everywhere, but emission concentrated in a small patch
    H = rng.uniform(0.0, 0.02, (n_rec, g.n_lat, g.n_lon))
    patch = g.bbox_mask(40.4, 40.6, -74.1, -73.9)        # small central box
    prior = np.where(patch, 5.0, 1e-3)                   # signal dominated by the patch

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "f.nc")
        _write_jacobian(path, n_rec, H)
        with JacobianFile(path) as jf:
            res = core_sizing([jf], g, prior, fractions=(0.8, 0.95))

    rows = res["rows"]
    assert [r["fraction_target"] for r in rows] == [0.8, 0.95]
    # boxes actually capture at least the requested share, and grow with the target
    for r in rows:
        assert r["captured_weighted"] >= r["fraction_target"] - 1e-9
    assert rows[1]["n_active"] >= rows[0]["n_active"]
    # the suggested 80% box sits within the emission patch's neighbourhood
    b = rows[0]["bbox"]
    assert 40.3 <= b[0] and b[1] <= 40.7 and -74.2 <= b[2] and b[3] <= -73.8
    # participation ratio is far below the full grid (signal is concentrated)
    assert 0 < res["participation_ratio"] < g.n_cells


def test_core_sizing_sums_flights():
    if not HAVE_DEPS:
        print("  skip test_core_sizing_sums_flights (deps missing)")
        return
    rng = np.random.default_rng(4)
    g = Grid(LAT, LON)
    H1 = rng.uniform(0, 0.02, (12, g.n_lat, g.n_lon))
    H2 = rng.uniform(0, 0.02, (18, g.n_lat, g.n_lon))
    prior = rng.uniform(0.1, 1.0, g.shape)

    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = os.path.join(tmp, "f1.nc"), os.path.join(tmp, "f2.nc")
        _write_jacobian(p1, 12, H1); _write_jacobian(p2, 18, H2)
        with JacobianFile(p1) as jf1, JacobianFile(p2) as jf2:
            from halo_oe.diagnostics import cell_sensitivity_field
            sens, _ = cell_sensitivity_field([jf1, jf2], g, prior)
            # combined sensitivity is the per-cell sum over BOTH flights' receptors
            expect = H1.reshape(12, -1).sum(0) + H2.reshape(18, -1).sum(0)
            assert np.allclose(sens, expect)
            res = core_sizing([jf1, jf2], g, prior, fractions=(0.9,))
            assert res["rows"][0]["captured_weighted"] >= 0.9 - 1e-9


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
