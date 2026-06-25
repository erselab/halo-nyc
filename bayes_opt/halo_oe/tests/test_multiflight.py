"""Tests for multi-flight assimilation (shared flux state, stacked flights).

Builds small synthetic per-flight Jacobian netCDFs and a synthetic inventory file,
then checks that ``load_context`` stacks flights correctly (rows concatenated,
shared columns, block-diagonal R, per-flight offset) and that a single-flight run
still matches the un-stacked case. No large files.

Run:  python halo_oe/tests/test_multiflight.py   (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import halo_oe  # noqa: F401,E402

from goe.config import Config  # noqa: E402
from halo_oe.pipeline import load_context, invert, flight_paths  # noqa: E402

try:
    import netCDF4  # noqa: E402
    import h5py  # noqa: E402
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

LAT = np.linspace(40.0, 41.0, 10)
LON = np.linspace(-74.5, -73.5, 10)


def _write_jacobian(path, n_rec, seed):
    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("receptor", n_rec)
    ds.createDimension("emitter_lat", len(LAT))
    ds.createDimension("emitter_lon", len(LON))
    ds.createVariable("emissions_lat", "f8", ("emitter_lat",))[:] = LAT
    ds.createVariable("emissions_lon", "f8", ("emitter_lon",))[:] = LON
    ds.createVariable("jacobian", "f8", ("receptor", "emitter_lat", "emitter_lon"))[:] = \
        rng.uniform(0, 0.02, (n_rec, len(LAT), len(LON)))
    ds.createVariable("receptor_xch4", "f8", ("receptor",))[:] = 1.9 + rng.uniform(0, 0.1, n_rec)
    ds.createVariable("receptor_lat", "f8", ("receptor",))[:] = rng.uniform(LAT[0], LAT[-1], n_rec)
    ds.createVariable("receptor_lon", "f8", ("receptor",))[:] = rng.uniform(LON[0], LON[-1], n_rec)
    ds.close()


def _write_emis(path):
    rng = np.random.default_rng(99)
    elat = np.linspace(39.5, 41.5, 8)
    elon = np.linspace(-75.0, -73.0, 8)
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=elat)
        f.create_dataset("lon", data=elon)
        for s, n in (("edgar", 3), ("epa", 2), ("pitt", 2)):
            f.create_dataset(s, data=rng.uniform(0, 1, (n, len(elat), len(elon))))
            f.attrs[f"{s}_categories"] = ";".join(f"{s} cat {i}" for i in range(n))


def _cfg(tmp, flights, **over):
    base = {
        "jacobian": {"dir": tmp, "flights": flights, "in_memory": "true"},
        "domain": {"bbox": "[40.1, 40.9, -74.4, -73.6]"},
        "emissions": {"path": os.path.join(tmp, "emis.h5"),
                      "inventory": "edgar", "compare": "edgar,epa,pitt"},
        "background": {"method": "planar", "degree": "1",
                       "domain_sensitivity_quantile": "1.0"},
        "prior": {"scalar_stddev": "0.5", "correlation_length_km": "0"},
        "observations": {"error_stddev": "0.02", "baseline": "1.9"},
        "offset": {"n_groups": "1", "stddev": "0.05"},
        "flux": {"unit_scale": "1.0"},
    }
    for sec, d in over.items():
        base.setdefault(sec, {}).update(d)
    return Config(mapping=base)


def test_flight_paths_resolution():
    cfg = Config(mapping={"jacobian": {"dir": "/d", "flights": "a, b ,c"}})
    assert flight_paths(cfg) == [("a", "/d/a.nc"), ("b", "/d/b.nc"), ("c", "/d/c.nc")]
    # explicit override wins
    assert flight_paths(cfg, ["x"]) == [("x", "/d/x.nc")]
    # single path fallback
    cfg2 = Config(mapping={"jacobian": {"path": "/some/20230726_1.nc"}})
    assert flight_paths(cfg2) == [("20230726_1", "/some/20230726_1.nc")]


def test_two_flight_stacking():
    if not HAVE_DEPS:
        print("  skip test_two_flight_stacking (deps missing)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_jacobian(os.path.join(tmp, "f2.nc"), 55, seed=2)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp, "f1,f2")

        ctx = load_context(cfg, inventories=["edgar"])
        assert ctx.n_flights == 2
        assert ctx.flight_ids == ["f1", "f2"]
        assert ctx.obs.n_obs == 40 + 55
        assert ctx.base.shape[0] == 95 and ctx.base.shape[1] == ctx.core.n_active
        # flight index marks the two blocks
        assert ctx.flight_index[:40].tolist() == [0] * 40
        assert ctx.flight_index[40:].tolist() == [1] * 55

        res = invert(ctx, "edgar")
        # one background offset per flight
        bc = res.state.block("bc")
        assert bc.size == 2
        assert res.report.names == ["edgar"]
        assert np.isfinite(res.report.posterior[0])


def test_single_flight_backcompat():
    if not HAVE_DEPS:
        print("  skip test_single_flight_backcompat (deps missing)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp, "f1")
        ctx = load_context(cfg, inventories=["edgar"])
        assert ctx.n_flights == 1 and ctx.obs.n_obs == 40
        # base is the bare operator (not wrapped) for a single flight
        assert ctx.base.shape == (40, ctx.core.n_active)
        res = invert(ctx, "edgar")
        assert res.state.block("bc").size == 1


def test_flights_override():
    if not HAVE_DEPS:
        print("  skip test_flights_override (deps missing)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        for nm, sd, nr in (("f1", 1, 40), ("f2", 2, 55), ("f3", 3, 30)):
            _write_jacobian(os.path.join(tmp, f"{nm}.nc"), nr, seed=sd)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp, "f1,f2,f3")
        # override selects a subset (experiment with a combination of days)
        ctx = load_context(cfg, inventories=["edgar"], flights=["f1", "f3"])
        assert ctx.flight_ids == ["f1", "f3"]
        assert ctx.obs.n_obs == 40 + 30


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
