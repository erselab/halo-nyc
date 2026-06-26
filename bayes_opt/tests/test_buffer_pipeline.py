"""End-to-end test of the buffer region through load_context / invert.

Builds a small synthetic Jacobian + inventory (as in test_multiflight) with a core
bbox strictly inside the domain so out-of-core cells exist, then checks that:

* ``load_context`` builds a Buffer and a buffer operator stacked over flights,
* ``invert`` adds a "buffer" block of the right size with the area-weighted prior
  mean, leaving the reported core total unchanged in structure, and
* the bundle round-trips the buffer geometry.

Run:  python tests/test_buffer_pipeline.py   (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402

from goe.config import Config  # noqa: E402
from halo_oe.pipeline import load_context, invert  # noqa: E402
from halo_oe.io_bundle import save_inversion, load_inversion  # noqa: E402

try:
    import netCDF4  # noqa: E402
    import h5py  # noqa: E402
    HAVE_DEPS = True
except ImportError:  # pragma: no cover
    HAVE_DEPS = False

LAT = np.linspace(40.0, 41.0, 12)
LON = np.linspace(-74.5, -73.5, 12)


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
            f.create_dataset(s, data=rng.uniform(0.1, 1, (n, len(elat), len(elon))))
            f.attrs[f"{s}_categories"] = ";".join(f"{s} cat {i}" for i in range(n))


def _cfg(tmp, flights, buffer):
    base = {
        "jacobian": {"dir": tmp, "flights": flights, "in_memory": "true"},
        # core covers only the interior -> a ring of out-of-core cells remains
        "domain": {"bbox": "[40.35, 40.65, -74.15, -73.85]"},
        "emissions": {"path": os.path.join(tmp, "emis.h5"),
                      "inventory": "edgar", "compare": "edgar,epa,pitt"},
        "background": {"method": "planar", "degree": "1",
                       "domain_sensitivity_quantile": "1.0"},
        "prior": {"scalar_stddev": "0.5", "correlation_length_km": "0"},
        "observations": {"error_stddev": "0.02", "baseline": "1.9"},
        "offset": {"n_groups": "1", "stddev": "0.05"},
        "buffer": buffer,
        "flux": {"unit_scale": "1.0"},
    }
    return Config(mapping=base)


def test_no_buffer_when_disabled():
    if not HAVE_DEPS:
        print("  skip (deps missing)"); return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_emis(os.path.join(tmp, "emis.h5"))
        ctx = load_context(_cfg(tmp, "f1", {"enabled": "false"}), inventories=["edgar"])
        assert ctx.buffer is None and ctx.buffer_op is None
        res = invert(ctx, "edgar")
        assert "buffer" not in res.state.names


def test_coarse_buffer_block():
    if not HAVE_DEPS:
        print("  skip (deps missing)"); return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp, "f1", {"enabled": "true", "mode": "coarse", "factor": "3",
                               "stddev": "1.0"})
        ctx = load_context(cfg, inventories=["edgar"])
        assert ctx.buffer is not None and ctx.buffer.n_super > 0
        assert ctx.buffer_op.shape == (40, ctx.buffer.n_super)
        res = invert(ctx, "edgar")
        blk = res.state.block("buffer")
        assert blk.size == ctx.buffer.n_super
        # buffer prior mean is the area-weighted inventory density (positive here)
        xa = res.problem.xa
        parts = res.state.unpack(xa)
        assert np.all(parts["buffer"] > 0)
        # core flux report still only reports the inventory total (buffer excluded)
        assert res.report.names == ["edgar"]


def test_buffer_stacks_over_flights():
    if not HAVE_DEPS:
        print("  skip (deps missing)"); return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_jacobian(os.path.join(tmp, "f2.nc"), 55, seed=2)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp, "f1,f2", {"enabled": "true", "factor": "3"})
        ctx = load_context(cfg, inventories=["edgar"])
        assert ctx.buffer_op.shape == (95, ctx.buffer.n_super)
        res = invert(ctx, "edgar")
        assert res.state.block("buffer").size == ctx.buffer.n_super
        assert np.isfinite(res.report.posterior[0])


def test_bundle_roundtrips_buffer():
    if not HAVE_DEPS:
        print("  skip (deps missing)"); return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp, "f1", {"enabled": "true", "factor": "3"})
        ctx = load_context(cfg, inventories=["edgar"])
        res = invert(ctx, "edgar")
        out = save_inversion(os.path.join(tmp, "bundle"), ctx, res)
        sv = load_inversion(out)
        assert sv.buffer is not None
        assert sv.buffer["value"].shape == (ctx.buffer.n_super,)
        assert sv.buffer["center_lat"].shape == (ctx.buffer.n_super,)
        assert "buffer" in sv.state.names
        # buffer posterior recoverable from the reloaded state
        assert sv.block("buffer").shape == (ctx.buffer.n_super,)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
