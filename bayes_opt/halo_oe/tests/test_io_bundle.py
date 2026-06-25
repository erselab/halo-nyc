"""Round-trip test for the saved-inversion bundle (save -> load -> re-aggregate).

Builds a small synthetic single-flight inversion, saves the bundle, reloads it,
and checks that the reloaded posterior reproduces the mean and supports exact
linear-functional aggregation (mean + covariance) without the forward operator —
the whole point of saving for post-hoc analysis.

Run:  python halo_oe/tests/test_io_bundle.py   (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
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

LAT = np.linspace(40.0, 41.0, 10)
LON = np.linspace(-74.5, -73.5, 10)


def _write_jacobian(path, n_rec, seed):
    rng = np.random.default_rng(seed)
    ds = netCDF4.Dataset(path, "w")
    ds.createDimension("receptor", n_rec)
    ds.createDimension("emitter_lat", len(LAT)); ds.createDimension("emitter_lon", len(LON))
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
    elat = np.linspace(39.5, 41.5, 8); elon = np.linspace(-75.0, -73.0, 8)
    with h5py.File(path, "w") as f:
        f.create_dataset("lat", data=elat); f.create_dataset("lon", data=elon)
        for s, n in (("edgar", 3), ("epa", 2), ("pitt", 2)):
            f.create_dataset(s, data=rng.uniform(0, 1, (n, len(elat), len(elon))))
            f.attrs[f"{s}_categories"] = ";".join(f"{s} cat {i}" for i in range(n))


def _cfg(tmp):
    return Config(mapping={
        "jacobian": {"dir": tmp, "flights": "f1", "in_memory": "true"},
        "domain": {"bbox": "[40.1, 40.9, -74.4, -73.6]"},
        "emissions": {"path": os.path.join(tmp, "emis.h5"), "inventory": "edgar",
                      "compare": "edgar,epa,pitt"},
        "background": {"method": "planar", "degree": "1", "domain_sensitivity_quantile": "1.0"},
        "prior": {"scalar_stddev": "0.5", "correlation_length_km": "0"},
        "observations": {"error_stddev": "0.02", "baseline": "1.9"},
        "offset": {"n_groups": "1", "stddev": "0.05"},
        "flux": {"unit_scale": "1.0"},
    })


def test_save_load_roundtrip():
    if not HAVE_DEPS:
        print("  skip test_save_load_roundtrip (deps missing)")
        return
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=1)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp)
        ctx = load_context(cfg, inventories=["edgar"])
        res = invert(ctx, "edgar")

        bundle = os.path.join(tmp, "bundle")
        save_inversion(bundle, ctx, res)
        # all expected pieces present
        for fn in ("factors.npz", "fields.nc", "layout.json", "report.json", "config.ini"):
            assert os.path.exists(os.path.join(bundle, fn)), fn

        loaded = load_inversion(bundle)
        # posterior mean reproduced exactly
        assert np.allclose(loaded.posterior.mean, res.posterior.mean)
        assert loaded.inventory == "edgar" and loaded.state.size == res.state.size

        # exact linear-functional aggregation without the operator
        rng = np.random.default_rng(5)
        A = rng.standard_normal((4, res.state.size))
        means, cov = loaded.estimate(A)
        ref_means = A @ res.posterior.mean
        ref_SAt = np.column_stack([res.posterior.cov_matvec(A[i]) for i in range(4)])
        assert np.allclose(means, ref_means)
        assert np.allclose(cov, A @ ref_SAt, atol=1e-8)

        # observation context is present and re-aggregation inputs are there
        for k in ("receptor_lat", "receptor_obs", "receptor_background", "enhancement", "modeled"):
            assert k in loaded.receptors and loaded.receptors[k].shape == (40,)
        assert len(loaded.group_fields) > 0
        assert "reduced_chi_square" in loaded.diagnostics


def test_reaggregate_to_total():
    """A saved category/total inversion can be re-summed to a domain total."""
    if not HAVE_DEPS:
        print("  skip test_reaggregate_to_total (deps missing)")
        return
    from halo_oe.flux import cell_areas_m2
    with tempfile.TemporaryDirectory() as tmp:
        _write_jacobian(os.path.join(tmp, "f1.nc"), 40, seed=2)
        _write_emis(os.path.join(tmp, "emis.h5"))
        cfg = _cfg(tmp)
        ctx = load_context(cfg, inventories=["edgar"])
        res = invert(ctx, "edgar")
        bundle = os.path.join(tmp, "b")
        save_inversion(bundle, ctx, res)
        loaded = load_inversion(bundle)

        # build a domain-total functional over the 'edgar' block from saved priors
        # (group_fields are stored on active cells)
        areas = loaded.core.from_field(cell_areas_m2(loaded.grid))
        prior_tot = sum(loaded.group_fields.values())
        a = np.zeros(loaded.state.size)
        a[loaded.state.slice("edgar")] = prior_tot * areas
        (mean,), cov = loaded.estimate(a[None, :])
        assert np.isfinite(mean) and cov[0, 0] >= 0
        # matches the saved report's total within rounding
        assert np.isclose(mean, res.report.posterior[0], rtol=1e-6)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
