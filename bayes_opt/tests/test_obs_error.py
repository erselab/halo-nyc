"""Tests for the HALO component-wise observation-error covariance.

Checks the local km projection, that R combines an independent measurement
diagonal with a correlated model-data-mismatch term, and that the diagonal /
correlated branches are selected by config. Synthetic; no real files.

Run:  python tests/test_obs_error.py   (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402

from goe.covariance import DiagonalCovariance, SparseCovariance  # noqa: E402
from halo_oe.obs_error import receptors_to_km, build_obs_error_covariance  # noqa: E402


class _Cfg:
    def __init__(self, d):
        self._d = d
    def get(self, s, k, default=None):
        return self._d.get((s, k), default)
    def get_float(self, s, k, default=None):
        v = self._d.get((s, k), default)
        return None if v is None else float(v)


def test_receptors_to_km():
    lat = np.array([40.0, 40.1, 40.0])
    lon = np.array([-74.0, -74.0, -73.9])
    xy = receptors_to_km(lat, lon)
    assert xy.shape == (3, 2)
    # 0.1 deg latitude ~ 11.1 km north
    assert abs(xy[1, 1] - xy[0, 1] - 11.1) < 0.3
    # 0.1 deg lon at ~40N ~ 8.5 km east
    assert abs(xy[2, 0] - xy[0, 0] - 8.5) < 0.5


def test_components_correlated_R():
    rng = np.random.default_rng(80)
    n = 60
    lat = 40.6 + 0.2 * rng.random(n)
    lon = -74.0 + 0.2 * rng.random(n)
    cfg = _Cfg({("observations", "measurement_stddev"): 0.01,
                ("observations", "mdm_stddev"): 0.02,
                ("observations", "mdm_correlation_length_km"): 8.0,
                ("observations", "error_inflation"): 1.0})
    R = build_obs_error_covariance(lat, lon, cfg)
    assert isinstance(R, SparseCovariance)
    d = np.diag(R.to_dense())
    # diagonal = measurement var + mdm var
    assert np.allclose(d, 0.01**2 + 0.02**2)
    # off-diagonal correlations are present (nearby receptors)
    assert R.to_dense().sum() > d.sum()
    x = rng.standard_normal(n)
    assert np.allclose(R.matvec(R.solve(x)), x, atol=1e-7)


def test_components_zero_correlation_is_diagonal():
    lat = np.linspace(40.6, 40.9, 20); lon = np.linspace(-74.1, -73.8, 20)
    cfg = _Cfg({("observations", "measurement_stddev"): 0.01,
                ("observations", "mdm_stddev"): 0.02,
                ("observations", "mdm_correlation_length_km"): 0.0,
                ("observations", "error_inflation"): 1.0})
    R = build_obs_error_covariance(lat, lon, cfg)
    assert isinstance(R, DiagonalCovariance)
    assert np.allclose(R.variances, 0.01**2 + 0.02**2)


def test_explicit_measurement_variance_and_inflation():
    lat = np.linspace(40.6, 40.9, 15); lon = np.linspace(-74.1, -73.8, 15)
    meas = np.full(15, 0.015**2)
    cfg = _Cfg({("observations", "mdm_stddev"): 0.0,
                ("observations", "mdm_correlation_length_km"): 0.0,
                ("observations", "error_inflation"): 4.0})
    R = build_obs_error_covariance(lat, lon, cfg, measurement_variance=meas)
    assert isinstance(R, DiagonalCovariance)
    assert np.allclose(R.variances, 4.0 * meas)   # inflation applied


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
