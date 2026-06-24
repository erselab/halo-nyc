"""Tests for the per-flight lower-envelope planar background (halo_oe.background).

Builds synthetic flights = a known planar background + a localized positive
"plume" + noise, and checks that the lower-envelope fit recovers the underlying
plane (ignoring the plume) and that distinct flights yield distinct backgrounds.

Run directly:  python halo_oe/tests/test_background.py
(from the bayes_opt directory, or with bayes_opt on PYTHONPATH).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import halo_oe  # noqa: F401,E402

from halo_oe.background import (  # noqa: E402
    constant_background, polynomial_design, fit_lower_envelope_surface,
    flight_background, receptor_background,
)


def test_polynomial_design_shapes():
    x = np.linspace(0, 1, 5)
    y = np.linspace(0, 1, 5)
    assert polynomial_design(x, y, 1).shape == (5, 3)   # 1, x, y
    assert polynomial_design(x, y, 2).shape == (5, 6)   # + x^2, xy, y^2


def _synthetic_flight(rng, n=400, plane=(2.00, 0.05, -0.03), plume_amp=0.15):
    lat = rng.uniform(40.3, 41.4, n)
    lon = rng.uniform(-74.9, -72.3, n)
    a, b, c = plane
    background = a + b * (lat - lat.mean()) + c * (lon - lon.mean())
    # a localized positive enhancement over part of the domain (one-sided)
    plume = plume_amp * np.exp(-(((lat - 40.7) / 0.15) ** 2 + ((lon - -74.0) / 0.2) ** 2))
    obs = background + plume + 0.005 * rng.standard_normal(n)
    return lat, lon, obs, background


def test_recovers_plane_under_plume():
    rng = np.random.default_rng(60)
    lat, lon, obs, true_bg = _synthetic_flight(rng)
    bg = flight_background(lat, lon, obs, degree=1, quantile=0.25, n_iter=6)
    # the fitted background should track the true plane, not the inflated mean
    err = bg - true_bg
    assert np.sqrt(np.mean(err ** 2)) < 0.02, np.sqrt(np.mean(err ** 2))
    # background must sit at or below the observations almost everywhere (lower envelope)
    assert np.mean(bg <= obs + 1e-9) > 0.9


def test_lower_envelope_below_ols_mean():
    rng = np.random.default_rng(61)
    lat, lon, obs, true_bg = _synthetic_flight(rng, plume_amp=0.3)
    bg = flight_background(lat, lon, obs, degree=1, quantile=0.2, n_iter=6)
    # a plain mean would be pulled up by the plume; the envelope fit stays lower
    assert bg.mean() < obs.mean()


def test_flight_dependence():
    rng = np.random.default_rng(62)
    # two flights with different background levels/gradients
    latA, lonA, obsA, _ = _synthetic_flight(rng, plane=(2.00, 0.05, -0.03))
    latB, lonB, obsB, _ = _synthetic_flight(rng, plane=(2.08, -0.02, 0.04))
    bgA = flight_background(latA, lonA, obsA)
    bgB = flight_background(latB, lonB, obsB)
    # the two flights produce clearly different background levels
    assert abs(bgA.mean() - bgB.mean()) > 0.05


def test_coeffs_evaluate_consistently():
    rng = np.random.default_rng(63)
    lat, lon, obs, _ = _synthetic_flight(rng)
    coeffs, design = fit_lower_envelope_surface(lat, lon, obs, degree=1)
    assert np.allclose(design @ coeffs, flight_background(lat, lon, obs))


class _FakeJac:
    def __init__(self, lat, lon, obs):
        self.receptor_lat, self.receptor_lon, self.receptor_obs = lat, lon, obs
        self.n_receptors = len(obs)


class _Cfg:
    def __init__(self, d):
        self._d = d
    def get(self, s, k, default=None):
        return self._d.get((s, k), default)
    def get_float(self, s, k, default=None):
        v = self._d.get((s, k), default); return None if v is None else float(v)
    def get_int(self, s, k, default=None):
        v = self._d.get((s, k), default); return None if v is None else int(v)


def test_receptor_background_dispatch():
    rng = np.random.default_rng(64)
    lat, lon, obs, _ = _synthetic_flight(rng)
    jf = _FakeJac(lat, lon, obs)

    planar = receptor_background(jf, _Cfg({("background", "method"): "planar"}))
    assert planar.shape == (len(obs),)
    assert planar.std() > 0  # spatially varying

    const = receptor_background(jf, _Cfg({
        ("background", "method"): "constant", ("background", "constant_value"): 1.9}))
    assert np.allclose(const, 1.9)

    # missing coordinates -> constant fallback
    jf2 = _FakeJac(None, None, obs)
    fb = receptor_background(jf2, _Cfg({("observations", "baseline"): 1.95}))
    assert np.allclose(fb, 1.95)


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
