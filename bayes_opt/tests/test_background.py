"""Tests for the per-flight lower-envelope planar background (halo_oe.background).

Builds synthetic flights = a known planar background + a localized positive
"plume" + noise, and checks that the lower-envelope fit recovers the underlying
plane (ignoring the plume) and that distinct flights yield distinct backgrounds.

Run directly:  python tests/test_background.py
(from the bayes_opt directory, or with bayes_opt on PYTHONPATH).
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402

from halo_oe.background import (  # noqa: E402
    constant_background, polynomial_design, fit_lower_envelope_surface,
    flight_background, receptor_background, domain_insensitive_mask,
    detect_legs, fit_leg_offsets,
    flag_leg_edge_discontinuities, flag_footprint_discontinuities,
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


class _FakeJacVar:
    """Mimics a netCDF4 Variable: supports ``var[i, :, :]`` row indexing."""
    def __init__(self, arr):
        self._arr = arr
    def __getitem__(self, key):
        return self._arr[key]


class _FakeDS:
    def __init__(self, arr, jac_var="jacobian"):
        self.variables = {jac_var: _FakeJacVar(arr)}


def _gaussian_footprint(ny, nx, cy, cx, sigma=2.0):
    yy, xx = np.mgrid[0:ny, 0:nx]
    return np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))


class _Cfg:
    def __init__(self, d):
        self._d = d
    def get(self, s, k, default=None):
        return self._d.get((s, k), default)
    def get_float(self, s, k, default=None):
        v = self._d.get((s, k), default); return None if v is None else float(v)
    def get_int(self, s, k, default=None):
        v = self._d.get((s, k), default); return None if v is None else int(v)
    def get_bool(self, s, k, default=None):
        v = self._d.get((s, k), default); return None if v is None else bool(v)


def test_fit_mask_excludes_points():
    """The surface fit ignores masked-out points but is evaluated everywhere."""
    rng = np.random.default_rng(65)
    n = 300
    lat = rng.uniform(40.3, 41.4, n)
    lon = rng.uniform(-74.9, -72.3, n)
    true = 2.0 + 0.05 * (lat - lat.mean()) - 0.03 * (lon - lon.mean())
    obs = true + 0.003 * rng.standard_normal(n)
    # the MAJORITY of receptors are "in-domain" and strongly elevated, so the
    # lower-envelope alone cannot exclude them (they dominate the low quantile too)
    in_domain = rng.random(n) < 0.7
    obs[in_domain] += 0.3
    fit_mask = ~in_domain   # fit only the out-of-domain receptors

    bg = flight_background(lat, lon, obs, degree=1, quantile=0.5, n_iter=4, fit_mask=fit_mask)
    # background should track the true plane, NOT be pulled up by the +0.3 receptors
    assert np.sqrt(np.mean((bg - true) ** 2)) < 0.02
    # contrast: fitting everything (no mask) is biased high
    bg_all = flight_background(lat, lon, obs, degree=1, quantile=0.5, n_iter=4)
    assert bg_all.mean() > bg.mean() + 0.02


def test_domain_insensitive_mask():
    ds = np.array([0.0, 1.0, 2.0, 3.0, 10.0])
    m = domain_insensitive_mask(ds, 0.5)  # keep lowest 50%
    assert m.tolist() == [True, True, True, False, False]


def test_receptor_background_uses_domain_sensitivity():
    rng = np.random.default_rng(66)
    n = 200
    lat = rng.uniform(40.3, 41.4, n)
    lon = rng.uniform(-74.9, -72.3, n)
    true = 2.0 + 0.04 * (lat - lat.mean())
    obs = true + 0.003 * rng.standard_normal(n)
    sens = rng.uniform(0, 1, n)
    hot = sens > 0.6
    obs[hot] += 0.4                      # domain-sensitive receptors carry enhancement
    jf = _FakeJac(lat, lon, obs)
    cfg = _Cfg({("background", "method"): "planar", ("background", "degree"): 1,
                ("background", "envelope_quantile"): 0.5, ("background", "n_iter"): 4,
                ("background", "domain_sensitivity_quantile"): 0.5})
    bg, offset = receptor_background(jf, cfg, domain_sensitivity=sens)
    # background should sit near the clean plane, not inflated by the hot receptors
    assert np.sqrt(np.mean((bg - true) ** 2)) < 0.03
    assert np.allclose(offset, 0.0)   # use_leg_offsets is off


def test_receptor_background_dispatch():
    rng = np.random.default_rng(64)
    lat, lon, obs, _ = _synthetic_flight(rng)
    jf = _FakeJac(lat, lon, obs)

    planar, planar_offset = receptor_background(jf, _Cfg({("background", "method"): "planar"}))
    assert planar.shape == (len(obs),)
    assert planar.std() > 0  # spatially varying
    assert np.allclose(planar_offset, 0.0)

    const, const_offset = receptor_background(jf, _Cfg({
        ("background", "method"): "constant", ("background", "constant_value"): 1.9}))
    assert np.allclose(const, 1.9)
    assert np.allclose(const_offset, 0.0)

    # missing coordinates -> constant fallback
    jf2 = _FakeJac(None, None, obs)
    fb, fb_offset = receptor_background(jf2, _Cfg({("observations", "baseline"): 1.95}))
    assert np.allclose(fb, 1.95)
    assert np.allclose(fb_offset, 0.0)


def _synthetic_track(n_per_leg, headings_deg, gap_s=200.0, step_s=5.0, start=(40.5, -74.5)):
    """A track flown along ``headings_deg`` (one leg per heading), separated by
    real time gaps, with receptors spaced ``step_s`` apart along each leg."""
    lat0, lon0 = start
    lats, lons, times = [], [], []
    t = 0.0
    lat, lon = lat0, lon0
    step_deg = 0.012   # ~ observed median step size
    for heading in headings_deg:
        for _ in range(n_per_leg):
            lats.append(lat); lons.append(lon); times.append(t)
            lat += step_deg * np.cos(np.radians(heading))
            lon += step_deg * np.sin(np.radians(heading)) / np.cos(np.radians(lat))
            t += step_s
        t += gap_s   # turn between legs
    return np.array(lats), np.array(lons), np.array(times)


def test_detect_legs_basic():
    # NE leg, then SW leg (heading reversal at a real time gap)
    lat, lon, t = _synthetic_track(50, [45.0, 225.0])
    leg_id = detect_legs(lat, lon, t)
    assert leg_id.max() + 1 == 2
    assert (leg_id[:50] == 0).all()
    assert (leg_id[50:] == 1).all()


def test_detect_legs_ignores_dropout_without_heading_flip():
    # single NE leg with a big time gap mid-leg but NO heading change
    lat, lon, t = _synthetic_track(100, [45.0])
    t = t.copy()
    t[50:] += 200.0   # a large gap that doesn't coincide with a turn
    leg_id = detect_legs(lat, lon, t)
    assert leg_id.max() + 1 == 1, "a time gap without a heading reversal must not split a leg"


def test_detect_legs_merges_tiny_legs():
    # NE, SW, NE with the middle leg far too short to be real (stray point at a turn)
    lat, lon, t = _synthetic_track(60, [45.0, 225.0, 45.0])
    # shrink the middle leg to 2 points
    lat = np.concatenate([lat[:60], lat[60:62], lat[120:]])
    lon = np.concatenate([lon[:60], lon[60:62], lon[120:]])
    t = np.concatenate([t[:60], t[60:62], t[120:] - (t[120] - t[62] - 5)])
    leg_id = detect_legs(lat, lon, t, min_leg_size=10)
    assert leg_id.max() + 1 == 2, "the 2-point middle leg should be merged into its neighbor"


def test_fit_leg_offsets_recovers_known_offsets():
    rng = np.random.default_rng(70)
    n_per_leg = 200
    true_offsets = [0.02, -0.03, 0.05]
    leg_id = np.repeat(np.arange(3), n_per_leg)
    # legs well separated in time (30 min apart), so precision (many points,
    # tiny noise variance) dominates the GP prior and each is recovered ~independently
    leg_time = np.repeat([0.0, 1800.0, 3600.0], n_per_leg)
    plane_background = np.full(leg_id.shape, 2.0)
    resid_noise = 0.003 * rng.standard_normal(leg_id.shape)
    offset_true = np.array(true_offsets)[leg_id]
    value = plane_background + offset_true + resid_noise
    fitted = fit_leg_offsets(value, plane_background, leg_id, leg_time, quantile=0.5)
    for k, true_off in enumerate(true_offsets):
        est = fitted[leg_id == k][0]
        assert abs(est - true_off) < 0.01, (k, est, true_off)


def test_fit_leg_offsets_pulls_sparse_leg_toward_neighbors():
    rng = np.random.default_rng(71)
    # two well-sampled legs at the same true offset (0.05), flanking a sparse
    # (2-point) leg close in time whose raw sample happens to read very differently
    leg_id = np.concatenate([np.zeros(200, dtype=int), np.ones(2, dtype=int), np.full(200, 2, dtype=int)])
    leg_time = np.concatenate([np.full(200, 0.0), [300.0, 300.0], np.full(200, 600.0)])
    plane_background = np.zeros(leg_id.shape)
    value = np.concatenate([
        0.05 + 0.003 * rng.standard_normal(200),
        [0.15, 0.16],           # a noisy, unrepresentative raw read from only 2 points
        0.05 + 0.003 * rng.standard_normal(200),
    ])
    fitted = fit_leg_offsets(value, plane_background, leg_id, leg_time, quantile=0.5)
    flank = fitted[leg_id == 0][0]
    sparse = fitted[leg_id == 1][0]
    # the sparse (n=2) leg's raw estimate (~0.155) is pulled sharply toward its
    # well-sampled, time-adjacent neighbors' consensus (~0.05): its variance is
    # inflated toward the prior scale by the reliability-gap term (n << 15),
    # so the GP prior -- not its own noisy 2-point read -- dominates
    assert abs(flank - 0.05) < 0.01
    assert sparse < 0.10, f"sparse leg should be pulled toward its neighbors, got {sparse}"
    assert sparse > flank, "some pull from its own (higher) raw read should remain"


def test_flag_leg_edge_discontinuities_flags_corrupted_edge():
    ny, nx, n_per_leg = 20, 20, 12
    footprints = np.zeros((2 * n_per_leg, ny, nx))
    for i in range(n_per_leg):                                    # leg 0: smooth sweep
        footprints[i] = _gaussian_footprint(ny, nx, cy=5, cx=2 + i)
    for i in range(n_per_leg):                                    # leg 1: smooth sweep
        footprints[n_per_leg + i] = _gaussian_footprint(ny, nx, cy=15, cx=2 + i)
    # corrupt only the LAST point of leg 0: a footprint at an unrelated location,
    # as if that release point's back-trajectory was still shaped by the turn
    footprints[n_per_leg - 1] = _gaussian_footprint(ny, nx, cy=0, cx=19)

    leg_id = np.array([0] * n_per_leg + [1] * n_per_leg)
    jf = _FakeJac(np.zeros(2 * n_per_leg), np.zeros(2 * n_per_leg), np.zeros(2 * n_per_leg))
    jf._ds = _FakeDS(footprints)
    jf._jac_var = "jacobian"

    flag = flag_leg_edge_discontinuities(jf, leg_id, relative_threshold=0.5, min_leg_size=6)
    assert flag[n_per_leg - 1], "the corrupted leg-end receptor must be flagged"
    assert flag.sum() == 1, f"only the corrupted receptor should be flagged, got {flag.sum()}"


def test_flag_leg_edge_discontinuities_clean_legs_unflagged():
    ny, nx, n_per_leg = 20, 20, 12
    footprints = np.zeros((2 * n_per_leg, ny, nx))
    for i in range(n_per_leg):
        footprints[i] = _gaussian_footprint(ny, nx, cy=5, cx=2 + i)
    for i in range(n_per_leg):
        footprints[n_per_leg + i] = _gaussian_footprint(ny, nx, cy=15, cx=2 + i)

    leg_id = np.array([0] * n_per_leg + [1] * n_per_leg)
    jf = _FakeJac(np.zeros(2 * n_per_leg), np.zeros(2 * n_per_leg), np.zeros(2 * n_per_leg))
    jf._ds = _FakeDS(footprints)
    jf._jac_var = "jacobian"

    flag = flag_leg_edge_discontinuities(jf, leg_id, relative_threshold=0.5, min_leg_size=6)
    assert not flag.any(), "smooth, uncorrupted legs must not be flagged"


def test_flag_footprint_discontinuities_default_off():
    jf = _FakeJac(np.zeros(5), np.zeros(5), np.zeros(5))
    flag = flag_footprint_discontinuities(jf, _Cfg({}))
    assert flag.shape == (5,)
    assert not flag.any()


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
