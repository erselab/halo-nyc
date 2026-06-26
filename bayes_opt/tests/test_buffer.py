"""Tests for the buffer-region geometry (coarse and mask modes).

Checks that the buffer partitions only out-of-core cells, that the coarse and mask
modes produce sensible super-cells, and that prior_mean / to_field behave.

Run:  python tests/test_buffer.py   (from bayes_opt/)
"""

from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import halo_oe  # noqa: F401,E402

from goe.config import Config  # noqa: E402
from adapters.gridded_state import Grid, GriddedState  # noqa: E402
from halo_oe.buffer import build_buffer  # noqa: E402


def _grid_core():
    g = Grid(np.linspace(40.0, 41.2, 13), np.linspace(-74.6, -73.4, 13))
    core = GriddedState(g, g.bbox_mask(40.4, 40.8, -74.2, -73.8), name="core")
    return g, core


def test_disabled_returns_none():
    g, core = _grid_core()
    assert build_buffer(g, core, Config(mapping={"buffer": {"enabled": "false"}})) is None


def test_coarse_buffer():
    g, core = _grid_core()
    cfg = Config(mapping={"buffer": {"enabled": "true", "mode": "coarse", "factor": "3"}})
    buf = build_buffer(g, core, cfg)
    assert buf is not None and buf.n_super > 0
    # membership: -1 exactly on core cells, >=0 on (some) out-of-core cells, never both
    memb = buf.membership.reshape(g.shape)
    assert np.all(memb[core.mask] == -1)
    assert np.all(memb[~core.mask] >= -1)               # outer cells may or may not be buffered
    assert (memb >= 0).sum() == (~core.mask).sum()      # all out-of-core cells assigned (no outer bbox)
    # geometry arrays sized by n_super
    for arr in (buf.center_lat, buf.center_lon, buf.cell_count, buf.area):
        assert arr.shape == (buf.n_super,)
    assert buf.cell_count.sum() == (~core.mask).sum()


def test_coarse_resolution_overrides_factor():
    g, core = _grid_core()
    native = abs(g.lat[1] - g.lat[0])
    cfg = Config(mapping={"buffer": {"enabled": "true", "resolution_deg": str(native*4)}})
    buf = build_buffer(g, core, cfg)
    # ~4x coarsening -> fewer super-cells than the ~146 out-of-core native cells
    assert 0 < buf.n_super < (~core.mask).sum()


def test_outer_bbox_limits_extent():
    g, core = _grid_core()
    cfg = Config(mapping={"buffer": {"enabled": "true", "factor": "2",
                                     "outer_bbox": "[40.3, 40.9, -74.3, -73.7]"}})
    buf = build_buffer(g, core, cfg)
    inside = g.bbox_mask(40.3, 40.9, -74.3, -73.7) & ~core.mask
    assert buf.cell_count.sum() == inside.sum()         # only the ring within outer bbox


def test_mask_mode():
    g, core = _grid_core()
    labels = np.zeros(g.shape, dtype=int)
    labels[:6, :] = 1          # region 1
    labels[6:, :] = 2          # region 2
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "labels.npy")
        np.save(path, labels)
        cfg = Config(mapping={"buffer": {"enabled": "true", "mode": "mask", "mask_file": path}})
        buf = build_buffer(g, core, cfg)
    assert buf.n_super == 2                              # two labels -> two super-cells
    memb = buf.membership.reshape(g.shape)
    assert np.all(memb[core.mask] == -1)                 # core excluded even where labelled


def test_prior_mean_and_to_field():
    g, core = _grid_core()
    cfg = Config(mapping={"buffer": {"enabled": "true", "factor": "3"}})
    buf = build_buffer(g, core, cfg)
    prior = np.ones(g.shape) * 2.0                       # uniform emission
    pm = buf.prior_mean(prior)
    assert pm.shape == (buf.n_super,)
    assert np.allclose(pm, 2.0)                          # area-weighted mean of a constant
    fld = buf.to_field(pm)
    assert fld.shape == g.shape
    assert np.all(np.isnan(fld[core.mask]))             # core cells are not buffer


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn(); print(f"  ok  {fn.__name__}")
    print(f"\n{len(fns)} tests passed.")


if __name__ == "__main__":
    _run_all()
