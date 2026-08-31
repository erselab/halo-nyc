"""Check the fitted per-leg background offsets for a systematic drift in time
over the course of each flight -- a possible signature of instrument thermal
response (e.g. warm-up after takeoff) rather than real boundary-layer
evolution or spatial background variation.

Background: HALO surveys NYC in a lawnmower pattern over a few hours, so
elapsed time since takeoff and geographic position are confounded within a
single flight -- a trend in the leg offsets against elapsed time could
equally be a spatial background gradient. Two things help separate the two:

0. Using true elapsed time since actual takeoff, not since the start of the
   NYC-clipped survey segment. The Jacobian-aligned flight files in
   `flight_data/` are clipped to the NYC lawnmower pattern only, dropping the
   climb-out/transit portion of each flight -- 37 to 71 minutes of real
   elapsed flight time, varying by flight (see `_full.h5`, the unclipped
   product, for `Nav_Data/gps_time`). Elapsed time here is always measured
   from that true takeoff time, recovered from the unclipped file, not from
   the first receptor in the clipped file.
1. Comparing multiple flights. Each flight's `use_leg_offsets` fit
   (`halo_oe.background.fit_leg_offsets`) already estimates one GP-smoothed
   additive offset per leg from the observations themselves (see
   `runs/legtest_legoffset_6flight`, saved per-receptor as
   `receptor_background_offset` in `fields.nc`). If a similar *shape* of
   drift (e.g. offset trending up over the first 30-60 minutes) appears
   across flights that fly different tracks, start at different times of
   day, and sample different air masses, a shared instrument-side cause
   (thermal drift) is a more plausible explanation than location-specific
   boundary-layer structure.
2. Elapsed-time-since-takeoff vs. absolute time-of-day. The six flights
   start at quite different UTC clock times. If the offset trend lines up
   in elapsed-since-takeoff across flights despite very different start
   times, that favors an instrument warm-up effect. If it lines up in
   absolute time-of-day instead, that favors a diurnal boundary-layer
   effect (real atmosphere, not instrument).
3. Aircraft altitude as a third candidate regressor. The NYC-clipped files
   carry `flight_alt` per receptor. Several flights descend meaningfully
   over the course of the NYC survey (not just during climb-out), which
   changes cabin/optical-bench temperature and pressure independent of
   elapsed time. If offset tracks altitude better than either time variable
   (and does so consistently in sign across flights), that points at a
   pressure/temperature-altitude effect rather than a fixed warm-up clock.
4. Surface (terrain) elevation and range-to-ground (AGL) as a fourth
   candidate. `UserInput/DEM_altitude` in the unclipped file gives ground
   elevation under each receptor (NYC-area terrain ranges from ~0 m at the
   coast/harbor to ~200 m inland), aligned to the clipped receptors by
   nearest GPS time. `flight_alt - DEM_altitude` gives the actual
   range-to-ground the laser/receiver saw, which is a lidar-relevant
   quantity distinct from MSL altitude: if a leg's background offset tracks
   terrain elevation or range-to-ground instead of (or in addition to) pure
   MSL altitude, that points at a surface-return or path-length effect
   (e.g. water vs. land vs. hilly terrain) rather than a cabin/instrument
   effect, since MSL altitude alone does not distinguish those.

This only reads the existing `legtest_legoffset_6flight` bundle (no re-solve,
no Jacobian) plus each flight's raw `time`/`lat`/`lon` from `flight_data/`,
and re-derives leg segmentation with the same config used to produce that
bundle, exactly as `leg_offset_check.py` does for its own purpose.

Run with:
    python3 scripts/leg_offset_drift_check.py
"""

from __future__ import annotations

import glob
import os
import sys

import h5py
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, ".")

from goe.config import Config
from halo_oe.background import detect_legs, _load_receptor_time, _resolve_flight_data_path
from halo_oe.io_bundle import load_inversion

BUNDLE = "runs/legtest_legoffset_6flight"
FLIGHT_DATA_DIR = "scratch_flight_data_1000m"
FULL_FLIGHT_DATA_DIR = "../flight_data"


def true_takeoff_hour(fid: str) -> float:
    """Real UTC takeoff hour for ``fid``, from the unclipped ``_full.h5`` product."""
    date, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    pattern = os.path.join(FULL_FLIGHT_DATA_DIR, f"*{date}*_F{fnum}_full.h5")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one unclipped file for {fid!r}, got {matches}")
    with h5py.File(matches[0], "r") as f:
        return float(np.nanmin(f["Nav_Data/gps_time"][:]))


def _load_receptor_alt(fid: str, flight_data_dir: str, receptor_lat, receptor_lon) -> np.ndarray:
    """Load ``flight_alt`` (m) for ``fid``, aligned to the Jacobian's receptors."""
    path = _resolve_flight_data_path(fid, flight_data_dir)
    with h5py.File(path, "r") as f:
        alt = np.asarray(f["flight_alt"][:], dtype=float)
        lat = np.asarray(f["lat"][:], dtype=float)
        lon = np.asarray(f["lon"][:], dtype=float)
    receptor_lat = np.asarray(receptor_lat, dtype=float)
    receptor_lon = np.asarray(receptor_lon, dtype=float)
    if lat.shape != receptor_lat.shape or not (
        np.allclose(lat, receptor_lat, atol=1e-4) and np.allclose(lon, receptor_lon, atol=1e-4)
    ):
        raise ValueError(f"flight_data file {path!r} receptor coordinates do not match for flight {fid!r}")
    return alt


def _load_receptor_surface_alt(fid: str, clock_h: np.ndarray) -> np.ndarray:
    """Ground (DEM) elevation (m) at each receptor, from the unclipped ``_full.h5``.

    The clipped, Jacobian-aligned product does not carry ``DEM_altitude``, so
    this matches each clipped receptor to its nearest sample (by GPS time,
    which is monotonic and shared) in the unclipped file, which does.
    """
    date, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    pattern = os.path.join(FULL_FLIGHT_DATA_DIR, f"*{date}*_F{fnum}_full.h5")
    matches = sorted(glob.glob(pattern))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected exactly one unclipped file for {fid!r}, got {matches}")
    with h5py.File(matches[0], "r") as f:
        t_full = f["Nav_Data/gps_time"][:, 0]
        dem = f["UserInput/DEM_altitude"][:, 0]
    idx = np.clip(np.searchsorted(t_full, clock_h), 0, len(t_full) - 1)
    return dem[idx]


WATER_SENTINEL = -0.5   # DEM_altitude is a clean bimodal -1.0 (water, no GLOBE/GEBCO land
                        # elevation) vs. >=~10 m (land) over the NYC domain -- nothing in between


def is_water(surface_m: np.ndarray) -> np.ndarray:
    return surface_m <= WATER_SENTINEL


def per_leg_table(inv, cfg, fid: str) -> dict:
    """Per-leg (elapsed_min, clock_hour, offset_ppm, n_pts) for one flight."""
    R = inv.receptors
    fi = inv.flight_ids.index(fid)
    sel = R["receptor_flight"].astype(int) == fi
    lat, lon = R["receptor_lat"][sel], R["receptor_lon"][sel]
    bg_offset = R["receptor_background_offset"][sel]

    clock_hours = _load_receptor_time(fid, FLIGHT_DATA_DIR, lat, lon)  # decimal UTC hours * 3600 -> seconds
    # _load_receptor_time returns seconds, but the source field is decimal UTC
    # hours of day (not elapsed since takeoff) -- recover both views here.
    clock_h = clock_hours / 3600.0
    takeoff_h = true_takeoff_hour(fid)
    elapsed_s = (clock_h - takeoff_h) * 3600.0
    alt_m = _load_receptor_alt(fid, FLIGHT_DATA_DIR, lat, lon)
    surface_m = _load_receptor_surface_alt(fid, clock_h)
    agl_m = alt_m - surface_m

    leg_id = detect_legs(
        lat, lon, clock_hours,
        gap_seconds=cfg.get_float("background", "leg_gap_seconds", default=8.0),
        min_leg_size=cfg.get_int("background", "leg_min_size", default=10),
        axis_deg=cfg.get_float("background", "leg_axis_deg", default=45.0),
    )

    legs = np.unique(leg_id)
    elapsed_min = np.zeros(legs.size)
    clock_mid = np.zeros(legs.size)
    offset = np.zeros(legs.size)
    alt_mid = np.zeros(legs.size)
    surface_mid = np.zeros(legs.size)
    agl_mid = np.zeros(legs.size)
    water_frac = np.zeros(legs.size)
    n_pts = np.zeros(legs.size, dtype=int)
    water = is_water(surface_m)
    for k, leg in enumerate(legs):
        m = leg_id == leg
        elapsed_min[k] = elapsed_s[m].mean() / 60.0
        clock_mid[k] = clock_h[m].mean()
        offset[k] = bg_offset[m].mean()
        alt_mid[k] = alt_m[m].mean()
        surface_mid[k] = surface_m[m].mean()
        agl_mid[k] = agl_m[m].mean()
        water_frac[k] = water[m].mean()
        n_pts[k] = m.sum()

    return dict(elapsed_min=elapsed_min, clock_hour=clock_mid, offset=offset, alt_m=alt_mid,
                surface_m=surface_mid, agl_m=agl_mid, water_frac=water_frac, n_pts=n_pts)


def main():
    inv = load_inversion(BUNDLE)
    cfg = Config(os.path.join(BUNDLE, "config.ini"))

    tables = {fid: per_leg_table(inv, cfg, fid) for fid in inv.flight_ids}

    print(f"{'flight':<14}{'n_legs':>7}{'  vs elapsed time (ppm/hr)':>28}{'      r':>8}{'       p':>10}"
          f"{'  vs altitude (ppm/km)':>24}{'      r':>8}{'       p':>10}"
          f"{'  vs surface elev (ppm/km)':>28}{'      r':>8}{'       p':>10}"
          f"{'  vs AGL (ppm/km)':>20}{'      r':>8}{'       p':>10}")
    fig, axes = plt.subplots(4, 1, figsize=(9, 18), constrained_layout=True, sharex=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(inv.flight_ids)))

    pooled_elapsed, pooled_alt, pooled_surface, pooled_agl, pooled_centered_offset = [], [], [], [], []
    for (fid, t), c in zip(tables.items(), colors):
        elapsed_h = t["elapsed_min"] / 60.0
        offset = t["offset"]
        alt_km = t["alt_m"] / 1000.0
        surface_km = t["surface_m"] / 1000.0
        agl_km = t["agl_m"] / 1000.0
        slope, intercept, r, p, se = stats.linregress(elapsed_h, offset)
        slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(alt_km, offset)
        slope_s, intercept_s, r_s, p_s, se_s = stats.linregress(surface_km, offset)
        slope_g, intercept_g, r_g, p_g, se_g = stats.linregress(agl_km, offset)
        print(f"{fid:<14}{len(offset):>7}{slope:>28.4f}{r:>8.3f}{p:>10.3g}"
              f"{slope_a:>24.4f}{r_a:>8.3f}{p_a:>10.3g}"
              f"{slope_s:>28.4f}{r_s:>8.3f}{p_s:>10.3g}"
              f"{slope_g:>20.4f}{r_g:>8.3f}{p_g:>10.3g}")

        axes[0].plot(t["elapsed_min"], offset, "o-", color=c, label=fid, ms=4)
        centered = offset - offset.mean()
        axes[1].plot(t["elapsed_min"], centered, "o-", color=c, label=fid, ms=4)
        axes[2].plot(alt_km, centered, "o", color=c, label=fid, ms=5)
        axes[3].plot(surface_km * 1000.0, centered, "o", color=c, label=fid, ms=5)
        pooled_elapsed.append(elapsed_h)
        pooled_alt.append(alt_km)
        pooled_surface.append(surface_km)
        pooled_agl.append(agl_km)
        pooled_centered_offset.append(centered)

    axes[0].set_xlabel("elapsed time since takeoff (min)")
    axes[0].set_ylabel("fitted leg offset (ppm)")
    axes[0].set_title("Per-leg background offset vs. elapsed flight time")
    axes[0].axhline(0, color="gray", lw=0.7)
    axes[0].legend(fontsize=8, ncol=2)

    axes[1].set_xlabel("elapsed time since takeoff (min)")
    axes[1].set_ylabel("leg offset minus flight mean (ppm)")
    axes[1].set_title("Same, each flight re-centered to its own mean (shape-only comparison)")
    axes[1].axhline(0, color="gray", lw=0.7)
    axes[1].legend(fontsize=8, ncol=2)

    axes[2].set_xlabel("aircraft altitude (km)")
    axes[2].set_ylabel("leg offset minus flight mean (ppm)")
    axes[2].set_title("Same, vs. aircraft altitude instead of time")
    axes[2].axhline(0, color="gray", lw=0.7)
    axes[2].legend(fontsize=8, ncol=2)

    axes[3].set_xlabel("surface (DEM) elevation under receptor (m)")
    axes[3].set_ylabel("leg offset minus flight mean (ppm)")
    axes[3].set_title("Same, vs. terrain elevation instead of time")
    axes[3].axhline(0, color="gray", lw=0.7)
    axes[3].legend(fontsize=8, ncol=2)

    plt.savefig("figures/leg_offset_drift_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("plot -> figures/leg_offset_drift_check.png")

    pooled_elapsed = np.concatenate(pooled_elapsed)
    pooled_alt = np.concatenate(pooled_alt)
    pooled_surface = np.concatenate(pooled_surface)
    pooled_agl = np.concatenate(pooled_agl)
    pooled_centered_offset = np.concatenate(pooled_centered_offset)
    slope, intercept, r, p, se = stats.linregress(pooled_elapsed, pooled_centered_offset)
    print(f"\npooled (flight-demeaned) offset vs elapsed time:  "
          f"slope={slope:+.4f} ppm/hr, r={r:+.3f}, p={p:.3g}, n={pooled_elapsed.size} legs")

    # Same regression against absolute clock time, to compare the two hypotheses.
    pooled_clock = np.concatenate([tables[fid]["clock_hour"] for fid in inv.flight_ids])
    slope_c, intercept_c, r_c, p_c, se_c = stats.linregress(pooled_clock, pooled_centered_offset)
    print(f"pooled (flight-demeaned) offset vs clock time:    "
          f"slope={slope_c:+.4f} ppm/hr, r={r_c:+.3f}, p={p_c:.3g}, n={pooled_clock.size} legs")

    slope_a, intercept_a, r_a, p_a, se_a = stats.linregress(pooled_alt, pooled_centered_offset)
    print(f"pooled (flight-demeaned) offset vs altitude:      "
          f"slope={slope_a:+.4f} ppm/km, r={r_a:+.3f}, p={p_a:.3g}, n={pooled_alt.size} legs")

    slope_s, intercept_s, r_s, p_s, se_s = stats.linregress(pooled_surface, pooled_centered_offset)
    print(f"pooled (flight-demeaned) offset vs surface elev:  "
          f"slope={slope_s:+.4f} ppm/km, r={r_s:+.3f}, p={p_s:.3g}, n={pooled_surface.size} legs")

    slope_g, intercept_g, r_g, p_g, se_g = stats.linregress(pooled_agl, pooled_centered_offset)
    print(f"pooled (flight-demeaned) offset vs AGL (range):   "
          f"slope={slope_g:+.4f} ppm/km, r={r_g:+.3f}, p={p_g:.3g}, n={pooled_agl.size} legs")


if __name__ == "__main__":
    main()
