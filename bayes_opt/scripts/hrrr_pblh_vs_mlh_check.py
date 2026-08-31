"""Does HRRR's own boundary-layer-height field (HPBL) -- the meteorology
STILT's transport is actually run on (`stilt/run_stilt_nyc_hrrr.r`'s
`met_path`) -- agree with HALO's lidar-measured mixed-layer height
(`CH4DataProducts/MixedLayerHeight`, used throughout mixed_layer_height_
check.py / footprint_mlh_correlation_check.py)? If STILT's underlying
meteorology has the wrong entrainment/boundary-layer depth, that's a much
more direct and actionable version of §29.4's footprint-breadth finding
than the lidar-only comparison could show on its own -- and it's an
independent check of the meteorology `mlht`/HPBL actually used to build the
Jacobians used throughout this whole investigation.

No local HRRR archive exists in this repo (checked: only stilt/
run_stilt_nyc_hrrr.r's `met_path` reference, pointing at a TACC scratch path
not present here, and no `.rds`/particle-trajectory output). Uses the
public NOAA HRRR Zarr archive on AWS Open Data instead
(s3://hrrrzarr, anonymous access, no credentials needed) -- the hourly
*analysis* (`anl`) HPBL field, `hrrrzarr/sfc/<date>/<date>_<hh>z_anl.zarr/
surface/HPBL/surface`, on HRRR's native ~3km Lambert Conformal grid, with
the static lat/lon grid (`hrrrzarr/grid/HRRR_latlon.h5`, ~19MB, cached
locally once) used to find the nearest grid cell to a representative NYC
point.

This is a first-pass, single-point sanity check (one representative
lat/lon per comparison, not a full spatial match to every receptor) -- see
the Verdict for what a fuller version would need.

Run with:
    python3 scripts/hrrr_pblh_vs_mlh_check.py
"""

from __future__ import annotations

import glob
import os

import h5py
import numpy as np
import s3fs
import xarray as xr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FULL_DIR = "../flight_data"
CLIP_DIR = "scratch_flight_data_1000m"
GRID_FILE = "scratch_hrrr/HRRR_latlon.h5"
WATER_SENTINEL = -0.5

FLIGHTS = ["20230726_1", "20230726_2", "20230728_1", "20230728_2", "20230805", "20230809"]
# representative NYC-area point (central urban/coastal domain, not a specific receptor)
REP_LAT, REP_LON = 40.75, -73.95


def nearest_hrrr_index():
    with h5py.File(GRID_FILE, "r") as f:
        lat = f["latitude"][:]
        lon = f["longitude"][:]
    d2 = (lat - REP_LAT) ** 2 + (lon - REP_LON) ** 2
    iy, ix = np.unravel_index(np.argmin(d2), d2.shape)
    return iy, ix, lat[iy, ix], lon[iy, ix]


def hrrr_hpbl(fs, date_str: str, hour: int, iy: int, ix: int) -> float:
    """HPBL (m) at one grid cell, one UTC hour, from the hourly analysis file."""
    path = (f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/"
            "surface/HPBL/surface")
    store = s3fs.S3Map(root=path, s3=fs, check=False)
    da = xr.open_zarr(store, consolidated=False)
    var = list(da.data_vars)[0] if da.data_vars else None
    arr = da[var] if var else da.to_array().isel(variable=0)
    return float(np.asarray(arr)[iy, ix])


def flight_window(fid: str):
    date, _, fnum = fid.partition("_")
    fnum = fnum or "1"
    full = sorted(glob.glob(os.path.join(FULL_DIR, f"*{date}*_F{fnum}_full.h5")))[0]
    clip = sorted(glob.glob(os.path.join(CLIP_DIR, f"*{date}*_F{fnum}_full_1000m.h5")))[0]
    with h5py.File(full, "r") as f:
        t = f["Nav_Data/gps_time"][:, 0]
        mlh = f["CH4DataProducts/MixedLayerHeight"][:, 0]
        dem = f["UserInput/DEM_altitude"][:, 0]
    tc = h5py.File(clip, "r")["time"][:]
    return date, t, mlh, dem, tc.min(), tc.max()


def main():
    iy, ix, glat, glon = nearest_hrrr_index()
    print(f"representative point: ({REP_LAT}, {REP_LON}) -> nearest HRRR cell "
          f"iy={iy}, ix={ix} at ({glat:.3f}, {glon:.3f})")

    fs = s3fs.S3FileSystem(anon=True)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True, sharey=False)
    print(f"\n{'flight':<12}{'mean HALO land-MLH':>20}{'mean HRRR HPBL':>16}{'HRRR/HALO':>11}{'bias (m)':>10}")

    for ax, fid in zip(axes.ravel(), FLIGHTS):
        date, t, mlh, dem, tc0, tc1 = flight_window(fid)
        land = dem > WATER_SENTINEL
        in_window = (t >= tc0) & (t <= tc1) & np.isfinite(mlh) & land
        tw, mw = t[in_window], mlh[in_window]

        hours = sorted(set(range(int(np.floor(tc0)), int(np.ceil(tc1)) + 1)))
        hrrr_t, hrrr_v = [], []
        for h in hours:
            if h < 0 or h > 23:
                continue
            try:
                v = hrrr_hpbl(fs, date, h, iy, ix)
            except Exception as e:
                print(f"  ! failed {fid} hour {h}z: {e}")
                continue
            hrrr_t.append(h)
            hrrr_v.append(v)
        hrrr_t, hrrr_v = np.array(hrrr_t, dtype=float), np.array(hrrr_v)

        nbins = max(3, int((tc1 - tc0) * 6))
        edges = np.linspace(tc0, tc1, nbins + 1)
        idx = np.clip(np.digitize(tw, edges) - 1, 0, nbins - 1)
        bin_t = 0.5 * (edges[:-1] + edges[1:])
        bin_med = np.array([np.median(mw[idx == k]) if np.any(idx == k) else np.nan for k in range(nbins)])
        ok = np.isfinite(bin_med)

        mean_halo = np.nanmean(bin_med[ok])
        # HRRR mean restricted to the same time window (interpolated to bin_t for a fair mean)
        hrrr_interp = np.interp(bin_t[ok], hrrr_t, hrrr_v)
        mean_hrrr = np.mean(hrrr_interp)
        print(f"{fid:<12}{mean_halo:>20.0f}{mean_hrrr:>16.0f}{mean_hrrr / mean_halo:>11.2f}"
              f"{mean_hrrr - mean_halo:>10.0f}")

        ax.plot(bin_t[ok] - 4, bin_med[ok], "o-", color="tab:blue", label="HALO lidar MLH (land-only)")
        ax.plot(hrrr_t - 4, hrrr_v, "s--", color="tab:red", label="HRRR HPBL (analysis)")
        ax.set_title(fid)
        ax.set_xlabel("local time, EDT (hr)")
        ax.set_ylabel("height (m)")
        ax.legend(fontsize=7)

    plt.savefig("figures/hrrr_pblh_vs_mlh_check.png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    print("\nplot -> figures/hrrr_pblh_vs_mlh_check.png")


if __name__ == "__main__":
    main()
