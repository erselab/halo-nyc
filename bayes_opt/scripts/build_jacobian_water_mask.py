"""Precompute and cache the land/water mask on the STILT Jacobian's native
~1666x1666 (~1km) footprint grid, for `halo_oe.background.
flag_water_affected_footprints` to load at solve time without a live network
fetch. All 6 flights share an identical Jacobian grid (verified directly,
not assumed), so one cached mask covers every flight.

Same construction as footprint_land_water_check.py §30.5/§30.6: HRRR's
static `LAND` field (`hrrrzarr/sfc/.../surface/LAND/surface`), nearest-
matched via KDTree to every Jacobian grid cell. Water = 1, land = 0,
flattened lat-major (matches `Grid.cell_centers()` / the Jacobian's own
row-flatten convention).

Run once (or whenever the Jacobian grid changes):
    python3 scripts/build_jacobian_water_mask.py
"""

from __future__ import annotations

import sys

import h5py
import numpy as np
import s3fs
import xarray as xr
from scipy.spatial import cKDTree

sys.path.insert(0, ".")

from adapters.jacobian_operator import JacobianFile

JAC_DIR = "../stilt/harvard_jacobians"
REFERENCE_FLIGHT = "20230726_1"
LAND_DATE_HOUR = ("20230726", 14)   # any date/hour -- LAND is a static terrain mask
OUT_PATH = "scratch_hrrr/jacobian_grid_water_mask.npy"


def main():
    fs = s3fs.S3FileSystem(anon=True)
    date_str, hour = LAND_DATE_HOUR
    path = f"hrrrzarr/sfc/{date_str}/{date_str}_{hour:02d}z_anl.zarr/surface/LAND/surface"
    store = s3fs.S3Map(root=path, s3=fs, check=False)
    land_field = np.asarray(xr.open_zarr(store, consolidated=False)["LAND"])

    with h5py.File("scratch_hrrr/HRRR_latlon.h5", "r") as f:
        hlat, hlon = f["latitude"][:], f["longitude"][:]
    tree = cKDTree(np.column_stack([hlat.ravel(), hlon.ravel()]))

    jf = JacobianFile(f"{JAC_DIR}/{REFERENCE_FLIGHT}.nc")
    glat, glon = jf.grid.cell_centers()
    jf.close()

    _, idx = tree.query(np.column_stack([glat, glon]), workers=-1)
    iy, ix = np.unravel_index(idx, hlat.shape)
    water_mask = 1.0 - land_field[iy, ix]   # water = 1, land = 0

    np.save(OUT_PATH, water_mask.astype(np.float32))
    print(f"saved {OUT_PATH}: shape={water_mask.shape}, water fraction of domain={water_mask.mean():.3f}")


if __name__ == "__main__":
    main()
