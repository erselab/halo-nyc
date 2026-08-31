"""One-off: per-receptor footprint centroid vs. the core domain, for each flight.

Answers: how many receptors per flight have a footprint centroid that clears
the core bbox entirely (the proposed background fit_mask), vs. the current
domain_sensitivity_quantile approach (fraction of total sensitivity outside core)?
"""
import sys, os, time
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import halo_oe  # noqa
from adapters.jacobian_operator import JacobianFile
from adapters.gridded_state import Grid, GriddedState
from goe.config import Config

cfg = Config('config.ini')
bbox = cfg.get_literal('domain', 'bbox')
jdir = cfg.get('jacobian', 'dir')
flights = [f.strip() for f in cfg.get('jacobian', 'flights').split(',')]
row_chunk = 32

lat_min, lat_max, lon_min, lon_max = bbox
print(f"core bbox: lat [{lat_min},{lat_max}]  lon [{lon_min},{lon_max}]", flush=True)

for fid in flights:
    out_npz = f"runs/centroid_{fid}.npz"
    if os.path.exists(out_npz):
        print(f"{fid}: already done, skipping", flush=True)
        continue
    path = os.path.join(jdir, fid + '.nc')
    t0 = time.time()
    jf = JacobianFile(path)
    rlat = np.asarray(jf.receptor_lat)
    rlon = np.asarray(jf.receptor_lon)
    grid = jf.grid
    core_mask = grid.bbox_mask(lat_min, lat_max, lon_min, lon_max).reshape(-1)  # (n_cells,) bool

    LAT, LON = np.meshgrid(grid.lat, grid.lon, indexing='ij')
    lat_flat = LAT.reshape(-1)
    lon_flat = LON.reshape(-1)

    n = jf.n_receptors
    total = np.zeros(n)
    inside = np.zeros(n)
    clat = np.full(n, np.nan)
    clon = np.full(n, np.nan)

    jac = jf._ds.variables[jf._jac_var]
    for i0 in range(0, n, row_chunk):
        i1 = min(i0 + row_chunk, n)
        block = np.asarray(jac[i0:i1, :, :]).reshape(i1 - i0, -1)
        block = np.nan_to_num(block, nan=0.0)
        s = block.sum(axis=1)
        total[i0:i1] = s
        inside[i0:i1] = block[:, core_mask].sum(axis=1)
        ok = s > 0
        if ok.any():
            idx = np.nonzero(ok)[0]
            clat[i0:i1][idx] = (block[idx] @ lat_flat) / s[idx]
            clon[i0:i1][idx] = (block[idx] @ lon_flat) / s[idx]
    jf.close()

    centroid_outside = ~((clat >= lat_min) & (clat <= lat_max) & (clon >= lon_min) & (clon <= lon_max))
    centroid_outside &= np.isfinite(clat)  # exclude zero-sensitivity receptors

    frac_outside = np.where(total > 0, 1.0 - inside / np.where(total > 0, total, 1), np.nan)

    np.savez(f"runs/centroid_{fid}.npz", receptor_lat=rlat, receptor_lon=rlon,
             centroid_lat=clat, centroid_lon=clon, total=total, inside=inside,
             frac_outside=frac_outside, centroid_outside=centroid_outside)

    print(f"{fid}: n={n}  centroid-outside-core={centroid_outside.sum()} "
          f"({100*centroid_outside.sum()/n:.1f}%)  "
          f"frac_outside>0.5={np.sum(frac_outside>0.5)} "
          f"frac_outside>0.9={np.sum(frac_outside>0.9)}  "
          f"[{time.time()-t0:.1f}s]", flush=True)

print("DONE", flush=True)
