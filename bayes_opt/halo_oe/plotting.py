import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from .flux import cell_areas_m2
from .io_bundle import load_inversion


def plot_posterior(inv: 'Inversion', out_path: str | None = None) -> None:
    """Plot posterior mean and 1σ for each scalar block in the inversion."""
    grid, core = inv.grid, inv.core
    m = core.mask
    rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1]+1, cols[0], cols[-1]+1
    EXTENT = [grid.lon[c0], grid.lon[c1-1], grid.lat[r0], grid.lat[r1-1]]
    crop = lambda f: f[r0:r1, c0:c1]

    cats = [b.name for b in inv.state.blocks if b.name not in ('bc', 'buffer')]
    std_parts = inv.state.unpack(inv.posterior.stddev())
    fig, ax = plt.subplots(2, len(cats), figsize=(4.2*len(cats), 8), constrained_layout=True, squeeze=False)
    for j, name in enumerate(cats):
        im0 = ax[0, j].imshow(crop(inv.field(name)), origin='lower', extent=EXTENT, aspect='auto',
                            cmap='RdBu_r', vmin=0.5, vmax=1.5)
        ax[0, j].set_title(f'posterior scalar: {name}'); fig.colorbar(im0, ax=ax[0, j], shrink=0.8)
        ustd = core.to_field(std_parts[name])
        im1 = ax[1, j].imshow(crop(ustd), origin='lower', extent=EXTENT, aspect='auto', cmap='viridis')
        ax[1, j].set_title(f'posterior 1\u03c3: {name}'); fig.colorbar(im1, ax=ax[1, j], shrink=0.8)
    for a in ax.ravel(): a.set_xlabel('lon'); a.set_ylabel('lat')
    if out_path:
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()

def plot_core_sizing(sizing_nc: str, out_path: str | None = None,
                     current_bbox=None) -> None:
    """Map the explained-enhancement field with the suggested core boxes.

    Reads the ``core_sizing.nc`` written by ``run_halo.py --size-core`` (per-cell
    ``explained_enhancement``, plus ``bbox_<pct>pct`` and ``participation_ratio``
    attributes) and draws the field (log scale) cropped to the largest suggested
    box, with each capture-fraction bbox and (optionally) the current core bbox
    overlaid. Saves a PNG to ``out_path``, else shows.
    """
    import netCDF4
    from matplotlib.colors import LogNorm

    with netCDF4.Dataset(sizing_nc) as ds:
        lat = np.asarray(ds['lat'][:]); lon = np.asarray(ds['lon'][:])
        ee = np.asarray(ds['explained_enhancement'][:])
        pr = float(getattr(ds, 'participation_ratio', np.nan))
        boxes = {int(k[len('bbox_'):-3]): [float(x) for x in np.atleast_1d(getattr(ds, k))]
                 for k in ds.ncattrs() if k.startswith('bbox_') and k.endswith('pct')}

    # crop to the largest suggested box (+ margin) so the field is legible
    allb = list(boxes.values()) + ([list(current_bbox)] if current_bbox is not None else [])
    latmin = min(b[0] for b in allb); latmax = max(b[1] for b in allb)
    lonmin = min(b[2] for b in allb); lonmax = max(b[3] for b in allb)
    mlat = (latmax - latmin) * 0.1 + 1e-9; mlon = (lonmax - lonmin) * 0.1 + 1e-9
    i0 = max(int(np.searchsorted(lat, latmin - mlat)), 0)
    i1 = min(int(np.searchsorted(lat, latmax + mlat)) + 1, lat.size)
    j0 = max(int(np.searchsorted(lon, lonmin - mlon)), 0)
    j1 = min(int(np.searchsorted(lon, lonmax + mlon)) + 1, lon.size)
    sub = ee[i0:i1, j0:j1]
    ext = [lon[j0], lon[j1 - 1], lat[i0], lat[i1 - 1]]
    pos = sub[sub > 0]
    vmin = float(pos.min()) if pos.size else 1e-12

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(np.where(sub > 0, sub, np.nan), origin='lower', extent=ext,
                   aspect='auto', cmap='viridis', norm=LogNorm(vmin=vmin, vmax=float(sub.max())))
    fig.colorbar(im, ax=ax, shrink=0.85, label='explained enhancement (per cell)')

    cmap = plt.get_cmap('autumn')
    lo, hi = (min(boxes), max(boxes)) if boxes else (0, 1)
    for frac in sorted(boxes):
        b = boxes[frac]
        ax.add_patch(Rectangle((b[2], b[0]), b[3] - b[2], b[1] - b[0], fill=False,
                     ec=cmap((frac - lo) / max(1, hi - lo)), lw=1.6, label=f'{frac}% capture'))
    if current_bbox is not None:
        b = current_bbox
        ax.add_patch(Rectangle((b[2], b[0]), b[3] - b[2], b[1] - b[0], fill=False,
                     ec='red', lw=2, ls='--', label='current core'))
    ax.set_xlabel('lon'); ax.set_ylabel('lat')
    title = 'Core sizing: explained enhancement + suggested boxes'
    if np.isfinite(pr):
        title += f'  (participation ratio {pr:.0f})'
    ax.set_title(title); ax.legend(loc='upper right', fontsize=8)
    if out_path:
        plt.savefig(out_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()


def plot_buffer_regions(bundle_dir: str, out_path: str | None = None) -> None:
    """Map the core and buffer regions with their prior mean and diagonal σ.

    A prior-only diagnostic: builds the grid, core mask, buffer super-cells and the
    prior mean and diagonal σ for each super-cell. Saves a PNG to out_path.
    """
    inv = load_inversion(bundle_dir)
    buf = inv.buffer
    if buf is None or 'buffer' not in inv.state.names:
        print('no buffer region in this inversion (enable [buffer] in the config to use one)')
    else:
        memb = np.asarray(buf['membership'])                       # (n_lat, n_lon), -1 off-buffer
        prior_b = inv.state.unpack(inv.xa)['buffer']               # prior mean per super-cell
    post_b = inv.block('buffer')                               # posterior mean
    post_sd = inv.state.unpack(inv.posterior.stddev())['buffer']
    n_super = post_b.size
    print(f'buffer: {n_super} super-cells over {(memb >= 0).sum()} native cells '
          f'(mode in config; geometry from the bundle)')

    grid, core = inv.grid, inv.core

    def to_grid(vals):
        f = np.full(memb.shape, np.nan)
        ok = memb >= 0
        f[ok] = np.asarray(vals)[memb[ok]]
        return f

    # window covering core + buffer
    reg = (memb >= 0) | core.mask
    ii, jj = np.where(reg)
    pad = 1
    i0, i1 = max(ii.min() - pad, 0), min(ii.max() + pad + 1, grid.n_lat)
    j0, j1 = max(jj.min() - pad, 0), min(jj.max() + pad + 1, grid.n_lon)
    ext = [grid.lon[j0], grid.lon[j1 - 1], grid.lat[i0], grid.lat[i1 - 1]]
    cropb = lambda f: f[i0:i1, j0:j1]
    # core extent (for an outline box)
    clat, clon = core.active_lat, core.active_lon
    core_box = Rectangle((clon.min(), clat.min()), clon.max() - clon.min(),
                         clat.max() - clat.min(), fill=False, ec='red', lw=1.5)

    fig, ax = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    vmax = np.nanmax([np.nanmax(to_grid(prior_b)), np.nanmax(to_grid(post_b))])
    for a, (title, fld, cmap, vlim) in zip(ax, [
            ('buffer prior flux density', to_grid(prior_b), 'viridis', (0, vmax)),
            ('buffer posterior flux density', to_grid(post_b), 'viridis', (0, vmax)),
            ('buffer posterior 1σ', to_grid(post_sd), 'magma', (None, None))]):
        kw = {} if vlim[0] is None else dict(vmin=vlim[0], vmax=vlim[1])
        im = a.imshow(cropb(fld), origin='lower', extent=ext, aspect='auto', cmap=cmap, **kw)
        a.add_patch(Rectangle((clon.min(), clat.min()), clon.max() - clon.min(),
                              clat.max() - clat.min(), fill=False, ec='red', lw=1.5))
        a.set_title(title); a.set_xlabel('lon'); a.set_ylabel('lat'); fig.colorbar(im, ax=a, shrink=0.8)
    plt.suptitle('Buffer super-cells (red box = core domain)'); plt.show()

    # prior vs posterior per super-cell (update direction + uncertainty)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    ax[0].errorbar(prior_b, post_b, yerr=post_sd, fmt='o', ms=4, alpha=0.5, lw=0.8)
    lim = [0, float(np.nanmax([prior_b.max(), post_b.max()])) * 1.05 + 1e-12]
    ax[0].plot(lim, lim, 'k--', lw=1)
    ax[0].set_xlabel('prior flux density'); ax[0].set_ylabel('posterior flux density')
    ax[0].set_title('buffer super-cells: prior vs posterior')
    upd = np.where(post_sd > 0, (post_b - prior_b) / post_sd, 0.0)
    ax[1].hist(upd, bins=25)
    ax[1].axvline(0, color='k', lw=1)
    ax[1].set_xlabel('(posterior - prior) / posterior 1σ'); ax[1].set_ylabel('super-cells')
    ax[1].set_title('how far each super-cell moved from prior')
    if out_path:
        plt.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()

def plot_residuals(bundle_dir: str, out_path: str = None) -> None:
    """Plot residuals and diagnostics from a saved inversion bundle.
    """
    inv = load_inversion(bundle_dir)    
    R = inv.receptors
    rlat, rlon = R['receptor_lat'], R['receptor_lon']
    z, modeled = R['enhancement'], R['modeled']
    resid = z - modeled
    flag = R.get('outlier_flag', np.zeros_like(z)).astype(bool)
    flight = R.get('receptor_flight', np.zeros_like(z, dtype=int)).astype(int)
    fin = np.isfinite(resid) & ~flag

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    for a, (title, val, cmap) in zip(ax, [('enhancement z', z, 'viridis'),
            ('modeled  Hx\u0302', modeled, 'viridis'), ('residual  z - Hx\u0302', resid, 'RdBu_r')]):
        vlim = np.nanmax(np.abs(resid)) if cmap == 'RdBu_r' else None
        kw = dict(vmin=-vlim, vmax=vlim) if vlim else {}
        s = a.scatter(rlon, rlat, c=val, s=22, cmap=cmap, **kw)
        if flag.any():
            a.scatter(rlon[flag], rlat[flag], s=80, facecolors='none', edgecolors='k', label='outlier')
            a.legend(loc='upper left')
        a.set_title(title); a.set_xlabel('lon'); a.set_ylabel('lat'); fig.colorbar(s, ax=a, shrink=0.85)
   
    if out_path:
        plt.savefig(out_path+'/'+'residuals_map.png', bbox_inches='tight')
    else:
        plt.show()

    # robust mismatch scale (MAD) and normalized residuals
    med = np.median(resid[fin])
    sigma_mad = 1.4826 * np.median(np.abs(resid[fin] - med))
    nresid = (resid - med) / sigma_mad

    chi2r = inv.diagnostics.get('reduced_chi_square', float('nan'))
    print(f'reduced chi-square (saved): {chi2r:.3f}   (~1 = error model consistent)')
    print(f'residual: mean {np.mean(resid[fin]):+.4f}  rms {np.sqrt(np.mean(resid[fin]**2)):.4f} ppm  '
        f'robust sigma(MAD) {sigma_mad:.4f} ppm')
    for f in np.unique(flight):
        sel = fin & (flight == f)
        fid = inv.flight_ids[f] if f < len(inv.flight_ids) else str(f)
        print(f'  flight {fid:<14} n={sel.sum():4d}  bias {np.mean(resid[sel]):+.4f}  '
            f'rms {np.sqrt(np.mean(resid[sel]**2)):.4f} ppm')

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True)
    # (a) normalized-residual histogram vs N(0,1)
    ax[0].hist(nresid[fin], bins=30, density=True, alpha=0.7)
    xx = np.linspace(-4, 4, 100)
    ax[0].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'k--', label='N(0,1)')
    ax[0].set_xlabel('residual / sigma_MAD'); ax[0].set_ylabel('density')
    ax[0].set_title('normalized residual'); ax[0].legend()
    # (b) residual vs modeled (bias / heteroscedasticity)
    ax[1].scatter(modeled[fin], resid[fin], s=14, alpha=0.6); ax[1].axhline(0, color='k', lw=1)
    ax[1].set_xlabel('modeled enhancement (ppm)'); ax[1].set_ylabel('residual (ppm)')
    ax[1].set_title('residual vs modeled')
    # (c) along-track autocorrelation, computed within the largest flight (index order ~ track)
    big = np.bincount(flight[fin]).argmax()
    r = resid[fin & (flight == big)]
    r = (r - r.mean()) / (r.std() + 1e-12)
    maxlag = min(40, max(2, r.size - 1))
    ac = np.array([1.0 if k == 0 else float(np.mean(r[:-k] * r[k:])) for k in range(maxlag)])
    ax[2].stem(range(maxlag), ac)
    ax[2].axhline(0, color='k', lw=1)
    for h in (1, -1):
        ax[2].axhline(h * 1.96 / np.sqrt(r.size), color='r', ls=':', lw=1)
    ax[2].set_xlabel('receptor-index lag'); ax[2].set_ylabel('autocorr')
    ax[2].set_title(f'residual autocorrelation (flight {inv.flight_ids[big] if big < len(inv.flight_ids) else big})')
    if out_path:
        plt.savefig(out_path+'/'+'residuals_autocorr.png', bbox_inches='tight')
    else:
        plt.show()