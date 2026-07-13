import os
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

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
        plt.savefig(os.path.join(out_path, 'posterior.png'), dpi=300, bbox_inches='tight')
        plt.close(fig)
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

def _flights_present(flight_ids, flight_index):
    """(fid, boolean selection mask) for each flight id that has receptors."""
    return [(fid, flight_index == f) for f, fid in enumerate(flight_ids) if (flight_index == f).any()]


def _time_binned_autocorr(r, t, max_lag_s=180.0, bin_width_s=5.0):
    """Residual autocorrelation binned by real elapsed-time lag, not sample index.

    Sample spacing is not uniform: within a leg it is a near-constant ~4-5s,
    but consecutive *samples* straddling a turn are minutes apart (see
    :func:`halo_oe.background.detect_legs`). An index-lag ACF silently treats
    both cases as "lag 1", which understates how quickly correlation actually
    decays along a leg and contaminates small lags with turn-adjacent pairs
    that have nothing to do with along-track correlation length. Binning by
    ``t[j] - t[i]`` instead keeps those turn-spanning pairs in their own
    (large, sparse) bins where they belong.

    ``r`` must already be standardized (zero mean, unit variance) and ``t``
    sorted ascending, same length, same order. Returns ``(lag_centers_s, ac)``
    with ``lag_centers_s`` starting at 0 (a synthetic, definitional 1.0, as is
    conventional for an ACF plot) followed by the real bins out to
    ``max_lag_s``; bins with no pairs are NaN.
    """
    n = len(r)
    nbins = max(1, int(np.ceil(max_lag_s / bin_width_s)))
    edges = np.linspace(0.0, max_lag_s, nbins + 1)
    sums = np.zeros(nbins)
    counts = np.zeros(nbins, dtype=int)
    for i in range(n):
        j = i + 1
        while j < n and t[j] - t[i] <= max_lag_s:
            b = min(int((t[j] - t[i]) / bin_width_s), nbins - 1)
            sums[b] += r[i] * r[j]
            counts[b] += 1
            j += 1
    ac = np.divide(sums, counts, out=np.full(nbins, np.nan), where=counts > 0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return np.concatenate([[0.0], centers]), np.concatenate([[1.0], ac])


def _residual_map_row(ax_row, fid, rlat, rlon, z, modeled, resid, flag, show_titles) -> None:
    """One flight's spatial maps (enhancement / modeled / residual) into a row of axes."""
    for a, (title, val, cmap) in zip(ax_row, [('enhancement z', z, 'viridis'),
            ('modeled  Hx\u0302', modeled, 'viridis'), ('residual  z - Hx\u0302', resid, 'RdBu_r')]):
        vlim = np.nanmax(np.abs(resid)) if cmap == 'RdBu_r' else None
        kw = dict(vmin=-vlim, vmax=vlim) if vlim else {}
        s = a.scatter(rlon, rlat, c=val, s=18, cmap=cmap, **kw)
        if flag.any():
            a.scatter(rlon[flag], rlat[flag], s=70, facecolors='none', edgecolors='k', label='outlier')
            a.legend(loc='upper left', fontsize=7)
        if show_titles:
            a.set_title(title)
        a.set_xlabel('lon'); a.figure.colorbar(s, ax=a, shrink=0.85)
    ax_row[0].set_ylabel(f'flight {fid}\nlat')


def _mismatch_diagnostics_row(ax_row, fid, modeled, resid, fin, show_titles, t=None) -> None:
    """One flight's normalized-residual histogram, residual-vs-modeled, and
    along-track autocorrelation into a row of axes (mirrors the notebook's
    mismatch section).

    ``t``, if given, is this flight's per-receptor elapsed time (seconds,
    same length/order as ``modeled``/``resid``); the autocorrelation panel
    then bins by real time lag (see :func:`_time_binned_autocorr`) instead of
    assuming uniform sample spacing. Falls back to sample-index lag when
    unavailable (e.g. no ``flight_data_dir`` configured for this bundle).
    """
    med = np.median(resid[fin])
    sigma_mad = 1.4826 * np.median(np.abs(resid[fin] - med))
    nresid = (resid - med) / sigma_mad
    print(f'  flight {fid:<14} n={fin.sum():4d}  bias {np.mean(resid[fin]):+.4f}  '
          f'rms {np.sqrt(np.mean(resid[fin]**2)):.4f} ppm  robust sigma(MAD) {sigma_mad:.4f} ppm')

    # (a) normalized-residual histogram vs N(0,1)
    ax_row[0].hist(nresid[fin], bins=30, density=True, alpha=0.7)
    xx = np.linspace(-4, 4, 100)
    ax_row[0].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), 'k--', label='N(0,1)')
    ax_row[0].set_xlabel('residual / sigma_MAD'); ax_row[0].set_ylabel(f'flight {fid}\ndensity')
    if show_titles:
        ax_row[0].set_title('normalized residual')
    ax_row[0].legend(fontsize=7)
    # (b) residual vs modeled (bias / heteroscedasticity)
    ax_row[1].scatter(modeled[fin], resid[fin], s=12, alpha=0.6); ax_row[1].axhline(0, color='k', lw=1)
    ax_row[1].set_xlabel('modeled enhancement (ppm)'); ax_row[1].set_ylabel('residual (ppm)')
    if show_titles:
        ax_row[1].set_title('residual vs modeled')
    # (c) along-track autocorrelation within this flight, by real time lag when
    # available (see _time_binned_autocorr), else by sample-index lag
    r = resid[fin]
    r = (r - r.mean()) / (r.std() + 1e-12)
    if t is not None:
        lags, ac = _time_binned_autocorr(r, t[fin])
        keep = np.isfinite(ac)
        ax_row[2].stem(lags[keep], ac[keep])
        xlabel = 'time lag (s)'
    else:
        maxlag = min(40, max(2, r.size - 1))
        ac = np.array([1.0 if k == 0 else float(np.mean(r[:-k] * r[k:])) for k in range(maxlag)])
        ax_row[2].stem(range(maxlag), ac)
        xlabel = 'receptor-index lag'
    ax_row[2].axhline(0, color='k', lw=1)
    for h in (1, -1):
        ax_row[2].axhline(h * 1.96 / np.sqrt(r.size), color='r', ls=':', lw=1)
    ax_row[2].set_xlabel(xlabel); ax_row[2].set_ylabel('autocorr')
    if show_titles:
        ax_row[2].set_title('residual autocorrelation')


def plot_residuals(bundle_dir: str, out_path: str = None) -> None:
    """Plot residuals and model-data-mismatch diagnostics, per flight.

    Reproduces the notebook's observation-diagnostics and mismatch sections
    (spatial maps of enhancement/modeled/residual, normalized-residual
    histogram, residual-vs-modeled, and along-track autocorrelation), but
    computed separately for each flight rather than aggregated over all
    observations \u2014 each flight has its own background fit and track geometry,
    so pooling residuals across flights can mask flight-specific bias or
    correlation structure. Each diagnostic type is written to a single file
    (``residuals_map.png``, ``residuals_autocorr.png``) with one row per flight.
    """
    inv = load_inversion(bundle_dir)
    R = inv.receptors
    rlat, rlon = R['receptor_lat'], R['receptor_lon']
    z, modeled = R['enhancement'], R['modeled']
    resid = z - modeled
    flag = R.get('outlier_flag', np.zeros_like(z)).astype(bool)
    flight = R.get('receptor_flight', np.zeros_like(z, dtype=int)).astype(int)
    sels = _flights_present(inv.flight_ids or ['0'], flight)
    n = len(sels)

    chi2r = inv.diagnostics.get('reduced_chi_square', float('nan'))
    print(f'reduced chi-square (saved): {chi2r:.3f}   (~1 = error model consistent)')

    # elapsed time per flight, for a real time-lag autocorrelation axis (falls
    # back to sample-index lag per flight if flight_data_dir isn't configured
    # or a flight's raw file/coordinates aren't available)
    flight_data_dir = None
    cfg_path = os.path.join(bundle_dir, 'config.ini')
    if os.path.exists(cfg_path):
        from goe.config import Config
        flight_data_dir = Config(cfg_path).get('background', 'flight_data_dir', default=None)
    times = {}
    if flight_data_dir:
        from .background import _load_receptor_time
        for fid, sel in sels:
            try:
                times[fid] = _load_receptor_time(fid, flight_data_dir, rlat[sel], rlon[sel])
            except (FileNotFoundError, ValueError) as e:
                print(f'  flight {fid}: no elapsed time for autocorrelation ({e}); using index lag')

    fig, ax = plt.subplots(n, 3, figsize=(16, 4.2 * n), constrained_layout=True, squeeze=False)
    for i, (fid, sel) in enumerate(sels):
        _residual_map_row(ax[i], fid, rlat[sel], rlon[sel], z[sel], modeled[sel], resid[sel],
                           flag[sel], show_titles=(i == 0))
    fig.suptitle('Observation diagnostics by flight')
    if out_path:
        plt.savefig(os.path.join(out_path, 'residuals_map.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    fig, ax = plt.subplots(n, 3, figsize=(16, 4.2 * n), constrained_layout=True, squeeze=False)
    for i, (fid, sel) in enumerate(sels):
        fin = np.isfinite(resid[sel]) & ~flag[sel]
        _mismatch_diagnostics_row(ax[i], fid, modeled[sel], resid[sel], fin, show_titles=(i == 0),
                                   t=times.get(fid))
    fig.suptitle('Model-data mismatch by flight')
    if out_path:
        plt.savefig(os.path.join(out_path, 'residuals_autocorr.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _background_row(ax_row, fid, rlat, rlon, obs, bg, flag, show_titles) -> None:
    """One flight's observed/background maps and their scatter relationship."""
    vmin, vmax = np.nanmin(obs), np.nanmax(obs)
    for a, (title, val) in zip(ax_row[:2], [('observed XCH4', obs), ('fitted background', bg)]):
        s = a.scatter(rlon, rlat, c=val, s=18, cmap='viridis', vmin=vmin, vmax=vmax)
        if flag.any():
            a.scatter(rlon[flag], rlat[flag], s=70, facecolors='none', edgecolors='k', label='outlier')
            a.legend(loc='upper left', fontsize=7)
        if show_titles:
            a.set_title(title)
        a.set_xlabel('lon'); a.figure.colorbar(s, ax=a, shrink=0.85)
    ax_row[0].set_ylabel(f'flight {fid}\nlat')

    ax_row[2].scatter(obs, bg, s=12, alpha=0.6)
    lim = [float(np.nanmin(obs)), float(np.nanmax(obs))]
    ax_row[2].plot(lim, lim, 'k--', lw=1, label='1:1')
    ax_row[2].set_xlabel('observed XCH4'); ax_row[2].set_ylabel('fitted background')
    if show_titles:
        ax_row[2].set_title('background rides the lower envelope')
    ax_row[2].legend(fontsize=7)


def plot_background(bundle_dir: str, out_path: str = None) -> None:
    """Plot the background surface subtracted from observations, per flight.

    Each flight gets its own lower-envelope planar (or higher-degree) fit (see
    :mod:`halo_oe.background`); this maps the observed column and the fitted
    background at each receptor, plus their scatter relationship, so the fit can
    be checked against the raw data flight by flight. One file
    (``background.png``) with a row per flight.
    """
    inv = load_inversion(bundle_dir)
    R = inv.receptors
    rlat, rlon = R['receptor_lat'], R['receptor_lon']
    obs, bg = R['receptor_obs'], R['receptor_background']
    flag = R.get('outlier_flag', np.zeros_like(obs)).astype(bool)
    flight = R.get('receptor_flight', np.zeros_like(obs, dtype=int)).astype(int)
    sels = _flights_present(inv.flight_ids or ['0'], flight)
    n = len(sels)

    fig, ax = plt.subplots(n, 3, figsize=(16, 4.2 * n), constrained_layout=True, squeeze=False)
    for i, (fid, sel) in enumerate(sels):
        _background_row(ax[i], fid, rlat[sel], rlon[sel], obs[sel], bg[sel], flag[sel],
                         show_titles=(i == 0))
    fig.suptitle('Background plane by flight')
    if out_path:
        plt.savefig(os.path.join(out_path, 'background.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _flux_block_prior_density(inv, name):
    """Prior emission density for a gridded state block, on active cells.

    Category blocks map directly onto a super-category prior field; a
    single-block total-inventory run has no matching category key, so its
    prior density is the sum of all super-category fields (== the inventory
    total), mirroring the post-hoc aggregation helper in the analysis notebook.
    """
    if name in inv.group_fields:
        return inv.group_fields[name]
    return sum(inv.group_fields.values())


def plot_flux_summary(bundle_dir: str, out_path: str = None) -> None:
    """Summarize prior vs posterior integrated fluxes, with uncertainty.

    Two files:

    * ``flux_maps.png`` \u2014 prior flux density, posterior flux density, and
      posterior 1\u03c3 flux density, one column per gridded state block (category
      fields if the inversion was decomposed, else the single inventory total).
    * ``flux_totals.png`` \u2014 a grouped bar chart of the saved flux report (prior
      vs. posterior \u00b1 1\u03c3), at whatever granularity was solved: just the total,
      or per-category totals plus the domain total when decomposed.

    This is exactly the information in ``inv.report`` / ``inv.field`` \u2014 no
    re-solve, no functional beyond what the bundle already stores.
    """
    inv = load_inversion(bundle_dir)
    _plot_flux_maps(inv, out_path)
    _plot_flux_bars(inv, out_path)


def _plot_flux_maps(inv, out_path) -> None:
    grid, core = inv.grid, inv.core
    m = core.mask
    rows = np.where(m.any(1))[0]; cols = np.where(m.any(0))[0]
    r0, r1, c0, c1 = rows[0], rows[-1] + 1, cols[0], cols[-1] + 1
    EXTENT = [grid.lon[c0], grid.lon[c1 - 1], grid.lat[r0], grid.lat[r1 - 1]]
    crop = lambda f: f[r0:r1, c0:c1]

    cats = [b.name for b in inv.state.blocks if b.name not in ('bc', 'buffer')]
    std_parts = inv.state.unpack(inv.posterior.stddev())

    fig, ax = plt.subplots(3, len(cats), figsize=(4.2 * len(cats), 12),
                            constrained_layout=True, squeeze=False)
    for j, name in enumerate(cats):
        prior_field = core.to_field(_flux_block_prior_density(inv, name))
        post_field = inv.field(name) * prior_field          # posterior scalar x prior density
        std_field = core.to_field(std_parts[name]) * prior_field   # per-cell 1sigma, ignoring cross-cell covariance
        vmax = float(np.nanmax([np.nanmax(prior_field), np.nanmax(post_field)]))

        im0 = ax[0, j].imshow(crop(prior_field), origin='lower', extent=EXTENT, aspect='auto',
                              cmap='viridis', vmin=0, vmax=vmax)
        ax[0, j].set_title(f'prior flux: {name}'); fig.colorbar(im0, ax=ax[0, j], shrink=0.8)
        im1 = ax[1, j].imshow(crop(post_field), origin='lower', extent=EXTENT, aspect='auto',
                              cmap='viridis', vmin=0, vmax=vmax)
        ax[1, j].set_title(f'posterior flux: {name}'); fig.colorbar(im1, ax=ax[1, j], shrink=0.8)
        im2 = ax[2, j].imshow(crop(std_field), origin='lower', extent=EXTENT, aspect='auto', cmap='magma')
        ax[2, j].set_title(f'posterior 1\u03c3 flux: {name}'); fig.colorbar(im2, ax=ax[2, j], shrink=0.8)
    for a in ax.ravel(): a.set_xlabel('lon'); a.set_ylabel('lat')
    fig.suptitle(f"Flux density ({inv.report.get('unit_label', '')} per m\u00b2)")
    if out_path:
        plt.savefig(os.path.join(out_path, 'flux_maps.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def _plot_flux_bars(inv, out_path) -> None:
    report = inv.report
    names = report['names']
    prior = np.asarray(report['prior'])
    posterior = np.asarray(report['posterior'])
    post_sd = np.asarray(report['posterior_stddev'])
    unit = report.get('unit_label', '')

    x = np.arange(len(names)); w = 0.35
    fig, ax = plt.subplots(figsize=(max(1.8 * len(names), 4) + 2, 5), constrained_layout=True)
    ax.bar(x - w / 2, prior, width=w, label='prior')
    ax.bar(x + w / 2, posterior, width=w, yerr=post_sd, capsize=4, label='posterior \u00b1 1\u03c3')
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha='right')
    ax.set_ylabel(f'integrated flux ({unit})')
    ax.set_title('prior vs posterior flux, by category'); ax.legend()
    if out_path:
        plt.savefig(os.path.join(out_path, 'flux_totals.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


def plot_leg_offsets(bundle_dir: str, out_path: str = None) -> None:
    """Map the kriged per-leg background offset, per flight.

    Only meaningful for a bundle solved with ``[background] use_leg_offsets =
    true``; the offset is exactly what :func:`halo_oe.background.fit_leg_offsets`
    added on top of the flight-wide plane (see ``receptor_background_offset``
    in the bundle's ``fields.nc``) — saved directly rather than re-derived, so
    this needs no re-fit and no second (plane-only) run to diff against. Prints
    a message and does nothing if the bundle predates this field, or if every
    flight's offset is exactly zero (leg offsets were off). Checks the bundle's
    own saved config.ini first and skips the (expensive) full bundle load
    entirely when ``use_leg_offsets`` was off, rather than materializing
    factors.npz just to find there is nothing to plot.
    """
    cfg_path = os.path.join(bundle_dir, 'config.ini')
    if os.path.exists(cfg_path):
        from goe.config import Config
        if not Config(cfg_path).get_bool('background', 'use_leg_offsets', default=False):
            print('use_leg_offsets was off for this run; skipping (no bundle load needed)')
            return

    inv = load_inversion(bundle_dir)
    R = inv.receptors
    if 'receptor_background_offset' not in R:
        print('no receptor_background_offset in this bundle (predates leg-offset '
              'background support, or was never re-saved since)')
        return
    rlat, rlon = R['receptor_lat'], R['receptor_lon']
    offset = R['receptor_background_offset']
    flight = R.get('receptor_flight', np.zeros_like(offset, dtype=int)).astype(int)
    flight_ids = inv.flight_ids or ['0']

    if np.allclose(offset, 0.0):
        print('receptor_background_offset is all zero (use_leg_offsets was off for this run)')
        return

    sels = _flights_present(flight_ids, flight)
    n = len(sels)
    vmax = float(np.nanmax(np.abs(offset)))

    fig, ax = plt.subplots(1, n, figsize=(5.5 * n, 5), constrained_layout=True, squeeze=False)
    ax = ax[0]
    for i, (fid, sel) in enumerate(sels):
        s = ax[i].scatter(rlon[sel], rlat[sel], c=offset[sel], s=16, cmap='RdBu_r',
                          vmin=-vmax, vmax=vmax)
        ax[i].set_title(f'flight {fid}'); ax[i].set_xlabel('lon')
        fig.colorbar(s, ax=ax[i], shrink=0.85, label='leg offset (ppm)')
    ax[0].set_ylabel('lat')
    fig.suptitle('Kriged per-leg background offset (added on top of the flight-wide plane)')
    if out_path:
        plt.savefig(os.path.join(out_path, 'leg_offsets.png'), bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()