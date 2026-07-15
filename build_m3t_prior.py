"""Build an M3T methane prior for the HALO NYC inversion.

`halo_oe` already expects an M3T prior — `config.ini` has ``[emissions] path =
../m3t_option_1.nc4`` and ``inventory = m3t``, and `run_halo.py` defaults to it.
This script produces that file. Nothing in `halo_oe` changes: the prior mechanism
is data-driven (``halo_oe.emissions.category_priors_on_grid`` reads a
``<inventory>`` dataset out of the file at ``[emissions] path``).

**M3T is an independent tool; this is HALO-specific glue.** halo-nyc depends on
M3T's *output format*, not its code. The default path here — ``pack`` — reads an
M3T output directory and never imports ``m3t``, so it runs anywhere (the halo
``analysis`` env). What it encodes is HALO-specific: the Pittsburgh category
taxonomy the inversion understands, the HALO grid, µmol units, and the file
schema.

Two steps:

1. **Run M3T** (in M3T's own environment, however you like) over the NYC domain.
   ``--print-domain`` prints the exact ``domain``/``domain_res`` to pass to
   ``m3t.ch4_inventory_build`` so its grid coincides with the HALO grid and no
   resampling is introduced. (For convenience, ``run`` will do this for you if
   ``m3t`` is importable — see below — but that is optional.)
2. **Pack** the M3T outputs into the prior file::

       python build_m3t_prior.py pack --m3t-out /path/to/m3t_run/out --proxy vulcan

The prior carries M3T's categories at their **native granularity** with M3T's own
names (``M3T_CATEGORIES`` below) — **no grouping is done here**. Collapsing
categories into process super-groups is ``halo_oe``'s job, driven by
``config.ini``'s ``[category_groups]`` keyword map, which changes between
experiments. So the file must stay fine-grained and let the inversion regroup it,
exactly as ``edgar`` (20 categories) and ``epa`` (26) do. ``M3T_PRIOR.md`` gives a
``[category_groups]`` block that groups M3T's labels.

Units: M3T writes **nmol m⁻² s⁻¹**; the Jacobian is ``ppm per µmol m⁻² s⁻¹``
(halo_oe README), so we divide by 1000.

``--proxy`` (vulcan/aces) is why the target is ``m3t_option_1.nc4`` not
``m3t.nc4``: a different M3T config is a different, equally-valid prior. Build each
to its own ``m3t_option_N.nc4`` and compare posteriors, as the three existing
inventories are compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

# --------------------------------------------------------------------------- #
# M3T's own categories -> the output file each is read from.
#
# **No grouping happens here.** Grouping into process super-categories is
# ``halo_oe``'s job, driven by ``config.ini``'s ``[category_groups]`` keyword map,
# which changes between experiments — so this file must carry M3T's categories at
# their native granularity and let the inversion regroup them. Each entry is one
# M3T output file (``<run>/out/<stem>.nc``); the label is M3T's own name for it,
# not a group. ``{proxy}`` is the CO2 proxy label (Vulcan / ACES).
#
# These are the finest *additive* fields M3T writes to top level (they sum to the
# whole inventory), so nothing is coarsened or double-counted. Stationary
# combustion is kept split into its fossil-fuel and wood totals (two distinct M3T
# files) rather than summed, for the same reason. See M3T_PRIOR.md on going finer
# still (M3T aggregates its gridded-EPA remainder into GEPA_thermo/non_thermo).
# --------------------------------------------------------------------------- #
M3T_CATEGORIES: dict[str, str] = {
    # M3T's bottom-up sectors (one output file each)
    "landfills": "landfills",
    "natural_gas_transmission": "natural_gas_transmission",
    "natural_gas_distribution": "natural_gas_distribution",
    "wastewater": "wastewater",
    "stationary_combustion_fossil_fuel": "Stationary_combustion_sector_fossil_fuel_total_{proxy}_bystate",
    "stationary_combustion_wood": "Stationary_combustion_sector_wood_total_{proxy}_bystate",
    "wetlands": "wetlands",
    # the gridded-EPA remainder, at M3T's finer breakdown (out/remaining_gepa/),
    # so a dedicated inventory can be swapped in per source group when building
    # the prior. These six partition the GEPA remainder exactly.
    "GEPA_oil_gas_upstream": "remaining_gepa/GEPA_oil_gas_upstream",
    "GEPA_coal": "remaining_gepa/GEPA_coal",
    "GEPA_livestock": "remaining_gepa/GEPA_livestock",
    "GEPA_crop_ag": "remaining_gepa/GEPA_crop_ag",
    "GEPA_industrial_landfill": "remaining_gepa/GEPA_industrial_landfill",
    "GEPA_other": "remaining_gepa/GEPA_other",
}

NMOL_TO_UMOL = 1e-3  # nmol m^-2 s^-1 (M3T)  ->  µmol m^-2 s^-1 (Jacobian units)
_PROXY_LABEL = {"vulcan": "Vulcan", "aces": "ACES"}


# --------------------------------------------------------------------------- #
# Target grid
# --------------------------------------------------------------------------- #
def target_grid(reference: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read the HALO grid (ascending ``lat``, ``lon`` cell centres) from a reference file."""
    with h5py.File(reference, "r") as f:
        lat = np.asarray(f["lat"][:], dtype="float64")
        lon = np.asarray(f["lon"][:], dtype="float64")
    if lat[0] > lat[-1] or lon[0] > lon[-1]:
        raise ValueError("reference lat/lon must be ascending")
    return lat, lon


def m3t_domain(lat: np.ndarray, lon: np.ndarray) -> tuple[tuple, tuple]:
    """M3T ``(domain_bbox, domain_res)`` whose cell centres coincide with ``lat``/``lon``.

    M3T's ``make_grid`` anchors ``(xmin, ymin)`` and centres cells at
    ``min + (i+0.5)*res``, so ``xmin = lon[0] - xres/2`` reproduces the reference
    centres exactly. Pass these straight to ``m3t.ch4_inventory_build``.
    """
    xres = float(np.round(np.diff(lon).mean(), 10))
    yres = float(np.round(np.diff(lat).mean(), 10))
    xmin, ymin = float(lon[0]) - xres / 2, float(lat[0]) - yres / 2
    bbox = (xmin, ymin, xmin + len(lon) * xres, ymin + len(lat) * yres)
    return bbox, (xres, yres)


# --------------------------------------------------------------------------- #
# Pack an M3T output directory  (no m3t dependency)
# --------------------------------------------------------------------------- #
def _read_category(out_dir: Path, stem: str, proxy_label: str,
                   lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Read one M3T output raster onto the target grid, in µmol m⁻² s⁻¹.

    Reindexed onto the reference ``lat``/``lon`` (ascending) with nearest matching,
    so the field lines up with the file's coordinate vectors whatever M3T's
    internal (north-up) orientation. When M3T was run on the coinciding grid
    (:func:`m3t_domain`) this is an exact copy; otherwise it is a nearest regrid.
    """
    path = out_dir / f"{stem.format(proxy=proxy_label)}.nc"
    if not path.exists():
        raise FileNotFoundError(
            f"expected M3T output {path.name} in {out_dir} — was that sector "
            "enabled, and is --proxy correct for the run?"
        )
    ds = xr.open_dataset(path, decode_coords="all")
    da = ds[next(iter(ds.data_vars))].astype("float64")
    ydim, xdim = da.dims[-2], da.dims[-1]
    da = da.rename({ydim: "y", xdim: "x"}).sortby("y").sortby("x")
    da = da.interp(y=lat, x=lon, method="nearest")
    return np.nan_to_num(da.values, nan=0.0) * NMOL_TO_UMOL


def pack(m3t_out: Path, reference: Path, out_path: Path, *, proxy: str,
         inventory_name: str = "m3t", year: int | None = None) -> Path:
    """Pack an M3T output directory into the halo_oe prior file. No ``m3t`` import."""
    lat, lon = target_grid(reference)
    proxy_label = _PROXY_LABEL[proxy]

    categories = list(M3T_CATEGORIES)
    stack = np.stack(
        [_read_category(m3t_out, M3T_CATEGORIES[c], proxy_label, lat, lon)
         for c in categories]
    )  # (n_category, n_lat, n_lon), µmol m^-2 s^-1

    write_prior(out_path, stack, categories, lat, lon, inventory_name,
                proxy=proxy, year=year)
    _report(stack, categories, lat, lon, out_path)
    return out_path


def write_prior(out_path, stack, categories, lat, lon, inventory_name, *, proxy, year):
    """Write the flat-HDF5 schema ``halo_oe.emissions`` reads."""
    with h5py.File(Path(out_path), "w") as f:
        f.create_dataset("lat", data=np.asarray(lat, dtype="float64"))
        f.create_dataset("lon", data=np.asarray(lon, dtype="float64"))
        f.create_dataset(inventory_name, data=np.asarray(stack, dtype="float64"))
        f.attrs[f"{inventory_name}_categories"] = "; ".join(categories)
        f.attrs["units"] = "umol m-2 s-1"
        f.attrs["source"] = "M3T (Modular Methane Mapping Tool)"
        f.attrs["m3t_proxy"] = proxy
        if year is not None:
            f.attrs["m3t_inventory_year"] = year


def _cell_area_m2(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Spherical-cap band area (m²) per cell — for a sanity budget only (no m3t)."""
    R = 6.371e6
    dlon = np.deg2rad(float(np.diff(lon).mean()))
    dlat = float(np.diff(lat).mean())
    edges = np.deg2rad(np.concatenate([[lat[0] - dlat / 2], lat + dlat / 2]))
    band = R * R * dlon * (np.sin(edges[1:]) - np.sin(edges[:-1]))  # per lat row
    return np.repeat(band[:, None], lon.size, axis=1)


def _report(stack, categories, lat, lon, out_path) -> None:
    """Print a per-category budget in Gg CH4/yr, a magnitude sanity check."""
    area = _cell_area_m2(lat, lon)
    # µmol m⁻² s⁻¹ x m² = µmol/s; then µmol/s -> Gg/yr
    umol_s_to_gg_yr = 1e-6 * 16.043 * 1e-9 * (3600 * 24 * 365)
    gg = (stack * area).sum(axis=(1, 2)) * umol_s_to_gg_yr
    print(f"\nWrote {out_path}")
    print(f"  grid {lat.size}x{lon.size}, {len(categories)} categories, µmol m⁻² s⁻¹")
    for c, g in sorted(zip(categories, gg), key=lambda t: -t[1]):
        print(f"    {c:28s} {g:8.3f} Gg CH4/yr")
    print(f"    {'TOTAL':28s} {gg.sum():8.3f} Gg CH4/yr")


# --------------------------------------------------------------------------- #
# Optional: run M3T for you  (imports m3t; needs M3T's env + companion data)
# --------------------------------------------------------------------------- #
def run_m3t(run_dir: Path, tigerlines: Path, reference: Path, *, proxy: str,
            year: int) -> Path:
    """Run M3T over the NYC domain on the HALO grid; return its output directory.

    Convenience only: importing ``m3t`` couples this to M3T's environment. The
    ``pack`` path does not need it. Requires the M3T companion data staged into
    ``run_dir/in`` (see M3T/python/notebooks/nyc_demo.py::prepare_run_dir).
    """
    try:
        import geopandas as gpd

        import m3t
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "run: importing M3T failed. M3T is a separate package — install it "
            "(`pip install -e /path/to/M3T/python`) into an env that also has "
            "h5py, or run M3T yourself and use `pack --m3t-out <dir>` instead.\n"
            f"  ({exc})"
        )

    lat, lon = target_grid(reference)
    bbox, res = m3t_domain(lat, lon)
    cfg = m3t.Config()
    cfg.Use_Vulcan, cfg.Use_ACES = (proxy == "vulcan"), (proxy == "aces")
    ctx = m3t.ch4_inventory_build(
        run_directory=run_dir, inventory_year=year, domain=bbox, domain_res=res,
        domain_crs="epsg:4326", tigerlines=gpd.read_file(tigerlines, layer=str(year)),
        config=cfg,
    )
    return Path(ctx.output_directory)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv=None):
    # common options live on a parent parser so they work on either side of the
    # subcommand (`build_m3t_prior.py pack --proxy aces` reads naturally)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reference", type=Path, default=Path("nyc_ch4_emissions.h5"),
                        help="file whose lat/lon define the HALO grid")
    common.add_argument("--out", type=Path, default=Path("m3t_option_1.nc4"))
    common.add_argument("--proxy", choices=["vulcan", "aces"], default="vulcan")
    common.add_argument("--inventory-name", default="m3t")
    common.add_argument("--year", type=int, default=2019)

    p = argparse.ArgumentParser(description=__doc__, parents=[common],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("pack", parents=[common],
                        help="pack an existing M3T output directory (no m3t import)")
    sp.add_argument("--m3t-out", required=True, type=Path, help="M3T run's out/ directory")

    sr = sub.add_parser("run", parents=[common],
                        help="run M3T on the HALO grid, then pack (needs m3t + companion data)")
    sr.add_argument("--run-dir", required=True, type=Path)
    sr.add_argument("--tigerlines", required=True, type=Path)

    sub.add_parser("print-domain", parents=[common],
                   help="print the M3T domain/res that matches the HALO grid")

    args = p.parse_args(argv)

    if args.cmd == "print-domain":
        lat, lon = target_grid(args.reference)
        bbox, res = m3t_domain(lat, lon)
        print("Run M3T with:")
        print(f"  domain     = {bbox}")
        print(f"  domain_res = {res}")
        print(f"  domain_crs = 'epsg:4326'   grid {lat.size}x{lon.size}")
        return

    if args.cmd == "run":
        out_dir = run_m3t(args.run_dir, args.tigerlines, args.reference,
                          proxy=args.proxy, year=args.year)
    else:
        out_dir = args.m3t_out

    pack(out_dir, args.reference, args.out, proxy=args.proxy,
         inventory_name=args.inventory_name, year=args.year)


if __name__ == "__main__":
    main()
