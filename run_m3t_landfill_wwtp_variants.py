"""Run M3T for landfills + wastewater only, with every method variant enabled,
over the exact domain/grid build_m3t_prior.py uses for the HALO prior --
so we get all 3 landfill (GHGRP reported/generation_first/collection_first)
and all 8 wastewater (CWNS|DMR x GHGI|Moore x septic national|state) sector-
total rasters in one run, directly comparable to what's already in
m3t_option_1.nc4.

Mirrors M3T/python/notebooks/nyc_demo.py::prepare_run_dir for staging the
companion data, and build_m3t_prior.py::run_m3t for the domain/grid.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd

import m3t
from m3t.config import Config

REPO = Path("/scratch/scrowel3_lab/halo-nyc")
DATA = Path("/scratch/scrowel3_lab/M3T_Processed/M3T_Processed")
RUN_DIR = REPO / "m3t_variant_run"
YEAR = 2019

_COMPANION = [
    "DMR_data.csv", "combined_wastewater_NLCD.tif", "Total_national_septic_area.csv",
    "wastewater_state_septic_area.csv", "processed_NWI_data", "Watersheds.gpkg",
    "combined_NLCD_downscaled_wetcharts.tif", "combined_county_tigerlines.gpkg",
    "Vulcan_v4.0", "ACES V2.0", "EIA",
]


def prepare_run_dir() -> Path:
    (RUN_DIR / "in" / "GHGRP").mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "in" / "NEI").mkdir(parents=True, exist_ok=True)
    for name in _COMPANION:
        link, target = RUN_DIR / "in" / name, DATA / name
        if target.exists() and not link.exists():
            link.symlink_to(target)
    for name in ("facility_data.csv", "Oil_and_gas_W.csv"):
        link, target = RUN_DIR / "in" / "GHGRP" / name, DATA / "GHGRP" / name
        if target.exists() and not link.exists():
            link.symlink_to(target)
    return RUN_DIR


def main():
    import build_m3t_prior as bp

    lat, lon = bp.target_grid(REPO / "nyc_ch4_emissions.h5")
    bbox, res = bp.m3t_domain(lat, lon)
    print(f"domain={bbox} res={res} grid={lat.size}x{lon.size}")

    run_dir = prepare_run_dir()

    cfg = Config()
    # only landfills + wastewater
    for k in ("Process_natural_gas_distribution", "Process_natural_gas_transmission",
              "Process_stationary_combustion", "Process_wetlands_and_inland_waters",
              "Process_remaining_sectors_from_gridded_EPA"):
        setattr(cfg, k, False)
    cfg.Process_landfills = True
    cfg.Process_wastewater = True
    # all landfill GHGRP method variants
    cfg.landfill_ghgrp_reported = True
    cfg.landfill_ghgrp_generation_first = True
    cfg.landfill_ghgrp_collection_first = True
    # all wastewater source x method x septic-kind variants
    cfg.Wastewater_use_CWNS = True
    cfg.Wastewater_use_DMR = True
    cfg.Wastewater_Municipal_Method_GHGI = True
    cfg.Wastewater_Municipal_Method_Moore = True
    cfg.Wastewater_national_septic = True
    cfg.Wastewater_state_septic = True

    states = gpd.read_file(DATA / "combined_state_tigerlines.gpkg", layer=str(YEAR))

    ctx = m3t.ch4_inventory_build(
        run_directory=run_dir, inventory_year=YEAR, domain=bbox, domain_res=res,
        domain_crs="epsg:4326", tigerlines=states, config=cfg,
    )
    print("sectors run:", ctx.shared["sectors_run"])
    print("output dir:", ctx.output_directory)
    out_files = sorted(p.name for p in Path(ctx.output_directory).glob("*.nc"))
    print("top-level output files:")
    for f in out_files:
        print(" ", f)


if __name__ == "__main__":
    main()
