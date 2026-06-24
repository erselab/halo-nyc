# halo-nyc

Scripts and notebooks for analyzing **HALO–STAQS** airborne methane (XCH₄) column
observations collected over the New York City area in July/August 2023, and for
comparing them against atmospheric transport simulations.

The workflow combines:

- **WRF-STILT** modeling (driving trajectories from a WRF-GHG meteorology field or HRRR),
- **column-averaged footprints** built from per-altitude STILT footprints,
- **boundary conditions** sampled from WRF output along STILT trajectories, and
- **Bayesian / scalar inversions** that use the footprints, emissions inventories, and
  boundary conditions to constrain NYC CH₄ fluxes.

> Author of the analysis code: Sean Crowell. Many scripts import a shared helper module
> `wrf-stilt-utils.py` from a sibling `../wrf-stilt` repository (not included here), and
> assume the TACC/Stampede directory layout (`/scratch/...`, `/work2/...`).

---

## End-to-end pipeline (how the pieces fit)

```
flight HDF5 (raw XCH4)
   │  flight_data/coarsen_track_files.py        (spatially average flight track)
   ▼
coarsened track HDF5
   │  flight_data/create_csv.py / create_csv.py (build STILT receptor CSV)
   ▼
STILT receptor CSV ──► stilt/run_stilt_*.r      (run STILT, one run per altitude)
   │                        via create_fp.sh / sample_all_alts.sh (SLURM)
   ▼
per-altitude footprints + trajectory .rds files
   │  halo_boundary_sampler.py                  (sample WRF along trajectories → boundary pts)
   ▼
boundary-point HDF5 files (./bnd/...)
   ├─ halo_column_xgas_bc.py    → column XCO2/XCH4 boundary conditions (xbnd.h5)
   └─ halo_column_fp.py         → pressure-weighted COLUMN footprints
   │
   ▼ (emissions side)
nyc_ch4_emissions.h5
   │  emissions_resampler.py    (regrid inventories onto footprint grid)
   ▼
nyc_emissions_regrid.h5
   │  halo_column_enh.py / enhancement_calc.py  (footprint × emissions → enhancements)
   ▼
modeled enhancements ──► inversions (bayes_opt/, *.ipynb)
```

---

## Top-level Python scripts

| Script | Purpose |
|---|---|
| `halo_boundary_sampler.py` | Core sampler. For each flight, finds STILT trajectory `.rds` files at a given receptor altitude, screens trajectory endpoints to the WRF∩STILT domain, then samples WRF output (`pressure`, `psfc`, `CO2_BCK/ANT/BIO`, `CH4_BCK/ANT`) at each trajectory boundary point. Writes trajectory + boundary-point HDF5 files under `./trj/` and `./bnd/`. Runs in parallel via `wrf-stilt-utils`. **Usage:** `python halo_boundary_sampler.py <domain> <altitude> <flt1> <flt2> ...` |
| `boundary_condition_pipeline.py` | Earlier/alternate version of the boundary sampler (same idea, simpler domain handling using a bounding box from the WRF grid). Largely superseded by `halo_boundary_sampler.py`. |
| `halo_column_xgas_bc.py` | Collapses the per-level boundary-point samples into **column-averaged (pressure-weighted) XCO₂/XCH₄ boundary conditions** for each receptor, writing `bnd/wrf_<domain>/<flt>_xbnd.h5`. **Usage:** `python halo_column_xgas_bc.py <domain> <flt1> ...` |
| `halo_column_fp.py` | Builds **pressure-weighted column-average footprints** by combining the individual per-altitude STILT footprints for a receptor using the layer pressures from the boundary files. Writes column-footprint netCDF/HDF5 (`*_col_*foot`). **Usage:** `python halo_column_fp.py <domain> <flt1> ...` |
| `emissions_resampler.py` | Nearest-neighbor **regrids emission inventories** (`edgar`, `epa`, `pitt` categories) from `nyc_ch4_emissions.h5` onto the STILT footprint grid; writes `nyc_emissions_regrid.h5`. Also defines `compute_xgas_enhancements` (footprint × emissions). |
| `halo_column_enh.py` | Convolves the regridded emissions with each flight's column footprints to compute **per-category modeled XCH₄ enhancements (`dxch4`)** per observation; writes `<flt>_dxch4.h5`. **Usage:** `python halo_column_enh.py <domain> <flt1> ...` |
| `enhancement_calc.py` | Lightweight companion that reads `nyc_emissions_regrid.h5` and the saved column footprints and computes enhancements (helper variant of the above; partially a work-in-progress). |
| `create_csv.py` | Converts coarsened flight HDF5 (`*300m*.h5`) into a STILT **receptor CSV** (`obid, lati, long, zagl, UTC_date, UTC_time`). |
| `create_halo_receptor_csv.py` | Splits a flight receptor CSV into smaller chunked receptor files (N receptors per file) for batched STILT runs. **Usage:** `python create_halo_receptor_csv.py <flight_csv> <n_per_file> <save_dir>` |

## Shell / SLURM job scripts

| Script | Purpose |
|---|---|
| `create_fp.sh` | SLURM batch job: activates the `stilt2` conda env and runs `run_stilt_nyc_hrrr.r` for a receptor file across a fixed list of altitudes (50 m … 10 km) to generate footprints. |
| `sbatch_sample_wrfout.sh` | SLURM batch job: runs `halo_boundary_sampler.py` for a given domain + altitude across all six flights. |
| `sample_all_alts.sh` | Submits `sbatch_sample_wrfout.sh` once per altitude (fan-out over all altitudes). **Usage:** `./sample_all_alts.sh <domain>` |
| `run_job.sh` | SLURM batch job: runs `halo_column_xgas_bc.py` for domain `d01` across all flights. |

## `stilt/` — STILT model run drivers (R)

UATAQ STILT executables (`Ben Fasoli` template), parameterized by `--alt` (receptor
altitude) and `--file` (receptor CSV). Each variant points at a different meteorology
source / domain:

- `run_stilt_nyc_hrrr.r` — driven by **HRRR** meteorology.
- `run_stilt_nyc_wrfghgd01.r` / `run_stilt_nyc_wrfghgd02.r` — driven by **WRF-GHG** met, domains d01 / d02.
- `run_stilt_pittdomain.r` — Pittsburgh-domain variant.
- `single_ob.csv` — example single-receptor input CSV.
- `harvard_jacobians/` — externally supplied Jacobian/footprint netCDFs (git-ignored).

## `flight_data/` — observation preprocessing

- `coarsen_track_files.py` — Reads raw HALO XCH₄ HDF5 (`CH4DataProducts/XCH4_clear`, `Nav_Data`),
  applies a rolling spatial average along-track to a target resolution `dx` (m), trims to
  per-flight time windows, and writes coarsened `lat/lon/time/xch4` HDF5.
  **Usage:** `python coarsen_track_files.py <dx_meters> <file1.h5> ...`
- `create_csv.py` — Builds STILT receptor CSVs from coarsened `*_1000.0m.h5` track files.
- `*.h5` — flight data products (raw `staqs-HALO-XCH4_*`, coarsened `*_500m/1000m`, `*subCH4*`).
- Notebooks: `compare_data_versions.ipynb`, `coarsen_halo_tracks.ipynb`, `plot_halo_xch4_obs.ipynb`.

## `bayes_opt/` — Bayesian (geostatistical) flux inversion

A self-contained LPDM-style Bayesian inversion package solving
`shat = sprior + HQTᵀ·(HQHT+R)⁻¹·(z − Hsp)`:

| File | Role |
|---|---|
| `lpdm.py` / `ctl.py` | Core `lpdm` class: reads `config.ini`, manages domain/grid/H-matrix bookkeeping. |
| `mklm.py` | Builds the land-mask / region arrays for the inversion domain (from CarbonTracker regions). |
| `hsplit.py` | Steps through footprint netCDFs and writes the **H matrix** as per-timestep sparse slices. |
| `make_sc.py` | Precomputes the **spatial-distance** matrix used to build the spatial covariance. |
| `hsigma.py` | Multiplies H slices by per-cell **sigma** to form Hσ. |
| `hq.py` | Computes **HQ and HQHT** (Yadav & Michalak 2012 Kronecker form), parallelized. |
| `make_z.py` | Builds `zhsp.txt`: observations minus prior-convolved H and background. |
| `inversion.py` | Solves the Bayesian equation → posterior fluxes (`shat_flux.npy/.nc`). |
| `apost.py` | Computes the **posterior uncertainty** covariance. |
| `config.ini` / `config.ini.sample` | Inversion configuration (domain, resolution, correlation lengths, file paths). |
| `*_obs.txt`, `*_dxco2.txt`, `r_*.txt`, `bkg.txt`, `receptors.txt`, `obs.txt` | Inputs: per-inventory obs vectors, prior dXCO₂, observation-error variants, backgrounds, receptor lists. |
| `prep_nyc_inputs.ipynb`, `plot_ctl_results.ipynb`, `sanity_check_ct.ipynb` | Input prep, result plotting, sanity checks. |

## `tropomi/` — satellite cross-validation

Notebooks validating against TROPOMI XCH₄: `TROPOMI_val.ipynb`, `TROPOMI_val-CAMS.ipynb`,
`TROPOMI_val-bremen.ipynb`.

## Top-level analysis notebooks

| Notebook | Focus |
|---|---|
| `halo_data_analysis.ipynb` | General exploration of the HALO XCH₄ observations. |
| `halo_stilt_analysis.ipynb` | Comparing HALO observations to STILT simulations. |
| `halo_nyc_flux_estimate.ipynb` | Deriving NYC CH₄ flux estimates. |
| `halo_versus_model_enhancements_800m.ipynb` | Observed vs. modeled enhancements at 800 m. |
| `nyc_ch4_scalar_constraint-halo.ipynb` | Scalar (single-factor) constraint on NYC CH₄ emissions. |
| `boundary_xgas_halo.ipynb` | Inspecting the column-XGAS boundary conditions. |
| `compare_footprints.ipynb` | Comparing footprint products. |
| `create_stilt_receptors.ipynb` | Building STILT receptor sets. |

## Data files

- `nyc_ch4_emissions.h5` — NYC CH₄ emission inventories (`edgar`, `epa`, `pitt`) with category metadata; input to `emissions_resampler.py`.

## Notes / caveats

- Hard-coded TACC paths (`/scratch/07351/...`, `/work2/07655/...`) and conda env names
  (`stilt2`, `analysis`) must be adjusted for other systems.
- Several scripts (`enhancement_calc.py`, `create_halo_receptor_csv.py`) contain
  work-in-progress fragments and may need cleanup before reuse.
- `stilt/harvard_jacobians/` is git-ignored (see `.gitignore`).
</content>
</invoke>
