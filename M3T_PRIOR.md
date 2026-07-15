# Using an M3T inventory as the HALO prior

[M3T](../M3T) (the Modular Methane Mapping Tool) is an **independent** package that
produces gridded, sectoral methane emissions for anywhere in CONUS. This directory
treats it as an external tool: halo-nyc depends on M3T's *output format*, not its
code, and all the HALO-specific choices (the NYC grid, the Pittsburgh category
taxonomy, µmol units, the file schema) live here in
[`build_m3t_prior.py`](build_m3t_prior.py).

`halo_oe` is already wired for this — `bayes_opt/config.ini` has:

```ini
[emissions]
path      = ../m3t_option_1.nc4
inventory = m3t
```

So the whole integration is producing `m3t_option_1.nc4`. No `halo_oe` code changes.

## Workflow

The two steps run in **different environments on purpose** — M3T in its own env,
the packing in halo's `analysis` env — so neither imports the other.

**1. Run M3T over NYC, on the HALO grid.** Get the exact domain first:

```bash
conda run -n analysis python build_m3t_prior.py print-domain
#   domain     = (-80.0, 39.5, -72.02, 43.5)
#   domain_res = (0.03, 0.02)
#   domain_crs = 'epsg:4326'   grid 200x266
```

Then run M3T however you normally do (its own env, with the companion data staged
into the run's `in/` — see `M3T/python/notebooks/nyc_demo.py::prepare_run_dir`),
passing that `domain`/`domain_res` to `m3t.ch4_inventory_build`. That writes the
sector rasters to `<run>/out/`.

*(Convenience: if `m3t` and `h5py` are importable in one env and the companion data
is staged, `build_m3t_prior.py run --run-dir <dir> --tigerlines <…>.gpkg` does this
step and the next in one shot. It is optional and the only path that imports M3T.)*

**2. Pack the M3T outputs into the prior file** (halo `analysis` env; no `m3t`):

```bash
conda run -n analysis python build_m3t_prior.py pack \
    --m3t-out /path/to/m3t_run/out --proxy vulcan --out m3t_option_1.nc4
```

This reads M3T's documented output rasters, maps them to the Pittsburgh category
taxonomy, converts **nmol → µmol m⁻² s⁻¹** (the Jacobian's units), and writes the
flat-HDF5 schema `halo_oe.emissions` reads. It prints a per-category budget in
Gg CH₄/yr as a sanity check.

**3. Verify** it's a drop-in (loads through the real `halo_oe` loaders):

```bash
conda run -n analysis python verify_m3t_prior.py --file m3t_option_1.nc4
```

**4. Run the inversion** — nothing new:

```bash
cd bayes_opt && python run_halo.py config.ini --inventory m3t
```

## The `_option_N` naming

`--proxy vulcan|aces` selects M3T's CO₂ downscaling proxy; each is a different,
equally-valid M3T prior. Build each to its own file (`m3t_option_1.nc4` = Vulcan,
`m3t_option_2.nc4` = ACES, say) and compare posteriors, exactly as EDGAR/EPA/Pitt
are compared. Any other M3T config choice can spawn further options the same way.

## Categories, and who groups them

**The prior file carries M3T's categories at their native granularity; it does no
grouping.** Collapsing categories into process super-groups is `halo_oe`'s job,
driven by `config.ini`'s `[category_groups]` keyword map — which changes between
experiments. So `build_m3t_prior.py` passes M3T's fields through under M3T's own
names (`M3T_CATEGORIES`), exactly as `edgar` (20 categories) and `epa` (26) are
carried fine and regrouped at inversion time. The 13 M3T categories — its seven
bottom-up sectors, plus the gridded-EPA remainder broken into six swappable
groups (from `out/remaining_gepa/`):

| category (label in the file) | M3T output |
|---|---|
| `landfills` | `landfills.nc` (municipal) |
| `natural_gas_transmission` | `natural_gas_transmission.nc` |
| `natural_gas_distribution` | `natural_gas_distribution.nc` |
| `wastewater` | `wastewater.nc` |
| `stationary_combustion_fossil_fuel` | `Stationary_combustion_sector_fossil_fuel_total_<proxy>_bystate.nc` |
| `stationary_combustion_wood` | `Stationary_combustion_sector_wood_total_<proxy>_bystate.nc` |
| `wetlands` | `wetlands.nc` |
| `GEPA_oil_gas_upstream` | `remaining_gepa/GEPA_oil_gas_upstream.nc` (petroleum systems + NG upstream + abandoned O&G) |
| `GEPA_coal` | `remaining_gepa/GEPA_coal.nc` |
| `GEPA_livestock` | `remaining_gepa/GEPA_livestock.nc` (enteric + manure) |
| `GEPA_crop_ag` | `remaining_gepa/GEPA_crop_ag.nc` (rice + field burning + composting) |
| `GEPA_industrial_landfill` | `remaining_gepa/GEPA_industrial_landfill.nc` |
| `GEPA_other` | `remaining_gepa/GEPA_other.nc` (mobile combustion + petrochemical + ferroalloy) |

The six `GEPA_*` fields partition the gridded-EPA remainder exactly, so each is a
**swap hook** — build a prior with a dedicated coal or oil-&-gas inventory by
replacing that one field. (M3T also still writes its `GEPA_thermo`/`non_thermo`/
`ind_landfill` aggregates, which these roll up to; the packer just reads the fine
ones.)

### Grouping M3T's labels

`halo_oe.groups.DEFAULT_KEYWORD_MAP` was tuned to the EDGAR/EPA/Pitt vocabularies,
so out of the box some M3T labels (e.g. `natural_gas_transmission`) fall into
`other`. Give `config.ini` a `[category_groups]` that understands M3T's names:

```ini
[category_groups]
natural_gas      = ng_, natural gas, natural_gas, fuel exploitation gas
coal             = coal
landfill         = landfill, solid waste, waste incineration, waste burning
wastewater       = wastewater, waste water
livestock        = livestock, enteric, manure
crop_agriculture = crop_ag, rice, agricultur, field burning, soils, composting
oil_gas_upstream = oil_gas, petroleum, oil, refineries, refining
wetlands         = wetland
combustion       = combustion, stationary, mobile, power industry, buildings, manufacturing, iron and steel, aviation, shipping, railways, transport, chemical
```

Order matters (first keyword match wins) — `crop_agriculture` sits **before**
`oil_gas_upstream` so `Agricultural soils` matches `soils` rather than the `oil`
in "s*oil*s". Verified against `halo_oe.groups`, this routes M3T as: `natural_gas`
(transmission, distribution), `oil_gas_upstream`, `coal`, `livestock`,
`crop_agriculture`, `landfill` (landfills + GEPA industrial), `wastewater`,
`combustion` (both fuels), `wetlands`, and `GEPA_other` → `other`.

Unlike the default map this **also** resolves EDGAR's and EPA's petroleum and
agriculture labels into these same fine groups (`Petroleum Systems *` →
`oil_gas_upstream`, `Enteric/Manure` → `livestock`, etc.) — which is what you want
for apples-to-apples prior comparison, since those inventories expose the same
processes. Pitt is unchanged. As always, grouping is config-driven: tune this per
experiment, and nothing in the prior file needs rebuilding when you do.
