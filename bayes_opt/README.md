# halo_oe — HALO NYC methane flux inversion

HALO-specific Bayesian flux inversion of column-averaged XCH₄ observations over
New York City. This package contains **only the HALO-specific glue**; the generic
linear-Gaussian optimal-estimation machinery lives in the separate
[`goe-inversion`](../../../goe-inversion) project and is used here purely by
import. Nothing HALO-specific belongs in that project.

```
halo-nyc/bayes_opt/halo_oe/      <- this package (imports goe + adapters)
/Volumes/Expansion/goe-inversion <- generic framework: goe/ (core) + adapters/
```

The dependency arrow is one-way: `halo_oe` → `goe-inversion`. Swapping in a
different problem means writing a different glue package, not editing the
framework.

---

## The model

Each HALO receptor is a column-averaged XCH₄ observation taken from an aircraft
at ~10–11 km. We write it as a background plus an enhancement driven by surface
fluxes through a precomputed column Jacobian `H` (the `stilt/harvard_jacobians/*.nc`
files, units ppm per µmol m⁻² s⁻¹):

```
x_obs = x_bg + H f + ε,      ε ~ N(0, R)
```

We solve for the flux `f` by Bayesian optimal estimation, parameterized as
dimensionless multiplicative scalars on a prior emission field, with a Gaussian
prior `N(xa, Sa)`:

```
x̂ = xa + Sa Hᵀ (H Sa Hᵀ + R)⁻¹ (z − H xa),     z = x_obs − x_bg
```

The solve is done in observation space (`n_obs ≈ 1271` receptors per flight ≪
`n_state`), so a multi-gigabyte dense Jacobian over millions of grid cells stays
tractable. The framework picks observation- vs state-space automatically.

---

## Pipeline (what runs, in order)

```
Jacobian file (.nc)  ─► JacobianFile            metadata only; big array read lazily
       │
       ├─ receptors (lat/lon/xch4) ─► background.py  per-flight planar background
       │                                  └─► z = x_obs − x_bg,  R   (observations)
       │
domain bbox ─► GriddedState (mask)  ─► jf.operator(active=mask)   ← the slow 13 GB read
       │                                  └─► H over active cells (base operator)
       │
nyc_ch4_emissions.h5 ─► emissions.py  ─► prior field(s) regridded onto Jacobian grid
       │                                  (one inventory; optionally per super-category)
       ▼
   StateSpace + BlockDiagonalCovariance ─► GaussianLinearProblem ─► solve
       ▼
   posterior maps  +  flux.py / decomposition.py  ─► integrated flux totals (+ uncertainty)
       ▼
   write_posterior(...)  ─► netCDF
```

The expensive Jacobian read is done **once** per run (`pipeline.load_context`) and
reused — including across the three inventories in `--compare` mode.

For **multiple flights** the flux state is shared and each flight contributes its
own block of observation rows (`goe.BlockRow`), its own background, error
covariance (`BlockDiagonalCovariance`), and background-offset group. Select flights
in the config (`[jacobian] flights`) or at the CLI (`--flights`); see below.

---

## Key concepts

### Inventories are alternatives, not additive

`nyc_ch4_emissions.h5` holds **three independent inventories** — `edgar`, `epa`,
`pitt` (Pittsburgh) — each a *complete* estimate of the same NYC emissions with
its own sub-category breakdown. **They are never summed.** A single inversion uses
one as the prior (`[emissions] inventory`); `--compare` inverts each separately
and tabulates the posteriors to show how prior-dependent the answer is.

### Multiple flights (shared flux, stacked observations)

A run assimilates one or more flights. The **flux state is shared** across flights;
each flight contributes its own observation rows (its Jacobian stacked with
`goe.BlockRow`), its own per-flight background, its own observation-error
covariance (the flights form a `BlockDiagonalCovariance` — errors correlate within
a flight, not across), and its own optimized **background-offset** group. This is
the lever that raises information content: degrees-of-freedom-for-signal is ~1–2
per flight, so combining flights is what makes the flux (and the `R`/`Sa` error
split) identifiable.

Flights are selected by id (the Jacobian file stem, e.g. `20230726_1`) via
`[jacobian] flights`, the `--flights` CLI override, or `load_context(cfg, ...,
flights=[...])` — so single days and arbitrary combinations are easy to run for
experiments. Outputs tag each receptor with its flight (`receptor_flight`).

### Background (per-flight planar, lower-envelope)

The Jacobian explains only the *enhancement*, so the inflow/free-tropospheric
background must be removed first. `background.py` fits a **per-flight, low-degree
(default plane) surface in (lat, lon)** to the **lower envelope** of the observed
columns (a low quantile of residuals, iteratively), so it tracks clean air rather
than riding up into the urban plume. Fitting per flight captures day/time
variation; the optimized background-offset block (`bc`) mops up a residual
constant per flight.

**Don't let in-domain data set the background.** The baseline must come from air
the inversion domain does not influence, or the in-domain enhancement leaks into
the background it is subtracted from (circular; suppresses the signal). For column
data, "in-domain" means a receptor whose Jacobian footprint touches the masked
cells, measured as each receptor's **domain sensitivity = row sum of the masked
`H`**. The planar fit is restricted to the least-sensitive fraction of receptors
(`[background] domain_sensitivity_quantile`); the surface is still evaluated at all
of them.

### Observation error (model-data mismatch)

The observation-error covariance `R` (`obs_error.py`, `[observations]
error_model = components`) is built from physical components rather than one
number: an independent per-receptor **measurement** variance plus a
**representation+transport** term. Because adjacent 1 km receptors have heavily
overlapping column footprints, that mismatch term is **correlated along-track** —
`R = diag(measurement) + correlated MDM`, a single `SparseCovariance`. A diagonal
`R` treats correlated residuals as independent and makes reduced χ² look far too
high; the correlation is often the largest single correctness gain.

The magnitudes are hyperparameters. `run_halo.py --tune` reports the Desroziers
consistency `r_scale` and the marginal-likelihood-optimal variance multipliers
(`goe.tuning`), with the config edits to apply them. With one flight the `R`/`Sa`
split is weakly identified (DOFS ~1–2) — tune `R` only until multi-flight data are
available.

### Category decomposition

Within **one** inventory, the sub-categories (landfills, wastewater, natural gas,
…) *are* additive sectors that sum to the inventory total, so the posterior can be
decomposed by category. Sub-categories are grouped into configurable
**super-categories** (`groups.py` + `[category_groups]`) by keyword matching.

Three attribution methods (`[decomposition] method`):

| method | what it solves | split is determined by | notes |
|---|---|---|---|
| `partition` | per-cell scalar on the inventory total | **prior** category variance (post-hoc) | cheap; data constrain the total only |
| `category_fields` | a per-cell scalar field **per category** | **data** + per-category covariance | **recommended**; spatially resolved, well-posed |
| `category_scalars` | one domain scalar per category + per-cell total correction | data (spatial fingerprints) | weakly identified from a single flight; prior-sensitive (can run away) |

**Per-category error structure** (`category_fields`, via `[category_spatial]`):
point sources with known locations (landfills, WWTPs) use a **diagonal**
covariance (only magnitude at known cells is uncertain); diffuse / spatially
uncertain sources (natural gas distribution, area combustion) use a **spatial**
covariance with a modest decorrelation length. This both stabilizes the total and
yields a physically defensible sectoral attribution.

> Caveat baked into the design: column XCH₄ poorly separates *co-located*
> categories (same footprint). The prior σ's therefore do real work in the split,
> and from a single flight the information content (DOFS) is ~1–2. Multi-flight
> data are needed before fine attribution is trustworthy.

### Buffer region (out-of-core emissions)

The receptors are sensitive to emissions **outside** the core mask; the
`--diagnose-domain` check shows a large fraction of explained enhancement can come
from beyond the core. If those out-of-core sources have nowhere to go, their signal
aliases into the core edge cells and the background, biasing the core total. The
**buffer** (`buffer.py`, `[buffer]`) gives them their own coarse flux degrees of
freedom: out-of-core native cells are grouped into **super-cells**, each carrying
one uniform flux unknown whose forward column is the *summed* Jacobian over its
native cells (built in the same streamed pass as the core operator — see
`JacobianFile.operator_with_buffer`, no second read).

Two ways to define the super-cells (`[buffer] mode`):

| mode | super-cells from | use for |
|---|---|---|
| `coarse` | tile the out-of-core ring by an integer `factor` (or target `resolution_deg`), optionally limited to `outer_bbox` | a generic "coarser resolution outside" |
| `mask` | an integer **label field** on the grid (`mask_file`, `.npy`/`.nc`); each positive label → one super-cell | arbitrary named buffer blocks / sectors |

The buffer is a **nuisance** state: its prior mean is the area-weighted inventory
flux density per super-cell (prior std `stddev`, relative to that mean), it absorbs
out-of-core signal and tightens the core through cross-covariance, but it is
**excluded from the reported core total**. The buffer block (`buffer`) and its
super-cell geometry are saved in the bundle for post-hoc inspection
(`SavedInversion.buffer`). Multi-flight runs stack the buffer operator across
flights exactly like the core.

---

## Layout

```
bayes_opt/
  run_halo.py        # CLI entry point (top level)
  config.ini         # all settings (see below)
  halo_oe/           # the Python package (importable modules only)
  tests/             # synthetic unit tests
  notebooks/         # walkthrough + bundle-analysis notebooks
  runs/              # all run_halo.py outputs land here
```

Run `run_halo.py` from the `bayes_opt/` directory; it adds that directory to the
path so `import halo_oe` works from any working directory.

### `halo_oe/` modules

| file | role |
|---|---|
| `__init__.py` | bootstraps `goe`/`adapters` onto `sys.path` (or use `GOE_INVERSION_PATH`) |
| `pipeline.py` | `load_context` (per-flight reads, stacked) + `flight_paths` + `invert` (all modes) |
| `background.py` | per-flight lower-envelope planar background (domain-insensitive receptors) |
| `obs_error.py` | component-wise `R`: measurement ⊕ along-track-correlated model-data mismatch |
| `emissions.py` | regrid inventory totals / per-sub-category fields onto the Jacobian grid |
| `groups.py` | configurable keyword grouping of sub-categories into super-categories |
| `decomposition.py` | the three attribution methods + per-category covariance builder |
| `buffer.py` | coarse out-of-core buffer super-cells (`coarse` tiling or `mask` label field) |
| `flux.py` | integrate scalars × prior × cell-area → totals with uncertainty (`linear_estimate`) |
| `io_bundle.py` | save/reload a complete inversion (prior+posterior, observations, factors) for post-hoc analysis |
| `diagnostics.py` | out-of-core sensitivity diagnostic (whether a buffer is needed) |

`run_halo.py` (top level) is the CLI: single run, `--compare`, `--inventory`,
`--flights`, `--tune`, `--diagnose-domain`, `--plot-buffer`. `notebooks/` holds
`halo_inversion_walkthrough.ipynb` (step-by-step, reads the same `config.ini`) and
`saved_bundle_analysis.ipynb` (post-hoc bundle reader). `tests/` holds the
synthetic unit tests (`test_flux`, `test_background`, `test_decomposition`,
`test_obs_error`, `test_multiflight`, `test_buffer`, `test_buffer_pipeline`,
`test_io_bundle`, `test_diagnostics`).

---

## Configuration (`config.ini`)

All settings are read from `config.ini`; the notebook reads the same file, so it
stays in sync with the CLI. Sections:

- `[jacobian]` — `dir` + `flights` (comma-separated flight ids, assimilated jointly;
  overridable with `--flights`), or a single `path` (back-compat); `in_memory`, `row_chunk`
- `[domain]` — `bbox = [lat_min, lat_max, lon_min, lon_max]` (the NYC core mask)
- `[emissions]` — `path`, `inventory` (primary prior), `compare` (list for `--compare`)
- `[background]` — `method` (planar|constant), `degree`, `envelope_quantile`, `n_iter`,
  `domain_sensitivity_quantile` (restrict the fit to domain-insensitive receptors; 1.0 = off)
- `[prior]` — `scalar_stddev`, `correlation_length_km` (per-cell total field)
- `[observations]` — `error_model` (simple|components), `error_stddev`, `error_inflation`,
  fallback `baseline`; for components: `measurement_stddev`, `mdm_stddev`,
  `mdm_correlation_length_km`
- `[offset]` — `n_groups`, `stddev` (per-flight background offset block)
- `[buffer]` — `enabled`, `mode` (coarse|mask); coarse: `factor` or `resolution_deg`,
  `outer_bbox`; mask: `mask_file`; `stddev`, `stddev_floor` (out-of-core buffer region)
- `[decomposition]` — `enabled`, `method` (partition|category_fields|category_scalars)
- `[category_groups]` — `group = keyword, keyword, …` (sub-category → super-category)
- `[category_uncertainty]` — `default` + per-group relative σ
- `[category_spatial]` — per-group decorrelation length km; `0` = diagonal (point source)
- `[tuning]` — `tune_R`, `tune_Sa` (used by `run_halo.py --tune`)
- `[flux]` — `unit_scale`, `unit_label`
- `[output]` — `dir` (where all outputs go; default `runs`, relative to the config
  file), `path` (posterior base filename), optional `bundle_dir`

Paths in `config.ini` are relative to the file's own directory. Inline `#`/`;`
comments after a value are supported (handled by `goe.config.Config`).

---

## Usage

Run from the `bayes_opt/` directory (`run_halo.py` puts itself on the path so the
`halo_oe` package imports regardless of the working directory).

```bash
# single inversion with the primary inventory (config [emissions] inventory)
python run_halo.py config.ini

# override the inventory
python run_halo.py config.ini --inventory epa

# compare all three inventories as alternative priors (one Jacobian read)
python run_halo.py config.ini --compare

# report model-data-mismatch diagnostics + max-likelihood error scales (non-destructive)
python run_halo.py config.ini --tune

# assimilate specific flights (single day, or any combination) for experiments
python run_halo.py config.ini --flights 20230726_1
python run_halo.py config.ini --flights 20230726_1,20230726_2,20230728_1

# is a buffer needed? fraction of receptor sensitivity outside the core (no solve)
python run_halo.py config.ini --diagnose-domain

# map the core + buffer regions with their prior mean and diagonal σ (no solve)
python run_halo.py config.ini --plot-buffer            # PNG next to [output] path
python run_halo.py config.ini --plot-buffer regions.png
```

The `--plot-buffer` map is a prior-only check (built from Jacobian metadata, no
large-array read): three panels over the core∪buffer window — prior mean flux
density, prior 1σ (diagonal), and relative σ (σ/|mean|) — with the core mask drawn
as a red box and buffer super-cell centers marked. Use it to sanity-check the
super-cell layout (`coarse` tiling or `mask` labels), the `outer_bbox` extent, and
how loose the buffer prior is relative to the core.

Decomposition is enabled via config (`[decomposition] enabled = true` and
`method = …`), not a flag. Output is a netCDF with posterior scalar fields, their
uncertainties, prior fields, and the integrated flux totals as attributes.

### As a library

```python
import halo_oe                       # wires goe + adapters onto the path
from goe.config import Config
from halo_oe.pipeline import load_context, invert

cfg = Config("config.ini")
# one or more flights (shared flux state); reads each Jacobian once
ctx = load_context(cfg, inventories=["pitt"], flights=["20230726_1", "20230726_2"])
res = invert(ctx, "pitt", decompose=True, method="category_fields")
print(res.report.as_table())                            # per-category totals + uncertainty
```

---

## Saving an inversion for post-hoc analysis

Re-reading the multi-gigabyte Jacobians for every new analysis is unnecessary:
**aggregation and disaggregation are linear functionals of the posterior**, so a
solved inversion can be saved and reused without the forward operator. Run once
with `--save` (or `[output] bundle_dir`):

```bash
python run_halo.py config.ini --flights 20230726_1,20230726_2 --save jul26_both
```

This writes a directory bundle: `factors.npz` (posterior mean + the covariance
factors `Sa`/`W = Sa Hᵀ`/Cholesky of `G`, which reproduce `aᵀx̂` and `aᵀŜa`
*exactly* — no operator), `fields.nc` (geometry, super-category prior fields on
active cells, per-receptor obs/background/enhancement/modeled/flight/outlier
flag), and `layout.json` / `report.json` / `config.ini`. A bundle is tens of MB
(dominated by `W`), even carrying the full cross-covariance.

Reload it instantly and re-analyze:

```python
from halo_oe.io_bundle import load_inversion
inv = load_inversion("runs/jul26_both")
inv.estimate(A)        # (A x̂, A Ŝ Aᵀ) for any functional A — exact, no Jacobian
inv.field("pitt")      # posterior scalar field on the grid
inv.group_fields       # super-category priors (active cells) for re-grouping
inv.receptors          # obs / background / enhancement / modeled / flight / flag
```

`inv.posterior`, `inv.state`, `inv.core`, `inv.grid`, and `inv.group_fields` plug
straight into the `flux` / `decomposition` helpers, so you can re-aggregate,
re-attribute by prior variance, or re-group categories with no re-solve. Bundles
are git-ignored (`*.npz`, `runs/`, `*_bundle/`) — keep them off GitHub.

## The walkthrough notebook (`halo_inversion_walkthrough.ipynb`)

A 12-step, runnable tour of one inversion, each stage exposed for inspection.
It reads the **same `config.ini`** as the CLI, so its results match
`run_halo.py`. It is **single-flight by design** (one Jacobian, ~2 min to run);
multi-flight assimilation is exercised via the CLI / `load_context(..., flights=[...])`.

### Configuration it runs with

As shipped (`config.ini`), the notebook's main inversion (steps 1–10) is:

- **State vector** — flight `20230726_1` (1271 receptors); NYC core mask
  `bbox = [40.4, 41.1, -74.3, -73.5]` (~6,942 active cells); prior inventory
  **Pittsburgh**. State = a per-cell multiplicative **scalar field on the
  Pittsburgh total** (prior mean 1) **+ one background offset** (`bc`), ≈ 6,943
  unknowns. (Step 12's `category_fields` decomposition expands this to a per-cell
  field *per* super-category + `bc`.)
- **Observations & background** — column XCH₄ minus a **per-flight planar
  background**: degree-1, lower-envelope (`envelope_quantile 0.25`, `n_iter 5`),
  **fit only on domain-insensitive receptors** (`domain_sensitivity_quantile 0.5`)
  and evaluated at all. One optimized background **offset** per flight
  (prior σ 0.02 ppm).
- **Uncertainties** — flux-scalar prior σ **0.5** with **5 km** spatial
  correlation; observation error `R` = **diagonal σ 0.02 ppm** (`error_model =
  simple`); outlier rejection **off** (`outlier_threshold = 0`).
- **Category decomposition** — grouping by keyword (Pittsburgh → natural_gas,
  landfill, wastewater, combustion, other); per-category prior uncertainty
  **0.5**; per-category prior-error structure: **natural_gas & combustion = 5 km
  spatial** (diffuse), **landfill, wastewater, other = diagonal** (point sources).

### Where the notebook intentionally goes beyond the config (to demonstrate options)

| Step | `config.ini` | The notebook also shows |
|---|---|---|
| 8b (model-data mismatch) | `error_model = simple` | builds the **`components` correlated `R`** and compares reduced χ² + error tuning |
| 8c (outliers) | `outlier_threshold = 0` (off) | **flags** at 4σ (`innovation`) and maps the flag (illustrative; not dropped) |
| 12 (decomposition) | `enabled = false` | **forces it on** with `method = category_fields` |

So steps 1–10 are faithful to the config; 8b/8c/12 are teaching overrides. Edit
`config.ini` (e.g. `error_model = components`, `outlier_threshold = 4`,
`decomposition.enabled = true`) and re-run to make the CLI behave like those steps.

---

## Tests

Synthetic, self-contained (no large files needed). Run any file directly:

```bash
python tests/test_flux.py
python tests/test_background.py
python tests/test_decomposition.py
python tests/test_obs_error.py
python tests/test_multiflight.py
```

Key invariants checked: the background fit recovers a plane under a synthetic
plume, is flight-dependent, and excludes domain-sensitive receptors; the flux
estimator matches a dense reference; both attribution modes (and `category_fields`)
sum exactly to the inventory total with prior totals equal to the direct integral;
per-category covariances are diagonal vs spatial per config; the component-wise `R`
combines an independent measurement diagonal with a correlated mismatch term.

---

## Status / units caveat

The pipeline runs end-to-end on real data (single flight). Integrated fluxes are
reported in **native units** (`prior-units × m²`) until the inventory's emission
units are confirmed and `[flux] unit_scale` is set (e.g. to convert µmol s⁻¹ →
kt CH₄ yr⁻¹).

### Remaining work toward a defensible number

1. **Multi-flight experiments** — multi-flight assimilation is implemented
   (`--flights`); run flight combinations to raise DOFS (≈1.2 for one flight,
   ≈2.5 for two) and identify the error split. Assimilating all six is the goal.
2. **Domain + buffer** — enlarge the core mask and add a coarse buffer ring so
   just-outside emissions don't alias inward.
3. **Error budget** — the component-wise, along-track-correlated `R` and the
   tuning hooks (`--tune`, `goe.tuning`) are in place; what remains is to *fit*
   the MDM magnitude and correlation length to data, best done with multi-flight
   so the `R`/`Sa` split is identifiable (DOFS ~1–2 per flight is too few).
4. **Units** — confirm inventory units, set `unit_scale`.
5. **Sensitivity** — vary background envelope quantile, prior widths, correlation
   lengths, MDM correlation length, and the category grouping.
