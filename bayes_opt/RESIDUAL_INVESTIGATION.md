# Residual & background investigation — findings and open questions

**Status as of 2026-08-31.** This documents an extended investigation into
systematic (non-noise-like) per-flight residual structure that survived MDM
tuning, covering the background-model improvements that came out of it, the
diagnostic ideas tried and ruled out along the way, and what remains
genuinely unexplained. It complements [README.md](README.md) (architecture),
[TUNING.md](TUNING.md) (the MDM tuning procedure itself), and — new as of
this update — [WIND_VARIABILITY.md](WIND_VARIABILITY.md) (a general-purpose
characterization of HRRR wind conditions across the 6 flights, kept as its
own document since it's useful beyond this investigation).

Sections are numbered in the order the investigation actually happened, and
each one's own **Verdict** states the conclusion in force today — later
sections sometimes supersede an earlier one's open question (cross-referenced
where that happens), and a few sections narrate a wrong initial read before
correcting it, kept because *how* the correction was caught is itself a
reusable lesson (collected in §25). The status below is the current summary;
read it first, then use the sections as the evidence trail behind it.

## Current status, per flight

| flight | status | mechanism |
|---|---|---|
| `726_1` | hotspot explained; dip open | hotspot = prior-shape misallocation (§5); dip = unexplained, footprint-shape ruled out (§7.4) |
| `726_2` | explained | leg-offset background correction (§2.2/§3) |
| `728_1` | open | broad correlation-length-limited gradient (§5); next step is comparing inventories (§26.1) |
| `728_2` | open | broad correlation-length-limited gradient, a Long Island sub-cluster; three explanations tested and ruled out (§22) |
| `805` | open | leg-to-leg banding; background modeling exhausted (§12); footprint-resolution ruled out (§20); land/water contrast present but doesn't close it (§28); one of the two larger HRRR-vs-HALO boundary-layer-depth mismatches at receptor resolution (§30.3, corrected from an earlier single-point read that had this backwards); by far the most light-and-erratic modeled wind of the six flights (§31), the most promising open lead |
| `809` | explained | footprint-shape/resolution error — genuinely broader intrinsic footprint decay length (§20), now linked to a real mechanism: the deepest boundary layer of the 6 flights, correlating with footprint breadth across all 6 (§29.4), the largest HRRR-vs-HALO depth bias even after correcting to receptor resolution (§30.3), real sub-hourly boundary-layer swings HRRR's hourly cadence cannot represent (§30.2), and the one flight that breaks the SW/NE spatial bias gradient every other flight shares (§30.4) — three independent axes on which `809` is a distinct meteorological regime, not an extreme point on a shared spectrum |

Two candidate explanations remain live for the flights still open
(`728_1`, `728_2`, `805`): missing/misclassified sources in the `m3t`
inventory (§21 found a real but geographically unrelated gap: 10 landfills
dropped by a non-reporter carry-forward rule) and systematic errors in the
raw column data itself, the one hypothesis this investigation has never
tested directly (§22's closing verdict; §28's elapsed-time/land-water,
§29's mixed-layer-height, and §30/§31's HRRR-meteorology regressors are all
indirect first passes at it, not a direct one). See §26 for the full,
current list of next steps.

## 1. Starting point: residuals are systematic, not noise

A 49-job sweep over `[observations] mdm_stddev` × `mdm_correlation_length_km`
was run to calibrate the error model so reduced χ² ≈ 1. Per-flight residual
maps from that sweep showed structure (banding, localized clusters) that
didn't look like noise. To settle whether that structure was real or an
artifact of a bad tuning point, we computed the per-receptor residual
mean/stddev **across the entire 48-bundle sweep**: if the pattern were just
under-tuned noise, it should wash out or shift as the error model changed.
It didn't — **over 99% of the residual variance was config-invariant**. That
result is what justified moving off MDM tuning entirely and investigating
the physical/model source of the mismatch instead.

## 2. Background-subtraction refinements (implemented, in production)

### 2.1 Per-flight planar background + domain-sensitivity `fit_mask` (pre-existing)

Lower-envelope polynomial fit per flight (`halo_oe.background.
fit_lower_envelope_surface` / `flight_background`), restricted to receptors
*insensitive* to the inversion domain (`[background] domain_sensitivity_
quantile`) so the fitted baseline can't be pulled up by the enhancement
the inversion is trying to retrieve.

### 2.2 Per-leg background offset (added during this investigation)

**Motivation:** a single plane per flight can't capture boundary-layer drift
*within* a flight between individual survey legs, because a leg boundary is a
time-ordered discontinuity, not a spatial one — legs revisiting similar
geography at different times can have different backgrounds a spatial-only
surface will never see.

- `detect_legs()` segments the track using real elapsed-time gaps (recovered
  from the raw `flight_data/*.h5` files — the Jacobian files carry no
  timestamps) combined with a check that the leg heading actually reverses
  (this survey flies every leg along a fixed SW/NE axis), so a mid-leg data
  dropout isn't mistaken for a turn. Spuriously short candidate legs get
  merged into the previous one.
- `fit_leg_offsets()` estimates one additive offset per leg via a **GP/kriging
  fit in elapsed time** (exponential kernel), not independent per-leg
  estimates — nearby legs' offsets are correlated (boundary-layer evolution
  is smooth), so a leg's noisy raw estimate is pulled toward its time-
  neighbors' consensus rather than shrunk independently to zero. Sparse legs
  get a "reliability gap" term added to their noise variance
  (`prior_stddev² · max(0, 1 − n/min_reliable_points)`) so a leg with too few
  points to trust its own quantile is pulled almost entirely from neighbors.
- Config: `[background] use_leg_offsets` (default `false`), `leg_gap_seconds`,
  `leg_min_size`, `leg_axis_deg`, `leg_offset_stddev`, `leg_correlation_time_s`,
  `leg_offset_noise_stddev`, `leg_min_reliable_points`.
- Validated on single-flight and all-6-flight real data; visibly cleaned up
  residual banding for some flights (see §3).

## 3. What per-leg offsets fixed, and what they didn't

- `20230726_1`, `20230726_2`: leg-offset correction cleaned residuals up well.
- `20230728_1`, `20230728_2`, `20230805`, `20230809`: retained substantial
  residual structure even with leg offsets on.

This split is what drove everything below: is the remaining structure (a)
more background-model error, (b) prior spatial-*shape* error, (c) a prior
spatial-*correlation-length* limitation, (d) a genuinely missing source, or
(e) a transport/footprint-resolution limitation?

## 4. Background-model ideas tested and ruled out

### 4.1 Wind-relative background (krige over downwind-projected distance, not elapsed time)

Motivating idea: legs flown upwind provide a natural "background" for legs
downwind. First test looked dramatically better than the time-based
version — but it was **circular**: it skipped the domain-sensitivity
`fit_mask`, so plume-contaminated receptors were informing the "background."
Once corrected (streamed the actual masked Jacobian for an honest
domain-sensitivity mask — ~11s for a 12.6GB file, since only active columns
are ever materialized), the wind-relative axis showed **no real advantage**
over the time-based one (~0.030 RMS either way), and the empirically-fit
gradient direction changed drastically once honest (44.6°/R²=0.61 unmasked →
107.4°/R²=0.12 masked) — most of the earlier "signal" was contamination, not
real background structure.

**Protocol established from this:** any new background-model idea must be
tested with the honest `fit_mask` from the first pass, never skipped "just to
get a quick read" — and always compare against a clear baseline, and inspect
maps, not just a summary RMS number.

### 4.2 Continuous (per-receptor) kriging, not per-leg-constant

Same GP idea as the leg offsets, but smoothed continuously per receptor
instead of one constant per leg. A modest, *real* improvement once tested
honestly (~11–12%). More informative: **RMS plateaued regardless of
correlation length**, from 0.05° to 10°. That ceiling is itself the finding —
it isolates a genuine, fixable **broad-scale** background component from a
persistent, **localized** residual feature that no amount of background
smoothing removes.

## 5. Diagnosing the localized residuals: prior shape vs. missing source

**Method:** look for dipole/checkerboard patterns in the posterior scalar
field (adjacent over- and under-allocation is a signature of a
*shape* error in the prior, even when the nearby total is plausible), overlay
against the prior's per-category fields to see whether any category has mass
near the residual at all, and cross-check against `[category_spatial]`
correlation lengths (`natural_gas = 5km`, `combustion = 5km`; `landfill`,
`wastewater`, `other`, default `= 0`, i.e. diagonal / point-source only) — this
sets a hard ceiling on how far apart cells can be jointly adjusted by the
flux-state posterior.

- **`20230726_1`:** clear, well-localized answer. A real sub-cluster
  misallocation inside `natural_gas`'s prior — a dipole in the posterior next
  to a cell carrying disproportionate prior mass. The 5km correlation length
  let the inversion partially relocate mass to cover the ~50km feature.
- **`20230728_1`:** inconclusive. **Zero prior mass in any category** at the
  residual's location — the flux state has no degrees of freedom there to
  produce signal with, regardless of correlation length. Either a genuinely
  missing/misclassified source in this inventory, or something else
  entirely (see §7).
- The `[category_spatial]` lengths directly explain the difference in
  outcome: `726_1`'s ~50km feature is within reach of several 5km-correlated
  cells acting together; `728_1`'s much broader (~170km) gradient is outside
  what any configured correlation length allows, independent of the prior's
  shape.

## 6. Outlier rejection (existing mechanism, currently off)

`goe.outliers.flag_outliers`, controlled by `[observations] outlier_threshold`
(currently `0`, off), `outlier_kind = innovation` (normalizes by the full
expected mismatch `H Sa Hᵀ + R`, not just `R`), `outlier_iterations`.
Discussed as a pragmatic fallback for individually bad points — but it is
**not** a fix for spatially-coherent systematic structure. A single gross
error and a shared bias across many nearby, correlated receptors look very
different diagnostically, and only the former is what outlier rejection is
built to catch.

## 7. Transport/footprint-resolution hypothesis

**Motivation:** STILT/HRRR transport resolution (~3km) is coarser than the
observation binning (~1km) — the model may be structurally unable to
reproduce sharp enhancements from strong point sources, independent of how
accurate the emission field is.

Four tests were run, in order of cost:

1. **Gradient-sharpness comparison** (`726_1`'s hotspot): the *modeled*
   enhancement is essentially flat across a ~200km along-track stretch where
   the *observed* data show a large, sharp real feature (a rise, a +0.127 ppm
   peak, a −0.023 ppm dip). Confirms the hypothesis is at least mechanistically
   plausible for this flight.
2. **Footprint centroid/spread** at the peak and dip receptors: both
   footprints are large (spread ~65–110km) with a sensitivity-weighted
   centroid ~164km from the receptor. **Caveat worth remembering:** this
   centroid distance is dominated by the long, low-sensitivity tail typical
   of *any* STILT footprint under a given day's meteorology — it is not a
   property specific to either receptor. Two receptors with wildly different
   observations having nearly the same centroid distance is not by itself
   diagnostic; don't over-read this metric on its own again.
3. **Direct footprint-shape similarity (cosine)**: the peak and dip
   footprints are actually nearly **orthogonal** (cosine ≈ 0.018) — genuinely
   different, not indistinguishable. This overturned an initial visual
   impression from the maps ("they look similar") that turned out to be
   wrong. Along the whole rise/peak/dip stretch, footprint similarity to the
   peak's own footprint decays smoothly over roughly ±15–20 receptors — a
   real, measurable "footprint correlation length" along-track.
4. **Leg-edge discontinuity, `726_1`'s dip specifically — corrected finding.**
   An initial plot appeared to show the dip receptor's footprint crashing to
   near-zero similarity with its immediate neighbor, suggesting a
   turn/release artifact. **This was wrong** — a plotting index misread: the
   crash shown was the ordinary, expected transition across the *actual* leg
   gap one receptor later, not a same-leg anomaly at the dip. Verified two
   independent ways: the true same-leg pair at the dip has cosine similarity
   0.705, right in line with that leg's own interior baseline of 0.741.
   **`20230726_1`'s dip remains genuinely unexplained.**

### General, rigorously verified result (all 6 flights, 52 legs)

Checked properly this time (targeted single-row Jacobian reads, cross-checked
two independent ways): most leg edges are statistically indistinguishable
from their own leg's interior baseline (median cosine ~0.76–0.80 for
start/end/interior alike). But a real minority tail exists — **~10% of
leg-starts and ~6% of leg-ends** show a genuine, large discontinuity (cosine
similarity to the immediate same-leg neighbor collapsing to well under half
of that leg's typical value). Consistent with the STILT release point still
being shaped by the turn maneuver rather than the leg's steady flight. Real
and worth guarding against — but **not** what explains the larger unexplained
features (`726_1`'s dip, `728_1`'s broad gradient) that motivated this whole
line of investigation.

## 8. New production feature: footprint-discontinuity QC flag

- `halo_oe/background.py`: `flag_leg_edge_discontinuities()` (compares each
  leg's first/last-pair footprint cosine similarity to that leg's own
  interior baseline) and `flag_footprint_discontinuities()` (the config-driven
  entry point, mirrors `receptor_background`'s leg-detection pattern).
- `halo_oe/pipeline.py`: new `InversionContext.discontinuity_mask`, computed
  per flight in `load_context`. `_solve_with_qc` takes a `pre_mask` that drops
  these receptors **before the first solve** — a hard, structural exclusion,
  applied before and independent of the residual-based `outlier_threshold`
  mechanism. Flagged receptors still show up in the saved bundle's
  `outlier_flag` alongside genuine outliers.
- Config (`[background]`, all **default off / unchanged behavior**):
  `flag_footprint_discontinuities = false`, `discontinuity_relative_threshold
  = 0.5`, `discontinuity_min_leg_size = 6`.
- Tests: 3 new cases in `tests/test_background.py` (synthetic corrupted-edge
  detection, clean-legs-stay-unflagged, default-off); full suite (50 tests
  across all test files) passes.
- Validated on real data: `20230726_1` with the flag on → 0 flagged
  (consistent with the corrected §7.4 finding); `20230726_2` → exactly 3
  flagged, matching the standalone diagnostic's prediction exactly.

## 9. Extending the prior-shape diagnostic to `728_2`, `805`, `809`

The §5 method (residual-cluster identification + per-category prior-mass
overlay + correlation-length reach) had only been worked through by hand for
`726_1` and `728_1`. Applied here to the three remaining flights, using the
leg-offset-corrected 6-flight bundle (`runs/legtest_legoffset_6flight`) and a
new script, `scripts/dipole_diagnostic.py` (plots: `runs/dipole_diagnostic_
<flight>_cluster<n>.png`).

**Two method bugs surfaced and were fixed before trusting any result** —
consistent with §25.6's rule that a surprising, consequential diagnostic
finding needs independent verification before being reported:

1. Connected-component clustering of flagged residual bins initially used
   4-connectivity (rook adjacency). But this survey flies every leg along a
   fixed SW/NE diagonal axis (§2.2), so consecutive same-leg bins are
   *diagonal* neighbors, not rook neighbors. 4-connectivity fragmented one
   real ~60km streak into a dozen spurious tiny "clusters" with no spatial
   coherence. Fixed by switching to 8-connectivity.
2. In `category_fields` mode, each category's *state block* is a
   **multiplicative scale factor** (prior = 1.0 uniformly for every cell) —
   the actual flux-density map used for grouping/plotting lives in a
   separate `group_fields` array with completely different units and scale.
   A first attempt diffed the state block directly against `group_fields`,
   producing a spurious uniform "posterior − prior ≈ 1.0" signal everywhere,
   independent of the real prior's magnitude or shape. The correct posterior
   flux perturbation is `(scale_factor − 1) × prior_density`.

**Findings, with the honest reach/prior-mass check applied to each cluster:**

- **`20230728_2`**: same character as `728_1` — a broad (~64km) negative
  cluster and a broad (~46km) positive cluster, both well beyond
  `natural_gas`'s 5km correlation-length reach. The negative cluster
  partially overlaps real `natural_gas` prior mass, and the posterior
  visibly pulled flux down there — but only over roughly the western third
  of the ~64km feature; the correlation length caps how far that adjustment
  can extend, leaving the rest of the band unexplained
  (**correlation-length-limited**, not a missing-prior case). The positive
  cluster (centroid ≈ 40.51, −74.14) has essentially **zero prior mass in
  all four categories** — the same "no degrees of freedom to produce signal
  with" signature as `728_1`'s residual, and a candidate for the same
  alternative-inventory comparison recommended there (§26.1).

- **`20230805`, `20230809`**: the dominant pattern here is visually
  different from `726_1`/`728_1`/`728_2` — not one or two broad gradients,
  but **leg-to-leg alternating-sign banding across nearly the entire
  flight** (adjacent survey legs biased oppositely; visible by eye in
  `residuals_map.png`, and this bundle already has per-leg background-offset
  correction, §2.2, turned on). The strongest localized clusters found
  within that banding (`805` at ≈40.69, −74.32; `809` at ≈40.72, −74.49 and
  ≈40.58, −74.64) mostly sit over **zero or near-zero prior mass in all four
  categories**, the same unexplained signature as `728_1` — but attributing
  them to prior shape would be premature: they sit inside a
  much-larger-amplitude leg-banding pattern that looks more like
  incompletely corrected background drift than a flux-shape problem
  specific to those individual cells.

**This changes the open-question framing for `805`/`809`**: the next
diagnostic step for these two flights should be background-side — did
`detect_legs` segment them correctly, and does the leg-offset GP's
`correlation_time_s` undershoot a real, fast boundary-layer change on these
specific days? — not more prior-shape work, which is the opposite of what
`726_1`/`728_1`/`728_2` needed. (Followed through in §12: the background-side
lead is now closed — the GP is not the problem, and background modeling for
these two flights is exhausted. §20 and §28 pick up where this section left
off, with the footprint-resolution and land/water-contrast angles.)

## 10. Checking whether the buffer region explains the residual bias

Prompted by a question about how the buffer (`halo_oe/buffer.py`, not
otherwise touched by this investigation before now) behaves when its prior
flux is zero: could the buffer's coarse resolution (`factor=10`, ~10km
super-cells) or finite `outer_bbox` be *causing* some of §9's unexplained
residual bias, rather than being an unrelated nuisance parameter? Checked with
three no-resolve diagnostics (script: `scripts/buffer_bias_check.py`; no inversion was
re-solved — everything here reads the existing `legtest_legoffset_6flight`
bundle plus one cheap `--diagnose-domain` Jacobian stream):

1. **Tile-edge proximity.** Rebuilt the buffer's prior (`build_buffer` +
   `category_priors_on_grid`, matched byte-for-byte against the bundle's saved
   `buffer_membership`) and compared posterior to prior per super-cell. All
   six §9 cluster centroids sit **63–103km inside the core**, i.e. nowhere
   near any buffer super-cell (buffer cells only exist outside the core mask)
   — a co-located tile-edge artifact cannot be the mechanism for any of them.
   The buffer's posterior barely moves from its prior anywhere in the domain
   near these clusters (|posterior − prior| / σ ≈ 0 at the nearest cell to
   each); the one visibly "hot" super-cell in the whole domain sits in the far
   west, ~300km from every flagged cluster.
2. **Out-of-core sensitivity vs. residual, all 6 flights.** Ran the existing
   `run_halo.py --diagnose-domain` (built for exactly this question — see
   `halo_oe/diagnostics.py`'s `out_of_core_sensitivity`) to get each
   receptor's fraction of *explained enhancement* originating outside the
   core mask, joined 1:1 against the bundle's residuals by flight+coordinate
   (counts and coordinates matched exactly for all 6 flights — a real join,
   not an assumption). Pearson correlation between that fraction and
   `|residual|` is weak or **negative** in every flight (`726_1`: −0.29,
   `726_2`: −0.03, `728_1`: +0.01, `728_2`: +0.16, `805`: −0.06, `809`:
   −0.17) — the scatter plots (`figures/buffer_bias_scatter.png`) show the
   largest residuals concentrated at *low* out-of-core fraction, the opposite
   of what the hypothesis predicts.
3. **Spatial check.** The out-of-core fraction (`figures/buffer_bias_map.png`)
   is a clean geographic effect — highest near the domain's NE corner (the
   part of every flight track physically closest to the core boundary),
   identically in *every* flight, including the already-explained `726_1`.
   None of the §9 cluster locations fall in that high-sensitivity region;
   they're all in the low-to-moderate part of the map.

**Verdict: the buffer region is not a plausible explanation for `728_2`'s
broad clusters or `805`/`809`'s leg-banding.** All three checks — spatial
proximity, correlation, and map inspection — point the same way, and none of
them required re-solving the inversion. Given the domain-truncation
correlation from check 2 above never turns positive and meaningful for any
flight, a buffer-disabled/coarsened ablation rerun (~31GB, real compute cost)
is not warranted by these results and is not planned as a follow-up unless
new evidence changes that.

## 11. Checking whether diffuse, multi-category-split prior emissions explain the residual bias

A second hypothesis, prompted by a question about how `category_fields` mode
assigns uncertainty when a cell's emission is small and roughly evenly split
across categories: in this mode, `Sa` is **block-diagonal across categories**
(`pipeline.py`'s `category_covariance`, `[category_uncertainty] default =
1.0` for every category in this config, no per-category override), so the
absolute-flux prior variance at a cell is `Σ_k rel_k² e_{k,i}²` — an
*independent* sum over categories, each scaled only by *its own* density. A
fixed total density split evenly across `N` categories gets `1/N` the
absolute-flux variance of the same total concentrated in one category (by
Cauchy–Schwarz: `Σ e_k² ≤ (Σ e_k)²` with equality only when one term
dominates). That shrinkage is a byproduct of specifying each category's
uncertainty relative to its own density rather than the cell's total density,
not an intentional "diffuse sources are better known" design choice — so a
cell whose true (underestimated) emission is spread thinly across several
categories would get an artificially tight prior band, one the inversion
can't stretch to correct it. Distinct from the zero-prior-mass cases already
found (§5/§9, no density in *any* category) — this targets cells with real,
but split, density.

**Checked** (script: `scripts/diffuse_prior_check.py`, no Jacobian read — pure
`group_fields` read from the bundle) by computing each active cell's
*concentration ratio* `Σ_k e_k² / (Σ_k e_k)²` (1.0 = one category dominates;
0.25 = split evenly across all 4) and comparing it, at both a tight 5km and
each cluster's own span-scale radius, across the six §9 clusters plus two
reference points from §5/§7: `726_1`'s peak (**explained** — a real
`natural_gas` sub-cluster misallocation) as a positive control, and `726_1`'s
dip (unexplained, footprint-shape already ruled out).

**Result is mixed, not a clean hit:**
- `805`'s two clusters (concentration 0.54–0.59) and `809`'s cluster1 (0.48)
  are meaningfully more diffuse than the explained `726_1` peak (0.68) and
  sit near the domain's more-diffuse decile (p10 = 0.47) — a plausible fit.
- But `728_2`'s **both** clusters (0.75, 0.84) and `809`'s cluster0 (0.76)
  are *at or above* the explained peak's own concentration — not diffuse by
  this measure, so the mechanism doesn't fit them.
- `726_1`'s still-unexplained dip is the *most* concentrated location checked
  (0.94 at 5km) — the opposite of diffuse, consistent with it needing a
  different explanation entirely (as §26.2's suggestion for `726_1`'s dip
  already assumed).
- Visually (`figures/diffuse_prior_overview.png`), low concentration (~0.4–0.6)
  is the **norm** across most of the rural/inland domain, not a rare property
  unique to the flagged clusters — only the dense urban/coastal patches are
  strongly one-category-dominated. So "this cell is diffuse" alone is a weak
  discriminator; most of the domain looks like that away from concentrated
  point-source patches, which tempers how much weight this mechanism alone
  can carry as *the* explanation.

**Verdict:** real, demonstrated mechanism, and a plausible partial
contributor for `805`'s clusters and `809`'s cluster1 specifically — but it
does not explain `728_2`'s broad gradient or `809`'s cluster0, which sit at
prior concentrations similar to or higher than an already-explained case. Not
a general answer to "why do these residuals persist," but worth keeping in
mind alongside the correlation-length limit (§9) as a compounding factor
where it does co-occur with low concentration.

## 12. Testing the leg-offset oversmoothing hypothesis for `805`/`809`

§9 reframed `805`/`809` as background-side, not prior-side, and §26.4
proposed the specific mechanism: maybe `fit_leg_offsets`'s GP
(`leg_correlation_time_s = 600s`) oversmooths a real, faster-varying
per-leg background signal. Tested directly (script: `scripts/leg_offset_check.py`,
no re-solve): re-derived `fit_leg_offsets`'s internal raw (pre-GP) per-leg
quantile estimate — exposing an intermediate value the production function
doesn't return — using data already in the bundle (`receptor_background` −
`receptor_background_offset` recovers the plane-only background;
`receptor_obs` the raw observation) plus one honest `fit_mask` re-derivation
per flight (`JacobianFile.receptor_column_sums`, the same cheap streamed
call used twice already in this investigation) and one small `flight_data/
*.h5` read per flight for elapsed time. The re-implementation was verified
exact: the re-derived GP-smoothed offset matched the bundle's saved
`receptor_background_offset` to 1e-17.

**Result: the oversmoothing hypothesis is not supported.** `raw` and
`smooth` are essentially identical for both flights — for `809`, `raw −
smooth` is ~10⁻⁵ ppm on *every* leg (all 10 legs are well-sampled, 31–149
eligible points, comfortably above `min_reliable_points=15`, so the GP has
little to pull toward neighbors); `805` matches closely too, with the one
exception being its first leg (0 eligible points, correctly filled in
entirely from neighbors — that's the estimator working as designed, not a
bug). A shorter `leg_correlation_time_s` would barely change the applied
correction, since there is essentially nothing being smoothed away.

**But a real leg-to-leg oscillation survives anyway**, even restricted to
the domain-insensitive ("clean air") receptors whose final residual isn't
confounded by flux-fitting: ±0.006 ppm (`805`) to ±0.013 ppm (`809`) across
legs (`figures/leg_offset_check_20230805.png`, `..._20230809.png`). Since `raw
≈ smooth` already, this isn't a good correction that got smoothed away —
the per-leg-constant quantile estimator itself never captured it, at any
smoothing setting. **This corroborates §4.2's earlier, independently-derived
finding** (continuous per-receptor kriging also plateaued regardless of
correlation length, 0.05°–10°) — two different methods (aggregate
receptor-level kriging vs. this leg-level, flight-specific check) converge
on the same conclusion: background modeling, in every form tried (plane,
per-leg constant, continuous kriging), has a real ceiling on these flights
that tuning a smoothing parameter further won't cross.

Leg segmentation itself looks reasonable for both flights: 10 legs each.
`figures/leg_segmentation_20230805.png` shows 8+ clean parallel lines.
`figures/leg_segmentation_20230809.png` shows what looks like legs revisiting
similar geography at different times — the exact scenario §2.2's per-leg
offset was built for, not an obvious segmentation bug, though a
time-ordered check would be needed to be fully certain.

**Verdict:** the background-side lead for `805`/`809` is now exhausted —
every background-model variant tried (plane, per-leg constant at any
smoothing setting, continuous kriging) plateaus at the same real, localized
residual. The next step for these two flights should shift back to
prior-shape/correlation-length territory (as already done for `728_2` in
§9), not further background tuning.

## 13. Along-track resolved-scale check: observed vs. modeled variability

A systematic version of §7's one-off gradient-sharpness comparison (which
only checked a single `726_1` hotspot/dip pair): comparing the along-track
variability of the observed enhancement `z` to the modeled `Hx̂` directly
measures the scale below which the model stops resolving real structure,
without needing a specific known feature to anchor on. Checked (script:
`scripts/along_track_scale_check.py`, no re-solve) with two real-elapsed-time-lag-
binned statistics, computed per flight and pooled: the along-track
**autocorrelation** (reusing `plotting._time_binned_autocorr` as-is) and
the along-track **structure function** `D(τ) = ⟨(x(t+τ)−x(t))²⟩` (new, same
binning), plus a noise-floor control — the same structure function
restricted to the domain-insensitive (`fit_mask`) receptors — to separate
real signal from the background-fitting noise §12 already found leaking
into "clean" receptors.

**Result: only `805` shows a clean, trustworthy transport-under-resolution
signature; `809` matches weakly; no other flight does.** Tabulating the
signed autocorrelation gap (`ac_modeled − ac_z`, script:
`scripts/along_track_acf_quant.py`) per lag bin, per flight:

| flight | mean gap, lag≤30s | mean gap, lag>30s | pattern |
|---|---|---|---|
| `805` | +0.17 | +0.27 | strong, consistent positive gap at every lag |
| `726_2` | +0.17 | +0.18 | equally strong positive gap, but see below — this one is an artifact |
| `809` | +0.02 | +0.09 | weak positive, only past ~12s |
| `728_2` | −0.02 | −0.11 | near zero at short lag, mildly negative at long lag |
| `728_1` | −0.17 | −0.17 | consistent *negative* gap |
| `726_1` | −0.15 | −0.26 | consistent negative gap, growing with lag |

`726_2` — already fully explained by leg-offset correction (§3) — looks like
a match for `805` in this table, but is not one: a follow-up check (script:
`scripts/along_track_outlier_check.py`) found `726_2` carries genuinely extreme
excursions (max robust z-score 8.9, 11 points beyond 4σ, vs. 3.3 and 2.9 for
`805`/`809`) that inflate its variance normalization and manufacture the
apparent gap — dropping its 5 most extreme receptors (0.4% of its data)
shrinks the long-lag gap by ~19% (+0.178 → +0.144), while the same test barely
moves `805`/`809` at all. Those 5 points are real point-source crossings, not
measurement artifacts (confirmed below), so this isn't "726_2 has bad data";
it's that a variance-normalized statistic like the ACF is unreliable for a
flight whose variance is dominated by a handful of large real events, which
is exactly `726_2`'s case and not `805`'s. `805`'s gap, by contrast, is
spread broadly across many modest excursions rather than riding on a few
dramatic ones, which is what makes it the trustworthy result here. `728_1`
shows the opposite sign from `805`/`809` entirely, matching `726_1` — the
autocorrelation gap does not track the known good/bad flight split beyond
`805` (and weakly `809`), and should not be read as a six-flight pattern. A
likely reason the sign is this unstable: the ACF normalizes by each series'
own variance, and `Hx̂` is often very flat (low true variance) except where
real flux signal exists, so how much structure a flight's footprints
happened to see that day can flip a low-variance series' standardized
autocorrelation in a way unrelated to model resolution — plausible, though
not tested directly.

**A second check, on excursion classification, gave a materially different
and more accurate description of the same result** (script:
`scripts/along_track_outlier_check2.py`). The "5 most extreme points" dropped from
`726_2` above are not measurement artifacts: a global mean/std threshold is
a poor outlier definition here, since a point on an otherwise unremarkable
stretch of background can be several sigma from the *flight-wide* mean
without being a real excursion, and a genuine point source is physically
extended, so it should imprint on more than one consecutive sample, unlike a
sensor glitch. Redefining excursions relative to each point's own local
neighborhood (median of same-leg samples 8–45s away) and classifying each as
CLUSTERED (a same-sign neighbor within 15s also shows a local excursion) vs.
ISOLATED (no such neighbor) found **90%+ of local excursions in every flight
are clustered** — `726_1`: 18/18, `726_2`: 37/41, `728_1`: 45/49, `728_2`:
54/58, `805`: 13/14, `809`: 15/16 — and dropping only the true isolated
points (0–4 per flight) changes essentially nothing anywhere (`z`'s stddev
moves ≤0.8%, the ACF gap ≤0.02, in every flight including `726_2`). So what
moved `726_2`'s gap above wasn't contamination by artifacts (nearly absent
from all six flights) — it was a handful of **genuine, large, real
point-source crossings** dominating its variance normalization. `805`'s
signal being spread across many modest real events rather than a few
dramatic ones is still the reason to trust it more; "broadly- vs.
narrowly-supported by real events" is the accurate framing, not
"outlier-robust."

**A separate, repeated confirmation fell out of the clustered-event
classification:** at literally every clustered (real) event across all six
flights, `modeled Hx̂` stays small (~0.01–0.03) while `z` swings far larger
(up to ±0.15) — the model essentially never reproduces these sharp real
features at anywhere near full amplitude, in *any* flight. This is §7's
original hotspot/dip finding confirmed at far higher sample size, though
since it happens in every flight — including ones already fully explained
another way — it doesn't by itself distinguish which flights have an
unresolved *net* bias problem.

**The structure function does not add anything beyond the autocorrelation
here.** `D(τ)` is in absolute (ppm²) units, and `Hx̂`'s overall magnitude is
much smaller than `z`'s throughout this investigation, so `D_modeled ≈ 0` at
every lag mostly re-states that known amplitude gap rather than showing a
scale/resolution difference. More informative: pooled and per-flight, `D_z`
(all receptors) and the `fit_mask`-only noise-floor control are **nearly
indistinguishable** at every lag, for every flight
(`figures/along_track_structure*.png`) — at the whole-flight pooled level,
observed along-track variability shows no excess structure attributable to
real plume signal beyond the background noise floor. That's most likely
dilution (most receptors on a long track are far from any point source at
any given moment, so a flight-wide average swamps the localized moments with
real signal) rather than evidence there's no real fine-scale plume structure
— but this test, as built, can't distinguish those two explanations. A
version restricted to domain-*sensitive* (plume-affected) receptors, rather
than pooled over the whole flight, is the natural follow-up for both
statistics if this angle is revisited.

**Verdict:** `805` has a clean, strong, lag-consistent along-track signature
matching transport under-resolution, confirmed to be spread broadly across
many modest excursions rather than riding on a handful of dramatic ones —
the single most solid finding this section produced. `809` matches weakly.
No other flight shows the signature; in particular `726_2`'s apparent match
is a variance-normalization artifact of its own large real point-source
events, not evidence of the same mechanism. This sharpens §7's hypothesis
for `805` specifically, well beyond the single hand-picked `726_1` pair it
was first tested on, but does not extend it to `809`/`728_1` as a group.
(Getting to this required two rounds of correcting an initial, more
sweeping claim — first a plot-based six-flight split that didn't survive
quantification, then an "outlier-sensitivity" framing that had the right
instinct but the wrong mechanism — both folded into the account above
rather than narrated separately; see §25.13 for the methodological lesson.)

## 14. Point-source amplification and MDM correlation-length tests

Prompted by the along-track findings (§13): could the model's failure to
reproduce sharp real excursions be fixed by (a) amplifying the specific
point source's prior flux, or (b) recognizing it as transport/representation
error that belongs in `R` rather than the prior? Both were tested, on
`805` first for (a) and pooled across all 6 flights for (b) — no re-solve
either way.

**(a) Single-cell amplification (scripts: `scripts/point_source_intersection_check.py`,
`scripts/point_source_amplification_check.py`).** Grouped `805`'s clustered
excursions (§13) into 11 discrete events and checked prior-mass overlap:
**0 of 11 have zero prior mass within 10km** — a real contrast with the
`728_1`-style "missing source" clusters in §5/§9, where some residual
locations had no mass in any category at all. Here a source is always
nearby; `modeled` is consistently ~2–5x smaller than peak `|z|` at every
event. That setup looked like a natural fit for "amplify the nearby
source" — but a proper regularized (Bayesian, not unconstrained-least-
-squares) 1-parameter update at the best-sensitivity candidate cell found
**every event's posterior is essentially unchanged from its prior**
(shift under 0.5σ in all 11 cases). The reason is quantitative: even at
peak sensitivity, a single 1km grid cell's leverage on any one receptor
(`H[i,c] × density_c` ≈ 1e-4 ppm per unit scale factor) is far too small
relative to realistic observation noise (`σ_obs = 0.02` ppm) for a handful
of nearby receptors to meaningfully constrain that one cell's flux. This
isn't evidence against a point-source explanation — it means single-cell
amplification, tested this way, lacks the statistical power to say
anything either way, and is consistent with *why* `natural_gas` carries a
5km correlation length in this model to begin with: only a pooled region of
cells, not an isolated pixel, is ever identifiable from a few receptors. An
unconstrained least-squares version of the same test was tried first and
produced wild, obviously-wrong numbers — up to a "needed" scale factor of
−185,848 — before being replaced with the regularized version above; a
reminder that an ill-conditioned small linear solve needs a prior-informed
regularizer, not just more averaging, when the sensitivity is this weak.

**(b) Empirical MDM correlation length near landfill/WWTP sources**
(script: `scripts/landfill_wwtp_coherence_check.py`). `landfill` and `wastewater`
are the two genuinely point-source categories here (`[category_spatial]`
pins both at 0), so an excursion whose nearby prior mass they dominate is
the cleanest available test of real point-source transport coherence.
Reused the §13 local-excursion/event detector, kept only landfill/WWTP-
dominated events (15 of 81 candidate events across all 6 flights — most
excursions are natural_gas/other-associated instead, and `809` contributed
none), then measured each event's along-track decay in real distance from
its peak, pooled into one empirical decay curve. Result: excursions decay
to half-max by **≈2.25km**, noticeably wider than the currently-configured
`[observations] mdm_correlation_length_km = 1.5`. The far tail (>4km)
settles into a persistent, mild negative plateau rather than returning to
zero — flagged as a likely artifact of the 45s (≈10km) local-baseline
window used to define "excursion" in the first place (a ringing effect of
local-median subtraction), not a real long-range anti-correlation, so only
the 0–3km decay should be trusted.

**Verdict:** (a) is a dead end as tested — not disproven, just
statistically powerless with this few receptors per event; a regional
(correlation-length-pooled) version would be needed to say anything
real, as already proposed and not yet built. (b) is a genuine, if modest
(15-event, single-flight-type) and preliminary signal that the current MDM
correlation length may be somewhat too short specifically near landfill/
WWTP sources — a real, data-grounded starting point for a locally-widened
`R`, distinct from the global correlation-length sweep §1 already found
insufficient, though not yet a large enough effect or sample to commit to
implementing without more data.

## 15. Joint prior/observation-error correlation-length sweep

§14's two threads (regional prior amplification, proposed but not built;
empirical MDM coherence length, preliminary) raise a real methodological
question: transport/representation error could in principle be captured
either by loosening the prior's spatial correlation length for landfill/
WWTP cells (`[category_spatial]`, currently pinned at 0) or by widening the
observation-error correlation length (`mdm_correlation_length_km`, currently
1.5km) — these are the two places a Gaussian linear model can absorb
spatially-coherent mismatch, and treating them as *fitting terms* rather
than guessing which one is "right" is the more principled approach: compute
the marginal likelihood of each landfill/WWTP event's local data under a
grid of both length scales together.

**Method (script: `scripts/joint_correlation_sweep.py`, no re-solve):** for each of
the 15 qualifying events (§14b), built a small closed-form Gaussian model
over its nearby candidate-category cells (prior `Sa[c,c'] = σ_prior² exp(-d/
L_prior)`) and elevated receptors (`R[i,i'] = mdm_stddev² exp(-d/L_obs) +
measurement_stddev² I`), marginalized out the local flux perturbation, and
computed `log p(b | L_prior, L_obs)` for `b = z − Hx̂` — summed across all 15
events (independent) at every point on a 4×4 grid (`L_prior` ∈ {0,1,2,3}km ×
`L_obs` ∈ {1.5,2,2.5,3}km).

**Result: the two knobs are not symmetric — one is real, one is inert.**
The likelihood is *exactly* invariant to `L_prior` at every grid point
(verified directly, not assumed: `diag(A Sa Aᵀ)` — the prior's actual
contribution to the model covariance — is `1e-16` to `1e-10` in nearly
every case, against `R`'s diagonal scale of `~7e-4`, six-plus orders of
magnitude smaller, regardless of correlation length or how many nearby
cells carry real density). This generalizes §14a's single-cell finding
rather than contradicting it: pooling sensitivity over a few km of
neighboring cells doesn't help, because the neighbors' own sensitivity ×
density products are just as negligible as the dominant cell's alone was.
`L_obs`, in sharp contrast, drives a real, monotonically increasing
preference: `Δlog-lik = +0.0 / +7.0 / +11.8 / +15.4` at `1.5 / 2.0 / 2.5 /
3.0km`, consistent with (and a more rigorous confirmation of) §14b's direct
decay-curve estimate. The preference is still rising at the edge of the
tested grid (3km) — the sweep shows the *direction* is right, not yet
*where it peaks*.

**Verdict:** a clean, decisive answer to "which side should capture this
error." The observation-error side is genuinely supported by the data; the
prior-correlation side is not, at any scale tested — not because widening
`[category_spatial]` for point-source categories is a bad idea in
principle, but because a landfill/WWTP cell's leverage on nearby receptors
is simply too small, in absolute terms, to be distinguished from
observation noise by a handful of receptors, regardless of how that
leverage is spatially pooled. Reinforces §14's recommendation to pursue a
locally-widened `R`, and closes off the regional-prior-amplification
follow-up (§26.8) as not worth building — the single-cell result it was
meant to extend turns out to generalize instead of being a special case.

**Extending `L_obs` to 10km: the trend keeps going, and that's a caveat, not
just a bigger number.** Re-ran with `L_obs` ∈ {1.5, 2, 2.5, 3, 4, 5, 6, 8,
10}km. Log-likelihood keeps improving at every step out to 10km
(`+0.0/+7.0/+11.8/+15.4/+20.4/+23.6/+25.8/+28.6/+30.1`) — growth
decelerating (roughly halving each doubling of range, consistent with
approaching a well-defined asymptote as `L_obs → ∞`, not a runaway
singularity) but not yet flat. Checked per-event, not just in aggregate:
13 of 15 events improve monotonically all the way to 10km, including
several with 6–7 receptors (enough pairs to meaningfully constrain
correlation, not just a 2-point degenerate fit) — so this is a broad,
real pattern, not an artifact of a couple of small-sample events.

That said, **10km is not a trustworthy point-source correlation-length
estimate, and shouldn't be read as one.** The "elevated receptors" that
define each event are, by construction, almost always drawn from the same
survey leg — and §12 already established that a real, leg-correlated
background bias survives leg-offset correction. A marginal-likelihood fit
like this one cannot distinguish "these receptors are correlated because
of point-source transport smearing" from "these receptors are correlated
because they share the same already-documented, unrelated leg-level bias"
— both look identical to this test. So the sweep's endpoint likely
conflates two different phenomena. The direction (wider than 1.5km) is
robust; the magnitude should be anchored to §14b's more targeted decay-
curve estimate (~2.25km half-max, measured as distance from each event's
own peak, which is far less confounded by generic leg-scale drift) rather
than to wherever this unconstrained sweep happens to stop being run.

**Leg timing, measured directly, confirms the confound is structurally
unavoidable in this test.** Across all 6 flights (52 legs, via
`detect_legs`): median leg duration ≈853s (~14min), median turn gap
(end of one leg to start of the next) ≈260s (~4.3min, range 161–731s),
median leg-to-leg period ≈1125s (~19min). An event's "elevated receptor"
window (±20s padding around a ~10–20s excursion) is tiny next to even the
*shortest* turn gap (161s) — so receptors within a single event are
essentially always confined to one leg and never cross a turn boundary.
The sweep therefore never has a same-event receptor pair spanning two legs
to test against, which is exactly why it can't separate point-source
correlation from leg-level bias: every event, by construction, only ever
samples within-leg correlation. Also useful context on its own: the turn
gap (161–731s) sits comfortably inside `leg_correlation_time_s = 600s`,
consistent with §12's finding that the leg-offset GP had little to smooth
between adjacent legs.

## 16. M3T method-variant sensitivity: does source methodology move the predicted enhancement?

New context: the `m3t` inventory (`m3t_option_1.nc4`, the prior for every run
in this investigation) is generated by
[M3T](/scratch/scrowel3_lab/M3T) (Modular Methane Mapping Tool), a 7-sector
bottom-up inventory builder, via `halo-nyc/build_m3t_prior.py`. Reading the
landfill sector's source (`M3T/python/src/m3t/sectors/landfills.py`)
surfaced a concrete, checkable mechanism for the §5/§9 "zero prior mass"
clusters: landfill emissions come from (a) GHGRP-reporting facilities (point
locations, three method variants — `reported`, `generation_first`,
`collection_first`), (b) LMOP facilities not already in GHGRP, and (c) the
*residual* `GHGI_total − GHGRP_total`, **spread evenly** over non-covered
landfills. A real landfill that neither reports to GHGRP nor appears in LMOP
contributes to the national total only via that diffuse residual term —
never at its true coordinates — which is exactly what "zero prior mass at
the real source's location" would look like in our diagnostics. Wastewater
has a similar three-stream structure (municipal CWNS/DMR × GHGI/Moore,
industrial GHGRP, septic).

**Method (scripts: `run_m3t_landfill_wwtp_variants.py`,
`scripts/m3t_variant_spatial_check.py`; one real M3T run, everything after is
no-resolve).** Landfills has 3 method variants; wastewater has 8
(`{CWNS,DMR} × {GHGI,Moore} × {national,state septic}`) — and both sectors'
code computes *every enabled variant in one run* rather than needing
separate runs, so a single M3T call (domain/grid matched exactly to the
Jacobian via `build_m3t_prior.py print-domain`, landfill + wastewater only,
all variant flags on) produced all 11 rasters directly. Regridded each onto
the Jacobian's native grid (nearest, same convention `build_m3t_prior.py`
uses), then for each of the 15 already-identified landfill/WWTP-dominated
events (§14b/§15), computed the predicted enhancement `H_i · density_variant`
at that event's elevated receptors under every variant of its dominant
category, and compared the variant-induced spread to the event's own
typical unexplained gap (`z − Hx̂`).

**Two distinct results, not one.** First, spatial pattern: landfill's
`reported` and `collection_first` are similar (domain-wide correlation
0.86) but `generation_first` is a genuinely different spatial pattern from
both (correlation 0.25–0.30) — the decay-model choice redistributes *where*
mass sits, not just the total. For wastewater, source (CWNS vs. DMR) is the
dominant axis of spatial difference (correlation ≈0.77–0.80 cross-source)
while method (GHGI/Moore) and septic kind (national/state) barely move the
pattern at all (≈0.98–1.00 within-source). Second, per-event impact: for 14
of the 15 events, variant choice moves the predicted enhancement by a small
fraction of that event's typical unexplained gap (median 3.2%, up to ~22%)
— not a meaningful contributor for most of what's still open.

**But one event is a dramatic, specific exception.** A `728_1` event at
≈(40.556, −74.179) shows a variant-induced prediction range **4.9× larger
than its own typical unexplained gap** — not a marginal effect, a
potentially sufficient one. Traced to a specific cell ~1–3km away
(40.562, −74.20) where `generation_first`'s estimate is **36× larger** than
`reported`'s or `collection_first`'s (16.6 vs. 0.46 µmol m⁻² s⁻¹) — one
facility whose HH-6 first-order-decay estimate is dramatically higher than
its as-reported GHGRP value or its HH-8 collection-efficiency estimate.
Unlike the diffuse, hard-to-act-on "maybe a source is missing" framing in
§5/§9, this is concrete and checkable: a specific facility, a specific
method disagreement, in the specific flight (`728_1`) that's been open the
longest.

**Verdict:** method-variant choice is not, in general, where this
investigation's unexplained residual mass lives — 14 of 15 events barely
move. But it is exactly the right explanation for at least one specific
case, found by testing rather than assumed, and the mechanism (GHGRP
non-reporting facilities falling into a diffuse residual term) gives a
concrete next step for the still-open `728_1`/`728_2` zero-prior-mass
clusters generally: check whether their locations coincide with landfills
present in GHGI/LMOP but absent from GHGRP reporting, the same way this
event's driving facility was found by direct inspection rather than guessed.

## 17. Footprint similarity as a joint function of space and time

New question: with §13–§16 all focused on background, prior-correlation, or
inventory-methodology explanations, how do the footprints *themselves* vary
along a flight track? If two receptors' footprints are similar enough
regardless of when they were sampled, a single shared rotation parameter
(nudging all footprints by the same angle to better fit the data) becomes a
plausible, cheap correction for a systematic wind-direction/transport-angle
error — but that's only worth testing if the underlying footprint shape is
reasonably stable across the flight, not drifting with time.

**Method (script: `scripts/footprint_similarity_space_time.py`).** For two flights
with contrasting residual character (`726_1`: localized hotspot+dip; `805`:
broad leg-to-leg banding), materialized the full (receptor × grid-cell)
footprint matrix on the native STILT grid (not core-restricted, matching
§7.4's convention), computed the full pairwise cosine similarity between
every receptor pair, and binned it jointly by great-circle distance (5km
bins to 150km) and elapsed-time gap (5min bins), masking any bin with fewer
than 5 pairs as unreliable.

**Result: distance dominates; time gap adds essentially nothing once
distance is fixed.** In both flights, high similarity is confined to
separations under ~15–20km — consistent with, and now confirmed by full
pairwise comparison rather than a single hand-picked pair, §7.3's original
along-track decay-length estimate. At the well-sampled fixed-distance
slices (~18km, ~42km), similarity vs. time gap is flat and noisy across the
*entire* flight duration (0 to 150–180min) for both flights — no detectable
within-flight drift in footprint shape at a given spatial separation.
`726_1` and `805` are structurally indistinguishable by this metric despite
their very different residual character. The one hint of a time effect sits
in the sparsest, least trustworthy bin: at ~2km separation, `805` shows only
3 qualifying points (similarity 0.50 → 0.40 → 0.36 as the time gap grows to
~2 hours) — suggestive of a mild decay for genuinely revisited corridors,
but far too thin a sample to treat as established.

**Verdict:** the stationarity result is good news for the rotation idea in
one specific sense — it argues against needing a *time-varying* correction,
since there's no sign of the footprint pattern drifting over the flight —
and supports trying a single, fixed rotation parameter instead of a more
complex one. But this test only compares footprints to *each other*; it
cannot say whether a real, fixed misalignment exists, since a uniformly
mis-rotated flight would look exactly as internally self-consistent as a
correctly-oriented one. That comparison against real data is §18.

## 18. Testing a footprint-rotation correction against real data

§17 established that a single shared rotation parameter is at least a
sensible thing to try (no within-flight drift to complicate it), but left
open whether the data actually support one. This tests it directly: does
rotating footprints about the receptor — a proxy for a systematic wind-
direction/transport-angle error — improve the fit to observations at
locations where a real, checkable answer is possible.

**Method (script: `scripts/rotation_check.py`, no re-solve).** Reused the same 15
landfill/WWTP-dominated events and candidate-cell neighborhoods from
§14b/§15. For each event receptor, read its raw Jacobian row and built a
local bilinear interpolator over a patch wide enough to contain every
candidate cell at every trial angle (rotation preserves distance from the
receptor, so the patch only needs to span the existing candidate-cell search
radius). For a grid of trial angles (−30° to +30°, 2° steps), resampled each
candidate cell's footprint value at its *pre-rotation* position — i.e. what
the unrotated footprint's value would have been, for the rotated footprint
to show this value here — rebuilt `A0_rot = H_rot × density`, and evaluated
the same closed-form marginal likelihood used in §15, holding `L_prior` and
`L_obs` at the current config baseline (0km / 1.5km) so only rotation
varies. `theta = 0` reproduces §15's own baseline log-likelihood exactly, as
a built-in sanity check.

**Result: no effect, and no consistent direction.** Pooled across all 15
events, the best achievable improvement anywhere in the ±30° range is
`Δlog-lik = +0.15` — and that "best" sits right at the edge of the tested
grid, with the *opposite* edge (−30°) giving almost the same gain (+0.13).
The curve is symmetric and still rising at both boundaries, with no interior
peak — the signature of a flat, uninformative likelihood surface, not a real
directional preference. Per-event optimal angles confirm this: of the 15
events, 6 prefer the extreme −30°, 6 prefer the extreme +30°, and only 3
land on an interior value — almost an even split between the two
boundaries, not the shared sign/magnitude a real wind-direction bias would
produce. For scale, §15's `L_obs` sweep bought `Δlog-lik = +30.1` over these
same events — rotation's total effect is about 200× smaller.

**Verdict:** no evidence that a footprint-rotation correction would help, at
least at these point-source events and within ±30°. Combined with §17's
stationarity result, the rotation-parameter idea is closed out as tested-
and-not-supported — a plausible-sounding transport-error mechanism ruled out
by a direct, cheap, no-resolve check, in the same spirit as §10's buffer
check. Scope caveat: this only tested compact point-source neighborhoods; it
says nothing about whether rotation would help `805`'s broad leg-to-leg
banding specifically, but that pattern was already attributed to background/
leg-offset structure (§12), not to point sources, so this isn't a surprising
gap.

## 19. Single-flight vs. joint-fit comparison, and footprint-sensitivity differences across flights

**Motivation.** Every diagnostic through §18 used the joint 6-flight
inversion, which fits **one shared set** of per-cell category scale factors
to all 6 days at once — implicitly assuming the same underlying emission
field (up to those scale factors) explains every day. Two related questions
had never been tested: does that joint-fitting assumption itself manufacture
or mask the persistent residual structure, and — a prerequisite for trusting
any answer to the first question — are the 6 flights' data even comparable,
given each day's footprints depend on that day's specific meteorology?

### Part A: footprint-sensitivity differences across flights

**Method** (script: `scripts/flight_sensitivity_check.py`, no re-solve — one
streamed `JacobianFile.cell_column_sums()` pass per flight, no operator
materialization). For each flight: the raw per-cell sensitivity map,
restricted to the core domain, pairwise cosine-compared across all 6
flights; and, using the same prior-density fields the inversion uses
(`group_priors_on_grid`), each flight's "expected signal" per category
(sensitivity × prior emission, core-integrated) — how much a given day's
data actually *sees* of each source category, independent of any solve.

**Result: real differences exist, and `809` stands out sharply.**
Pairwise cosine similarity of core-sensitivity maps ranges 0.44–0.85 across
the 6 flights — not identical, not orthogonal. `809`'s map is visually
distinct from the other five (`figures/flight_sensitivity_maps.png`): a large
part of the northern/western core (roughly north of 41.25°, west of −75°)
has **exactly zero** sensitivity, a real coverage gap, not just weak
sensitivity, while every other flight has at least some (~1e-8–1e-10)
sensitivity across nearly the whole domain. Per-category expected signal
confirms it: `809` has the lowest signal of all 6 flights for landfill,
natural_gas, *and* wastewater, and its relative composition is skewed — `other`
is 35% of its total signal vs. 14–22% for every other flight, not because it
saw more `other`-category sources but because its point-source-category
signal is so depressed.

**Verdict:** `809`'s single-flight posterior for landfill/natural_gas/
wastewater should be expected to sit close to the prior — not because those
emissions are actually near their prior values that day, but because the
data barely constrain them. Any single-flight-based comparison involving
`809` (including §20's Phase 3) needs this caveat; a big single-vs-joint gap
for `809` in one of those categories would mean "this day's data doesn't
inform this category," not "joint fitting biased this day."

### Part B: does joint fitting itself manufacture the residual structure?

**Method** (script: `scripts/run_single_flight_inversions.sh`, then
`scripts/single_vs_joint_check.py`, no re-solve for the comparison itself). Solved
each of the 6 flights **alone** (single-flight state, no shared information
across days), using `runs/legtest_legoffset_6flight/config.ini` — the exact
settings behind every diagnostic in this document — as the base config, with
`use_leg_offsets` pinned explicitly (matching the reference bundle; the live
`config.ini` had drifted to `false` and `mdm_stddev=0.03` by this point in
the investigation and was only partially reconciled — see §27's config
reference). Compared each flight's single-flight posterior against its own
slice of the joint 6-flight posterior, at both the per-receptor-residual and
per-category-flux level.

**Result 1: per-receptor residuals are essentially identical.** Correlation
between joint and single-flight residuals is r = 0.995–1.000 for all 6
flights, with bias/RMS matching to the 4th decimal in every case
(`figures/single_vs_joint_residual_scatter.png`). Whatever produces the
persistent residual structure in this investigation, it survives completely
unchanged when a flight is solved in total isolation from the other 5 —
joint fitting is not creating it.

**Result 2: category flux totals show the joint fit legitimately pooling
consistent evidence, except for one category.**

| category | joint scale | single-flight scale range (6 flights) |
|---|---|---|
| `other` | 1.000 | 1.000 (untouched, every flight) |
| `natural_gas` | 0.857 | 0.902 – 1.002 |
| `landfill` | 1.055 | **0.898 – 1.211** |
| `wastewater` | 0.482 | 0.507 – 0.975 |

For `natural_gas` and especially `wastewater`, every single flight
independently pulls the scale factor the *same direction* (down from 1.0,
never above) — they only disagree on magnitude. The joint fit (0.857, 0.482)
sitting further from prior than any individual flight is exactly what
correct pooling of agreeing evidence should do, not a symptom of forcing
incompatible days into a compromise. `landfill` is the exception: single-
flight scales span 0.898–1.211, straddling both sides of 1.0 — flights
genuinely disagree on *direction*, not just magnitude, which is the
signature of real day-to-day variability (or at least a materially noisier
per-flight constraint) rather than consistent evidence being diluted by
pooling. It doesn't show up as a residual problem (Result 1), but it's a
real, distinct finding in its own right.

**Cross-check against Part A.** `809` — the flight found to have weak/
restricted sensitivity to landfill, natural_gas, and wastewater — shows
single-flight scale factors of 0.978, 1.002, and 0.975 for exactly those
categories: essentially pinned to the prior, exactly as Part A predicted.
The two independent checks agree.

**Verdict:** joint 6-flight fitting is ruled out as an explanation for the
persistent residual structure — added to rotated winds (§17/§18) and
remaining background (§4/§12) on the ruled-out list. Real day-to-day
emission variability looks plausible for `landfill` specifically, worth
keeping in mind as a distinct, minor thread, but not residual-relevant. The
lasting, practical product of this section is the single-flight bundles
themselves (`runs/single_<fid>`) and the sensitivity caveat on `809`, both
reused directly in §20's Phase 3.

## 20. Attacking hypothesis (a): systematic footprint-shape errors beyond rotation

**Context.** With rotated winds (§17/§18) and remaining background (§4/§12)
both ruled out, three candidate explanations remained for the residual
structure still open at `805`/`809`: (a) systematic footprint-shape errors
beyond rotation, (b) unresolved point sources, (c) systematic errors in the
column data itself. (a) had the most existing, if scattered, supporting
evidence — §13's finding that only `805` (strongly) and `809` (weakly) show
`Hx̂` decorrelating more slowly along-track than `z` — and the most
already-built infrastructure to extend, so it was attacked first via a
three-phase plan, cheapest first, each phase's result scoping whether the
next was worth running.

**Phase 1 — is `805`/`809`'s footprint intrinsically broader than the other
flights?** (script: `scripts/footprint_similarity_space_time.py`, extended from 2 to
all 6 flights; run as its own SLURM allocation via
`scripts/run_phase1_footprint_similarity_sbatch.sh`, ~15GB per flight materialized
and freed in turn). Extracted each flight's along-track footprint
self-similarity half-max decay length (same convention as §14b's empirical
MDM decay estimate), pooled over all time gaps since §17 already found time
gap barely matters at fixed distance. Mechanism being tested: `Hx̂` is a
linear functional of the footprint against a smooth density field, so a
footprint that stays similar over a longer along-track distance would
mechanically produce exactly §13's decorrelation-length signature.

| flight | half-max decay length |
|---|---|
| `726_1` | 7.9 km |
| `728_1` | 8.6 km |
| `728_2` | 10.0 km |
| `805` | 12.1 km |
| `726_2` | 12.9 km |
| `809` | **19.6 km** |

**Split result.** `809` is a clean, unambiguous outlier — nearly 2.5× `726_1`'s
decay length, visibly separated from every other flight at every distance
from 20–150km, not just at the half-max crossing (`runs/
footprint_similarity_decay_1d.png`). `805` is **not** an outlier: its decay
length (12.1km) is shorter than `726_2`'s (12.9km), one of the four flights
this test predicted it should clearly exceed — its curve actually decays
*faster* than `726_2`'s through the 10–40km range. The mechanism this phase
tested confirms for `809`; it does not confirm for `805`.

**Phase 2 — restricting §13's ACF-gap check to plume-affected receptors
only** (script: `scripts/along_track_plume_restricted_check.py`; no Jacobian
read — reuses the saved joint bundle plus `scripts/along_track_outlier_check2.py`'s
clustered-event classification and `joint_correlation_sweep.
group_into_events`, every clustered excursion, not just landfill/WWTP-
dominated ones). §13 pooled its ACF-gap statistic over the whole flight,
which dilutes real localized under-resolution against a long track mostly
far from any source — flagged as its own biggest limitation but never
fixed. Restricting to each event's own extent (±30s padding) instead:

| flight | pooled gap (≤30s) | plume-restricted gap (≤30s) |
|---|---|---|
| `805` | +0.172 | **+0.399** |
| `809` | +0.019 | **+0.406** |
| `726_1` | −0.148 | −0.150 |
| `726_2` | +0.173 | **−0.008** |
| `728_1` | −0.169 | −0.207 |
| `728_2` | −0.016 | −0.000 |

**Clean confirmation, at both flights.** `805` and `809` both sharpen
substantially and land on nearly identical values (+0.399 / +0.406); every
control flight either stays null/negative or — `726_2` specifically — loses
the spurious positive match §13 already attributed to a handful of
variance-inflating real excursions rather than genuine under-resolution.
Long-lag (>30s) numbers were computed too but are not reported as evidence:
a direct pair-count check (not assumed) found the restricted receptor sets
give only ~15–100 pairs per long-lag bin, well under the ~800/bin threshold
§13's own bin-sparsity check established as trustworthy, and several
control flights swung to large spurious positive long-lag gaps under that
sparsity — exactly the kind of small-N artifact this investigation has
learned to check for rather than assume.

**Phase 3 — does artificially coarsening real footprints shrink modeled
amplitude at real events, more for `805`/`809` than a control?** (script:
`scripts/footprint_coarsening_check.py`; no re-solve — reads each flight's own
single-flight bundle, `runs/single_<fid>`, and streams small local Jacobian
patches only). For every clustered event's elevated receptors, Gaussian-
smoothed the raw footprint (σ = 0/1/2/3/5km, mass-conserving) and
recomputed the core contribution to `Hx̂` against that flight's own total
posterior density (`inv.block(name) * ` prior density, summed over
categories — the same construction `halo_oe/plotting.py::_plot_flux_maps`
uses), tracking the shift relative to the unsmoothed baseline. Tested `805`,
`809`, and `726_2` (already fully explained, §3) as a negative control.

| flight | mean \|shrinkage\| at σ=5km |
|---|---|
| `805` | 0.00025 |
| `726_2` (control) | 0.00018 |
| `809` | 0.00003 |

**Not a clean confirmation — a genuine complication.** `809` is *less*
sensitive to added coarsening than the control, and `805` only modestly
more. All three magnitudes are tiny — 0.1–3% of the ~0.01–0.03 ppm baseline
modeled amplitude at these events (§13), nowhere near enough to close the
real gaps (up to ±0.15 ppm) already documented. Most clustered events aren't
landfill/WWTP point sources specifically (this phase deliberately didn't
filter by category, unlike §14–16), so the underlying density near a typical
event is likely fairly diffuse — smoothing the footprint has comparatively
little sharp structure to redistribute, which limits this particular test's
power to detect an amplitude effect even where one might be real.

**Overall verdict: split, not uniform.** `809`'s case is well explained by
footprint-shape/resolution error — Phase 1 (a genuinely broader intrinsic
decay length) and Phase 2 (the along-track signature sharpening under
restriction) are both independently positive for it, and it can reasonably
be treated as closed. `805` is not — it shows the *same symptom* in Phase 2
(and was the stronger, cleaner signature in §13's original finding) but
neither Phase 1 (its footprint isn't intrinsically broad) nor Phase 3
(coarsening barely moves its modeled amplitude) supports footprint
resolution as the mechanism. Two flights with an identical along-track
diagnostic signature most likely have two different underlying causes.
`805`'s residual structure remains genuinely open, and should now be
pointed at (b) unresolved point sources or (c) systematic column-data
errors rather than further footprint-shape work — the next natural step
per the original three-hypothesis framing this plan was built to attack.

## 21. Attacking hypothesis (b): are real point sources missing from the prior?

**Motivation.** §20 closed with a specific recommendation for `805`: check
(b) unresolved point sources or (c) column-data errors, since footprint-shape
work (a) didn't explain it. Separately, reading M3T's sector code directly
(landfills.py, wastewater.py) to answer a general question about the prior-
building pipeline had already identified three plausible mechanisms for a
real facility to be invisible in the prior: a landfill in neither GHGRP nor
LMOP gets no representation at all; an unguarded inner join between a
facility's location table and its emissions table can silently drop a real
reporter; and `m3t_option_1.nc4` was built from one specific method/source
variant per sector (confirmed from the file's own NCO history attribute:
landfill = GHGRP `reported` + LMOP residual; wastewater = CWNS + Moore +
national septic), so a facility only present in an unused variant (e.g.
DMR, not CWNS) would be invisible in the *actual* prior even though the
code path to include it exists. This section tests all of that empirically.

**Method** (script: `scripts/emissions_point_source_audit.py`, no re-solve — reads
M3T's packaged reference datasets, the staged GHGRP facility-location table,
and `m3t_option_1.nc4` directly). For every real, named, geolocated facility
in the raw M3T inputs (GHGRP landfills + LMOP for the landfill sector; CWNS,
DMR, and GHGRP industrial for wastewater) that falls inside the prior's own
grid extent, looked up the built prior's value at that facility's own
coordinates (nearest cell) and flagged any landing on an exact zero.
Facility tables carry multiple years per facility; each facility's most
recent available year was taken as representative — a stated simplification,
since the mechanisms being tested are structural, not annual.

**Two bugs caught before trusting any result — worth recording as their own
lesson.** First pass: `m3t_option_1.nc4`'s `lat` array is descending, not
ascending (confirmed directly, not assumed — strictly monotonic). The
nearest-cell lookup used `np.searchsorted`, which silently assumes ascending
order; with a descending array it ran without error but returned wrong
indices everywhere, producing a nonsense "100% of facilities missing"
result. Caught only because it contradicted §16's already-established fact
that some facilities *are* represented — fixed by sorting the grid at load
time, plus adding a direct sanity check against a manually-verified nonzero
point (`fresh kills landfill`, expected ≈8.11) that now runs on every
execution. Second bug, same pass: the landfill audit didn't replicate
`compute_landfills`' own GHGRP/LMOP de-duplication
(`lmop[~lmop["GHGRP ID"].isin(ghgrp["facility_id"])]`), so a real facility
listed in both datasets under slightly different names (`fresh kills
landfill` / `Fresh Kills SLF`, same `GHGRP ID`) was flagged as missing under
its LMOP entry despite being correctly represented via its GHGRP entry.
Fixed by excluding LMOP rows whose `GHGRP ID` matches any GHGRP facility.

**Result: wastewater is essentially complete; landfill has 10 real gaps.**
1363 real wastewater facilities fall inside the domain; only **1** is not
represented — a DMR-only facility 158km south of the domain near the
Delaware border, clearly irrelevant. The "CWNS chosen over DMR" concern
raised going in doesn't matter empirically: nearly every DMR-only facility
still lands on a nonzero cell, most likely via nearby CWNS coverage or the
septic layer's broad area-based footprint. Landfill: 66 real facilities in
the domain, **10 genuinely missing** — all real, current-format GHGRP
facilities (not LMOP-diffuse cases), none within 5km of a known open
residual cluster, though two sit on Long Island moderately near `728_2`'s
cluster A: **Blydenburgh Road Landfill** (5.1km) and the **Town of
Smithtown Municipal Services Facility** (11.2km). Full lists: `runs/
emissions_point_source_audit_landfill.csv` / `_wastewater.csv`; map:
`figures/emissions_point_source_audit_map.png`.

**The mechanism behind all 10, diagnosed directly rather than guessed.**
Comparing the 10 missing facilities against two confirmed-represented ones
(`pennsauken sanitary landfill`, `edgemere landfill`) makes it unambiguous.
The represented pair report continuously through 2023, no gaps,
`reporting_status` blank throughout. All 10 missing facilities stopped
reporting years ago (last real emissions data 2012–2018) and carry
`reporting_status = "STOPPED_REPORTING_VALID_REASON"` from that point on.
That status is the exact discriminator: `landfills.py`'s non-reporter
carry-forward logic (lines 102–110) rescues only
`"STOPPED_REPORTING_UNKNOWN_REASON"` — and across the whole facility table,
that status accounts for just 757 of 21,187 non-null rows (4%) against
20,430 (96%) for `VALID_REASON`. The carry-forward mechanism was built for
a small minority case; the much larger group — facilities that stopped
reporting for an administratively "valid" reason (falling under a reporting
threshold, decommissioning a gas-collection system, etc.) — gets no
carry-forward and no representation, permanently, once their last active
year is behind the target inventory year.

**Verdict: a real, physically-motivated gap, not a bug.** A landfill that
stops GHGRP reporting for a "valid" administrative reason is treated by M3T
as having zero ongoing emissions from that point forward. Closed landfills
are well documented to keep generating methane from decomposing waste for
years to decades after they stop being required to report — so all 10
facilities found here are plausible, real, ongoing sources this
investigation's prior currently cannot see at all, regardless of which
GHGRP method variant is selected (§16's variant sensitivity test could never
have caught this, since it compares variants that all share the same
non-reporter handling). None is a confirmed smoking gun for `805` or `809`
specifically — the closest lead is `728_2`'s Long Island cluster — but this
closes out part of hypothesis (b) with a concrete, named mechanism, the same
style of result as §16's GHGRP-non-reporter finding, and identifies exactly
which 10 facilities would need independent verification (e.g. satellite or
state-level data) to check whether they're still physically emitting.

## 22. `728_2`'s Long Island gradient: three explanations tried, all ruled out

**Context.** §21 found two real, currently-zero landfills near `728_2`'s
Long Island residual — Blydenburgh Road Landfill (5.1km from the cluster)
and the Town of Smithtown Municipal Services Facility (11.2km) — both
excluded from the prior for the `STOPPED_REPORTING_VALID_REASON` reason
diagnosed there. This section tests whether they (or two alternative
mechanisms it led to) actually explain the residual.

### 22.1 Point-source test: do the two missing landfills explain it?

**Method** (script: `scripts/landfill_gap_sensitivity_check.py`, no re-solve). Same
regularized single-parameter Bayesian amplification test used throughout
this investigation (§14/§16/§18), but anchored to something better than an
arbitrary unit density for once: each facility's own last-reported GHGRP
emissions (before it stopped reporting) as the physically-motivated
reference magnitude, tested under both the `reported` and `generation_first`
methods (§16 already found these can differ 30x+). Prior on the
amplification `x`: `N(0, 1)` — skeptical, `x=0` matches the facility's
current (non-)reporting status, `x=1` means "still emits exactly at its own
historical rate."

**Result: underpowered, and the spatial pattern argues against it
independent of that.** Posterior amplification stayed within 0.7σ of zero
for both facilities under both reference magnitudes, with posterior
uncertainty barely shrunk from the prior — the data added almost no
information. Only 19 of 114 receptors near Blydenburgh Road (35 of 144 near
Smithtown) have any meaningful sensitivity to that one grid cell — the same
single-cell power problem §14a hit originally, now recurring with a much
better-motivated reference magnitude and the same result. More informative
than the fit itself: mapping the actual gap near both landfills shows a
smooth, **monotonic gradient spanning the entire ~50km leg** — strongly
negative in the southwest, strongly positive in the northeast, both
landfills sitting near the zero-crossing roughly by coincidence. A real
point source produces a localized bump decaying in both directions from its
own location, not a trend that runs one direction across an entire leg —
this pattern argues against a point-source explanation regardless of the
statistical power problem.

### 22.2 Background test: does continuous (per-receptor) kriging remove it?

That gradient's shape raised a specific, checkable question: leg-offset
fitting (already on in every bundle this investigation uses —
`use_leg_offsets=true`, confirmed directly) is `fit_leg_offsets`'
per-leg-**constant** kriging, GP-smoothed across *legs* in elapsed time. It
has no mechanism to represent a value varying continuously *along* a single
leg — a monotonic within-leg gradient is structurally invisible to it
regardless of tuning. §4.2 tested a genuinely continuous alternative (a
per-receptor spatial GP, not one constant per leg) for `728_1` and found a
real but plateauing ~11-12% RMS improvement — never re-run for `728_2`.

**Method** (script: `scripts/continuous_kriging_check.py`, no re-solve). Standard GP
regression (exponential kernel over great-circle distance) fit to the
plane-only residual (`receptor_background − receptor_background_offset`,
recovered directly from the saved bundle, no re-fit needed) at the
domain-insensitive (`fit_mask`) receptors, predicted continuously at every
receptor, correlation length swept 5–150km.

**Result: an attractive RMS number that doesn't survive inspection.** Short
lengths give a real-looking RMS improvement (-25.5% full-flight, -22.1% in
the landfill region at 5km) — but mapping it shows the gradient completely
unchanged; the improvement comes from smoothing elsewhere on the flight, not
from resolving this feature. More tellingly, performance gets **worse**
than the current baseline as the length grows past ~10km (+8.9% at 20km,
+32.8% at 150km) — the opposite of what genuine broad-scale background
structure should show, and inconsistent with §4.2's plateau for `728_1`.
That inconsistency is itself diagnostic: a real background process should
fit comparably well (or better) as the correlation length approaches its
own true scale, not degrade monotonically past a short cutoff. Short-length
"improvement" here looks like local overfitting/signal absorption — the
same circularity risk §4.1 already caught once — not genuine background
capture. **Verdict: not a background-explainable feature at any
correlation length tested; instituting the 5km version was explicitly
weighed and rejected** despite its RMS number, precisely because it doesn't
touch the actual problem and risks absorbing real signal.

### 22.3 Flux-side test: does widening `natural_gas`'s prior correlation length explain it?

Of point source / background / flux-side, only the flux-side mechanism —
via the category that's both already-populated *and* already has a nonzero
configured correlation length (`natural_gas`, 5km; `combustion` is
configured too but has zero actual cells in this M3T-based prior, confirmed
directly) — had never been tested against this specific residual using its
real mechanism, as opposed to a hypothetical point source or a
category-blind background surface.

**Method** (script: `scripts/natural_gas_correlation_length_check.py`, no
re-solve). Same regularized marginal-likelihood machinery as §15's
landfill/wastewater sweep, scoped to `natural_gas` cells and `728_2`
receptors near the gradient, with a much larger neighborhood than §15's
point-source tests needed (checked directly for tractable matrix size
before running: 40km cell radius → ~3,900 candidate cells was used; 100km
would have given ~20,000, intractable for the dense `Sigma = A Sa Aᵀ + R`
this test forms at every sweep point). `R`'s own correlation length held
fixed at its current configured value — only the prior side was in
question here.

**Result: technically significant, practically negligible — a clean null,
not a weak one.** Log-likelihood does improve monotonically with `L`, the
same no-peak-within-range pattern as §15's `L_obs` sweep — but the
magnitude is on a different scale entirely: `Δlog-lik = +1.35` at L=150km
(30× the current 5km) vs. §15's `+30.1` for the analogous `R`-side sweep,
about 22× smaller. The gap reduction makes it unambiguous: 0.1% at the
current length, rising to only 0.9% even at L=150km, and that negligible
reduction requires an implausibly large correction (`+160%` at some cells)
to achieve. Mapping the residual at the current length, the best-fit
L=150km, and the completely uncorrected gap side by side shows all three
**visually indistinguishable** — the gradient is untouched at any length
tested.

### Overall verdict

`728_2`'s Long Island gradient has now survived three independent,
well-motivated tests spanning every mechanism the current model has
available: a point source at real, physically-grounded facility locations
(§22.1), background modeling at both the constant-per-leg and continuous
per-receptor level (§22.2), and flux-side adjustment of the one already-
populated diffuse category with room to move (§22.3). None of hypothesis
(a) [ruled out for `728_2`-adjacent flights already, §20], (b) [tested here
directly], or standard background modeling explain it. This leaves (c)
systematic column-data errors — still completely untested, for any
flight — as the one remaining candidate in the original three-hypothesis
framing, or a mechanism outside it entirely.

## 23. Single-day runs with leg offsets, the "best" found parameters, and outlier filtering on/off

**Context: what "best parameters" honestly means.** Of every knob this
investigation tested, only one has a motivated alternative to its current
default — `mdm_correlation_length_km` (§14b's empirical ~2.25km vs. the
configured 1.5km, though §26's own suggestions list flags this explicitly
as preliminary, not implementation-ready). Everything else was *confirmed*
at its default, not improved on: `mdm_stddev`/`measurement_stddev` (§1's
49-job sweep), `category_spatial` `natural_gas=5km` (§15, §22.3), and
`category_uncertainty` default `1.0` (never beaten). Outlier filtering (§6)
has never been turned on anywhere in this investigation — `outlier_threshold
= 0` confirmed directly in every bundle used, including via an empirical
check (`outlier_flag` all-`False` across all 8,078 receptors in the joint
bundle) — so no established "on" threshold existed; `3.0σ` with
`outlier_kind = innovation` (already the configured kind, §6's own
recommended one) was used as a standard default.

**Method** (`scripts/run_single_flight_best_params.sh`: 12 real solves, 6 flights ×
outlier off/on, `mdm_correlation_length_km=2.25` and leg offsets pinned on
via the reference bundle's own config; `scripts/best_params_outlier_check.py`: no
re-solve, compares residual stats and category scale factors against each
other and against the earlier `runs/single_<fid>` baseline — 1.5km, no
outlier filter — so the correlation-length change and the outlier change
are each attributed separately, not conflated).

**Result 1: the outlier filter only trips for the already-explained
flights.** At 3σ innovation: `726_1` — 5 receptors flagged, `726_2` — 6,
`728_1` — 3. `728_2`, `805`, `809` — the three flights with persistent,
still-open residual structure — get exactly **zero** flags each. Their
residual structure isn't concentrated in a handful of points sharp enough
to cross a 3σ innovation threshold; it's smoothly spread out (consistent
with `728_2`'s broad ~50km gradient, §22, which by construction wouldn't
produce isolated spikes).

**Result 2: where it trips, the effect is real but modest, and biggest for
`726_2` specifically.**

| flight | n flagged (3σ) | RMS, filter off | RMS, filter on |
|---|---|---|---|
| `726_1` | 5 | 0.0286 | 0.0279 |
| `726_2` | 6 | 0.0189 | 0.0170 |
| `728_1` | 3 | 0.0342 | 0.0338 |
| `728_2` | 0 | 0.0310 | 0.0310 |
| `805` | 0 | 0.0157 | 0.0157 |
| `809` | 0 | 0.0198 | 0.0198 |

`726_2`'s ~10% RMS drop is the standout, consistent with §13's finding that
it has genuinely extreme excursions (max robust z-score 8.9) inflating its
variance normalization. The other three are identical before/after, exactly
as Result 1 predicts.

**Result 3: the correlation-length change (1.5→2.25km) barely moves
whole-flight RMS anywhere** — every flight's change is under 1% (e.g.
`728_2`: 0.0308 → 0.0310), consistent with §14b/§15's characterization that
its effect is concentrated at specific landfill/WWTP point-source events, a
small fraction of any flight's receptors — too small to register in a
whole-flight aggregate.

**Result 4: but it has a real, interpretable effect on individual category
scale factors** — it pulls the most extreme corrections back toward the
prior, e.g. `728_2`'s landfill scale factor 1.211 (1.5km) → 1.163 (2.25km);
`726_2`'s wastewater 0.709 → 0.750. Expected behavior of a longer MDM
correlation length: nearby correlated residuals count as less independent
evidence, so the fit is less aggressive everywhere it currently moves far
from prior. `other` stays pinned at exactly `1.000` in every one of the 18
bundles compared here (old, off, on × 6 flights) — still completely
unmoved by anything tried against it anywhere in this investigation.

**Verdict:** neither change meaningfully affects `728_2`/`805`/`809`
specifically — the outlier filter has nothing to catch there since their
residual structure isn't concentrated in extreme points, and the
correlation-length change only nudges category totals broadly, not the
underlying still-open residual pattern in those flights. Both matter more
for the already-explained flights than for the open ones — a useful
confirmation that "best parameters" here mostly means *confirmed harmless*,
not *found to fix anything*.

## 24. Why whole legs stand out in the posterior but never get flagged as outliers

**Motivation.** §23 found the production outlier filter never trips for
`728_2`/`805`/`809`. That raised the obvious follow-up: those flights
visibly show entire legs standing out in the posterior residual maps used
throughout this document — so why doesn't the filter catch *any* of that?

### 24.1 The filter is provably per-receptor, not leg-aware

Confirmed directly from source (`goe/outliers.py`), not inferred:

```python
diag_HSaHt = np.einsum("ki,ki->i", Ht, W)   # diag(H Sa Hᵀ) only
return diag_HSaHt + R.diagonal()             # off-diagonal terms discarded
```

`R` is built *with* the along-track MDM correlation (`mdm_correlation_length_km`)
— that correlation is used everywhere else in this investigation (the actual
solve, every marginal-likelihood test, §14–§22) — but the outlier check
specifically discards every off-diagonal entry and compares each receptor
only to its own variance. A leg that's coherently offset by a moderate
amount — every receptor individually only 1–2σ — never crosses a per-point
threshold, no matter how large that same offset is when the whole leg is
considered together: pooling `n` correlated receptors shrinks the effective
uncertainty by roughly `√n` (less than that, to the exact extent they're
correlated), turning a boring single-point deviation into a decisive
aggregate one that the filter has no mechanism to see.

### 24.2 Building the aggregate test directly (script: `scripts/leg_level_outlier_check.py`)

For every leg (`detect_legs`) in every flight, tested whether the leg's
*mean* posterior residual (`z − H·x̂` — the actual quantity in every
residual map in this document, not the filter's own pre-fit quantity, see
§24.3) differs from zero given the leg's real correlated `R`:

```
R_leg[i,j] = mdm_stddev² · exp(−dist(i,j)/mdm_correlation_length_km) + measurement_stddev² · [i=j]
Var(mean)  = (1ᵀ R_leg 1) / n²          # NOT mean(diag(R_leg))/n — must use the full matrix
leg_z      = mean(residuals in leg) / sqrt(Var(mean))
```

using the config's actual `mdm_stddev=0.025`, `measurement_stddev=0.01`,
`mdm_correlation_length_km=1.5` throughout — the same `R` used everywhere
else, just pooled correctly instead of only via its diagonal. (One
simplification: `R` alone stands in for the full `G = H Sa Hᵀ + R` the
production filter's default technically uses, justified by this
investigation's repeated finding, §15/§22.3, that the prior's contribution
to total variance is orders of magnitude smaller than `R`'s.)

**Result: decisive, and sparse in a very specific way.** Of ~52 legs across
the 6 flights, 14 have `|leg_z| > 3`. Only 6 legs have *any* individually-
flagged receptor at all (76 receptors total, out of the whole dataset), and
those 6 split into two different phenomena:

- **Partial catch** (4 legs: `726_1` leg 2, `728_2` leg 0, `728_1` leg 0,
  `728_1` leg 4) — these are exactly the legs with the *largest* leg-z
  scores (13.0, −10.1, −9.5, 8.3), large enough that a handful of their most
  extreme points also individually clear 3σ. But even here the filter only
  catches a minority: `728_1` leg 0 flags 34 of 141 receptors, leaving 107
  still carrying the same coherent bias untouched. Running the filter would
  trim the tail, not fix the leg.
- **Genuinely different problem** (`726_2` leg 5 and leg 3) — here the
  *leg-level* z-score is unremarkable (1.70, 0.52) but one or two sharp
  individual points (5.31σ, 3.17σ) get caught anyway — the filter correctly
  doing its intended job on real isolated spikes, consistent with §13's
  finding that `726_2` specifically has a few genuinely extreme excursions
  rather than broad coherent structure.

The **other 10 of the 14 significant legs** sit in a strict blind spot:
leg-z from 3.2 to 8.0, and not one receptor in any of them ever exceeds
~2.9σ individually (`726_1` leg 0, `809` leg 9, `728_2` leg 1, `728_1` legs
5/6, `726_1` legs 1/3, `809` leg 8, `805` leg 0, `726_2` leg 7). No
threshold on the per-receptor test could ever catch these without also
flagging huge numbers of ordinary points elsewhere — the information only
exists at the pooled level, not the single-receptor one. These legs span
`726_1`, `728_1`, `728_2`, and `809` — i.e. leg-coherent structure this
large isn't unique to the flights this investigation has focused on.

### 24.3 Which of these were already there before fitting? (script: `scripts/leg_level_prior_vs_posterior_check.py`)

Same pooled test, computed at both stages using the *same* `R_leg` (only
the mean shifts between stages): `prior_leg_z` from the innovation
`z − H·xa` (`prior_modeled`, saved directly, independent of `R` by
construction — this is the filter's own pre-fit quantity, in contrast to
§24.2's posterior-stage test), and `post_leg_z` from `z − H·x̂`.

**Result: close to an even split, with a real and informative asymmetry.**
Of 26 legs significant at the prior stage, 12 are substantially resolved by
fitting (43–95% reduction) but 13 remain significant — and several of those
get *worse*, not better: `728_2` leg 0 (−7.53σ → −10.06σ), `728_1` leg 0
(−6.84σ → −9.51σ), `726_1` leg 0 (−5.68σ → −7.99σ), `726_1` leg 1 (−3.06σ →
−4.57σ). This isn't random scatter: every one of the 12 resolved legs
started **positive**; every worsened leg is **negative**, and on the
prior-vs-posterior scatter plot the negative-going legs sit visibly beyond
the 1:1 line while the positive-going ones are pulled inside it toward
zero. Given this is the joint 6-flight bundle with one shared flux state,
the natural reading is real cross-leg (and likely cross-flight) tension:
whatever correction helps explain the positive legs is being pulled in a
direction that actively worsens these specific negative ones, not
independent per-leg noise. One case makes this concrete rather than
inferred: `728_1` leg 5 went from unremarkable (−1.35σ, not significant)
to −4.72σ *after* fitting — a leg the model had no problem with until the
fit created one.

Also notable: this is leg-specific, not flight-uniform, even within the
flights this document has treated as uniformly hard. `728_2` has both
extremes — leg 0 gets worse (−34%) while legs 6, 2, 4, and 7 resolve
cleanly (58–89% reduction). "`728_2` is unexplained" was always too coarse
a description; specific legs within it fit fine, and the genuinely stubborn
residual lives in a subset, not the whole flight.

### Could this improve the production filter?

The natural extension — a correlation-aware, pooled group test sitting
alongside the existing per-receptor one — is straightforward; §24.2/§24.3
already are that test. The real design question is what to *do* when it
fires, and the answer is **not** the same as the per-receptor filter's:
automatically dropping every receptor in a flagged leg would risk exactly
the failure mode §6 already warns against for single points, at a much
larger scale (150+ receptors, several σ of aggregate significance) — a
coherent leg-level bias is far more likely to be real structure (background,
emission, or transport) than a gross error, which is what outlier rejection
is actually for. §24.3 sharpens this further: roughly half of significant
legs *are* explainable by the existing flux model, so blanket rejection
would also discard cases the fit can legitimately handle. Worth keeping as a
diagnostic/flagging tool — not worth wiring into automatic rejection.

## 25. Major takeaways

1. The residual structure left after MDM tuning is real and systematic, not
   noise — proven via cross-config stability, not assumed.
2. Leg-to-leg background drift is real and fixable (leg-offset kriging), and
   materially helps 2 of 6 flights; the other 4 need a different explanation.
3. Any new background-model idea must be tested with the honest
   domain-sensitivity `fit_mask`, or its apparent benefit is likely just
   circularity/signal leakage — this happened twice in this investigation
   (wind-relative axis, initially; caught before being trusted).
4. Prior spatial-shape and spatial-correlation-length limits explain a
   meaningful share of what background modeling alone cannot fix:
   `[category_spatial]` correlation lengths set a hard ceiling on how far the
   flux state can "move" emissions to match an observed feature.
5. Footprints *can* discriminate nearby receptors well (cosine ≈ 0.02 for two
   receptors ~15–20km apart along-track) — the "transport can't tell these
   apart" hypothesis is not supported for this specific pair. Don't generalize
   that conclusion without checking; the flat-response gradient-sharpness
   result for the same pair still stands and needs a different explanation.
6. A visual read of a diagnostic plot is not a substitute for a direct numeric
   check. The corrected `726_1` dip finding (§7.4) is the clearest example in
   this investigation — when a diagnostic finding is surprising *and*
   consequential, verify it a second, independent way before acting on it or
   reporting it as fact.
7. Leg-edge footprint discontinuities (turn-affected release points) are a
   real, minority (~6–10% of legs), independently verifiable phenomenon — now
   a production QC option — but they do not explain the specific
   flights/features that motivated this investigation.
8. Net position: `726_1`'s original hotspot is explained (prior shape).
   `726_1`'s dip and `728_1`'s broad residual gradient remain genuinely open.
9. Extending §5 to the remaining three flights (§9) split them further:
   `728_2` is `728_1`-like (broad gradient, correlation-length-limited plus
   a zero-prior-mass sub-feature), but `805`/`809`'s dominant residual
   pattern is leg-to-leg alternating banding, not a broad gradient or a
   localized prior-shape defect — pointing back at background/leg-offset
   fitting rather than the flux prior for those two flights specifically.
10. The buffer region (§10) is ruled out as a cause of any of the §9
    residual patterns — three independent no-resolve checks (tile-edge
    proximity, correlation, spatial map) all came back negative. Worth
    remembering as a template: a plausible-sounding mechanism can often be
    screened cheaply (no re-solve) before committing to an expensive
    ablation rerun.
11. `category_fields`' block-diagonal-across-categories prior gives cells
    with the same total density but split thinly across categories a
    tighter absolute-flux uncertainty than concentrated cells (§11) — real,
    but only a partial fit to the open clusters (`805`, `809`'s cluster1),
    not `728_2` or `809`'s cluster0. A mechanism being real and demonstrable
    doesn't mean it's *the* explanation for every open case — check each
    location rather than assuming a plausible story generalizes.
12. Item 9's "pointing back at background/leg-offset fitting" lead for
    `805`/`809` was followed through (§12) and closed: the GP applies
    almost exactly what its own raw per-leg estimate says (`raw ≈ smooth`
    on every leg), so there was never a meaningful oversmoothing gap to
    tighten. Background modeling for these two flights is now exhausted —
    every variant tried (plane, per-leg constant, continuous kriging, §4.2)
    plateaus at the same real, localized residual. The lead correctly
    identified *where* to look but the specific mechanism (correlation-time
    oversmoothing) wasn't it; the useful outcome was ruling the whole
    background-side avenue out, converging with §4.2's independent ceiling.
13. The along-track autocorrelation check (§13) was first reported as a
    clean flight-level split (`805`/`809`/`728_1` vs. the rest) from a plot
    read, then corrected by tabulating the actual signed gap per lag: only
    `805` shows a strong, consistent transport-under-resolution signature;
    `809` weakly; `726_2` (already-explained) shows a gap of the *same sign
    and size as `805`*, and `728_1` shows the *opposite* sign entirely. The
    six-flight split doesn't track the known good/bad split and was
    over-generalized from appearance. Both along-track statistics tried
    here (ACF and structure function) turned out to have real, different
    confounds — normalization instability on low-variance series for the
    ACF, raw-amplitude dominance for the structure function — neither
    cleanly isolates "real signal at a given scale" as built. The catch
    itself is the reusable lesson: §25.6's rule applied to this
    investigation's own new work in the same session it was written, not
    just to older, already-cited cases. A further check explained *why*
    `726_2` had matched `805`: a handful of large excursions dominate
    `726_2`'s variance normalization while `805`'s gap is spread across many
    modest ones. That check was itself first framed wrong — a flight-wide
    mean/std-based "outlier" threshold doesn't distinguish a real excursion
    from a point that's merely on an elevated stretch of background, and
    dropping the "top 5 most extreme" turned out to mean dropping genuine
    point-source crossings, not measurement noise. Redefining excursions
    against each point's own local neighborhood, and requiring a same-sign
    neighbor within 15s (a minimum length scale) to count as a real,
    physically-extended event rather than a single-sample artifact, showed
    90%+ of extreme points in *every* flight are such multi-sample events —
    true artifacts are nearly absent from this dataset. Three corrections
    deep on the same finding, `805`'s conclusion held up each time; that
    kind of repeated, independent stress-testing — including the user
    pushing back twice more after the first correction — is what makes a
    single-flight finding trustworthy rather than another appearance to be
    corrected later. One more useful thing fell out along the way: `Hx̂`
    stays flat at essentially every one of these real clustered events, in
    every flight — a much larger-sample confirmation of §7's original
    hypothesis than the one hand-picked `726_1` hotspot it started from,
    even though it doesn't discriminate which flights have unresolved net
    bias.
14. §14's two follow-on tests split cleanly by statistical power, not by
    which sounded more promising going in: single-cell amplification
    (testing whether the *prior/state* side was wrong) had essentially zero
    power with a handful of receptors per event and came back uninformative
    rather than negative — a reminder to check whether a test *can* detect
    an effect before treating its null result as evidence against one.
    Pooling across many cells or many events, not one receptor or one
    pixel, is what gives any of these along-track checks real power — true
    for the leg-offset check (§12), the amplification check, and the MDM
    coherence-length check alike. The correlation-length question (should
    transport error live in `R` or `Sa`?) got a real, if preliminary,
    data-grounded answer in the direction the user's instinct suggested:
    landfill/WWTP excursions cohere over more distance (~2.25km) than the
    model currently assumes (1.5km) — modest evidence, not proof, but
    concrete enough to be worth a larger sample before deciding.
15. §16's M3T method-variant check is a clean illustration of the same
    "check before generalizing" discipline this whole document keeps
    re-learning: 14 of 15 events showed variant choice barely matters, which
    could have been reported as "not a real effect" — except the 15th event
    showed a 4.9x-larger-than-typical-gap swing, traced to one specific
    facility's decay-model disagreement. A single outlier that big, found by
    checking every event rather than a summary statistic, is worth exactly
    as much as the aggregate null result — neither should have been reported
    alone. Also the first time this investigation could trace a specific
    open residual to a specific, named mechanism in the prior's own
    construction (GHGRP non-reporting → diffuse residual term) rather than a
    statistical property of the inversion — a different, more actionable
    kind of finding than anything in §5–§15.
16. §17's footprint-similarity mapping showed distance, not elapsed time,
    is what governs how alike two receptors' footprints are — at fixed
    distance, similarity is flat across the whole flight duration, in both
    a hotspot-dominated flight (`726_1`) and a large-scale-banding one
    (`805`). Useful on its own (footprint shape is stationary within a
    flight, so a time-varying transport correction isn't motivated), and a
    good example of a test whose result is a precondition for a different
    hypothesis (§18) rather than a verdict on that hypothesis itself — a
    self-similarity check can support or undercut a *simpler* model for the
    error, but can't confirm the error exists.
17. §18's rotation test is this investigation's second clean "ruled out by
    a direct check" result (after §10's buffer). The tell that it was a real
    null, not an underpowered one: the improvement (`+0.15` log-lik) was
    tiny *and* symmetric *and* still rising at both edges of the tested
    range, and the per-event optimal angles split evenly between the two
    extremes rather than agreeing with each other — three independent
    signs of a flat, uninformative surface, not one summary number taken on
    faith. Contrast with §15's `L_obs` sweep on the identical 15 events,
    which found a real, ~200×-larger effect — the same events, the same
    machinery, a decisively different answer, which is what makes the
    rotation result trustworthy rather than another case of an underpowered
    test being mistaken for a negative one (§25.14's lesson, applied here
    with a test that had no such power problem).
18. §19 closed out a question this investigation had never actually tested
    despite using it as the foundation of every prior diagnostic: does
    fitting all 6 flights jointly, rather than each day alone, manufacture
    the persistent residual structure? No — single-flight residuals match
    the joint fit's almost exactly (r ≥ 0.995, every flight). The category-
    total comparison that answered *why* is the more interesting result:
    `natural_gas` and `wastewater` show every single flight agreeing on
    correction *direction*, so the joint fit moving further from prior than
    any individual day is genuine evidence pooling, not a forced compromise
    — `landfill` is the one category where flights actually disagree on
    direction (0.898–1.211×), a real, separate, minor finding about day-to-
    day variability that doesn't implicate the residual structure. The
    footprint-sensitivity check that motivated this section paid for itself
    twice: it predicted `809` would show near-prior single-flight scale
    factors for its weakly-sampled categories, and the actual solve
    confirmed it exactly — two independently-built checks agreeing is what
    makes each trustworthy, not either one alone.
19. §20's three-phase attack on hypothesis (a) is this investigation's first
    result that is genuinely split rather than a clean confirm/deny: `809`
    is well explained by footprint-shape/resolution error (a real, outlier-
    broad intrinsic decay length in Phase 1, confirmed independently by
    Phase 2's sharpened ACF signature); `805` shows the *same* along-track
    symptom in Phase 2 — and was the stronger, original signature in §13 —
    but neither Phase 1 (its footprint isn't intrinsically broad; it's
    shorter-decaying than an already-explained control flight) nor Phase 3
    (coarsening barely moves its modeled amplitude) supports the same
    mechanism for it. Worth remembering as its own lesson: two flights
    sharing one diagnostic signature does not mean they share one cause, and
    a plan built to explain both should be allowed to confirm only one.
20. §21's point-source audit is this investigation's clearest example yet of
    verifying a code-reading conclusion empirically before trusting it. Two
    real bugs surfaced only because the check was run and cross-checked, not
    just reasoned about: a descending (not ascending) `lat` array silently
    broke a `searchsorted`-based nearest-cell lookup and produced a nonsense
    "100% missing" result, caught only because it contradicted §16's
    already-established fact that some facilities *are* represented; and an
    unreplicated GHGRP/LMOP de-duplication step flagged an already-covered
    facility as missing under its LMOP alias. Once fixed, the result was
    much smaller and more useful than the initial code-reading guess
    suggested (10 real gaps out of 66 landfill facilities, not a systemic
    problem) — and, unusually for this investigation, the *mechanism* behind
    all 10 was fully diagnosable directly from data (`reporting_status =
    "STOPPED_REPORTING_VALID_REASON"`, a status M3T's own non-reporter
    carry-forward logic doesn't rescue) rather than left as a plausible
    story. A deliberate modeling choice in M3T, not a bug in this
    investigation's own pipeline — but a real, named, checkable gap all the
    same.
21. §28's leg-offset regressor sweep is a good example of a pooled-vs-
    per-flight result actively misleading in opposite directions depending
    which view is taken. Altitude looked like a strong explanation *within*
    `726_1`/`726_2` alone (r=0.74, 0.90) but vanished pooled across flights
    (r=0.10) once the collinearity with elapsed time inside a single
    monotonically-descending flight was accounted for — a within-flight
    correlation that isn't cross-flight evidence. The land/water result ran
    the other way: pooling all 6 flights' receptors together made a real,
    highly significant effect in every single flight (p from 0.016 to
    3e-34) disappear entirely (p=0.20), because the sign of the effect
    flips flight to flight — the per-flight table, not the pooled number,
    was the one that mattered. Same general lesson as §25.11 and §25.19
    (a plausible aggregate can hide or fabricate a real per-case pattern),
    now demonstrated in both directions in one section.
22. §29 is this investigation's first result built by correlating two
    independently-derived per-flight quantities that share no machinery —
    §20's footprint self-similarity decay length (pure STILT Jacobian
    geometry) and this section's mixed-layer height (pure lidar backscatter
    retrieval) — rather than testing one new mechanism against the existing
    residuals directly. That independence is what makes the r=0.91
    correlation meaningful despite only 6 data points: two unrelated
    methods agreeing on the same flight ordering is a different, stronger
    kind of evidence than one method's result looking plausible on its own.
    The explicit leave-one-out check (dropping `809`, the extreme point in
    both variables) is what separated the two MLH metrics that survived
    (mean and max depth, still positively correlated and rank-significant
    without `809`) from the one that didn't (within-survey MLH variability,
    which collapsed from r=0.81 to r=0.19) — the same "don't trust a pooled
    or small-sample number without stress-testing it" discipline as
    §25.14/§25.21, applied here to a correlation instead of a regression
    coefficient or a pooled group difference.
23. §31's wind-shift-timing hypothesis came back null, at both a coarse
    single-point-per-flight test and a properly-resolved per-leg one — and
    that null didn't leave `805` back at "no leads": the same data that
    ruled the hypothesis out also surfaced `805`'s uniquely light-and-erratic
    modeled wind (circular direction std 62° vs. 16–33° elsewhere), found
    only because the checks were run in enough detail to notice a pattern
    outside the specific hypothesis being tested. Confirms §29's
    methodological point from a different angle — a well-motivated
    hypothesis being wrong is still worth testing rigorously, because the
    byproduct of doing so can outlast the hypothesis itself.
24. §30.2→§30.3 is this investigation's most consequential single-point-
    sampling correction, and the clearest illustration yet of why §25.3's
    "any new background-model idea must be tested with the honest fit_mask"
    discipline generalizes beyond background modeling specifically: **any**
    single-point stand-in for a spatially heterogeneous ~250km domain is
    suspect, not just the background-fit case that discipline was
    originally written for. The single-point comparison didn't just have
    a noisier or weaker version of the right answer — it had the *specific
    claim about `805`* backwards (best depth agreement of any flight, used
    as evidence against depth bias explaining its residual, when receptor
    resolution shows `805` has the second-worst mismatch). The correlation
    result built on the same flawed single-point data (§30.2's footprint-
    breadth correlation, `r=0.74`) turned out to be trustworthy anyway
    (§30.3 reproduces it at `r=0.91`) — a reminder that a method's flaw
    doesn't automatically invalidate every result it produced; each claim
    needs checking on its own, not written off or trusted as a bundle.
25. §30.1's confirmation that STILT uses HRRR's `HPBL` close to directly
    (`zicontroltf=0`, `mlht` sampled per-particle, no override of HYSPLIT's
    default of reading mixing depth from the driving model) is the first
    time this investigation traced a footprint-behavior finding (§20, §29)
    back to a specific, named STILT/HRRR configuration choice rather than
    stopping at an empirical correlation. Worth remembering as a category
    of check distinct from everything else in this document: reading the
    actual transport-model configuration, not just re-analyzing its
    outputs, turned a correlation into a mechanism.
26. §30.4's per-flight (not per-day) faceting is what turned a modest
    pooled correlation (`r≈-0.17`, a few percent of variance) into a
    clean, spatially coherent, independently-repeated pattern (up to
    `r=-0.50` within a single flight) — the pooled number alone would have
    been easy to dismiss as noise. The reason this worked, and the reason
    it's trustworthy, is the same reason §25.11/§25.19/§25.21 keep
    recurring in this document: day-to-day (here, weather-to-weather)
    variability was diluting a real within-day signal, not creating a
    fake one, and the only way to tell the difference was to look at each
    flight on its own before pooling, not just check whether pooling
    changed a p-value. `809`'s clean break from the other five flights'
    shared pattern is the same principle again, one level up: a plausible
    reason to worry the whole SW/NE gradient was a track-geometry artifact
    (§30.3's leg-axis caveat) was substantially defused not by a new
    statistical test, but by noticing that the one flight sharing the
    exact same track geometry as the rest was also the one flight that
    *didn't* share the pattern.
27. §30.5's first pass is a useful reminder that even a mechanistically
    better proxy (footprint-weighted water fraction instead of receptor
    position) doesn't automatically help every relationship it's applied
    to — it made the HPBL-bias correlation nearly twice as strong but,
    at leg-level aggregation, appeared to do nothing for the CH4 side.
    Worth remembering as a general caution against assuming "the better
    version of this test will surely show something everywhere": whether
    a more careful measurement helps depends on whether measurement
    crudeness was actually the limiting factor for *that specific*
    relationship. But (see item 28) this particular null didn't survive
    either — a second lesson stacked on the first.
28. §30.5→§30.6 caught an aggregation artifact stacked on top of item 27's
    lesson: the leg-level CH4-residual null wasn't really a null — it was
    52 legs' worth of averaging hiding a real, `r=-0.165`,
    `p=2.5×10⁻⁵⁰` receptor-level (`n=8078`) effect. §30.6 then found *why*
    a linear correlation and a leg-average both nearly missed it: the
    relationship isn't a smooth dose-response like the HPBL bias
    (§30.6's HPBL bins decline steadily across the whole range) — it's a
    threshold effect, flat until roughly the top quartile of footprint
    water fraction and only clearly negative above that. A coarse,
    linear-only test is exactly the kind of check that misses a threshold
    effect concentrated in a minority of the data (here, mostly `728_1`/
    `728_2`) while a smooth dose-response elsewhere in the same
    investigation shows up easily in the same crude test — two real
    effects, same investigation, needing different levels of scrutiny to
    detect, discovered only because both were checked past the first
    (null-looking) pass rather than accepting either result at face value.

## 26. Suggestions for future analysis

1. **`728_1`'s and `728_2`'s zero-prior-mass clusters:** with zero prior mass
   at either residual location in every category, the next step is
   `run_halo.py --compare` against the alternative inventories (EPA,
   Pittsburgh) — if another inventory *does* place mass at either location,
   that's strong evidence of a genuinely missing or misclassified source in
   `m3t` specifically, not a modeling artifact.
2. **`726_1`'s dip:** now that the leg-turn-artifact hypothesis is ruled out,
   it deserves its own look — e.g. check the raw XCH4 measurement's other
   channels/QA flags around that receptor (could be a real, narrow
   atmospheric event), or check meteorology/mixing-height data at that
   specific time/place if available. The footprint-shape toolkit built this
   session (§7.3) already rules out a footprint-shape explanation, so the
   next step should look elsewhere.
3. **Turn on `flag_footprint_discontinuities`** in a real 6-flight run and
   check whether the ~8–12 total flagged receptors overlap with the flights
   that still show poor residuals (`728_1`/`728_2`/`805`/`809`). If they
   cluster there, it's a small but real contributing factor worth keeping on
   by default; if scattered, it's neutral hygiene with no diagnostic value.
4. ~~`805`/`809`'s leg-banding: background-side check~~ **Done, §12 —
   ruled out.** `raw ≈ smooth` on every leg for both flights, so
   `correlation_time_s` was never oversmoothing anything; leg segmentation
   also looked reasonable. Background modeling is exhausted for these two
   flights. Extending the §5 dipole/prior-mass-overlay diagnostic to their
   specific alternating-leg locations (only their strongest residual bins
   were checked in §9, not a full per-leg treatment) is still worth doing.
   §13's along-track check gives `805` specifically (and `809` more weakly)
   an additional, distinct lead — see item 7 — but that check did *not*
   hold up for `728_1` on closer inspection, so treat it as `805`-specific,
   not a shared `805`/`809`/`728_1` mechanism.
5. **Reconsider `[category_spatial]` for categories pinned at 0**
   (`landfill`, `wastewater`, `other`): if an unexplained residual location
   turns out to coincide with a landfill/WWTP whose prior location might be
   slightly off, a small nonzero correlation length (much smaller than
   `natural_gas`'s 5km) could let the inversion nudge it without
   over-smoothing genuinely diagonal point sources. `805`'s cluster 2 (§9,
   centroid ≈40.80, −74.37) and `809`'s cluster 0 both sit adjacent to real
   `other`-category prior mass that the zero correlation length can't reach
   — plausible first candidates to check if this is revisited.
6. **If none of the above resolve `728_1`/`728_2`-style residuals**, the
   `outlier_threshold` mechanism (§6) remains a defensible last resort — but
   only after confirming the flagged points are the same recurring problem
   locations, not scattered/arbitrary, so it isn't silently discarding real
   signal.
7. ~~An explicit footprint-coarsening test~~ **Done, §20 Phase 3 — not a
   clean confirmation.** Rescoped from this item's original `726_1`/`728_1`
   framing (superseded by §13's correction) to `805`/`809` with `726_2` as a
   negative control, exactly as this item anticipated. Result: coarsening
   real footprints up to 5km barely moves modeled amplitude at real events
   for any of the three flights, and `809` — the flight §20 Phase 1
   independently confirmed has a genuinely broader footprint — is actually
   *less* sensitive to added coarsening than the control, not more. The
   along-track *decorrelation-length* version of this idea held up (§20
   Phase 1/2); the *amplitude* version this item specifically proposed did
   not. Still separate from `726_1`'s dip, which remains footprint-shape-
   unrelated (§7.4).
8. **A regional (not single-cell) amplification test** (§14a): the
   single-cell version was statistically powerless (every event's posterior
   moved under 0.5σ from its prior) because one 1km cell's leverage on a
   handful of receptors is far too small relative to observation noise.
   Summing `A_i = H[i,c] × density_c` over all cells within the category's
   actual correlation length around a candidate location (5km for
   `natural_gas`), rather than one cell, would pool enough sensitivity to
   give a real answer — proposed but not yet built.
9. **A locally-widened `R` correlation length near landfill/WWTP sources**
   (§14b, confirmed and sharpened by §15's marginal-likelihood sweep):
   empirical decay of landfill/WWTP-dominated excursions reaches half-max
   at ≈2.25km vs. the configured `mdm_correlation_length_km = 1.5km`, and a
   likelihood-based sweep independently confirms wider is better (broad-
   based across 13/15 events, not a small-sample artifact) but keeps
   improving out to 10km without a clear peak — most likely because
   "elevated receptors" near an event are almost always on the same leg,
   and §12's already-documented leg-level background bias is
   indistinguishable from point-source correlation in this test. Use
   §14b's ~2.25km as the working magnitude estimate, not the sweep's
   unbounded endpoint. Based on only 15 qualifying events pooled across
   all 6 flights (most excursions are natural_gas/other-associated
   instead, and `809` contributed none), so still treat as preliminary,
   not implementation-ready. More events, and a way to separate the
   point-source-specific correlation from the generic leg-level one before
   committing to a spatially-varying `R` implementation, would be the
   needed next step. Distinct from the global correlation-length sweep §1
   already found insufficient — this is a targeted, source-type-specific
   version of the same idea, not a repeat of it.
10. **Identify the specific facility behind §16's `728_1` outlier** and
    check its GHGRP reporting status directly (cross-reference the driving
    cell, ≈40.562, −74.204, against `GHGRP/facility_data.csv` by
    coordinates). If it's a GHGRP reporter, the `reported`-vs-
    `generation_first` disagreement is a genuine methodology question
    (which decay/collection-efficiency equation is right for this specific
    facility) worth raising with M3T directly. If it turns out to be a
    non-reporter whose mass mostly lands in the GHGI-residual "spread
    evenly" term, that's a direct confirmation of §16's proposed mechanism
    for the zero-prior-mass clusters, and worth then checking systematically
    against `728_1`/`728_2`'s other unexplained locations (§5, §9) — do they
    also coincide with GHGI/LMOP-known but GHGRP-non-reporting landfills?
11. **`805`'s residual structure, now that (a) is closed out for it
    specifically (§20):** unlike `809`, none of the three footprint-shape
    tests support a resolution-based explanation for `805`, despite it
    having the original, cleanest along-track ACF signature in §13. The
    next step should be (b) or (c), not further footprint-shape work —
    concretely, check `805`'s specific residual locations against §16's
    GHGRP-non-reporter mechanism (unresolved point sources) the same way
    item 10 proposes for `728_1`/`728_2`, and separately look at `805`'s
    raw column-data QA/retrieval flags (systematic column-data errors,
    §20's hypothesis (c)) — the one candidate explanation this entire
    investigation has never yet tested directly, for any flight. §21's
    point-source audit is a first, general pass at the (b) half of this —
    it found 10 real missing landfills nationally but none within 5km of
    `805`'s own cluster (≈40.80,−74.37); the nearest hits are on Long
    Island, near `728_2` instead. `805` specifically still needs its own
    targeted look, and (c) remains completely untested for every flight.
12. **Read the raw XCH4 retrieval's own QA/channel flags directly** — still
    the one part of hypothesis (c) no script in this investigation has
    touched. §28 approached (c) indirectly via the already-fit leg offsets
    and found two real, independent leads instead of the column data
    itself: an elapsed-time-since-takeoff transient (significant pooled,
    p=0.031, but only individually significant in 2 of 6 flights) and a
    per-flight land/water contrast in the within-leg residual (significant
    in *every* flight, p as low as 3e-34, but sign-flipping by flight in a
    way that argues for a real, synoptically-varying atmospheric cause
    rather than a fixed instrument artifact — §28.5). Neither explains
    `805`, specifically, on its own: `805` is one of the two flights
    *without* an individually-significant elapsed-time trend, though it
    does show one of the two largest land/water splits (alongside `809`).
    Next step: check whether `805`'s land/water-flagged legs coincide with
    its specific open residual locations (§9), and separately pull the
    actual raw retrieval QA flags (cloud screening, weighting-function
    quality, SSE confidence) at those same locations — direct access to
    (c) rather than another indirect proxy.
13. **Extend §29.4's footprint-depth correlation beyond 6 points.** The
    mean/max mixed-layer-height correlation with footprint decay length
    (r=0.79–0.91, Spearman-significant even excluding `809`) is credible
    but structurally limited to one point per flight from this campaign
    alone. ~~A meteorological reanalysis product (e.g. HRRR's own
    boundary-layer-height field...) would give an independent depth
    estimate to cross-check against the lidar-derived MLH used here~~
    **Done, §30 — HRRR's `HPBL` pulled and compared directly, confirmed
    (§30.1) to be essentially the mixing depth STILT's transport actually
    uses.** Reproduces the correlation (r=0.74, Spearman p=0.042) using
    the real driving meteorology, and finds a systematic, same-sign
    depth-overestimate bias on every flight (§30.2) — but at only one
    domain point per flight, still not the many-times/many-locations
    version this item originally asked for; that remains open if more
    statistical power is wanted than six flight-level numbers can give.
14. **Check whether `805`'s open residual sits in a shallower-boundary-
    layer regime than `809`'s.** §29's status update already notes `805`'s
    boundary layer is less extreme than `809`'s, consistent with §20's
    finding that footprint resolution explains `809` but not `805` — worth
    confirming directly (not just by contrast with §29.2's summary table)
    whether `805`'s specific residual locations (§9) sit during its
    survey's shallower or deeper MLH legs, which would sharpen (or rule
    out) a partial footprint-breadth contribution to `805`'s still-open
    case. (§30.2 initially seemed to add a second, independent reason to
    deprioritize depth bias as `805`'s driver — its HRRR-vs-HALO agreement
    looked like the best of any flight — but §30.3's receptor-resolution
    correction reversed that specifically: `805` actually has the
    second-worst mismatch. That removes a reason to rule depth *out*
    without adding one to rule it *in*; this item's original framing
    stands on its own regardless.)
15. **Check `805`'s specific residual locations against its wind-field
    instability, not just its wind-shift timing.** §31 ruled out the
    original wind-*shift*-timing-mismatch hypothesis (no leg-level
    correlation), but surfaced `805`'s uniquely light and directionally
    erratic modeled wind (circular direction std 62° vs. 16–33° elsewhere)
    as an unexamined, better-targeted candidate. Not yet tested against
    `805`'s specific open residual locations (§9) the way item 14 proposes
    for boundary-layer depth — the natural next step, and currently the
    single most promising untested lead for `805` specifically.
16. **Pull the real STILT particle-trajectory `mlht` values, if
    accessible, instead of an independently-fetched copy of HRRR's
    `HPBL`.** §30.1 establishes with high confidence (from this repo's own
    run-script configuration) that STILT uses HRRR's mixing depth close to
    directly, but the actual per-particle `mlht` field from the original
    runs (which would live in `out_hrrr/` on the HPC scratch space named
    in `stilt/run_stilt_nyc_hrrr.r`, not available in this local repo) has
    never been read. That would remove any remaining ambiguity about
    whether the Zarr-archive copy of `HPBL` used in §30.2 matches what
    STILT's ARL-converted input actually contained, and would settle
    §30.1's one open caveat (drawn from HYSPLIT/STILT's documented general
    behavior, not verified against the STILT package's own source, which
    isn't checked out in this repo).
17. **Read the raw XCH4 retrieval's own QA/channel flags directly — now
    the single most overdue item on this list.** Every meteorological
    thread pursued in §28–§31 (elapsed time, land/water, mixed-layer
    height, HRRR boundary-layer bias, wind speed/direction/shift) has come
    back either null or only a partial, indirect proxy for hypothesis (c).
    None of them has actually opened the column-data files' own QA fields.
    That remains completely untouched for every flight, and is now the
    most direct remaining path to (c) rather than another indirect
    meteorological angle.
18. **Check whether `728_1`/`728_2`'s specific open residual locations (§9)
    coincide with their highest-footprint-water-fraction receptors.**
    §30.6 found a real, threshold-like effect of footprint water fraction
    on the post-fit residual, concentrated above roughly the top quartile
    of footprint water fraction and statistically significant specifically
    for `728_1`/`728_2` (§30.5's majority-water-vs-land table) — but this
    was tested against the *whole-flight* residual, not against those two
    flights' own already-identified broad-gradient clusters (§5, §9). If
    the high-water-fraction receptors driving this effect spatially
    overlap those clusters, that would be a much stronger, spatially
    specific result than the current whole-flight statistical one; if they
    don't overlap, the effect is real but unrelated to this document's
    original motivating residuals for those two flights, and would need
    its own explanation for what it *is* connected to.

## 27. Configuration reference

The findings above are scattered across 27 sections built up over an
extended investigation; this pulls every `config.ini` knob that was
actually exercised into one place, organized by section header, each with
what testing (or just using) it told us and where to read the detail. Not a
new finding — a map of the ones already made.

**`[background]`** — background subtraction (§2)
- `method = planar`, `domain_sensitivity_quantile = 0.5`: the flight-wide
  plane and the `fit_mask` it's restricted to (§2.1). The `fit_mask`
  protocol — any new background idea must be tested with it honestly, never
  skipped for a quick read — is the single most-repeated lesson in this
  investigation (§4.1, established; violated and caught again in §25.6).
- `use_leg_offsets = true`: this investigation's main background addition (§2.2).
  Fixed `726_1`/`726_2` cleanly; `728_1`/`728_2`/`805`/`809` retained
  substantial structure regardless (§3) — the split that motivated
  everything from §4 onward.
- `leg_gap_seconds = 8.0`, `leg_min_size = 10`, `leg_axis_deg = 45.0`
  (`detect_legs` params): segmentation checked directly for `805`/`809` and
  found reasonable (§12). Measured leg timing directly (§15 addendum, 52
  legs across all 6 flights): median leg duration ≈853s, turn gap ≈260s
  (range 161–731s), leg-to-leg period ≈1125s.
- `leg_offset_stddev = 0.05`, `leg_correlation_time_s = 600.0`,
  `leg_offset_noise_stddev = 0.02`, `leg_min_reliable_points = 15`
  (`fit_leg_offsets` GP params): tested directly for `805`/`809` (§12) —
  the GP-smoothed offset matched its own raw per-leg estimate almost
  exactly on every leg (`raw ≈ smooth`), so `correlation_time_s` was never
  oversmoothing anything; background modeling is exhausted for these two
  flights regardless of this parameter's value.
- `flight_data_dir`: required to recover real elapsed time (the Jacobian
  files carry none) — the dependency underlying every along-track analysis
  from §2.2 through §15.
- `flag_footprint_discontinuities = false` (default/off),
  `discontinuity_relative_threshold = 0.5`, `discontinuity_min_leg_size =
  6`: new production QC feature (§8) for turn-affected release points
  (~6–10% of legs, real but doesn't explain this investigation's
  motivating flights). Never turned on in a real multi-flight run — still
  open (§26.3).

**`[category_spatial]`** — prior spatial correlation length (§5, §14, §15)
- `default = 0`, `natural_gas = 5`, `combustion = 5` (km): sets the hard
  ceiling on how far the posterior can relocate flux between cells.
  Directly explains the §5 split — `726_1`'s ~50km feature was within
  reach of several 5km-correlated `natural_gas` cells acting together;
  `728_1`'s ~170km gradient wasn't, regardless of prior shape.
- `landfill`/`wastewater`/`other` implicitly pinned at the `0` default
  (true point sources in this model): tested whether loosening this would
  help explain landfill/WWTP-associated excursions (§14a single-cell,
  §15's swept extension to regional pooling) — found genuinely inert at
  every scale tried (0–3km), not just unhelpful: `diag(A·Sa·Aᵀ)`, the
  prior's actual contribution, was `1e-16`–`1e-10` against `R`'s `~7e-4`,
  six-plus orders of magnitude too small to matter regardless of
  correlation length or how many neighboring cells carry real density.

**`[category_uncertainty]`** — relative prior uncertainty per category
- `default = 1.0` (relative stddev on each cell's scale factor): drives
  §11's diffuse-emissions finding directly — because `Sa` is block-diagonal
  across categories, a fixed total density split evenly across N
  categories gets `1/N` the absolute-flux variance of the same total
  concentrated in one, an artifact of specifying uncertainty relative to
  each category's own density rather than the cell's total. Real, but only
  a partial fit to the still-open clusters (§11's verdict). Also the
  `sigma_prior` used directly in §14a/§15's regularized point-source tests.

**`[observations]`** — error model (§1, §6, §14, §15)
- `error_model = components`, `mdm_stddev = 0.025`, `measurement_stddev =
  0.01`: the components of `R` used directly in §15's joint sweep kernel.
- `error_stddev = 0.02`: used as `sigma_obs` in §14a's single-cell
  amplification test — the number that made a single grid cell's leverage
  (`~1e-4` ppm/unit-scale) look negligible by comparison.
- `mdm_correlation_length_km = 1.5`: the original subject of the 49-job
  sweep (§1) that found **>99% of residual variance config-invariant**
  across 1–20km — the result that justified moving off MDM tuning entirely
  and into everything from §2 onward. Revisited in a *targeted* form in
  §14b/§15: landfill/WWTP-associated excursions empirically cohere over
  ≈2.25km (half-max), and a marginal-likelihood sweep confirms wider is
  robustly better (broad-based across events) — but the sweep's own
  endpoint (still improving at 10km) isn't trustworthy as a magnitude
  estimate, since it's confounded with leg-level bias that has nothing to
  do with point sources (§15 addendum). Not a repeat of §1's global
  sweep — that tested one uniform value everywhere; this is source-type-
  specific and still preliminary (15 events, not implementation-ready).
- `outlier_threshold = 0` (off), `outlier_kind = innovation`: discussed
  (§6) as a pragmatic fallback for individually bad points, explicitly
  **not** a fix for spatially-coherent systematic structure. Never turned
  on; remains a defensible last resort (§26.6) only after confirming
  flagged points recur at the same locations.

**`[buffer]`** — out-of-core flux representation (§10)
- `enabled = true`, `mode = coarse`, `factor = 10`, `outer_bbox = [39, 43,
  -77.5, -70]`, `stddev = 1.0`, `stddev_floor = 0.0`: tested as a possible
  cause of the residual bias across three independent no-resolve checks
  (tile-edge proximity, out-of-core-sensitivity correlation, spatial
  mapping) — ruled out cleanly on all three; not revisited since.

**`[domain]`** — core extent
- `bbox = [39.9, 42.1, -76.9, -72.5]`: the region every correlation-length
  reach, cluster-to-boundary distance, and out-of-core sensitivity number
  in §5, §9, and §10 is measured relative to.

**`[flux]` / `[decomposition]`**
- `unit_scale = 1.0`; `method = category_fields`: the decomposition mode
  actually run throughout — each category gets its own per-cell
  **multiplicative scale factor** (prior mean `xa = 1.0` uniformly), a
  distinct quantity from the `group_fields` density map used for
  grouping/plotting. Conflating the two was a real bug caught mid-session
  (§9) — any new diagnostic touching posterior flux must multiply by
  density to get an actual flux perturbation, not diff the raw state
  block against the density map directly.

## 28. Leg-offset regressors: elapsed-time drift, altitude, terrain, and land/water

§20's three-phase attack closed out hypothesis (a) (footprint shape) for
`809` but not `805`, and §25's takeaways leave hypothesis (c) — systematic
errors in the raw column data itself — as the one candidate this entire
investigation had never tested, for any flight. This section is a first,
indirect attempt at (c): rather than the raw XCH4 retrieval and its QA
flags directly (not yet accessed by any script in this investigation), it
asks whether the already-fit **leg offsets** (§2.2) — the per-leg additive
background level, estimated independently of the flux solve — carry a
systematic dependence on anything instrument- or surface-side, as opposed
to being pure atmospheric noise. A dependence there would be indirect
evidence for something upstream of the inversion (instrument response or a
surface-dependent retrieval effect) rather than a modeling gap in
`halo_oe` itself.

### 28.1 Elapsed time since *actual* takeoff, not since the NYC-clipped window starts

**Correction made before anything else:** the Jacobian-aligned flight files
in `flight_data/` are clipped to the NYC lawnmower survey only — they drop
each flight's climb-out and transit leg entirely. The gap between true
takeoff (`Nav_Data/gps_time`, minimum, in the unclipped `_full.h5` product)
and the first receptor in the clipped file is not small or uniform: 63.0
min (`726_1`), 37.1 min (`726_2`), 38.2 min (`728_1`), 36.8 min (`728_2`),
70.8 min (`805`), 65.3 min (`809`). Any elapsed-time analysis using the
clipped file's own first timestamp as "time zero" is silently measuring
time since an arbitrary, flight-varying point, not time since takeoff — a
mistake worth flagging explicitly since it is easy to make and does not
fail loudly. All elapsed-time numbers below are corrected for this.

### 28.2 Method (script: `scripts/leg_offset_drift_check.py`, no re-solve)

Reads `runs/legtest_legoffset_6flight`'s saved `receptor_background_offset`
(the already-fit, GP-smoothed leg offset — §2.2) directly from the bundle
for all 6 flights; no Jacobian, no re-fit. For each flight, re-derives leg
segmentation (`detect_legs`, same config) and averages the offset, elapsed
time (§28.1), `flight_alt`, and — newly pulled from the unclipped file —
`UserInput/DEM_altitude` (ground elevation, matched to each clipped
receptor by nearest GPS time) per leg, giving one row per leg (52 legs
across 6 flights). Every regression below pools across flights after
subtracting each flight's own mean offset, so it tests whether a *shape*
of dependence is shared across flights with very different absolute
background levels, tracks, and start times — not whether any one flight
alone shows a trend.

### 28.3 Results: elapsed time and terrain elevation are real; altitude and clock time are not

| regressor | pooled r | pooled p | note |
|---|---|---|---|
| elapsed time since takeoff | −0.299 | 0.031 | only 2 of 6 flights individually significant (`726_1` p=0.038, `726_2` p=0.0008); `728_2` marginal (p=0.060) |
| absolute clock time (UTC) | −0.120 | 0.398 | not significant — argues against a diurnal/boundary-layer explanation |
| aircraft altitude (MSL) | +0.098 | 0.49 | not significant pooled, despite looking strong *within* `726_1`/`726_2` alone (r=0.74, r=0.90) — those two flights simply descend monotonically through their whole survey, so altitude and elapsed time are collinear within them; across flights with different descent profiles the altitude relationship does not hold up |
| surface (DEM) elevation under receptor | −0.306 | 0.027 | as strong as elapsed time, and **not** the same signal: corr(elapsed time, surface elevation) = +0.08 pooled, and a joint regression keeps both terms significant independently (elapsed time p=0.040, surface elevation p=0.035, joint R²=0.17, n=52) |
| altitude above ground level (range-to-target) | +0.107 | 0.448 | not significant — rules out a laser-range/path-length explanation specifically |

Two independent, non-redundant leads survive: elapsed time since takeoff,
and terrain elevation under the receptor. Both are modest in absolute
size (elapsed time: −0.007 ppm/hr; terrain: −0.25 ppm/km, i.e. roughly a
0.02–0.05 ppm swing across the NYC domain's ~0–200m relief) and neither
alone explains much variance — but both are statistically real and clearly
distinct mechanisms, worth carrying forward separately rather than
collapsing into one story.

### 28.4 Is the terrain-elevation effect actually a land/water split? (script: `scripts/land_water_background_check.py`)

`DEM_altitude` is cleanly bimodal over the NYC domain: land receptors carry
a real GLOBE elevation (≈10m and up), water receptors (harbor, rivers,
Long Island Sound) carry a flat `-1.0` sentinel with nothing in between —
so a simple threshold (`WATER_SENTINEL = -0.5`) gives an unambiguous
per-receptor land/water flag, motivating a direct check of whether §28.3's
continuous terrain-elevation result is really a land/water contrast.

**Leg-level (each leg's water fraction vs. its fitted offset, same pooling
as §28.3): no.** Pooled alone: r=+0.057, p=0.69. Added to the elapsed-time
regression it contributes nothing (p=0.51, vs. elapsed time still p=0.028).
So §28.3's terrain-elevation result is driven by continuous relief among
*land* legs (e.g. coastal-flatland vs. higher inland terrain), not a
land/water step.

**Receptor-level, within-leg (the sharper test): yes, and strongly — but
with a sign that flips by flight.** For every receptor, take the
pre-leg-smoothing residual (`obs - plane_background`, the same quantity
`fit_leg_offsets` operates on — §2.2) and subtract that leg's own mean,
removing the leg's overall fitted level entirely; this asks only whether
water and land receptors disagree systematically *within* a leg, using
every receptor (not one number per leg) and needing no cross-flight
pooling to have power:

| flight | n water | n land | mean water (ppm) | mean land (ppm) | diff (ppm) | t | p |
|---|---|---|---|---|---|---|---|
| `726_1` | 297 | 822 | −0.0025 | +0.0009 | −0.0034 | −2.41 | 0.016 |
| `726_2` | 261 | 776 | +0.0032 | −0.0011 | +0.0043 | +2.66 | 0.008 |
| `728_1` | 281 | 720 | −0.0048 | +0.0019 | −0.0067 | −3.55 | 0.0004 |
| `728_2` | 297 | 832 | −0.0125 | +0.0045 | −0.0169 | −10.26 | 6e-23 |
| `805` | 400 | 945 | +0.0029 | −0.0012 | +0.0041 | +5.07 | 5e-7 |
| `809` | 245 | 876 | +0.0114 | −0.0032 | +0.0146 | +13.39 | 3e-34 |

Every flight shows a highly significant land/water contrast within its own
legs — this is not noise. But the sign is not consistent: negative for
`726_1`/`728_1`/`728_2`, positive for `726_2`/`805`/`809`, with no evident
pattern by date, by flight-of-day (F1 vs. F2), or by local time of day.
Naively pooling all 6 flights' receptors together (ignoring flight
identity) makes the effect vanish (diff=−0.0008 ppm, t=−1.29, p=0.20) —
the signs cancel — which is why the per-flight table above, not a single
pooled number, is the result that matters here; see
`figures/land_water_background_check.png` for both views side by side.

### 28.5 Interpretation

A land/water contrast that is real and large on every flight but flips
sign between flights is inconsistent with a fixed instrument or retrieval
artifact (a genuine surface-reflectivity or lidar-return bias tied to
water vs. land should point the same direction every time it's present).
It is much more consistent with a real, synoptically-varying atmospheric
effect — e.g. land/sea-breeze circulation or day-specific boundary-layer
contrast between the harbor/Sound and the urban/inland surface, either of
which would plausibly flip sign with wind direction and synoptic pattern
on a given flight day. That reframes this finding: not a bug in the
background model or retrieval to fix, but a real atmospheric structure
that a single per-leg constant offset (§2.2) is not built to represent,
and a plausible *contributor* — not the resolution — to `805`'s and
`809`'s still-open residual structure, since both show the two largest
land/water splits of the six flights.

Net addition to hypothesis (c): still not directly tested (no script in
this investigation has yet read the raw XCH4 retrieval's own QA/channel
flags), but two indirect leads are now on the table with real statistical
support — an elapsed-time-since-takeoff transient present in roughly half
the flights, and a per-flight, sign-varying land/water contrast present in
all six — that a future column-data QA check should be run alongside, not
instead of.

## 29. Mixed-layer height: a real land/water mechanism, and a footprint-breadth correlation

§28 left the land/water contrast in the leg offsets attributed to "a real,
synoptically-varying atmospheric effect" without a measured mechanism
behind it. This section pulls a real boundary-layer-depth product
(`CH4DataProducts/MixedLayerHeight`, a lidar backscatter-derived mixed-layer
height, `m AGL`, present in the unclipped flight files but not used
anywhere else in this investigation) to test that directly, motivated by a
question about whether the six flights' split into three morning surveys
(~9:30am–12:45pm EDT: `726_1`, `728_1`, `805`) and three afternoon surveys
(~1–4:45pm EDT: `726_2`, `728_2`, `809`) shows a "complex transport"
signature — and finds one, though not primarily a morning/afternoon one.

### 29.1 The dominant MLH signal is land/water, not time of day

**Method** (script: `scripts/mixed_layer_height_check.py`, no re-solve, no
Jacobian). For each flight, matched `MixedLayerHeight` to the NYC survey
window (the clipped file's own time range) and binned it into 10-minute
medians across that window, split into the morning/afternoon groups above.

**Result: every flight's raw MLH time series is a sawtooth, not a smooth
diurnal curve** (`figures/mixed_layer_height_check.png`) — swings of several
hundred meters on a ~10–15 minute cycle, too fast to be boundary-layer
growth. That cycle matches the lawnmower survey's leg-crossing cadence, not
time of day, and a direct check confirms why: **every one of the 6
flights, without exception, shows the boundary layer running substantially
shallower over water than over land** — 407 to 625m shallower, every
flight individually significant at `p ≈ 0` (Welch's t, `t` from −46 to
−163, `n` in the thousands per class per flight). This is a real,
physically-expected marine/coastal-vs-continental contrast, and it
directly grounds §28.4's land/water finding: every flight really does fly
through a systematically different boundary-layer depth over water, giving
a physical mechanism for why *some* land/water CH4-background contrast
should exist on every flight, even though (§28.5) the specific *sign* of
that contrast varies flight to flight in a way more consistent with
day-specific wind direction than a fixed depth effect on its own (§29.3
tests, and rules out, MLH depth itself as a direct predictor of that sign).

### 29.2 After removing the land/water step, morning flights show real growth; afternoon flights do not behave uniformly

**Method.** Same binning, restricted to land-only receptors
(`DEM_altitude > -0.5`) to remove the §29.1 step and isolate any real
within-flight temporal trend, with a linear fit against elapsed local time.

| flight | group | land MLH start → end (m) | slope (m/hr) | p |
|---|---|---|---|---|
| `726_1` | morning | 553 → 689 | +149 | 0.078 |
| `728_1` | morning | 564 → 1129 | +207 | 0.007 |
| `805` | morning | 868 → 1259 | +86 | 0.012 |
| `726_2` | afternoon | 1059 → 1647 | +157 | 0.018 |
| `728_2` | afternoon | 1435 → 1427 | −6 | 0.934 |
| `809` | afternoon | 1810 → 1608 | −144 | 0.071 |

All three morning flights show real (mostly significant) boundary-layer
growth through their survey window — the expected morning-growth
signature, present on every morning flight. The afternoon flights are the
more genuinely "complex" group, and not uniform: `726_2` is still growing
significantly as late as 4:42pm; `728_2` has already plateaued by 2:17pm;
`809` — the flight whose survey starts closest to solar noon (1:01pm) and
runs into mid-afternoon (4:09pm), matching the original question most
closely — starts at the *highest* boundary-layer depth of any flight all
day (1810m) and *declines* through its survey window, a transitioning
regime none of the other five flights show.

### 29.3 MLH itself does not predict leg-level CH4 bias, linear or quadratic

**Method** (script: `scripts/mlh_bias_regression_check.py`, no re-solve). Per leg,
per flight, regressed both the fitted leg background offset (§2.2/§28) and
the posterior residual (`z − Hx̂`) against land-only MLH and against
elapsed time since takeoff, fitting linear and quadratic models (pooled
across flights, flight-demeaned, same convention as §28) and F-testing
whether the quadratic term earns its keep over a straight line.

**Result: a clean null for MLH itself, in both bias targets, at both
orders.** Background offset vs. MLH: `R² = 0.003`, `p = 0.71` (linear);
quadratic term `p = 0.99`. Posterior residual vs. MLH: `R² = 0.006`,
`p = 0.59`; quadratic term `p = 0.88`. Visually
(`figures/mlh_bias_regression_check.png`) both MLH panels are flat noise —
no relationship at any curvature tried. This is worth stating plainly: even
though land/water crossing drives a large, universal boundary-layer-depth
swing (§29.1), that depth's *magnitude* does not leave a detectable,
functional-form-agnostic imprint on either bias target — if boundary-layer
structure drives the land/water CH4 effect, it is not through mean leg-level
depth in any simple form, more likely through wind/advection direction (as
§28.5 already suspected).

Elapsed time reproduces §28's linear result for the background offset
exactly (`slope = −0.0069` ppm/hr, `R² = 0.089`, `p = 0.031`); a quadratic
term adds curvature (`R² → 0.121`) but is not significant (`p = 0.19`).
The one place curvature shows up at all is the posterior residual against
elapsed time (linear `p = 0.15`, not significant; quadratic term
`p = 0.073`, `R² : 0.041 → 0.102`) — a hump shape, worth flagging as
suggestive but not established, since at `n = 52` pooled legs it is
visibly shaped by a couple of points at the elapsed-time extremes
(`728_1`'s two earliest legs, `809`'s latest legs) rather than a trend most
legs participate in.

### 29.4 Footprint breadth correlates with boundary-layer depth across all 6 flights

**Motivation.** §20 Phase 1 found `809`'s footprint self-similarity decay
length (19.6km) a clean outlier against the other five flights (7.9–12.9km),
with no mechanism offered beyond "a real, outlier-broad intrinsic decay
length." §29.2 independently found `809` also has the deepest and only
declining boundary layer of the six flights. Two independently-derived
per-flight quantities — one built purely from STILT footprint cosine
similarity, the other purely from the lidar backscatter product, with no
shared machinery — agreeing would be real evidence that boundary-layer
depth drives footprint breadth, not a coincidence specific to `809`.

**Method** (script: `scripts/footprint_mlh_correlation_check.py`, no re-solve). Six
data points (one per flight): §20's decay length against each flight's
mean and max land-only MLH during its survey window (§29.2's summary
statistics), Pearson correlation plus an explicit leave-one-out check
dropping `809` — the extreme point in both variables — to separate a real
trend across all 6 flights from a coincidence involving only `809`.

| metric | r (all 6) | p (all 6) | r (excl. `809`) | p (excl. `809`) | Spearman ρ | p |
|---|---|---|---|---|---|---|
| mean land-only MLH | +0.79 | 0.061 | +0.68 | 0.20 | +0.83 | 0.042 |
| max land-only MLH | **+0.91** | **0.013** | +0.84 | 0.076 | +0.89 | 0.019 |
| MLH std during survey | +0.81 | 0.051 | +0.19 | 0.76 | +0.60 | 0.21 |

**Result: real, and not just a `809` artifact — with one exception that
is.** Peak boundary-layer depth correlates strongly with footprint decay
length (`r = 0.91`), and the relationship survives dropping `809`
(`r = 0.84`, Spearman still significant at `p = 0.019`) — the other five
flights' footprint decay lengths track their boundary-layer depths in the
same direction too (`728_1`: shallowest BL, shortest decay length; `726_2`:
deep BL, long decay length), which is the more convincing part of this
result than `809` alone. Mean MLH is a similar, slightly weaker story.
**MLH variability during the survey (std) is the one metric that does
*not* survive** — it looks comparably strong pooled (`r = 0.81`) but
collapses to `r = 0.19` with `809` excluded, i.e. that particular
correlation really is just "`809` is extreme in both quantities," not a
pattern the other five flights participate in — flagged explicitly since
it would have been easy to over-report alongside the two metrics that
hold up.

**Verdict:** this connects §20's empirical, meteorology-free
footprint-resolution finding to a real, physically coherent mechanism for
the first time — deeper daytime boundary layers plausibly permit more
turbulent horizontal dispersion, broadening STILT's footprint and worsening
along-track resolution — and shows `809` sits at the extreme end of a real
6-flight trend rather than being an isolated anomaly. With only 6 data
points this is a credible, worth-recording lead rather than a proven
result; a larger multi-day dataset (or an independent meteorological
reanalysis of boundary-layer depth, rather than this campaign's own lidar
retrieval) would be needed to treat it as established.

## 30. HRRR's own boundary-layer height vs. HALO's lidar MLH — the mixing depth STILT actually uses

§29 correlated footprint breadth with HALO's *independently measured*
boundary-layer depth. This section asks the more mechanistically direct
question: does HRRR's own `HPBL` — confirmed below to be essentially the
mixing-depth field STILT's transport actually runs on, not just a loosely
related external quantity — agree with HALO's lidar MLH, and does that
matter for the still-open flights specifically?

### 30.1 STILT uses HRRR's `HPBL` close to directly

Confirmed from this repo's own STILT run configuration
(`stilt/run_stilt_nyc_hrrr.r`, and its sibling `run_stilt_*.r` scripts),
not assumed: `varsiwant` includes `mlht`, the per-particle mixing-depth
value HYSPLIT/STILT samples along every trajectory, and `zicontroltf <- 0`
disables STILT's mixing-depth smoothing correction (a known STILT feature
for limiting artificial jumps at hourly meteorology updates). With it off,
STILT uses the **raw, un-smoothed, step-function mixing depth straight
from HRRR's own hourly field** for every trajectory in this investigation
— combined with HYSPLIT's default of reading mixing depth from the driving
model's own field (nothing in these run scripts overrides that default to
force an internally-recomputed diagnostic instead), this makes HRRR's
`HPBL` the operative vertical-mixing boundary shaping every Jacobian used
throughout this document, not an independent point of comparison.
(Caveat: this is drawn from HYSPLIT/STILT's documented general behavior
and this repo's own config, not verified against the STILT R package's
source directly — it isn't checked out in this repo.)

### 30.2 A single-point comparison suggested a one-directional bias — corrected below, this was wrong

**First pass** (script: `scripts/hrrr_pblh_vs_mlh_check.py`, no re-solve). Pulled
hourly analysis `HPBL` from the public NOAA HRRR Zarr archive on AWS
(`s3://hrrrzarr`, anonymous access; no HRRR data exists locally in this
repo otherwise) at **one representative NYC-domain point** (40.75, −73.95),
matched to each flight's survey-window hours, and compared against §29.2's
land-only binned-median HALO MLH. That comparison showed HRRR running
higher than HALO on every single flight (+4% to +31%), with `805` the
smallest-bias flight — and an initial verdict was drawn from it (HRRR
over-predicts boundary-layer depth systematically; `805`'s good agreement
argues against depth bias explaining its residual).

**That verdict does not survive testing at receptor resolution, and is
retracted here rather than left standing.** §30.3 redoes the comparison
matching every individual receptor to its own nearest HRRR grid cell and
hour, instead of one point for the whole flight — the obvious fix given
§29.1 already established real, large spatial HPBL structure (the
land/water split) across this same ~250km domain that a single point
cannot see. The corrected result reverses the specific, single-point
finding above; keeping the original framing here (rather than silently
replacing it) is deliberate, per §25.6's rule that a surprising,
consequential finding needs independent verification before being acted
on — this is that verification, and it changed the answer.

### 30.3 Receptor-resolution result: no systematic depth bias; a real land/water split instead

**Method** (script: `scripts/hrrr_pblh_vs_mlh_receptor_check.py`, no re-solve).
Every receptor (all 6 flights, `n=7459`) matched to its own nearest HRRR
grid cell via a KDTree over the static ~1059×1799 HRRR lat/lon grid, and
its own nearest HRRR analysis hour — not one point/hour standing in for
the whole flight. Compared against that same receptor's own HALO lidar MLH
(nearest-GPS-time matched, same convention as `scripts/mlh_bias_regression_check.py`).
Each unique `(date, hour)` HPBL grid fetched once and reused across every
receptor/flight that falls in it, to keep remote reads manageable.

| flight | n receptors | mean HALO | mean HRRR | ratio | bias | (single-point bias, §30.2) |
|---|---|---|---|---|---|---|
| `726_1` | 1256 | 573m | 539m | 0.94 | **−34m** | +107m |
| `726_2` | 1127 | 1257m | 1155m | 0.92 | **−102m** | +311m |
| `728_1` | 928 | 710m | 755m | 1.06 | +45m | +264m |
| `728_2` | 1286 | 1273m | 1151m | 0.90 | **−121m** | +235m |
| `805` | 1350 | 942m | 775m | 0.82 | **−167m** | +40m |
| `809` | 1512 | 1497m | 1816m | 1.21 | +318m | +343m |

**Result: the systematic, one-directional bias is gone.** Pooled across
all 7459 receptors, mean bias is −2m (median −14m) — essentially zero.
Four of six flights now show HRRR running *shallower* than HALO, not
deeper — the opposite sign from §30.2's single-point result for 3 of
those 4 (`726_1`, `726_2`, `728_2`). The scatter (`runs/
hrrr_pblh_vs_mlh_receptor_check.png`, left panel) shows a real, significant
positive correlation between HALO and HRRR (`r=0.667, p≈0`) with
substantial two-sided scatter around the 1:1 line — over- and
under-estimates roughly balance out in aggregate, which is exactly why
the single-point sample (one location, unluckily biased toward
over-estimation at that specific cell) looked like a clean one-directional
effect when it wasn't one.

**`805`'s specific verdict reverses, and the "good agreement argues
against depth bias" claim from §30.2 does not hold.** At receptor
resolution `805` has the *second-worst* relative mismatch of the six
flights (−18%, behind only `809`'s +21%) — not the best, as the
single-point sample suggested. The negative cross-check argument in the
original §30.2 verdict is retracted; `805`'s residual and its
boundary-layer depth bias are, if anything, now more plausibly connected,
not less. This doesn't reopen depth bias as `805`'s confirmed explanation
— no test in this document has directly regressed `805`'s specific
residual locations against its receptor-level HPBL bias — but it removes
a piece of evidence that had been used to argue against it.

**A real, different, and more specific finding replaces the retracted
one: land vs. water receptors show significantly different HRRR bias**
(`t=6.94, p=5×10⁻¹²`). Land receptors are roughly unbiased (+22m); water
receptors show HRRR under-predicting by −81m on average
(`figures/hrrr_pblh_vs_mlh_receptor_check.png`, right panel). This is a
plausible, specific mechanism — HRRR's ~3km native grid likely cannot
resolve the actual marine/harbor/Sound boundary-layer structure the way
it resolves land, especially given how narrow much of the relevant water
(East River, the Sound's coastline) is relative to that grid spacing — not
a general entrainment-scheme depth bias. Worth noting: this land/water
HPBL bias is a different quantity from §29.1's land/water *contrast in
HALO's own MLH* (a real atmospheric signal) or §28.4's land/water
*contrast in the fitted CH4 background offset* (still unexplained,
sign-flipping by flight) — three related but distinct land/water findings
now on record; not yet checked against each other directly.

**Correlation with §29.4's footprint-breadth result holds up, and
slightly strengthens, at receptor resolution.** Per-flight mean HRRR HPBL
(receptor-resolution) vs. §20's footprint decay length: `r=0.91, p=0.013`
— matching §29.4's HALO-based result almost exactly, and effectively
unchanged from §30.2's single-point number (`r=0.74`) despite that
number's own flight-level bias values being substantially wrong. Per-flight
mean HALO MLH at receptor resolution (all receptors, not just land-only):
`r=0.81, p=0.049`, also consistent with §29.4. **This is the one part of
§30's original analysis that survives the correction intact** — the
footprint-breadth/boundary-layer-depth relationship was never dependent on
getting the single point right.

**Verdict.** §30.2's "HRRR systematically over-predicts, `805` agrees
best" finding was an artifact of an unrepresentative single sampling
point and is withdrawn. In its place: no systematic depth bias at
receptor resolution (pooled bias ≈ 0), `805` actually has one of the
*larger* relative mismatches (removing it as counter-evidence against
depth bias, though not establishing it as the cause either), a real and
specific land/water HPBL bias plausibly tied to HRRR's native grid
resolution near the coast, and the boundary-layer-depth/footprint-breadth
correlation (§29.4) confirmed to be robust to this correction. Caveat
carried forward: lidar-derived MLH (aerosol-backscatter-gradient proxy)
and model HPBL (turbulence-scheme diagnostic) remain not-identically-defined
quantities even at receptor resolution — a documented, general
boundary-layer-verification phenomenon — so the residual scatter in the
1:1 plot is not purely "HRRR error."

### 30.4 Beyond land/water: a real southwest-to-northeast gradient, and `809` breaks it

§30.3's land/water split raised the natural next question: is the
mismatch structured *within* land (or water) too, as a continuous spatial
gradient, rather than just a two-category step?

**Method** (script: `scripts/hrrr_pblh_vs_mlh_gradient_check.py`, no re-solve;
extends §30.3's per-receptor matching by keeping each receptor's lat/lon
and terrain elevation instead of collapsing to a summary). Pooled
receptor-level linear correlations against latitude/longitude, a
land-only regression against terrain elevation (linear + quadratic), and
a spatially binned mean-bias map (0.15°/0.2° cells, minimum-count
thresholded) — computed pooled, per flight (6 panels, not per day —
`726_1`/`726_2` and `728_1`/`728_2` kept separate), and for the
morning/afternoon groups (§29.2).

**Result: a real, moderate-strength gradient pooled, but a strong and
strikingly consistent one within individual flights.** Pooled: bias vs.
latitude `r=-0.18` (`p=4e-54`), vs. longitude `r=-0.17` (`p=1e-47`) —
highly significant given `n=7459`, but each explains only a few percent
of receptor-level variance on its own; the binned map
(`figures/hrrr_pblh_vs_mlh_gradient_check.png`) shows why the raw correlation
understates it — a clean ~500–600m swing from strongly positive
(HRRR too deep) in the domain's southwest (inland NJ/Staten Island) to
strongly negative (too shallow) in the northeast (toward Long Island
Sound), once per-receptor noise is averaged out spatially. Land-only bias
vs. terrain elevation is real but small and non-monotonic (`r=+0.05,
p=0.0005` linear; quadratic term `p=2e-6` — bias rises slightly with
elevation to ~150m, then turns back down at the highest elevations
sampled, thin data there).

**Per-flight breakdown (`figures/hrrr_pblh_vs_mlh_gradient_by_flight.png`,
one panel per flight, not per day) confirms this is a real, repeatable
regional pattern — not a pooling artifact, and not a single-day fluke.**
Five of six flights (`726_1`, `726_2`, `728_1`, `728_2`, `805`)
independently show the *same* qualitative SW-positive/NE-negative
gradient, and the correlation is markedly stronger within a single flight
than pooled (e.g. `728_1` vs. longitude: `r=-0.50, p=8e-59`) — day-to-day
noise was diluting the pooled number, not manufacturing a false pattern.
Both the morning and afternoon groups
(`figures/hrrr_pblh_vs_mlh_gradient_morning_afternoon.png`) show the same
underlying gradient too, ruling out a time-of-day-specific explanation.

**`809` is the one flight that breaks the pattern, not a stronger version
of it.** Instead of a SW/NE gradient, its map is almost uniformly positive
(HRRR too deep nearly everywhere it flew), and it's the only flight whose
longitude correlation flips sign (`r=+0.235`, vs. negative on every other
flight). This is consistent with, and adds to, `809`'s already-established
pattern of meteorological distinctiveness (§29.2's deepest and only-
declining boundary layer, §30.3's largest single-flight depth bias,
`WIND_VARIABILITY.md`'s highest wind speed) — one more way this flight's
day was a genuinely different regime from the other five, not an extreme
point on a shared spectrum.

**Resolves §30.3's leg-axis caveat, at least partially.** §30.3 flagged
that the gradient runs along nearly the same axis as the survey's own
NE/SW leg orientation (`leg_axis_deg=45°`), raising the possibility that
part of the pattern reflected flight-track geometry (e.g. along-leg
position or timing) rather than real geography. That the gradient's
*character* is shared across five independently-flown days, while `809`
— flying essentially the same track geometry as the others — shows a
completely different spatial pattern instead of the same one more
strongly, argues for a real, day-varying meteorological signal being
sampled along that track, not a track-shape artifact. Not a complete
resolution — a formal regression against along-leg vs. cross-leg position
was not built — but a meaningfully strong piece of evidence against the
artifact explanation.

**Verdict.** A real, spatially coherent, physically plausible gradient
(consistent with a genuine regional HRRR bias pattern, e.g. differential
skill inland vs. near the Sound) sits underneath §30.3's land/water split,
confirmed repeatable across 5 of 6 independently-flown days. `809`'s
break from that pattern is now a third independent axis (alongside
boundary-layer depth and wind character, §29 and §31) on which it stands
apart from the other five flights — worth treating as its own distinct
meteorological regime for any future analysis, not folded into a
"6-flight average" picture.

### 30.5 Footprint composition (not receptor position) predicts the HPBL bias much more strongly — and a real, if narrower, CH4 effect too

§28.4/§29/§30.3/§30.4 all classified a receptor by the terrain *directly
under the aircraft* at the moment of measurement. That's not actually the
physically relevant quantity — a receptor's measured enhancement (and,
mechanistically, the mixing depth relevant to it) reflects its **footprint**:
the upwind STILT sensitivity field, which can be mostly over water even for
a receptor currently flying over land, or vice versa, depending on that
day's wind. This section tests the more direct version of the land/water
question for both biases this investigation has tracked: does *footprint*
water fraction (continuous, not binary) predict the CH4 background bias
(§28.4) and/or the HRRR-HPBL-vs-HALO-MLH bias (§30.3) better than receptor
position did?

**Method** (script: `scripts/footprint_land_water_check.py`, no re-solve —
reads each flight's full Jacobian once, ~10s for a 12.6GB file, via the
existing `JacobianFile.receptor_column_sums` streaming infrastructure).
Built a water mask on the Jacobian's own ~1666×1666 (~1km) footprint grid
by nearest-matching every cell (KDTree, <1s for 2.78M cells) to HRRR's own
static `LAND` field (`hrrrzarr/sfc/.../surface/LAND/surface`, same source/
resolution as §30's `HPBL` and §31's wind, for consistency). For each
receptor, `receptor_column_sums(weights={"water": water_mask})` gives the
fraction of its total footprint-weighted sensitivity that falls over water
— continuous in `[0, 1]`, unlike the binary label used everywhere else in
this investigation. This is a **footprint-sensitivity-weighted** fraction,
not a simple count of land vs. water grid cells the footprint's extent
happens to cover: for each receptor it is `Σ_cell H[receptor, cell] ·
water_mask[cell] / Σ_cell H[receptor, cell]`, i.e. weighted by the actual
STILT sensitivity (`H`, the Jacobian row) at every cell. A receptor whose
footprint barely grazes a large water body gets a small water fraction
(low sensitivity there); one whose footprint is concentrated over a
smaller but high-sensitivity water area gets a larger one — the correct
way to ask "how much of what this receptor actually measured came from
over water," as opposed to "how much water geography did the footprint's
bounding area happen to contain." Regressed leg-mean footprint water fraction against
leg background offset and posterior residual (flight-demeaned, same
convention as §28/§29), and receptor-level footprint water fraction
against the receptor-level HRRR-HALO HPBL bias (§30.3's construction,
recomputed here to pair with each receptor's own footprint).

**Result: a much stronger, cleaner signal for the meteorological bias.**
Receptor-level footprint water fraction vs. HPBL bias: `r=-0.30`,
`p=5×10⁻¹⁵⁹` (`n=7459`) — nearly double the strength of §30.4's lat/lon
gradients (`r≈-0.18`) and vastly more significant, with the same sign
(more water in the footprint → HRRR increasingly *under*-predicts relative
to HALO). This is a real improvement in mechanistic specificity, not just
a restatement of the land/water split: it ties the mismatch to a
receptor's actual upwind transport history rather than to where the plane
happened to be at that instant, and is consistent with (rather than
redundant with) §30.4's spatial gradient — footprint composition is a more
direct proxy for "how much marine air did this measurement actually
integrate over" than static position or coordinates.

**Result at leg-level aggregation: apparently a clean null for the CH4
side — but this turned out to be an aggregation artifact, corrected
below.** Leg-mean footprint water fraction vs. leg background offset:
`r=+0.15`, `p=0.28` (`n=52`, not significant); vs. leg-mean posterior
residual: `r=-0.03`, `p=0.83` (nothing). Per-flight, five of six flights
show a positive but individually non-significant trend on the background
offset (`r=0.21`–`0.53`, `n=8`–`10` legs each — too few to reach
significance alone); `809` is again the outlier, flipping negative
(`r=-0.43`), consistent with its pattern of standing apart in §29–§31.

**Corrected: the posterior residual does have a real receptor-level
effect — averaging down to 52 legs washed it out.** Re-run at full
receptor resolution (all `n=8078` receptors, flight-demeaned, no leg
averaging): footprint water fraction vs. posterior residual gives
`r=-0.165`, `p=2.5×10⁻⁵⁰` — small (`r²≈3%`) but statistically
unambiguous given the sample size. The leg-background-offset result does
**not** get the same correction: it stays null at receptor resolution too
(§30.6 below shows no consistent trend at any water percentage). So the
two CH4-side quantities behave differently — the post-fit residual
(the quantity every "is this flight explained" claim in this document is
actually about) carries a real, if modest, footprint-water signature; the
upstream background-offset estimate does not.

**A majority-water-vs-majority-land split (closer to §28.4's original
binary framing) is only testable for half the flights**, because
footprints that are truly >50% water are rare (mean footprint water
fraction per flight ranges only 7–20%, per §30.5's own numbers):

| flight | n majority-water | mean diff (water − land), ppm | p |
|---|---|---|---|
| `726_1` | 0 | — | untestable |
| `726_2` | 27 | +0.0035 | 0.20 (opposite sign, n.s.) |
| `728_1` | 13 | **−0.0117** | **0.0007** |
| `728_2` | 18 | **−0.0095** | **0.044** |
| `805` | 0 | — | untestable |
| `809` | 0 | — | untestable |
| pooled | 58 | −0.0040 | 0.067 (marginal) |

`728_1` and `728_2` — both still-open flights — show a real, individually
significant negative effect; `726_2` (already explained) trends the
opposite direction but isn't significant; `726_1`, `805`, `809` never
have enough majority-water footprint receptors to test at all. This
matches, rather than contradicts, the earlier land/water investigations:
the effect is real but concentrated in specific flights, not a uniform
six-flight pattern — the same character as §28.4's sign-flipping result,
just now localized to a subset of flights instead of alternating sign
across all six.

### 30.6 The two biases have different shapes: a smooth dose-response for HPBL, a threshold effect for the CH4 residual

**Method** (script: `scripts/footprint_land_water_bins_check.py`, no
re-solve, no network access — pure re-analysis of §30.5's saved per-
receptor dataset, `figures/footprint_land_water_fractions.csv`). Binned
all three targets (post-fit residual, leg background offset, HRRR-HALO
HPBL bias) against footprint water fraction using ~8 equal-count quantile
bins (roughly 1000 receptors per bin, for even statistical power across a
skewed distribution — most receptors sit under 20% footprint water
fraction, so fixed-width bins above ~50% are nearly empty) plus fixed
10%-wide bins to show where the data actually concentrates.

**Result: the HPBL bias is a clean, near-monotonic dose-response across
the whole range.** Quantile-bin means fall steadily from `+88m` (lowest
water-fraction bin) to `-370m` (highest, >26% water), declining through
every intermediate bin with tight confidence intervals (`n≈900-1000` per
bin throughout) — not a threshold effect concentrated at one end, and not
an artifact of the linear fit reported in §30.5: the real relationship
looks genuinely close to linear across the observed range.

**Result: the CH4 posterior residual behaves differently — flat, then a
late, sharp drop.** The first six quantile bins (0-21% water) wobble
between roughly `-0.001` and `+0.005` ppm with overlapping confidence
intervals — no discernible pattern there. It turns clearly and
significantly negative only in the top two bins (21-26% water: `-0.0026`
ppm; 26-76% water: `-0.0100` ppm, the tightest and most negative point on
the plot). This explains §30.5's majority-water-vs-land table directly:
the effect is real but concentrated at the *high* end of footprint water
fraction, which is exactly why it only shows up as testable/significant
for `728_1`/`728_2` — the two flights whose footprints reach furthest
into that high-water regime — and is invisible in a plain linear
correlation or a coarse leg-level average that mixes high- and low-water
receptors together.

**Result: the leg background offset shows no pattern at any water
percentage.** Bin means bounce between `-0.006` and `+0.002` ppm with
overlapping confidence intervals throughout the whole range
(`figures/footprint_land_water_bins_check.png`, middle panel) — confirms
the null holds everywhere, not just when tested linearly or at a single
50% split.

**Verdict.** Footprint composition is now the best-supported single
predictor found in this investigation for the HRRR-vs-HALO boundary-layer
mismatch — stronger than raw position, terrain elevation, or the lat/lon
gradient (§30.4) — reinforcing that §30's bias is a real, physically
specific effect (plausibly HRRR's ~3km grid failing to resolve marine/
coastal boundary-layer structure) rather than noise, and shaped like a
smooth dose-response rather than a threshold. On the CH4 side the picture
is more nuanced than either the original binary test (§28.4) or the first
pass at this section suggested: the background-offset estimate genuinely
shows nothing, at any resolution or water percentage tried; but the actual
post-fit residual carries a real, if modest, effect that only appears once
a footprint's water fraction crosses roughly the top quartile — a
threshold-like, not smooth, relationship, and concentrated in `728_1`/
`728_2` specifically. §28.4's original sign-flipping land/water contrast
in the background offset remains unexplained by any static land/water
measure tried so far (receptor position, §28.4; footprint composition,
§30.5) — still consistent with §28.5's read that the background-offset
effect specifically is wind-*direction*-dependent rather than a fixed
spatial pattern. The post-fit-residual effect found here is a distinct,
new, and more actionable lead: worth checking directly against `728_1`/
`728_2`'s specific open residual locations (§9, §26 item 1).

## 31. HRRR 10m wind vs. residual bias: no leg-level relationship, but `805` is uniquely light and erratic

Motivated by whether wind changes during a survey could create a mismatch
between STILT's modeled transport and the real flow, manifesting as
elevated residuals on legs near a modeled wind shift. Full per-flight and
per-leg wind characterization (method, rotation correction, caveats, and
the complete 52-leg table) lives in
[WIND_VARIABILITY.md](WIND_VARIABILITY.md), kept separate since it's
useful beyond this specific question; this section covers only the
residual-relevant tests and their results.

**Method** (scripts: `scripts/hrrr_wind_bias_check.py`, single point/hour per
flight; superseded by `scripts/hrrr_wind_leg_change_check.py`, each leg's own
receptor-centroid location and nearest HRRR hour — see
WIND_VARIABILITY.md for why the per-leg version is preferred). Regressed
leg background offset and posterior residual against wind speed and
against `(u, v)` components jointly, and specifically tested whether a
bigger leg-to-leg wind-direction *change* coincides with a bigger
`|residual|` on that leg.

**Result: no support for the wind-shift-mismatch hypothesis, at either
resolution tested.** Signed regressions: offset vs. speed `r=+0.03,
p=0.84`; vs. `(u,v)` jointly `R²=0.007, p=0.85`. Posterior residual vs.
speed `r=-0.003, p=0.98`; vs. `(u,v)` jointly `R²=0.006, p=0.86`. The
specific leg-to-leg-change test: `|posterior residual|` vs. `|Δ wind
direction|` gives `r=-0.24` (`p=0.11`, not significant, and the *wrong*
sign for the hypothesis — bigger direction jumps trend toward smaller, not
larger, residuals); `|Δ wind speed|` similarly null (`r=-0.10, p=0.51`).
This held at both the earlier single-point-per-flight resolution and the
per-leg-location, per-leg-hour resolution — not a resolution artifact of
the coarser first pass.

**But `805`'s modeled wind is a real, distinct outlier, independent of
that null result.** Its circular direction standard deviation (62°) is
2–4× every other flight's (16–33°; full table in
[WIND_VARIABILITY.md](WIND_VARIABILITY.md)), swinging through nearly the
full compass leg to leg while every other flight rotates smoothly and
modestly. Its wind is also among the lightest of the six flights (mean
2.2 m/s, several legs under 1 m/s, where direction becomes numerically
noisy — a caveat on the precise degree values, not on the light-and-unsteady
character itself).

**Verdict.** The specific mechanism proposed (wind-shift timing mismatch
between legs) isn't supported by the leg-level data. `805`'s uniquely
light-and-erratic modeled wind is nonetheless a new, distinct, and
better-targeted candidate on its own terms: a light-and-variable flow is
inherently harder for any hourly NWP analysis to represent accurately
(small absolute errors in a weak, unsteady wind have an outsized effect on
transport direction), independent of boundary-layer depth. (An earlier
draft of this verdict additionally leaned on §30.2's finding that `805`
had the best HRRR-vs-HALO depth agreement of any flight, treating this as
a second, complementary reason depth-independent instability specifically
should get priority; §30.3's receptor-resolution correction reversed that
premise — `805` is actually one of the worse depth-agreement flights — so
that additional support is withdrawn. The light-and-erratic-wind lead
itself doesn't depend on it and stands regardless.) This reframes `805`'s
open status from "no remaining leads" to "simple wind-shift timing is
ruled out; wind-field *instability* itself, not yet directly tested
against `805`'s specific residual locations, is a candidate alongside
boundary-layer depth (§30.3), not in place of it" — still an indirect
proxy for hypothesis (c), not a direct test of the column data, which
remains completely untouched.
