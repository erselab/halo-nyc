# Residual & background investigation — findings and open questions

This documents an extended investigation into systematic (non-noise-like)
per-flight residual structure that survived MDM tuning, covering the
background-model improvements that came out of it, the diagnostic ideas tried
and ruled out along the way, and what remains genuinely unexplained. It
complements [README.md](README.md) (architecture) and [TUNING.md](TUNING.md)
(the MDM tuning procedure itself).

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

### 2.2 Per-leg background offset (new this session)

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

**Motivation** (user's insight): STILT/HRRR transport resolution (~3km) is
coarser than the observation binning (~1km) — the model may be structurally
unable to reproduce sharp enhancements from strong point sources, independent
of how accurate the emission field is.

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
new script, `dipole_diagnostic.py` (plots: `runs/dipole_diagnostic_
<flight>_cluster<n>.png`).

**Two method bugs surfaced and were fixed before trusting any result** —
consistent with §16.6's rule that a surprising, consequential diagnostic
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
  alternative-inventory comparison recommended there (§17.1).

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
`726_1`/`728_1`/`728_2` needed.

## 10. Checking whether the buffer region explains the residual bias

Prompted by a question about how the buffer (`halo_oe/buffer.py`, §-independent
of this investigation until now — zero prior mentions) behaves when its prior
flux is zero: could the buffer's coarse resolution (`factor=10`, ~10km
super-cells) or finite `outer_bbox` be *causing* some of §9's unexplained
residual bias, rather than being an unrelated nuisance parameter? Checked with
three no-resolve diagnostics (script: `buffer_bias_check.py`; no inversion was
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
   −0.17) — the scatter plots (`runs/buffer_bias_scatter.png`) show the
   largest residuals concentrated at *low* out-of-core fraction, the opposite
   of what the hypothesis predicts.
3. **Spatial check.** The out-of-core fraction (`runs/buffer_bias_map.png`)
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

**Checked** (script: `diffuse_prior_check.py`, no Jacobian read — pure
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
  different explanation entirely (as §17.2's suggestion for `726_1`'s dip
  already assumed).
- Visually (`runs/diffuse_prior_overview.png`), low concentration (~0.4–0.6)
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

§9 reframed `805`/`809` as background-side, not prior-side, and §17.4
proposed the specific mechanism: maybe `fit_leg_offsets`'s GP
(`leg_correlation_time_s = 600s`) oversmooths a real, faster-varying
per-leg background signal. Tested directly (script: `leg_offset_check.py`,
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
legs (`runs/leg_offset_check_20230805.png`, `..._20230809.png`). Since `raw
≈ smooth` already, this isn't a good correction that got smoothed away —
the per-leg-constant quantile estimator itself never captured it, at any
smoothing setting. **This corroborates §4.2's earlier, independently-derived
finding** (continuous per-receptor kriging also plateaued regardless of
correlation length, 0.05°–10°) — two different methods (aggregate
receptor-level kriging vs. this leg-level, flight-specific check) converge
on the same conclusion: background modeling, in every form tried (plane,
per-leg constant, continuous kriging), has a real ceiling on these flights
that tuning a smoothing parameter further won't cross.

Leg segmentation itself looks reasonable for both flights (10 legs each;
`runs/leg_segmentation_20230805.png` shows 8+ clean parallel lines;
`runs/leg_segmentation_20230809.png` shows what looks like legs revisiting
similar geography at different times — the exact scenario §2.2's per-leg
offset was built for, not an obvious segmentation bug, though a
time-ordered check would be needed to be fully certain).

**Verdict:** the background-side lead for `805`/`809` is now exhausted —
every background-model variant tried (plane, per-leg constant at any
smoothing setting, continuous kriging) plateaus at the same real, localized
residual. The next step for these two flights should shift back to
prior-shape/correlation-length territory (as already done for `728_2` in
§9), not further background tuning.

## 13. Along-track resolved-scale check: observed vs. modeled variability

A systematic version of §7's one-off gradient-sharpness comparison (which
only checked a single `726_1` hotspot/dip pair): if background varies
slowly, comparing the along-track variability of the observed enhancement
`z` to the modeled `Hx̂` directly measures the scale below which the model
stops resolving real structure — without needing a specific known feature to
anchor on. Checked (script: `along_track_scale_check.py`, no re-solve) with
two real-elapsed-time-lag-binned statistics, computed per flight and pooled:
the along-track **autocorrelation** (reusing `plotting._time_binned_autocorr`
as-is) and the along-track **structure function** `D(τ) = ⟨(x(t+τ)−x(t))²⟩`
(new, same binning). A noise-floor control — the same structure function
computed on `z` restricted to the domain-insensitive (`fit_mask`)
receptors — separates real signal from the background-fitting noise §12
already found leaking into "clean" receptors.

**Autocorrelation: an initial visual read overclaimed a clean split, and a
follow-up numeric check (script: `along_track_acf_quant.py`) corrected it —
another instance of §16.6's rule.** The first pass eyeballed the six panels
and reported `805`, `809`, `728_1` all showing `Hx̂` decorrelating more
slowly than `z` (the transport-under-resolution signature), with
`726_1`/`726_2`/`728_2` not. Tabulating the actual signed gap
(`ac_modeled − ac_z`) per lag bin instead of reading the plot did not
support that split:

| flight | mean gap, lag≤30s | mean gap, lag>30s | pattern |
|---|---|---|---|
| `805` | +0.17 | +0.27 | strong, consistent positive gap at every lag |
| `726_2` | +0.17 | +0.18 | equally strong positive gap — **same sign and magnitude as `805`**, despite being an already-explained flight |
| `809` | +0.02 | +0.09 | weak positive, only past ~12s |
| `728_2` | −0.02 | −0.11 | near zero at short lag, mildly negative at long lag |
| `728_1` | −0.17 | −0.17 | consistent *negative* gap — opposite sign from what was claimed |
| `726_1` | −0.15 | −0.26 | consistent negative gap, growing with lag — opposite sign |

Only `805` is an unambiguous, strong match for the transport-under-resolution
signature at every lag. `809` matches weakly. `726_2` — already fully
explained by leg-offset correction (§3) — shows a gap of the same sign and
magnitude as `805`, and `728_1` (one of the flights the first pass claimed
*did* show the signature) actually shows the opposite sign, matching
`726_1`. **The split does not track the known good/bad flight split at all**
and should not have been generalized from a plot impression.

A likely confound: the ACF normalizes by each series' own variance, and
`Hx̂` is often very flat (low true variance) except where real flux signal
exists — how much real structure a flight's footprints happened to see
varies a lot flight-to-flight, which can destabilize a low-variance series'
standardized autocorrelation in a way unrelated to "is the model too
smooth." This plausibly explains why the sign flips without tracking flight
quality, though it wasn't tested directly.

**Follow-up: is the surviving `805` signal itself an artifact of a few large
point-source excursions in `z`?** Checked directly (script:
`along_track_outlier_check.py`) rather than assumed. Two distinct
mechanisms were in play: bin sparsity (a single extreme pair dominating a
lag bin with few pairs) and per-series variance inflation (a few large `z`
values shrinking the whole standardized series). Bin sparsity turned out not
to matter — every lag bin has 800+ pairs for all three flights checked, far
too many for one pair to dominate. Variance inflation, however, is real and
sharply different between flights: `726_2` has genuinely extreme excursions
(max robust z-score 8.9, 11 points beyond 4σ), and dropping just its 5 most
extreme receptors (0.4% of that flight's data) shrinks its overall stddev by
9% and its long-lag ACF gap by ~19% (+0.178 → +0.144). `805` and `809` have
no comparably extreme points (max robust z-score 3.3 and 2.9), and the same
drop-5 test barely moves either flight's numbers at all. **This explains why
`726_2` matched `805`'s signature despite being an already-explained
flight** — `726_2`'s apparent match is concentrated in a handful of large
excursions inflating its variance normalization, while `805`'s is spread
broadly across many modest ones, making it a meaningfully more trustworthy
standalone finding than the raw gap-table comparison alone suggested.

**Important correction to that framing, from a follow-up check the user
specifically requested** (script: `along_track_outlier_check2.py`): the
"5 most extreme points" dropped above were *not* measurement artifacts.
Global mean/std-based thresholds are a poor outlier definition here, since a
point sitting on a slowly-varying but otherwise unremarkable stretch of
background can be several sigma from the *flight-wide* mean without being a
real excursion at all — and a genuine point source is physically extended,
so it should imprint on more than one consecutive sample, unlike a sensor
glitch. Redefining excursions relative to each point's own local
neighborhood (median of same-leg samples 8–45s away, excluding the point's
immediate vicinity so a multi-sample plume doesn't bias its own baseline)
and classifying each as CLUSTERED (a same-sign neighbor within 15s also
shows a local excursion) vs. ISOLATED (no such neighbor) found that **90%+
of local excursions in every flight are clustered** — `726_1`: 18/18,
`726_2`: 37/41, `728_1`: 45/49, `728_2`: 54/58, `805`: 13/14, `809`: 15/16.
Dropping only the true isolated points (0–4 per flight) changes essentially
nothing anywhere (`z`'s stddev moves ≤0.8%, the ACF gap ≤0.02, in every
flight including `726_2`). So the earlier "outlier-sensitivity" framing was
wrong in an important way: what made `726_2`'s gap move wasn't contamination
by artifacts — real single-sample measurement artifacts are nearly absent
from all six flights — it was removing a handful of **genuine, large,
real point-source crossings**. `805`'s signal being spread across many
modest real events rather than a few dramatic ones is still the right
reason to trust it more, but "outlier-robust" was the wrong frame; "broadly-
vs. narrowly-supported by real events" is the accurate one.

A striking pattern fell out of this that's worth its own note: at literally
every clustered (real) event across all six flights, `modeled Hx̂` stays
small (~0.01–0.03) while `z` swings far larger (up to ±0.15) — the model
essentially never reproduces these sharp real features at anywhere near
full amplitude, in *any* flight. This is a strong, repeated confirmation of
§7's original hypothesis at far higher sample size than the single `726_1`
hotspot it was first based on — but since it happens in every flight,
including ones whose overall bias was already resolved another way, it
doesn't by itself distinguish which flights have an unresolved *net* bias
problem from which don't.

**The structure function has the same amplitude-dominance problem as
before, and is not a safe substitute for the (also-flawed) ACF here.**
`D(τ)` is in absolute (ppm²) units, and `Hx̂`'s overall magnitude is much
smaller than `z`'s throughout this whole investigation (every residual
map's modeled colorbar has been visibly narrower than its enhancement
colorbar) — so `D_modeled ≈ 0` at every lag mostly just re-states that known
amplitude gap, not a scale/resolution difference. More surprising: pooled
and per-flight, `D_z` (all receptors) and the `fit_mask`-only noise-floor
control are **nearly indistinguishable** at every lag, for every flight
(`runs/along_track_structure*.png`) — i.e., at the whole-flight pooled
level, observed along-track variability doesn't show excess structure
attributable to real plume signal beyond the background noise floor. That's
most likely a dilution effect (most receptors on a long track are far from
any point source at any given moment, so a flight-wide average swamps the
localized moments with real signal) rather than evidence there's no real
fine-scale plume structure — but this test, as built, can't distinguish
those two explanations. A targeted version restricted to the
domain-*sensitive* (plume-affected) receptors specifically, rather than
pooled over the whole flight, would be the natural follow-up for both
statistics if this angle is revisited — neither tool as currently built
cleanly isolates "real signal at a given scale" from "how much true
variance this series happens to have."

**Verdict:** weaker than first reported, but the surviving part is on solid
footing. `805` alone has a clean, strong, lag-consistent along-track
signature matching transport under-resolution, confirmed to be spread
broadly across its many (mildly) extreme receptors rather than riding on a
handful of dramatic ones; `809` matches weakly; the rest of the six-flight
split does not hold up against the quantitative check and should not be
treated as established — `726_2`'s apparent match to `805` was concentrated
in a handful of large, *real* point-source events (not measurement
artifacts — see below) that dominate its variance normalization, which is
what made the original six-flight split look cleaner than it was. This
sharpens §7's hypothesis for `805` specifically beyond the single
hand-picked `726_1` pair it was first tested on, but does **not** extend it
to `809`/`728_1` as a group the way first
claimed. The lesson about over-trusting a plot (§16.6) applied to this
investigation's own new work, not just historical cases — worth remembering
the next time a multi-panel comparison looks clean at a glance, and worth
following every such correction with an outlier-sensitivity check before
trusting whatever signal survives.

## 14. Point-source amplification and MDM correlation-length tests

Prompted by the along-track findings (§13): could the model's failure to
reproduce sharp real excursions be fixed by (a) amplifying the specific
point source's prior flux, or (b) recognizing it as transport/representation
error that belongs in `R` rather than the prior? Both were tested, on
`805` first for (a) and pooled across all 6 flights for (b) — no re-solve
either way.

**(a) Single-cell amplification (scripts: `point_source_intersection_check.py`,
`point_source_amplification_check.py`).** Grouped `805`'s clustered
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
cells, not an isolated pixel, is ever identifiable from a few receptors.
(An unconstrained least-squares version of the same test was tried first
and produced wild, obviously-wrong numbers — up to a "needed" scale factor
of −185,848 — before being replaced with the regularized version; a
reminder that an ill-conditioned small linear solve needs a prior-informed
regularizer, not just more averaging, when the sensitivity is this weak.)

**(b) Empirical MDM correlation length near landfill/WWTP sources**
(script: `landfill_wwtp_coherence_check.py`). `landfill` and `wastewater`
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

**Method (script: `joint_correlation_sweep.py`, no re-solve):** for each of
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
follow-up (§16 item 8, prior numbering) as not worth building — the
single-cell result it was meant to extend turns out to generalize instead
of being a special case.

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

## 16. Major takeaways

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
   check. The corrected `726_1` dip finding (§7.4) is the clearest example
   this session — when a diagnostic finding is surprising *and*
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
    itself is the reusable lesson: §16.6's rule applied to this
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

## 17. Suggestions for future analysis

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
7. **An explicit footprint-coarsening test** (spatially smooth/coarsen STILT
   footprints to ~3km and see whether that alone reproduces the flat model
   response seen in the gradient-sharpness test) was proposed early on but
   never run. §13's along-track check, after correction, only robustly
   motivates this for `805` (`809` more weakly) — the initial three-flight
   grouping with `728_1` did not survive a quantitative check (`728_1`
   actually showed the opposite-signed gap) and should not be used to scope
   this test. Separate from `726_1`'s dip specifically, which turned out not
   to be footprint-shape-related.
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
