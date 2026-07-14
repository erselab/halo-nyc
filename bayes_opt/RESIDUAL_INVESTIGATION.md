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
consistent with §14.6's rule that a surprising, consequential diagnostic
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
  alternative-inventory comparison recommended there (§15.1).

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
  different explanation entirely (as §15.2's suggestion for `726_1`'s dip
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

§9 reframed `805`/`809` as background-side, not prior-side, and §15.4
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
another instance of §14.6's rule.** The first pass eyeballed the six panels
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

**Verdict:** weaker than first reported. `805` alone has a clean, strong,
lag-consistent along-track signature matching transport under-resolution;
`809` matches weakly; the rest of the six-flight split does not hold up
against the quantitative check and should not be treated as established.
This still sharpens §7's hypothesis for `805` specifically beyond the single
hand-picked `726_1` pair it was first tested on, but does **not** extend it
to `809`/`728_1` as a group the way first claimed. The lesson about
over-trusting a plot (§14.6) applied to this investigation's own new work,
not just historical cases — worth remembering the next time a multi-panel
comparison looks clean at a glance.

## 14. Major takeaways

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
    itself is the reusable lesson: §14.6's rule applied to this
    investigation's own new work in the same session it was written, not
    just to older, already-cited cases.

## 15. Suggestions for future analysis

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
