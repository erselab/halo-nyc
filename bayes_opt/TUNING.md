# Error-model tuning guide

A practical guide to choosing the observation-error (`R`) and prior (`Sa`)
covariances for the HALO CH₄ inversion, and to reading the `--tune` diagnostics.
The inversion is only as trustworthy as these covariances: they set how much the
data are allowed to move the fluxes and how tight the reported uncertainties are.

> TL;DR — the goal is **reduced χ² ≈ 1** *and* the **Desroziers `r_scale` ≈ 1** at
> the same time. If they disagree, it is a *structure* problem (correlation length
> or prior scale), not a magnitude problem — fix the structure before rescaling.

---

## 1. What you are tuning

The model is `z = H f + ε`, solved as
`x̂ = xa + Sa Hᵀ (H Sa Hᵀ + R)⁻¹ (z − H xa)`. Two covariances are hyperparameters:

| | config | knobs | meaning |
|---|---|---|---|
| **`R`** (observation error) | `[observations]` | `measurement_stddev`, `mdm_stddev`, `mdm_correlation_length_km` (and `error_inflation`) | per-receptor retrieval noise ⊕ along-track-correlated model–data mismatch (MDM) |
| **`Sa`** (prior) | `[prior]`, `[category_*]` | `scalar_stddev`, `correlation_length_km`, per-category `[category_uncertainty]` / `[category_spatial]` | how far, and how smoothly, fluxes may depart from the inventory |

`R` has two parts: a **magnitude** (the std-devs, i.e. its diagonal/trace) and a
**shape** (`mdm_correlation_length_km`, the off-diagonal structure). The diagnostics
below probe these differently — that is the whole point of having more than one.

---

## 2. Running the diagnostics

```bash
python run_halo.py config.ini --tune
```

`--tune` is **non-destructive**: it runs the normal inversion, writes the usual
`posterior.nc`, then *additionally* prints diagnostics and *suggested* rescalings.
It never edits the config or re-solves — you read the numbers and decide.

Apply suggestions per-run with `--set` (no file editing):

```bash
python run_halo.py config.ini --set observations.error_inflation=0.7
python run_halo.py config.ini --tune --set tuning.tune_Sa=true
python run_halo.py config.ini --tune --set observations.mdm_correlation_length_km=3
```

By default `--tune` optimizes `R` only (`[tuning] tune_R=true`, `tune_Sa=false`);
turn on `tune_Sa` to let it move the prior too.

---

## 3. The four numbers and how to read them

```
reduced chi-square (current): 0.516
Desroziers R consistency r_scale: 1.934  (>1 => assumed R too small)
max-likelihood scales: alpha_R=0.517  alpha_Sa=1.000  (log-likelihood 3848.18)
-> set [observations] error_inflation = 0.517  (or scale mdm/measurement stddev by 0.719)
```

**reduced χ²** — `(z−Hx̂)ᵀ R⁻¹ (z−Hx̂) / n_obs`, the observation-space posterior
residual measured *through the `R⁻¹` metric*.
- ≈ 1 → fit consistent with `R`. **This is the target.**
- < 1 → fit is "too good": residuals smaller than `R` claims (R too large, or the
  prior is too loose and the analysis is over-fitting, or R's correlations are too
  long — see §4).
- \> 1 → residuals larger than `R` claims (R too small, or the prior is too tight
  to let the analysis fit the data).

**alpha_R** — the marginal-likelihood-optimal multiplicative scale on `R`. It is the
*principled* magnitude fix and usually ≈ reduced χ² (both live in the `R⁻¹` metric),
so they move together. `R → alpha_R · R` drives χ² toward 1.

**Desroziers `r_scale`** — `tr(R_est) / tr(R_assumed)` with
`tr(R_est) = (z−Hx̂)·(z−Hxa)`. A **trace/magnitude** check on `R`. `> 1` ⇒ assumed
`R` too small; iterating `R → r_scale·R` converges toward consistency. (The report
also implies a companion relation for the prior, `d_o·d_b ≈ tr(H Sa Hᵀ)`.)

**alpha_Sa** — optimal scale on the prior. **Shown as `1.000` whenever
`tune_Sa=false`** (it was held fixed) — that is not a result, just "not tuned."

---

## 4. When the diagnostics agree vs disagree

**They agree (both say the same direction).** Easy case. If reduced χ² ≈ alpha_R and
`r_scale ≈ 1/alpha_R` point the same way, it really is a magnitude problem. Apply
the scale and move on:

```bash
# example: χ²≈1.8, alpha_R≈1.8, r_scale≈1.8  -> R genuinely too small
python run_halo.py config.ini --set observations.error_inflation=1.8
# (equivalently scale measurement_stddev and mdm_stddev by sqrt(1.8)≈1.34)
```

**They disagree (the example above).** reduced χ² = 0.516 and alpha_R = 0.517 say
"halve R," but `r_scale` = 1.934 says "double R." A metric diagnostic (`R⁻¹`) and a
trace diagnostic (trace) pointing **opposite** ways means the **magnitude is not the
problem** — the *structure* is. Two usual suspects:

1. **`mdm_correlation_length_km` too long.** Long correlations make `R⁻¹` permissive
   (χ² collapses below 1) while the diagonal variances stay too small (trace check
   wants them bigger → `r_scale` > 1). This is the most common cause of this exact
   contradiction.
2. **Prior `Sa` mis-scaled.** χ² < 1 also happens when the prior is too loose and the
   analysis over-fits; the likelihood blamed `R` only because `Sa` was frozen.

Do **not** blindly apply `error_inflation = 0.517` here: it would force χ² to 1 *and
halve your posterior uncertainties*, confidence the conflicting `r_scale` says you
have not earned. Instead, find the structure:

```bash
# sweep the correlation length; watch chi-square and r_scale move TOWARD each other
python run_halo.py config.ini --tune --set observations.mdm_correlation_length_km=2
python run_halo.py config.ini --tune --set observations.mdm_correlation_length_km=20

# let the prior tune too, and see whether it (not R) absorbs the misfit
python run_halo.py config.ini --tune --set tuning.tune_Sa=true
```

When χ² ≈ 1 **and** `r_scale` ≈ 1 together, the error model is consistent and a
single `alpha_R` (if still ≠ 1) is finally meaningful to apply.

---

## 5. A workflow

1. **Start sane.** `measurement_stddev` ≈ the retrieval's stated 1σ; `mdm_stddev`
   a bit larger (transport/representation error); `mdm_correlation_length_km` ~ the
   along-track footprint overlap scale; `scalar_stddev` = how many × the inventory
   you'd believe (e.g. 0.5 = ±50%).
2. **Run `--tune`.** Read §3.
3. **If the diagnostics agree:** apply the magnitude fix (`error_inflation` or scale
   the std-devs), re-run `--tune`, confirm χ² ≈ 1.
4. **If they disagree:** treat it as structure (§4) — sweep
   `mdm_correlation_length_km`, and/or `--set tuning.tune_Sa=true`, until χ² and
   `r_scale` agree near 1. *Then* apply any residual magnitude scale.
5. **Sanity-check the prior pull.** Compare the observation-only χ² (`--tune`) with
   how far the posterior moved from the prior. A low obs-χ² with large prior
   departure points at `Sa`, not `R`.
6. **Lock it in.** Once happy, write the chosen values into `config.ini` (the
   `--set` values you converged on). Saved bundles record the *effective* config,
   so a `runs/<name>/config.ini` always documents what produced a result.

---

## 6. Caveats specific to this problem

- **Single flight ⇒ weak identifiability.** With one flight the `R` vs `Sa` split is
  poorly constrained (DOFS ~1–2); `--tune` will happily move either. Prefer tuning
  `R` only from one flight, and revisit `Sa` once you assimilate several flights
  (`--flights a,b,c`), where the joint fit constrains the split better.
- **Column XCH₄ barely separates co-located sectors.** Per-category prior σ
  (`[category_uncertainty]`, `[category_spatial]`) does real work in the
  decomposition; tune those deliberately, not just the global `scalar_stddev`.
- **χ² < 1 is not "good."** It means over-confident inputs or over-fitting, and it
  *understates* posterior uncertainty. Aim for ≈ 1, not the smallest χ².
- **Outliers first.** Gross errors inflate residuals and corrupt every diagnostic.
  Set `[observations] outlier_threshold` (innovation check) so the tuning numbers
  reflect the data you actually trust.
- **Buffer interactions.** If out-of-core signal has nowhere to go it leaks into the
  background/edge and can masquerade as model–data mismatch. Run
  `--diagnose-domain`; if the out-of-core fraction is large, enable `[buffer]`
  before reading too much into `R` tuning.

---

See also: `README.md` (§ "Observation error", "Usage" `--tune`/`--set`),
`halo_oe/obs_error.py` (how `R` is built), `goe/tuning.py` (the diagnostics), and
`notebooks/saved_bundle_analysis.ipynb` § 4 (model–data mismatch on a saved run).
