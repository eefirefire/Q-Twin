# Q-Twin: Known Limitations (master reference)

Consolidates every "HONEST CAVEAT," open question, and documented limitation
across the project into one place, organized by pipeline stage, for Week 5's
proposal writing. Each entry cites its source file — this document
summarizes, it doesn't replace the originals as the source of truth.

## Data / ingestion

- **`15Mar_No.16` excluded, not scored FAILURE**: ~20,600 Hz CHI-to-probe
  discontinuity, corroborated by the chip's absence from the lab's own
  success-rate tracking. *Source: `clarifying_questions.md` item 9.*
- **Success-rate denominator (45 chips vs. 57 trials) still open** — Eva/the
  teacher input outstanding. *Source: `clarifying_questions.md` item 1.*
- **Target-stage +63.49 Hz/10 µM anchor point unverified against real data**
  — real 10 µM measurements are noisy and don't cleanly reproduce it.
  *Source: `clarifying_questions.md` item 15.*
- **A stale Drive-artifact incident** (multiple old `chip_summary.csv`
  copies with pre-Δf-fix values) was found and traced — confirmed the
  actual pipeline was never affected (single git-tracked source), but the
  stale Drive copies still need manual removal (no delete/move tool
  available). *Source: `Task0_Data_Integrity_Confirmation.md`.*

## Stage 0 — Gatekeeper

- **100% accuracy is not evidence of a sophisticated model.**
  `DIVERGENT_REPLICATES` is caught by construction; `early_displacement_30s`
  correlates 0.9997 with the label-defining quantity. *Source:
  `gatekeeper_metrics.txt`.*
- **Near-boundary accuracy drops to 91.7%**, with SUCCESS recall falling to
  44% within 3 Hz of the −0.5 Hz threshold — the real-world 100% only means
  "validated on the chips this dataset has." *Source:
  `gatekeeper_boundary_stress_test.txt`.*
- **Random Forest genuinely does beat logistic regression** (100% vs.
  93.9%) — the suspicion that RF was "an elaborate way of restating a
  simple rule" did not hold up; the relationship isn't purely linear.
  *Source: `gatekeeper_logistic_comparison.txt`.*
- **BLIND hold-out result: holds at 100%** (n=11, first genuinely
  untouched-by-any-design-decision validation). *Source:
  `holdout_validation_results.txt`.*
- **Redefining the biomarker (slope → displacement) bought +47.8 points**
  on a simple threshold rule alone, before any RF/divergence-flag
  machinery. Could not build a matched synthetic-trained comparison for
  the old slope biomarker (never modeled in the generator, since it was
  already proven non-predictive in Week 1). *Source: `ablation_results.txt`.*

## Stage 2a — Sequence model (LSTM / LSTM+Attention / TCN)

- **Official LSTM: 75.8% accuracy**, `DIVERGENT_REPLICATES` recall=1.00 but
  precision=0.47 — over-triggers rather than misses (a different failure
  mode than the original Week 3 prototype, whose problem was the reverse).
  155 synthetic training curves is small for a 3-class sequence problem —
  this is a feasibility prototype, not a production classifier. *Source:
  `lstm_metrics.txt`.*
- **LSTM+Attention and TCN both reach 87.9%**, `DIVERGENT_REPLICATES`
  precision improved to 0.636. Not yet promoted to the official model —
  flagged for explicit sign-off. A within-comparison LSTM re-run scored
  57.6% under a different seed, showing real hyperparameter-selection
  noise at this dataset size — treat any single run's exact decimal as
  uncertain. *Source: `lstm_tcn_comparison.txt`.*
- **BLIND hold-out result: DROPS sharply to 45.5%** (n=11) vs. 75.8% on
  the 33-chip validation set — the first genuinely blind evidence that
  this model's reported accuracy may not generalize as well as the
  extensively-iterated-against validation numbers suggested. The single
  most important honest finding of the testing week. *Source:
  `holdout_validation_results.txt`.*
- **A flattened-feature Random Forest baseline, properly tuned, scores
  63.6%** (max_depth/n_estimators swept via internal split; an earlier
  untuned version reported 51.5% and was caught understating the
  baseline's real ceiling during independent review) — still clearly below
  the official LSTM (75.8%) and LSTM+Attention/TCN (87.9%), confirming the
  sequence architecture earns real value over a naive baseline, on a
  fairer/harder-to-challenge comparison than the first version made.
  *Source: `benchmark_comparison.txt`.*
- **Attention weights do NOT clearly concentrate in the first 15s**
  (mean weight 0.225, seed 1; 0.294 average — uniform baseline 0.333) —
  does not independently confirm the Week 1 "first 30 seconds matter most"
  biomarker insight the way hoped. **Verified across 6 seeds during
  independent review** (originally left as a hedged single-run
  observation) — the below-uniform direction holds consistently every
  time, upgrading this from suggestive to a genuinely reproducible
  finding. *Source: `attention_weight_analysis.txt`.*
- **Learning-rate/dropout sweep confirms, doesn't improve on, Week 4's
  existing defaults** (tied at 0.548 internal accuracy, not beaten).
  Sequence window (45s/60pts) reconfirmed as the coverage-optimal choice —
  a 60s window would drop/extrapolate at least one real replicate.
  *Source: `hyperparameter_sweep.txt`.*

## Stage 2b — Concentration regressor

- **Probe-stage regression is structurally ill-posed, not undertuned.**
  The concentration-response curve is non-monotonic (peaks at 10 µM); a
  single Δf can map to two real concentrations. No polynomial degree fixes
  this. *Source: `regression_metrics.txt`.*
- **Option C (adding early_displacement_30s as a second feature) does NOT
  meaningfully help** (MAE unchanged: 6.56 → 6.56 µM) — the early biomarker
  correlates too strongly (0.9997) with the endpoint value to add real
  disambiguating information. A genuine negative result, not a failed
  attempt hidden. **Option A (restrict to <10 µM) adopted instead**: MAE
  2.34 µM on 5 real chips below 10 µM. *Source: `regression_curve_shape_fix.txt`.*
- **CAUGHT DURING INDEPENDENT REVIEW (2026-09-11): Option A was "adopted"
  only in the report, never actually wired into the running pipeline.**
  `pipeline_api.py` (used by both the Streamlit mockup and
  `holdout_validation.py`) was still silently training and serving the
  OLD, unscoped, acknowledged-flawed probe model. Fixed: `pipeline_api.py`
  now trains the probe model on the <10 µM subset. A first fix attempt
  (gate individual predictions by whether the input delta_f fell inside
  the <10 µM subset's observed range) was itself found to be a second bug
  in the same review pass: per-chip noise makes that range (-367 to +56 Hz)
  almost as wide as the FULL unrestricted range (-394 to +56 Hz) — 30 of 33
  real chips would pass the gate regardless of their true concentration, so
  it rejected almost nothing. Removed. The correct fix: the model always
  returns a prediction, but every caller must treat it as conditional on
  "assumes true concentration <10 µM" — an assumption that cannot be
  verified from the reading alone, only stated as a caveat
  (`PROBE_SCOPE_CAVEAT` in `pipeline_api.py`). *Source: this review.*
- **Target-stage interpolation near 7 µM fails plausibility** (predicts
  10.8–12.8 µM across all 5 polynomial degrees, outside the 5–10 µM
  plausible range) — may reflect the unverified +63.49 Hz anchor point
  rather than a model flaw; flagged as a candidate for real lab validation
  in the funded phase. *Source:
  `Proposal_Notes_Target_Stage_Interpolation.md`.*
- **BLIND hold-out MAE (post-fix, Option A model)**: probe 1.62 µM
  evaluated honestly on the 4 hold-out chips truly <10 µM (n=4, very small),
  vs. 4.59 µM if the same model is naively applied to all 11 hold-out chips
  regardless of true concentration — a real, measurable cost of ignoring
  the scope caveat. Target 3.77 µM (n=3, also very small). All three numbers
  are illustrative given the tiny n, not conclusive. *Source:
  `holdout_validation_results.txt`.*

## Replicate divergence (cross-cutting)

- **Real replicate pairs flip sign at the 30s read 38% of the time (8/21)
  vs. 4% for the pre-fix synthetic generator** — traced to shared-trend
  vs. independent-kinetics, fixed via per-replicate target jitter
  (`REPLICATE_KINETIC_JITTER_STD=50.0` Hz). Fixing this also surfaced and
  required fixing a second, previously-masked inconsistency (synthetic
  biomarker nulled on divergence; real data never does). *Source:
  `Task4_Replicate_Divergence_Investigation.md`.*
- **The gatekeeper's real contribution isn't raw accuracy on divergent
  chips** (naive threshold got 0/7 wrong anyway in this ablation) — it's
  flagging that a measurement is unreliable BEFORE a confident label gets
  attached, a reliability signal a naive threshold silently discards even
  when it happens to still guess right. *Source: `ablation_results.txt`.*

## Biomarker / threshold consistency

- **Biomarker classification accuracy corrected to 97.7% (43/44), flat
  across all four windows** (15/30/45/60s) — an earlier "97.5%"
  description was mislabeled ("threshold-at-0" when the number already
  used -0.5 Hz), caught and fixed during a finalization-pass consistency
  check. A colleague's independent re-verification reported a different,
  window-varying pattern that could not be reproduced under either of two
  baseline definitions tried — flagged back as a specific, reproducible
  disagreement rather than silently adopted. *Source:
  `clarifying_questions.md` item 8, `biomarker_window_reconciliation.txt`.*

## Meta

- **Two real version-consistency incidents caught this project**: a stale
  threshold value lingering in a draft after Eva's Q4 update, and multiple
  conflicting `chip_summary.csv` Drive copies with genuinely different
  values for the same chips (traced to early Week 1 push artifacts, never
  touched the actual pipeline). Both were caught by direct verification
  against raw data, not assumed correct. *Source:
  `Task0_Data_Integrity_Confirmation.md`.*
