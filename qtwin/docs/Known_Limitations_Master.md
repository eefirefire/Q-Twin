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

### DETERMINISM BUG: every Stage 2a result before 2026-09-12 needs re-reading through this lens

**`torch.manual_seed()` alone does not make CPU-trained PyTorch models
bit-reproducible.** PyTorch's default multi-threaded CPU execution
(MKL/oneDNN intra-op parallelism) has non-deterministic floating-point
reduction order — thread scheduling varies run to run, and small
differences compound over training into meaningfully different final
weights. Caught while regenerating an attention-weight plot: the exact
same script and seeds produced different numbers on every fresh run, once
even flipping a "CONSISTENT across 6 seeds / VERIFIED ROBUST" claim to
"INCONSISTENT" (see the attention-weights entry below). Investigating the
scope found this affected **every torch training script in the
project** — `model_trainer.py`, `model_trainer_augmented.py`,
`lstm_tcn_comparison.py`, `hyperparameter_sweep.py`,
`augmented_architecture_comparison.py`, `augmented_ensemble.py` — all of
which only ever called `torch.manual_seed()`. (sklearn-based scripts —
`gatekeeper_model.py`, `regression_model.py`, `benchmark_comparison.py`,
`loco_cv_gatekeeper.py` — set `random_state` with no `n_jobs`, i.e.
already single-threaded, and were unaffected; confirmed reproducible
throughout.) Fixed by adding `torch.set_num_threads(1)` +
`torch.use_deterministic_algorithms(True)` to all six scripts, verified
by running each 2–3 times and confirming bit-identical output files.

**The practical impact was large, not cosmetic — two headline results
this project reported were artifacts of thread-scheduling luck, not real
effects, and both had already driven a wrong decision before the bug was
caught:**

1. **The single-replicate augmentation's "45.5% → 72.7% hold-out"
   improvement did not survive.** Under proper determinism: pre-augmentation
   plain LSTM = 0.818 / 33-chip, 0.545 / hold-out. Post-augmentation plain
   LSTM (the model that was promoted to official on the strength of this
   claim) = 0.576 / 33-chip, 0.545 / hold-out. **Zero real hold-out
   improvement, and a real regression in 33-chip accuracy.** The
   augmentation technique itself may still have conceptual merit (the
   root-cause diagnosis — real `SINGLE_REPLICATE` chips being
   out-of-distribution — was independently visible in the 33-chip set,
   not just a hold-out artifact), but the specific numeric claim was
   false, generated by an unlucky/lucky thread-scheduling draw, not by the
   technique. **This directly contradicts a "keep the augmented LSTM
   official" instruction given in this conversation before the bug was
   found — retracted.**
2. **LSTM+Attention/TCN's apparent loss to the plain LSTM on augmented
   data also did not survive.** Under proper determinism (same augmented
   310-sequence set, evaluated on both 33-chip and hold-out): plain LSTM =
   0.788/0.545, LSTM+Attention = 0.727/0.727, TCN = 0.788/0.727. Attention
   and TCN both genuinely beat the plain LSTM on hold-out. **This also
   contradicts a "promote Attention" instruction given in this
   conversation, on different (correct, for once) grounds than
   originally proposed** — the attention-weight visualization does NOT
   support the "model independently rediscovered the early-signal
   insight" story (see below), so Attention was not promoted either;
   **retracted for the wrong reason first, right reason second.**

**PROMOTED (2026-09-12, final): TCN, trained on the single-replicate-augmented
310-sequence set, is the official Stage 2a model**, replacing the
augmented plain LSTM. Decision, made after the corrected numbers above:
TCN tied for best 33-chip AND best hold-out simultaneously in the
architecture comparison (0.788/0.727) — the only candidate best-or-tied
on both metrics — with no interpretability complication the way Attention
has. Not LSTM+Attention (same hold-out, lower 33-chip in that comparison,
plus the attention-weight caveat below). Not the ensemble (identical
numbers to TCN alone, 3x inference cost, zero net gain). Not
pre-augmentation plain LSTM (best 33-chip at 0.818, but worst hold-out at
0.545 — a wide gap consistent with overfitting to the 33-chip set, not
real generalization). **Honest note: the actually-promoted TCN artifact
(trained standalone, its own dedicated run) scores 0.727/0.727, not the
comparison script's in-sequence 0.788/0.727** — training TCN as the only
model in a fresh process consumes the shared PyTorch RNG differently than
training it third-in-sequence after LSTM and Attention within one script,
even with the same seed and the same determinism fix. Both are
individually bit-reproducible; "deterministic" means reproducible given
the same RNG-consumption history, not identical across scripts with
different call sequences. `pipeline_api.py`'s `load_lstm()` was made
architecture-aware (`lstm_config.json`'s new `"architecture"` field) to
support this promotion. *Source: `promote_tcn_official.py`,
`tcn_official_metrics.txt`, `pipeline_api.py`.*

- **Superseded (twice) — pre-augmentation plain LSTM: 0.818 / 33-chip,
  0.545 / hold-out** (deterministic numbers; originally reported as 0.758
  with no standalone hold-out figure). *Source: `lstm_metrics.txt`.*
- **Superseded — single-replicate-augmented plain LSTM (briefly official):
  0.576 / 33-chip, 0.545 / hold-out** (deterministic; originally
  celebrated as 0.758/0.727, a thread-scheduling artifact — see above).
  *Source: `lstm_augmented_metrics.txt`.*
- **Root-cause diagnosis behind the augmentation attempt kept as
  background, not retracted**: real `SINGLE_REPLICATE` chips scored only
  56.25% in the 33-chip validation set (vs. 90–100% for
  CONCORDANT/DIVERGENT chips) because synthetic training data always
  averaged two replicates per sequence, producing lower point-noise than
  any genuine single-replicate curve (`Var(mean of 2 iid) = Var/2`). That
  diagnosis is still independently visible in the data; the fix built on
  top of it (155 additional single-replicate synthetic sequences) simply
  didn't measurably help once evaluated under real determinism.
- **LSTM+Attention and TCN both reach 0.788 (33-chip)** on the
  PRE-augmentation comparison (deterministic; originally reported as an
  identical 87.9% for both — under determinism they no longer tie: LSTM
  baseline=0.727, Attention=0.788, TCN=0.727). *Source:
  `lstm_tcn_comparison.txt`.*
- **A flattened-feature Random Forest baseline, properly tuned, scores
  63.6%** (sklearn, unaffected by the determinism bug — same number
  before and after) — still clearly below every sequence model tested
  (0.727–0.818 pre-augmentation, 0.727–0.788 on augmented data),
  confirming the sequence architecture earns real value over a naive
  baseline. *Source: `benchmark_comparison.txt`.*
- **Attention weights do NOT clearly concentrate in the first 15s**
  (mean weight 0.299, seed 20260907; 0.307 average across 6 seeds —
  uniform baseline 0.333; peak weight occurs at t=38.1s, LATE in the
  window) — does **not** independently confirm the Week 1 "first 30
  seconds matter most" biomarker insight; if anything the model weights
  the LATE part of the curve more. This is the finding that already
  caught the determinism bug (see above) and, separately, the reason
  Attention wasn't promoted even though its accuracy briefly looked
  competitive — the "independent rediscovery" narrative this finding
  would need to support isn't there. **Verified across 6 seeds** under
  the fixed, deterministic settings — the below-uniform-early direction
  holds consistently every time. *Source:
  `attention_weight_analysis.txt`.*
- **Learning-rate/dropout sweep: previously "confirms, doesn't beat,
  defaults" — now a real (if narrow) win.** Under determinism, `lr=0.001,
  dropout=0.4` reaches 0.645 internal-tuning-split accuracy vs. the
  existing default combo's 0.613 (previously both were reported tied at
  0.548). Sequence window (45s/60pts) reconfirmed as the coverage-optimal
  choice — a 60s window would drop/extrapolate at least one real
  replicate. *Source: `hyperparameter_sweep.txt`.*

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
- **`holdout_validation_results.txt` re-run (2026-09-12) with the promoted
  single-replicate-augmented LSTM** — its LSTM hold-out line now reads
  0.727 (gap vs. 33-chip accuracy: +0.031), matching
  `lstm_augmented_metrics.txt` exactly. Regression/gatekeeper numbers were
  unaffected (deterministic re-run, only the LSTM predictions changed).
  This closes the staleness gap flagged right after the LSTM promotion.

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

## Reviewer-anticipated validation (added 2026-09-12)

Responding directly to a set of reviewer-anticipated questions (physics-
informed synthetic data validation, strict blind validation strategy,
comparison against standard algorithms, clinical real-world limitations).
Each item below is either a genuinely new check run against real data, or
an explicit, honest statement of what wasn't attempted and why -- not
fabricated to look complete.

- **LOCO-CV (Leave-One-Chip-Out Cross-Validation), gatekeeper**: **1.000
  accuracy across all 44 real chips** (each fold trains a fresh RF on the
  other 43 real chips only, no synthetic data — a different, complementary
  experiment to the synthetic-trained/real-validated official gatekeeper,
  answering "does early_displacement_30s + is_divergent separate these
  classes on real data alone, across every chip available," not "does the
  synthetic training pipeline generalize"). Strengthens the single
  11-chip hold-out claim with a full-dataset sweep. *Source:
  `loco_cv_gatekeeper.py`, `loco_cv_gatekeeper_results.txt`.*
- **LOCO-CV, LSTM: explicitly NOT attempted**, stated plainly rather than
  silently skipped. Each fold would train on only ~43 real curves for a
  3-class sequence problem — already known to be too small for stable
  LSTM training (this project's synthetic augmentation exists specifically
  because small/skewed training data causes real generalization problems,
  see the Stage 2a promotion above) — so a 44-fold LOCO-CV would produce
  noisy, not-more-informative numbers at real computational cost. Flagged
  as a real open gap for a future phase with more real chips, not
  quietly omitted.
- **PROMOTED FROM CAVEAT TO FINDING (2026-09-12): 45/47 real SUCCESS
  curves are already ≥70% of their own final value at the very FIRST
  recorded sample (t=0.62s) — real binding is essentially complete before
  the instrument's acquisition window starts.** This independently
  explains a result already reported elsewhere in this document:
  `early_displacement_30s` correlating 0.9997 with the full-run endpoint
  `delta_f_probe` isn't a coincidence or a redundant-feature artifact —
  if binding saturates in well under a second, a t=30s reading measures
  the *same* already-reached steady-state value the endpoint does, just
  earlier and with less accumulated drift/noise. Genuine mechanistic
  support for why Week 1's central biomarker discovery works, worth
  stating as a synthesis finding in the proposal. *Source:
  `biological_plausibility_check.py`,
  `biological_plausibility_check.txt`.*
- **Separate, narrower scope limitation (attempting to independently fit
  the generator's k_obs against real curves)**: this specific check
  **degenerated** — no real rise phase is left in the data (per the
  finding above) to fit a rate constant against, so k_obs
  (pseudo-first-order rate, 0.18 +/- 0.05 /s) cannot currently be
  validated against real curves the way a Kd table would. A real, honest
  scope limit on ONE generator parameter — it does not diminish the
  finding above, which stands on its own. *Source: same files.*
- **BUG CAUGHT DURING SELF-REVIEW (2026-09-12): the check above's own
  CHI-stage baseline computation was wrong on first two attempts.** V1
  grouped CHI rows by `chip_id` alone and took the chronologically-last
  row — for chips with two separate CHI recordings (e.g. `20Mar_No.1`,
  CHI files ~39 Hz apart), this silently mixed one replicate's baseline
  into the other's probe curve, producing delta_f values up to 35.4 Hz
  off from `chip_summary.csv`'s own authoritative value. V2 paired
  `(chip_id, replicate)` exactly — better, but still up to 9.3 Hz off for
  chips where CHI and probe have a different replicate count (e.g.
  `15Mar_No.17`: 1 CHI replicate, 2 probe replicates). Fixed by exactly
  replicating `ingest_raw_curves.py`'s own logic (average CHI endpoints
  across CHI's own replicates per chip, apply that one value to every
  probe replicate) — verified to reproduce `chip_summary.csv`'s
  `delta_f_probe` to <1e-8 Hz across all 29 checked SUCCESS chips. The
  core finding (curves already ~100% risen at the first sample) was
  robust to the bug: 46/47 became 45/47, same conclusion. *Source:
  `biological_plausibility_check.py`'s `load_chi_baselines` docstring.*
- **Wasserstein distance, supplementary to the required K-S tests**: added
  alongside (not replacing) the two REQUIRED K-S tests in
  `ks_validation_report.txt`. Delta-f probe: 17.70 Hz (3.7% of real
  range); kinetic biomarker: 12.31 Hz (2.6% of real range) — both small
  relative to the real data's own spread, consistent with (not just
  duplicating) the existing K-S PASS results. *Source:
  `wasserstein_supplementary_check.py`,
  `wasserstein_supplementary_check.txt`.*
- **Temperature / sensor-drift effect: real data exists and was checked**,
  not left as unverified prose. At matched 10 µM concentration, success
  rate drops 75% (RT, n=12) → 67% (37C, n=6) → 50% (60C, n=6), and 60C
  chips show much tighter signal spread (std 32 Hz vs. RT's 129 Hz) — a
  real but statistically inconclusive signal (K-S p=0.26, Mann-Whitney
  p=0.21 with these small n) that heat may degrade probe binding.
  Pre-probe CHI baseline stays near 0 Hz at every temperature (no gross
  instrument drift). None of the three models use temperature as a
  feature, and the synthetic generator has no temperature dimension —
  flagged as a real scope limitation for deployment outside RT. *Source:
  `temperature_effect_analysis.py`, `temperature_effect_analysis.txt`,
  `Clinical_Limitations_Discussion.md`.*
- **Matrix effect / background interference: no real data exists to
  evaluate this, and none was fabricated.** Only 2 real NC chips exist,
  both in the same clean-buffer condition as every other chip — there is
  no real complex-matrix (e.g. synthetic urine) comparison condition in
  this dataset at all. Stated as an open Phase 2 validation gap requiring
  new wet-lab data, not answered with an untested assumption. *Source:
  `Clinical_Limitations_Discussion.md`.*

## Meta

- **Two real version-consistency incidents caught this project**: a stale
  threshold value lingering in a draft after Eva's Q4 update, and multiple
  conflicting `chip_summary.csv` Drive copies with genuinely different
  values for the same chips (traced to early Week 1 push artifacts, never
  touched the actual pipeline). Both were caught by direct verification
  against raw data, not assumed correct. *Source:
  `Task0_Data_Integrity_Confirmation.md`.*
