# Q-Twin: Technical Methods and Results (draft)

**Status:** Draft technical sections, written by Evin per Week 4 Task 5. Ready
to merge with Eva's biomedical sections for the Week 5 proposal. Numbers below
are pulled directly from the committed metrics files in `qtwin/models/` and
`qtwin/data/` as of this writing (2026-08-28), not from memory — see the file
citation at the end of each subsection to independently re-check any number.

**[EVA: BIOMEDICAL BACKGROUND SECTION PLACEHOLDER]** — assay chemistry,
PCA3/QCM biosensor mechanism, and clinical motivation belong here. Not
drafted by Evin; left intentionally blank rather than guessed at.

---

## 1. Methods

### 1.1 Data and the four-component pipeline

The system is built from four components, each with its own validated output:

1. **Ingestion** (`ingest_raw_curves.py`): parses 45 raw QCM chip runs (3
   experimental sessions, 14/15/20/21 March 2026) into `chip_summary.csv`
   (one row per chip: Δf per stage, kinetic biomarker, replicate concordance
   status, SUCCESS/FAILURE label) and `raw_timeseries_master.csv` (full
   per-replicate curves). One chip (`15Mar_No.16`) is excluded, not scored
   FAILURE, for a documented instrument fault (see 1.2).
2. **Synthetic data generation** (`curve_generator.py` +
   `generate_probe_batch.py`/`generate_target_batch.py`): produces 155
   synthetic curves per stage (probe, target), calibrated against the real
   45-chip distribution, validated by Kolmogorov-Smirnov tests against real
   data before being used for training (see 1.4).
3. **Three predictive models** (Stage 0 gatekeeper, Stage 2a sequence model,
   Stage 2b concentration regressor), trained on synthetic data, validated
   against 33 of the 44 valid real chips (11 reserved as a blind hold-out
   set, `holdout_chips.txt`, not used in any training or validation this
   phase).
4. **Interactive review layer** (`app/streamlit_app.py`): runs all three
   models against any selected real chip side by side, for qualitative
   inspection during development — not a claimed result on its own.

### 1.2 Δf computation and data exclusion

For a given stage, Δf is defined as that stage's **endpoint** resonance
frequency minus the **prior** stage's endpoint frequency (e.g. probe-endpoint
minus CHI-endpoint) — a cross-stage difference, not the stage's own internal
drift. This corrected an early implementation bug (documented in
`clarifying_questions.md` item 2); the corrected formula reproduces the
published −62.96 Hz reference at 10 µM to within ~1 Hz (−62.04 Hz measured,
n=3 chips per concentration point).

`15Mar_No.16` is excluded (not scored FAILURE) because its CHI-to-probe
transition shows a ~20,600 Hz discontinuity — verified directly against the
raw curve (CHI ends at 9,996,705.24 Hz, probe starts at 9,976,105.68 Hz) and
consistent with this chip's absence from the lab's own internal success-rate
tracking, i.e. a pre-existing, independently-corroborated instrument fault,
not a result of this pipeline's own processing.

*Source: `clarifying_questions.md` items 2, 9; `qtwin/data/chip_summary.csv`.*

### 1.3 SUCCESS/FAILURE threshold and the kinetic biomarker

**Label:** SUCCESS if `delta_f_probe <= -0.5 Hz`; FAILURE otherwise. The
−0.5 to 0 Hz band is treated as within instrument baseline noise rather than
a real signal (`FAILURE_THRESHOLD_HZ = -0.5` in `constants.py`).

**Kinetic biomarker (`early_displacement_30s`):** the probe curve's own value
at t = 30 s (linearly interpolated), minus the same CHI-stage endpoint
baseline `delta_f_probe` uses — i.e. an early read of the same underlying
signal, not a different one. Correlation with the full-run `delta_f_probe`
across the 44 valid chips: **r = 0.9997**. Threshold classification against
the current −0.5 Hz SUCCESS/FAILURE rule (not the old 0 Hz cutoff — an
earlier draft of this section mislabeled the test as "threshold-at-0";
caught and corrected 2026-08-30, see `clarifying_questions.md` item 8)
scores **97.7% (43/44 scoreable chips)** at the 30 s window, and holds flat
at 97.7% across 15/30/45/60 s windows — this single number is the reason
the Stage 0 gatekeeper (1.5) performs as well as it does.

**Replicate concordance rule:** two replicate readings `a`, `b` (for either
the displacement biomarker or the endpoint value) are called `CONCORDANT`
and averaged if they share sign **and** `|a-b|/max(|a|,|b|) <= 0.5`;
otherwise the chip is flagged `DIVERGENT_REPLICATES`. Implemented identically
in `ingest_raw_curves.py` (real data) and `curve_generator.py` (synthetic
data) via the same tolerance constant
(`REPLICATE_CONCORDANCE_RELATIVE_TOLERANCE = 0.5`).

*Source: `clarifying_questions.md` items 4, 8, 12; `constants.py`.*

### 1.4 Synthetic data generator and validation

Probe-stage curves follow pseudo-first-order (Langmuir) association
kinetics — `trend(t) = final_Δf · (1 − e^(−k_obs·t))` — plus a mean-reverting
(Ornstein-Uhlenbeck) secondary drift term and point-to-point measurement
noise, both calibrated against real chip-to-chip residuals. `final_Δf`
itself is drawn from a fitted concentration-response curve (asymmetric
split-Gaussian in log-concentration, peaking at 10 µM at −62.96 Hz) plus a
bootstrapped real residual.

**Replicate independence (Week 3 Task 4 fix, implemented Week 3 follow-up):**
each of a chip's two replicates now draws an *independently jittered* target
value (`final_Δf + N(0, 50 Hz)`) rather than sharing one chip-level target —
modeling two physically independent immobilization spots rather than two
noisy readings of the same event. This was a direct fix for a measured gap:
real replicate pairs disagree (by the Section 1.3 rule) 57–64% of the time
depending on which biomarker is checked, while the pre-fix generator only
produced 12–13% organic disagreement. Post-fix, the synthetic organic
disagreement rate is 57.6% (probe) / 64.2% (target), matching the real
range. Full root-cause analysis:
`Task4_Replicate_Divergence_Investigation.md`.

**Validation gate:** two required two-sample Kolmogorov-Smirnov tests must
pass (p > 0.05) before any synthetic batch is used for training: (1) probe
delta_f distribution vs. real (currently p = 0.16), (2) kinetic biomarker
distribution vs. real (currently p = 0.26). Both pass on the current,
post-fix batch. A third, supplementary target-stage test is informational
only and currently fails (p = 0.006) — see 2.4.

*Source: `curve_generator.py`; `qtwin/data/ks_validation_report.txt`;
`Task4_Replicate_Divergence_Investigation.md`.*

### 1.5 Model architectures and training

**Stage 0 gatekeeper** (`gatekeeper_model.py`): `RandomForestClassifier`
(200 trees, max depth 6, balanced class weight). Features: the displacement
biomarker (1.3) plus a binary replicate-divergence flag. Trained on 155
synthetic probe curves, validated on the 33 real non-hold-out chips.

**Stage 2a sequence model** (`model_trainer.py`, `sequence_architectures.py`):
takes the full probe-stage curve shape — a 60-point sequence resampled onto a
fixed 0–45 s grid (`resample_curve()`), not just the single 30 s biomarker
value. Three architectures compared (Week 4 Task 4):
- **LSTM** (baseline): single-layer LSTM, final hidden state → linear head.
- **LSTM+Attention**: same LSTM backbone, but an additive-attention layer
  learns to weight whichever time region actually distinguishes a curve,
  rather than compressing the whole 45 s window into one final state.
- **TCN**: dilated causal 1D-convolution stack, global average pooling.

All three use class-weighted cross-entropy loss (inverse class frequency)
and are tuned via an internal 80/20 stratified split of the *synthetic*
training data only — hyperparameters are never selected by looking at real
validation accuracy, the same discipline applied to the hold-out set and the
K-S gate.

**Stage 2b concentration regressor** (`regression_model.py`): polynomial
regression (`PolynomialFeatures` + `LinearRegression`), degree chosen by
5-fold cross-validation, trained separately per stage on synthetic CLEAN-class
curves only.

*Source: `gatekeeper_model.py`, `model_trainer.py`,
`sequence_architectures.py`, `lstm_tcn_comparison.py`, `regression_model.py`.*

---

## 2. Results

*Every number below is reproduced from a committed `qtwin/models/*.txt` file
— the "HONEST CAVEAT" wording in each is preserved deliberately, not
softened for this draft. Per the Week 4 plan: these caveats are a
credibility asset, not a weakness to edit out.*

### 2.1 Stage 0 gatekeeper

**100% accuracy** on 33 real non-hold-out chips (18 SUCCESS, 8 FAILURE, 7
DIVERGENT_REPLICATES — precision/recall/F1 all 1.000 per class).

**Honest caveat, preserved as written:** this is not a hard classification
problem and the perfect score should not be read as evidence of a
sophisticated model. `DIVERGENT_REPLICATES` is caught by construction (the
same concordance check used as a feature), and the displacement biomarker
correlates 0.9997 with the value that literally defines the label. Zero of
the 33 validation chips originally sat within 2 Hz of the decision boundary
— meaning this 100% never tested the hard, near-threshold case (see 2.5).

*Source: `gatekeeper_metrics.txt`.*

### 2.2 Stage 2a sequence model (LSTM / TCN comparison)

Official Task 3 baseline (`lstm_metrics.txt`, hidden_size=16, epochs=220,
class-weighted): **75.8%** accuracy. `DIVERGENT_REPLICATES` recall = 1.00
but precision = 0.47 — the model over-triggers on divergence rather than
missing it (this is a *different* weak point than the one identified in the
original Week 3 prototype, whose problem was the reverse: missing
divergence, 4/7 misclassified as SUCCESS, before the Task 4 replicate-jitter
fix and class-weighted retrain changed the failure mode).

Week 4 Task 4 comparison (`lstm_tcn_comparison.py`, independently re-tuned
run for fair three-way comparison): **LSTM+Attention and TCN both reach
87.9% accuracy** with `DIVERGENT_REPLICATES` precision improved to 0.636
(up from 0.333–0.47) and FAILURE recall at 1.00 (up from 0.50–0.62).
**Recommendation, not yet acted on:** promote LSTM+Attention to the official
Stage 2a model — same recurrent family as the current deliverable, the
improvement is consistent across every tracked metric, and the change is
easy to audit. Held back pending explicit sign-off, the same standard
applied to the Task 4 generator fix.

**Honest caveat, preserved as written:** 155 synthetic training curves for a
3-class problem is small for any sequence model — this remains a
feasibility prototype, not a production classifier. A within-comparison
LSTM re-run scored notably lower (57.6%) than the official saved baseline
under a different random seed, which is itself a finding: this dataset is
small enough that hyperparameter selection is measurably seed-sensitive, so
any single run's exact decimal carries real uncertainty. The qualitative
result (attention/TCN both help) is more robust than any specific number.

*Source: `lstm_metrics.txt`, `lstm_tcn_comparison.txt`.*

### 2.3 Stage 2b concentration regressor

Probe stage: degree-1 polynomial, **MAE 7.09 µM** (31 real chips). Target
stage: degree-3 polynomial, **MAE 5.35 µM** (7 real chips).

**Honest caveat, preserved as written:** the probe-stage concentration-response
curve is non-monotonic (peaks at 10 µM, weakens 20–40 µM) — a single Δf value
can correspond to two different real concentrations, making "predict
concentration from probe Δf" a genuinely ill-posed inverse problem
regardless of model tuning. The probe MAE improved substantially from an
earlier run (16.56 µM) after the Task 4 batch regeneration — verified this
is a side effect of the new random draw (all 5 polynomial degrees converge
to a similar MAE now, vs. a degree-specific fit before), **not** a causal
improvement from the divergence fix itself. Treat the exact number as noisy;
the qualitative non-monotonicity limitation is structural and unaffected by
which random draw the training data happens to be.

### 2.4 Target-stage interpolation limitation (proposal-ready)

> The target-stage concentration regressor does not reliably interpolate
> near 7 µM: it predicts 10.8–12.8 µM depending on polynomial degree (all
> five degrees checked directly, not just the CV-selected one), outside the
> plausible 5–10 µM range. This may reflect the externally-published
> +63.49 Hz/10 µM anchor point itself being unverified against this
> project's own real target-stage data (10 µM real measurements are noisy
> and don't cleanly reproduce that number either) rather than a flaw in the
> regression approach. Flagged as a candidate for direct lab validation in
> the funded phase: a real 7 µM target-stage measurement would distinguish
> "the model doesn't interpolate well" from "the anchor point itself needs
> re-deriving," which the current dataset (10 real target-stage chips, none
> at 7 µM) cannot resolve on its own.

*Source: `Proposal_Notes_Target_Stage_Interpolation.md`, `regression_metrics.txt`.*

### 2.5 Gatekeeper boundary stress test

Motivated directly by 2.1's caveat: 60 synthetic chips with true Δf drawn
within ±3 Hz of the −0.5 Hz decision boundary, run against the exact same
trained gatekeeper.

**Accuracy drops to 91.7%** overall, with SUCCESS recall specifically
falling to **44%** (the model over-predicts FAILURE for near-boundary
SUCCESS chips). Per-band accuracy (small n per band, noisy):
0–0.5 Hz from threshold: 80%; 0.5–1 Hz: 0%; 1–2 Hz: 100%; 2–3 Hz: 60%.

**Interpretation, preserved as written:** this confirms the concern the
original caveat raised. The gatekeeper's real-world 100% should be read as
"validated on the chips this dataset actually has," not "reliable
arbitrarily close to −0.5 Hz" — a genuine performance boundary, reported
honestly rather than patched away. (A first version of this test
accidentally applied the Task 4 replicate jitter to the ±3 Hz target band,
overwhelming it entirely — caught and fixed before trusting the result.)

*Source: `gatekeeper_boundary_stress_test.txt`.*

### 2.6 Replicate-divergence root cause

Real probe replicate pairs flip **sign** at the 30 s read 38% of the time
(8/21 dual-replicate chips) vs. 4% for the pre-fix synthetic generator —
traced to the generator sharing one target value across both replicates and
adding only independent *noise*, not independent *kinetics*. Fixed via
per-replicate target jitter (1.4); not a concordance-rule bug (identical
rule applied to both real and synthetic data).

*Source: `Task4_Replicate_Divergence_Investigation.md`.*

---

## 3. Summary table

| Model | Metric | Value | Real n | Honest caveat headline |
|---|---|---|---|---|
| Stage 0 Gatekeeper | Accuracy | 100% | 33 | Not a hard problem; near-boundary accuracy is 91.7% (2.5) |
| Stage 2a LSTM (official) | Accuracy | 75.8% | 33 | DIVERGENT_REPLICATES precision only 0.47 |
| Stage 2a LSTM+Attention (comparison) | Accuracy | 87.9% | 33 | Recommended, not yet promoted to official model |
| Stage 2a TCN (comparison) | Accuracy | 87.9% | 33 | Matches LSTM+Attention; fewer parameters |
| Stage 2b Regressor (probe) | MAE | 7.09 µM | 31 | Structurally ill-posed (non-monotonic curve) |
| Stage 2b Regressor (target) | MAE | 5.35 µM | 7 | 7 µM interpolation fails plausibility check (2.4) |

---

**[EVA: RESULTS-TO-CLINICAL-SIGNIFICANCE PLACEHOLDER]** — how these
technical results map to the assay's intended clinical use case, sensitivity/
specificity framing for a diagnostic context, and comparison to the TISIIF
paper's reported performance. Not drafted by Evin; left intentionally blank.

**[EVA: DISCUSSION/LIMITATIONS BIOMEDICAL FRAMING PLACEHOLDER]** — biological
plausibility of the replicate-independence finding (2.6) and the
target-stage anchor-point ambiguity (2.4), from a biochemistry standpoint
rather than a modeling standpoint. Not drafted by Evin; left intentionally
blank.
