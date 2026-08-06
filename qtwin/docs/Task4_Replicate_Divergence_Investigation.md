# Week 3 Task 4: Why real replicates diverge so much more than synthetic ones

**Author:** Evin (solo, Eva out until Sunday). Originally written as an
investigation with the fix deferred pending Eva's sign-off; **updated
2026-08-06 after Eva's Week 3 review explicitly authorized implementing it
now** ("that's a technical call, not a biology one... you're not asking
permission first, just showing the work like always"). The fix described
below IS implemented as of this update -- see "Implementation" at the end.

## The gap, re-measured directly (not from memory)

`clarifying_questions.md` item 12 already flagged this during Week 2: applying
Eva's Q7 concordance rule (same sign AND `|a-b|/max(|a|,|b|) <= 0.5`) to the
**rate** biomarker gives 14 of 22 real dual-replicate probe chips (63.6%)
flagged `DIVERGENT_REPLICATES`. Re-running the identical rule on the
**displacement** biomarker (`early_displacement_30s`, the one actually used by
the Week 3 gatekeeper and LSTM) gives a similar picture: 12 of 21 dual-replicate
chips (57.1%) diverge.

The synthetic generator, by contrast, produces organic (non-forced) divergence
at a much lower rate:

| | n (dual-replicate) | DIVERGENT_REPLICATES | rate |
|---|---|---|---|
| Real probe, rate-based (item 12) | 22 | 14 | 63.6% |
| Real probe, displacement-based (re-measured here) | 21 | 12 | 57.1% |
| Synthetic probe, displacement-based, organic only | 151 | 20 | 13.2% |
| Synthetic target, displacement-based, organic only | 151 | 18 | 11.9% |

("Organic" = excludes the 4 intentionally-corrupted forced-divergent rows per
batch, which exist specifically to be caught and aren't a fair comparison to
real chips' natural variability.)

So the gap is real, reproducible, and consistent across two different
biomarker definitions and both stages: **real chips diverge roughly 4-5x more
often than the synthetic generator produces.**

## Root cause: sign flips, not just noisy magnitudes

Relative-difference-based divergence (`> 0.5`) can happen two ways: two
same-sign values far apart, or two opposite-sign values (which the rule always
flags regardless of magnitude, per Eva's Q7). Splitting these out is where the
real explanation shows up:

| | n | sign-flip rate |
|---|---|---|
| Real probe (dual-replicate, re-measured) | 21 | **38.1%** (8/21) |
| Synthetic probe (organic) | 151 | 4.0% (6/151) |
| Synthetic target (organic) | 151 | 2.6% (4/151) |

Real replicate pairs flip sign more than **9x** as often as synthetic ones.
Examples straight from the data (early_displacement_30s, Hz):

- `21Mar_No.9`: R1 = -1.22, R2 = +0.91 -- one replicate reads net binding,
  the other reads net drift-away, at the exact same 30s timepoint on the
  same chip.
- `20Mar_No.1`: R1 = -30.47, R2 = +2.65
- `21Mar_No.1`: R1 = -61.65, R2 = +17.04

This lines up with a structural difference in how the two datasets are built.
`generate_probe_batch.py`'s `make_replicate_pair()` gives both replicates of a
synthetic chip the **same `final_value`** (one chip-level draw from
`sample_clean_final_value`/`sample_defective_final_value`) and the same
`k_obs` **distribution**, then lets them diverge only through independent
per-timestep measurement noise and independent OU secondary drift on top of
that shared trend. That's a good model of *instrument* noise, but it treats
"replicate 1" and "replicate 2" as two noisy readings of the *same*
underlying binding event.

Real replicates are not that -- they are two physically separate
immobilization spots on the same chip, each with its own (apparently quite
variable) effective binding kinetics. The sign-flip evidence supports that
directly: a shared-trend-plus-noise model essentially cannot flip sign at an
early, still-near-baseline timepoint like 30s unless the noise amplitude is
comparable to or larger than the (still small) trend value itself -- which is
exactly what's *not* built into the current noise parameters (`POINT_NOISE_STD_LOW/HIGH`,
`SECONDARY_DRIFT_STEP_STD=0.6`), since those were tuned to match the real
**endpoint** value distribution (`_real_probe_residuals()`), not real
early-timepoint replicate-pair spread.

## What this is NOT

- **Not a bug in the concordance rule itself.** The same `check_replicate_concordance()`
  function and 0.5 threshold are applied identically to both real and synthetic
  data (verified in `generate_probe_batch.py`/`generate_target_batch.py` vs.
  `ingest_raw_curves.py`) -- the rule isn't miscounting on either side.
- **Not a re-run of item 12's open question.** Item 12 asked whether 64% divergence
  meant "genuinely noisy instrument" or "threshold too strict" and left both open.
  This investigation doesn't resolve that question (it's a real, separate
  judgment call for Eva/the teacher) -- it answers the narrower question Task 4
  asked: *why does the synthetic generator not reproduce that rate*, which is a
  generator-design gap, not a threshold-tuning question.

## Implementation (2026-08-06)

Implemented as `curve_generator.jitter_replicate_final_value()`: each
replicate now gets its own independent target value (`final_value + N(0,
REPLICATE_KINETIC_JITTER_STD)`) before its curve is generated, instead of
both replicates sharing one chip-level `final_value`. Applied in both
`generate_probe_batch.py` and `generate_target_batch.py`'s
`make_replicate_pair()`.

**`REPLICATE_KINETIC_JITTER_STD` calibration.** Swept 10-110 Hz, checking
organic (non-forced) divergence rate and the two required K-S tests
(`assemble_batch.py`) at every value:

- A first pass matched the target divergence rate well (~68 Hz) but broke the
  **biomarker** K-S test outright (p<0.002 at every nonzero jitter tried).
  Root cause turned out to be a separate, pre-existing inconsistency this
  investigation surfaced rather than caused: synthetic
  `early_displacement_30s` was `NaN` on `DIVERGENT_REPLICATES`
  (`check_replicate_concordance()` returned `np.nan`), while real data's own
  `early_displacement_30s` always averages both replicates regardless of
  concordance status (`ingest_raw_curves.py` uses `.mean()` unconditionally,
  with `displacement_replicate_status` as a separate caveat column, never a
  reason to null the value). That mismatch was invisible while synthetic
  divergence was rare (~12%), but once jitter made divergence realistically
  common, the synthetic biomarker K-S sample shrank into an increasingly
  biased "replicates happened to agree" subset and failed against real data's
  full, unbiased population.
- Fixed `check_replicate_concordance()` to always return the mean (matching
  real data's convention exactly), then re-swept 42-58 Hz with 5 seeds each,
  checking both required tests' worst-case p-value every time.

**Final value: `REPLICATE_KINETIC_JITTER_STD = 50.0` Hz.** Organic divergence
rate on the actual regenerated batches: **57.6%** (probe), **64.2%** (target)
-- both land inside the real 57.1-63.6% range this investigation measured.
K-S validation (`ks_validation_report.txt`): both required tests PASS
(probe delta_f p=0.16, biomarker p=0.26).

**Downstream effects, tracked honestly rather than only reporting the fix
itself:**
- **Gatekeeper (Task 1):** unaffected, still 100% on real validation --
  it's no longer leaning on a NaN-encodes-divergence shortcut either (see its
  updated docstring), and the underlying `early_displacement_30s` signal is
  still strong enough on its own.
- **Regression (Task 2):** probe-stage MAE dropped from 16.56 to ~7.1 uM
  across all 5 polynomial degrees -- verified this is a side effect of the
  regenerated data's different RNG draw (shifted by the new per-replicate
  jitter call), not evidence the non-monotonicity problem went away; the
  qualitative ill-posedness is still real and still documented in
  `regression_metrics.txt`.
- **LSTM (Task 3):** accuracy dropped from 87.9% to 51.5% initially --
  matching the real divergence rate shrank `FAILURE` to 13/155 training rows
  (weak-signal/`DEFECTIVE_CHIP` curves are the ones most prone to sign-flips
  under the new jitter, so they get swept into `DIVERGENT_REPLICATES`
  disproportionately). Added class-weighted loss (matching the gatekeeper's
  `class_weight="balanced"`) and extended the internal-tuning-split sweep to
  cover `epochs` as well as `hidden_size` -- recovered to 75.8%, still below
  the pre-fix 87.9% but a real, understood, honestly-reported number, not
  patched by peeking at real-validation accuracy to choose hyperparameters.
  Full detail in `lstm_metrics.txt`.
