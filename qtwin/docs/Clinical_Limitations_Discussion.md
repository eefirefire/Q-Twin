# Clinical / real-world limitations (Discussion-section draft)

Written in response to a reviewer-anticipated question set: "how will
this system handle challenges in real-world samples?" Two sub-questions
--Background Interference/Matrix Effect, and Sensor Drift/Temperature
Variation-- are addressed here. One has real data behind it; the other
does not, and this document says so plainly rather than writing generic
reassurance prose for either.

## Sensor Drift & Temperature Variation

**This one has real data.** `raw_timeseries_master.csv` includes a
`temperature` field (RT / 37C / 60C) for a subset of chips, mostly at
matched 10 uM concentration -- a de facto small temperature-sweep
sub-experiment inside the existing dataset, not something collected
specifically for this analysis. See
`temperature_effect_analysis.py` / `temperature_effect_analysis.txt` for
the full run.

Findings, restricted to 10 uM chips so concentration isn't confounded
with temperature:

| Temperature | n | Success rate | delta_f_probe mean (std) |
|---|---|---|---|
| RT   | 12 | 75% (9/12) | -84.5 Hz (128.9) |
| 37C  | 6  | 67% (4/6)  | -73.2 Hz (166.1) |
| 60C  | 6  | 50% (3/6)  | -9.4 Hz (32.2)   |

Success rate drops monotonically with temperature, and 60C chips show a
much smaller and tighter signal magnitude (std 32 Hz vs. RT's 129 Hz) --
consistent with heat plausibly degrading probe binding. Neither a K-S
test nor a Mann-Whitney U test between RT and pooled 37C+60C reaches
p<0.05 (p=0.26 and p=0.21 respectively) -- **with n=6 per hot group this
is genuinely underpowered**, so this is reported as a real but
inconclusive signal, not a proven effect. The pre-probe CHI-stage
baseline stays near 0 Hz at every temperature, so there's no evidence of
gross sensor drift from temperature alone before binding starts -- if
anything is happening, it's biological (weaker/less specific binding),
not an instrument artifact.

**Scope limitation, stated plainly**: none of this project's three models
(gatekeeper, LSTM, Stage 2b regressor) take temperature as an input
feature, and `curve_generator.py`'s synthetic data has no temperature
dimension at all -- every synthetic curve implicitly assumes one unstated
condition (effectively RT, since that's the majority of the real
calibration data). The `-0.5 Hz` `FAILURE_THRESHOLD_HZ` has not been
re-derived per temperature. If the funded phase deploys this assay
outside a controlled RT lab setting, these models have not been validated
for that and should not be assumed to transfer without a larger,
purpose-built temperature-sweep dataset (more than 6 chips per
non-RT condition).

## Background Interference / Matrix Effect

**This one does not have real data, and no experiment was run to
manufacture some.** This project's real dataset has exactly two negative
control (NC) chips (`21Mar_No.7`, `21Mar_No.29`), both run in the same
clean-buffer condition as every other chip -- there is no real chip in
this dataset run in an actual complex biological matrix (e.g. real or
synthetic urine with competing proteins/ions), and no comparison
condition exists to measure a matrix effect against. `BACKGROUND_SOUP`
in the synthetic generator is bootstrapped from those same 2 real NC
chips' clean-buffer values (`curve_generator.sample_background_soup_final_value`)
-- it represents "no specific binding in clean buffer," not "signal in a
realistic sample matrix."

**Honest limitation for the Discussion section**: this project cannot
currently distinguish a true PCA3-target signal from a matrix-driven
false signal, because it has never been tested against one. This is a
real, open question for the funded phase, not something this analysis
step can resolve retroactively from the existing dataset -- it needs new
wet-lab data (chips run in synthetic urine or an equivalent complex
matrix, ideally both with and without PCA3 target present) before any
model here can be said to handle it. Recommended framing for the
proposal: flag this explicitly as a validation study needed in Phase 2,
not as a solved problem or an untested assumption left implicit.

## Summary for the Discussion section

- Temperature: a real, small, inconclusive signal exists in this
  project's own data suggesting a possible negative temperature effect on
  assay performance; models are not currently temperature-aware; flagged
  as needing a larger dedicated dataset before deployment outside RT.
- Matrix effect: no real data exists to evaluate this at all; this is an
  open validation gap for Phase 2, stated as such rather than glossed
  over or answered with an untested assumption.
