# Week 3 Task 4: Why real replicates diverge so much more than synthetic ones

**Author:** Evin (solo, Eva out until Sunday). Data-driven investigation, not a
fix -- see Recommendation at the end for why a fix is deferred rather than
silently applied.

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

## Recommendation (not implemented this pass)

The generator would need per-replicate kinetic variability, not just
per-replicate noise on a shared trend -- e.g. drawing `k_obs` (and possibly a
fraction of `final_value` itself) independently per replicate rather than once
per chip, with the independent component sized large enough to occasionally
flip sign at the 30s read. That's a change to `curve_generator.py`'s core
generation function, which would regenerate `probe_synthetic_batch.csv` /
`target_synthetic_batch.csv` and everything downstream of them (the two K-S
tests, the gatekeeper, the regression models, the LSTM, and the hold-out
split's proportions).

Given how much of Weeks 2-3 has already been built and validated on top of the
current batch, and that Eva is out until Sunday, this is flagged as a
recommendation for a deliberate, sign-off'd regeneration rather than something
to silently change mid-Task-4. If Eva/the teacher confirm this read is right,
the concrete next step would be a new `--replicate-kinetic-jitter` parameter on
`generate_association_curve`/`generate_baseline_drift_curve`, re-validated with
the same K-S tests before anything downstream is retrained on it.
