# Data Sanity Audit — Week 2 Synthetic Batch (DRAFT, Evin's first pass)

**Status: rough draft, written while Eva is out until Sunday.** This is my
own honest read of the synthetic curves and the numbers behind them, not a
polished sign-off — meant as something real for Eva to react to and edit
Sunday rather than a blank page. Everything below is either something I
looked at directly (plots, numbers recomputed from the actual files) or
explicitly marked as my own judgment call, not stated as settled fact.

## What this covers

`synthetic_batch_v1.csv` — 310 curves across 4 classes (100 CLEAN_PCA3_TARGET,
55 DEFECTIVE_CHIP on the probe stage; 100 CLEAN_PCA3_TARGET_HYB, 55
BACKGROUND_SOUP on the target stage), plus 8 intentionally-corrupted
replicate pairs. Full generation details in `curve_generator.py`,
`generate_probe_batch.py`, `generate_target_batch.py`.

## Probe stage (CLEAN_PCA3_TARGET / DEFECTIVE_CHIP) — looks solid

Both required K-S tests pass comfortably (Delta-f p=0.85, biomarker
p=0.30 — both n>>0.05). More important than the p-values: I looked at the
actual curve shapes, not just the statistics.

- **CLEAN_PCA3_TARGET** curves keep a real binding *signature* even at low
  concentration — e.g. the 5 µM example shows a clean fast initial drop
  (0-15s) before settling, the same qualitative shape as the 10 µM and
  20 µM examples, just smaller in magnitude. That's because the probe-stage
  trend is already meaningfully non-zero at 5 µM (~-11.6 Hz mean in the real
  data), so the signal doesn't get lost in the added noise. This looks like
  genuine binding kinetics, not just "a number that happens to be negative."

- **DEFECTIVE_CHIP** curves look like plausible failed/non-specific binding
  — a slow rise to a positive plateau with realistic wander, not chaos and
  not a clean exponential. Visually distinct from both CLEAN_PCA3_TARGET
  (wrong direction, no fast initial drop) and BACKGROUND_SOUP (has a real
  upward trend, not just flat noise).

**My honest take: I'd trust this half of the batch for Week 3 model
training as-is.**

## Target stage (CLEAN_PCA3_TARGET_HYB / BACKGROUND_SOUP) — the real concern

This is the one I'd want Eva's eyes on before anything downstream trusts it.

- The supplementary K-S test for this stage **fails** (p=0.0058) — the
  synthetic target-stage curve is statistically distinguishable from our
  own real target-stage data. This isn't one of the two tests Task 5
  required to pass, so it didn't block the milestone, but it's not nothing.

- Looking at the actual plots, not just the statistic: at 5 µM,
  **CLEAN_PCA3_TARGET_HYB and BACKGROUND_SOUP look visually
  indistinguishable** — both wander noisily around/below zero with no
  clear directional trend, similar magnitude (roughly -2 to -6 Hz). This is
  because the target-stage trend at 5 µM is tiny (~-0.55 Hz by the current
  model) — small enough that the added drift noise dominates and erases
  whatever "clean" shape the trend was supposed to produce. At 10 µM the two
  classes separate clearly (CLEAN_PCA3_TARGET_HYB clearly rises to +60 to
  +120 Hz; BACKGROUND_SOUP stays near zero) — the problem is specifically at
  low concentration.

- **Why this matters for Week 3:** if a classifier trains on this batch,
  it's very plausible it does fine on 10 µM examples and badly on 5 µM
  ones for the target stage specifically — not because the classifier is
  bad, but because the training labels themselves aren't visually separable
  at that concentration. Worth checking per-concentration accuracy in Week 3
  rather than only an aggregate number, if the target-hybridization
  generator ends up feeding the gatekeeper/regression/LSTM models directly.

- This traces back to the underlying issue already flagged in
  `clarifying_questions.md` item 15 and `Biochemical_Guardrails.docx`
  Section 2b: the +63.49 Hz / 10 µM target-inversion number is per
  Eva's/the published paper's spec, not independently confirmed against our
  own real target-stage data (which only has n=4 chips at 10 µM and doesn't
  cleanly show the pattern). **This is the item flagged for Eva's input when
  she's back — not something to resolve without her.**

## Replicate concordance — matches the real-data behavior it's modeled on

Checked the intentional-artifact example directly: Replicate 1 is a clean
binding curve, Replicate 2 is flat then spikes chaotically starting well
after the 30s biomarker window — same pattern as the real
21Mar_No.29 case this rule was built from. Correctly flagged
DIVERGENT_REPLICATES.

One gap worth naming plainly: real chips diverge on replicate concordance
64% of the time (14/22 multi-replicate chips), but this generator's curves
only diverge ~12.6% spontaneously (before the 8 intentional examples). That
means either real QCM replicate noise is higher than this model captures,
or the 0.5 relative-difference tolerance is too strict for real instrument
behavior. I don't think this is resolvable from the data alone — it needs
someone who's actually run the instrument to say which explanation is more
plausible.

## Known statistical gap, not hidden

The "% of final value reached by 30s" statistic (the reason the 30s
biomarker works as an early proxy at all) is wider in synthetic data than
real (std ~0.34 vs real 0.11), traced to a specific, understood cause:
chips whose true endpoint lands very close to the -0.5 Hz threshold make
this ratio mathematically unstable (dividing by near-zero). Doesn't affect
either required K-S test. Full diagnosis in `curve_generator.py`'s
`SECONDARY_DRIFT_STEP_STD` comment.

## Bottom line (my read, not a final verdict)

- Probe stage: solid, would ship as-is.
- Target stage: usable for concentrations >= 10 µM, genuinely weak at
  5 µM specifically — recommend either (a) documenting this as a known
  limitation and having Week 3 check per-concentration performance rather
  than trusting an aggregate number, or (b) holding off on using the
  target-hybridization synthetic data for anything concentration-sensitive
  until Eva can weigh in on the +63.49 Hz number (item 15). I lean toward
  (a) so this doesn't block Week 3's start, but that's a judgment call, not
  a fact — Eva should override this if she disagrees.
- Nothing here changes the two things that were actually required to pass
  (probe-stage Delta-f and biomarker K-S tests) — both still pass. This
  audit is about what's *not* covered by those two tests, not a claim that
  they're wrong.
