# Proposal note: target-stage interpolation limitation

Written per Eva's Week 3 review feedback -- reframed from an open question
into a documented limitation ready to drop into the Week 5 proposal draft,
since the underlying fact has already been independently verified (checked
5 polynomial degrees directly, not just the CV-selected one; see
`regression_metrics.txt`), and Eva's remaining input (whether to schedule a
follow-up lab batch) is a later scheduling conversation, not a blocker here.

## For the proposal

> The target-stage concentration regressor does not reliably interpolate
> near 7 uM: it predicts ~10.8-12.8 uM depending on polynomial degree (all
> five degrees checked, 1 through 5), outside the plausible 5-10 uM range.
> This may reflect the externally-published +63.49 Hz/10 uM anchor point
> itself being unverified against this project's own real target-stage data
> (see `clarifying_questions.md` item 15 -- the real 10 uM measurements are
> noisy and don't cleanly reproduce that number either) rather than a flaw
> in the regression approach. We flag this as a candidate for direct lab
> validation in the funded phase: a real 7 uM target-stage measurement would
> let us tell apart "the model doesn't interpolate well" from "the +63.49 Hz
> anchor point itself needs re-deriving," which the current dataset (10
> real target-stage chips, none at 7 uM) cannot resolve on its own.

## Why this is a limitation, not an open question, as of this writing

- The implausible prediction is confirmed independent of model tuning:
  degrees 1-5 all land outside the 5-10 uM plausibility band (see
  `regression_metrics.txt` for the exact per-degree values from the current
  synthetic batch).
- The test itself is a self-consistency check against the generator's own
  trend function, not against real data -- there is no real 7 uM
  measurement to compare against, so "the model is wrong" and "the anchor
  point is wrong" are both live explanations that current data can't
  distinguish. That ambiguity is real, not something more polynomial-degree
  tuning would resolve (Task 2's own scope was explicitly Polynomial
  Regression, not a switch to a different model family).
- What's left open is not "is this a real limitation" (yes, verified) but
  "which of the two root causes is it" -- and that requires new real lab
  data, which is a scheduling/resourcing decision for Eva/the teacher in a
  later conversation, not something this analysis can settle.
