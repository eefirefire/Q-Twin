# Task 0: Single-source data integrity confirmation

**One-line confirmation:** Single-source data verified in the codebase/repo
(chip_summary.csv, chip_index.csv, cross_check_report.csv all freshly
regenerated from raw files, byte-identical to the previously-committed
versions, no model was ever trained on stale data) — **with one open item**:
the stale/duplicate Drive copies identified in the prior incident still need
manual removal, since I have no delete/move capability for Drive files.
Testing week proceeds from here with that caveat carried forward, not
silently dropped.

## 1. Fresh regeneration (done, not just re-checked)

Deleted `chip_summary.csv`, `chip_index.csv`, `cross_check_report.csv`, and
`raw_timeseries_master.csv` locally, then regenerated all four from raw
source files only (`ingest_raw_curves.py` -> `build_chip_index.py` ->
`cross_check_index.py`). Result: `git status` came back completely empty --
byte-identical to what was already committed. This confirms the local
pipeline has only ever had one version of each file; there was never a
"which version is correct" ambiguity in the codebase itself.

Spot-checked the three chips from the incident directly against the fresh
regeneration:

| Chip | delta_f_probe |
|---|---|
| 21Mar_No.3 | -56.94 (matches the raw-data-verified correct value) |
| 21Mar_No.1 | -19.14 |
| 21Mar_No.28 | -48.205 |

## 2. Which models were trained on which data

**All of them, always, on the correct data.** `gatekeeper_model.py`,
`regression_model.py`, and `model_trainer.py` all read
`qtwin/data/chip_summary.csv` directly off disk -- the single, git-tracked
file. There is no code path anywhere in this repo that reads from a Drive
copy, a cached snapshot, or an alternate CSV. The multiple-version problem
found in the prior incident existed **only** in a stale Drive folder from
early Week 1 pushes (5 versions of `chip_summary.csv`, 4 of `chip_index.csv`,
4 of `cross_check_report.csv`, one of them under a different Google account)
-- it never touched the actual training pipeline, so **no model needs
re-running because of this incident.**

This is worth stating plainly rather than assuming: the incident was a
documentation/artifact-hygiene problem (old Drive pushes never cleaned up),
not a data-pipeline correctness problem. Re-verified, not assumed.

## 3. What's still outstanding (not silently resolved)

I do not have a delete or move tool for Google Drive files, and permanent
deletion isn't something performed unilaterally even where a tool exists.
The exact stale files were identified and handed off in the prior turn:

- `chip_summary.csv` -- Aug 2 03:48 CSV and the Aug 2 05:56 Google Sheet
  (different account, evarin.srison@gmail.com) -- both stale/wrong
- `chip_index.csv` -- Aug 2 03:47 -- stale
- `cross_check_report.csv` -- Aug 2 03:47 -- stale
- Three additional Aug 3 timestamp groups (06:20 / 07:32-33 / 10:08-09) are
  all *correct* but redundant -- worth collapsing to just the 10:08-09 set

**This item is NOT closed.** Proceeding to Task 1 per the plan's own
instruction that Task 0 blocks on the *pipeline* being clean (confirmed),
while flagging that the Drive cleanup itself needs a human with delete
access -- not deferring the finding, just being honest about what a
tool-constrained agent can and can't finish alone.
