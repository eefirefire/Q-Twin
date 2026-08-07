# Task 9: Full consistency audit

## File-level duplicate check

- **Local repo**: `chip_summary.csv`, `chip_index.csv`, `cross_check_report.csv`
  each exist exactly once, git-tracked, regenerated from scratch and
  verified byte-identical (see `Task0_Data_Integrity_Confirmation.md`).
- **Drive**: confirmed 5 versions of `chip_summary.csv`, 4 of `chip_index.csv`,
  4 of `cross_check_report.csv` exist in an old, pre-subfolder-structure
  folder from early Week 1 pushes. Two are stale (Aug 2 pushes, one under a
  different Google account). Handed off the exact file list/links for
  manual removal — no delete/move tool available to close this out
  directly. **Not fully resolved**, carried forward honestly rather than
  marked done.

## Cross-document number checks

Grepped every doc for the key numbers that have previously drifted:

- **-0.5 Hz threshold**: consistent everywhere it's stated
  (`constants.py FAILURE_THRESHOLD_HZ`, `clarifying_questions.md`,
  `build_guardrails_doc.js`, all metrics files).
- **Biomarker classification accuracy (97.5% -> 97.7%)**: all remaining
  "97.5%" mentions are explicitly historical/explanatory (describing the
  correction itself), not presented as current. Current canonical number
  (97.7%, 43/44, flat across windows) consistent in
  `clarifying_questions.md`, `build_guardrails_doc.js`, and
  `Week4_Technical_Methods_Results_Draft.md`.
- **Success rate (65.9% vs. 68.2%)**: 68.2% only appears once, explicitly
  labeled as the superseded 0 Hz-cutoff figure.
- **LSTM accuracy (75.8% official / 87.9% comparison / 45.5% hold-out)**:
  every mention is contextualized with which run/split it refers to,
  including in the newly-added `Known_Limitations_Master.md`.
- **Replicate concordance formula** (same-sign AND relative-difference
  <=0.5): identical wording/constant
  (`REPLICATE_CONCORDANCE_RELATIVE_TOLERANCE=0.5`) in
  `curve_generator.py`, `ingest_raw_curves.py`, and both doc sources.

## README

Found and fixed: `qtwin/README.md` was still titled "Week 1" and claimed
`models/`, `app/` were empty placeholders for "Weeks 2+" -- badly stale
given both are now populated with the full trained-model and Streamlit
mockup output. Rewritten to describe the current pipeline end-to-end,
including the testing-week scripts.

## What was NOT exhaustively checked

`Biochemical_Guardrails.docx`/`.pdf` themselves were not re-downloaded and
diffed against the `.js` source in this pass (the `.js` source was kept in
sync during the biomarker-figure fix, but the compiled document wasn't
regenerated, since Eva was actively finalizing it at the time -- see that
prior turn's reasoning). Worth a direct diff once Eva's finalization pass
is done, rather than assumed in sync.
