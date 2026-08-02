# Clarifying questions on the guardrails draft (Evin's Task 6)

Sent ahead of Week 2. These are the "DRAFT — needs Eva's review" flags from
`Biochemical_Guardrails.docx`, pulled out into one list. **Update (2026-08-02):
found and fixed a real bug in the Δf computation — see item 2, and the new
item 8, which is now the most important open question in this list.**

1. **Success-rate mismatch (also Eva's Task 2):** the teacher's number
   (40/57 = 70.18%) vs the TISIIF paper's 70.72%, vs the 30/45 = 70.0%
   SUCCESS rate now computed here (after the Δf fix in item 2 — it was 44.4%
   before the fix). 70.0% and 70.18% are now very close. The remaining gap
   is very likely a chip-count (45) vs trial-count (57) difference: the
   "Success rate" sheet in `data/raw/All results_PCA3.xlsx` counts 57
   individual Probe-Chi/Target-Probe measurements (some chips contribute 2
   values when they have 2 replicates, e.g. 20 Mar's "RT-1-1"/"RT-1-2" style
   rows), not 45 physical chips. Eva/the teacher should confirm this is the
   right way to reconcile it, and which denominator (chip-level or
   trial-level) Week 2 should standardize on.

2. **Crowding reference number — RESOLVED, was a bug on our end.** The guide
   cites −62.96 Hz at 10 µM, room temperature. Original Δf computation (each
   stage's own start-to-end drift) gave +3.44 Hz at 10 µM — didn't match.
   Cross-checking against `All results_PCA3.xlsx` (Mean&SE sheet) showed the
   lab's actual formula is: a stage's ENDPOINT frequency minus the PRIOR
   stage's endpoint frequency (e.g. probe-endpoint minus chi-endpoint), not
   that stage's own internal drift. Fixed in `ingest_raw_curves.py` — now
   reproduces -62.04 Hz at 10 µM, matching -62.96 Hz almost exactly (~1 Hz
   residual, within instrument noise). No further action needed here, but
   see item 8 — the fix has a knock-on consequence for the biomarker.

3. **Crowding onset:** with the corrected Δf, the concentration-sweep table
   (0/5/10/20/40 µM) shows binding turns clearly negative between 0 and 5 µM
   and peaks in magnitude at 10 µM, then the magnitude *shrinks* at 20-40 µM
   without going positive again in this dataset. So "crowding onset" may be
   better described as "binding plateaus/weakens above 10 µM" rather than a
   full sign inversion — worth confirming against lab notes whether a true
   inversion (positive Δf) was ever observed at higher concentrations.

4. **Failure threshold:** with the corrected Δf, is a hard 0 Hz cutoff still
   right? The boundary is actually cleaner now — the two closest chips to
   zero are 21Mar_No.9 (-0.18 Hz, SUCCESS) and 21Mar_No.8 (+0.10 Hz,
   FAILURE), about 0.3 Hz apart. Still worth confirming whether a small
   buffer is wanted for instrument noise.

5. **NC curve interpretation:** the draft in Section 4 calls the NC drift
   "thermal/baseline drift, not binding" — that's Evin's inference from the
   plots, not a lab-verified fact. Eva should confirm from lab notes/context.
   Note the corrected Δf for both real NC chips (No.7: +3.29 Hz, No.29:
   +22.615 Hz) is now unambiguously positive/FAILURE — the negative-control
   result reads cleaner after the fix, not weaker.

6. **Biomarker concentration trend:** superseded by item 8 — the biomarker
   doesn't cleanly track outcome at all anymore, so a concentration
   breakdown of it isn't very meaningful yet without redefining it first.

7. **Biomarker false positive on a real NC chip:** chip 21Mar_No.29 still
   has the 2nd-fastest early binding rate in the dataset (-0.071 Hz/s)
   despite its corrected full-run Δf being clearly positive (+22.615 Hz,
   FAILURE). Still traced to one noisy replicate declining smoothly for 30s
   while the other is flat with a step artifact. This is now understood as
   one specific case of the broader problem in item 8, rather than an
   isolated edge case.

8. **NEW, most important: the kinetic biomarker doesn't predict the
   corrected outcome.** The original "SUCCESS ≈ -0.0149 Hz/s vs FAILURE ≈
   +0.0187 Hz/s" claim in Section 5(c) was measured against SUCCESS/FAILURE
   labels from the *buggy* Δf — which was itself a same-stage slope, so it's
   not surprising it correlated with another same-stage slope (the
   biomarker). Against the corrected, lab-formula-matching Δf: SUCCESS chips
   average +0.0061 Hz/s, FAILURE chips average -0.0024 Hz/s (backwards from
   the original hypothesis, heavily overlapping, correlation -0.16 — down
   from 0.75 against the old Δf). Before Week 2 trains anything on this
   biomarker, Eva/the teacher need to decide: redefine it (e.g. as the early
   slope of probe-relative-to-chi-baseline, matching how Δf itself is now
   computed, rather than the probe curve's own internal slope), or drop it
   as a feature until a version that actually correlates with outcome is
   found. As currently defined, it would likely hurt a trained model more
   than help.
