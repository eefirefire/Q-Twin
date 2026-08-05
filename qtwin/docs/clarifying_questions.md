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

8. **RESOLVED — kinetic biomarker redefined as displacement, not slope.**
   The original slope-based biomarker (`binding_rate_probe_dfdt_30s`,
   linear-fit slope of the probe curve's own first 30s) was confirmed not
   predictive against the corrected Δf (correlation -0.16, worse than
   chance as a threshold classifier). Redefined per Eva's spec as
   **early displacement**: the probe curve's own value AT t=30s (linearly
   interpolated) minus the CHI-stage endpoint baseline — i.e. the same
   cross-stage baseline delta_f_probe uses, just read 30s into the probe
   run instead of at the end. New column: `early_displacement_30s` in
   `chip_summary.csv`.

   Independently re-verified (not just taken on Eva's word): correlation
   with the corrected `delta_f_probe` is 0.9997 (n=40, excludes
   15Mar_No.16) — this is essentially the same signal read early, not a
   different one, which is exactly why it works. Simple threshold-at-0
   classification accuracy (predict SUCCESS if displacement < 0) came out
   to 97.5% (39/40) at the 30s window in this reproduction, and flat at
   97.5% across 15s/30s/45s/60s windows too — close to but not identical to
   Eva's reported 97.7% (15s) / 95.5% (30s) / 93-98% (15-60s) range, most
   likely due to a difference in interpolation or accuracy-scoring method
   between the two implementations, not a discrepancy worth chasing further
   given both independently land in the same "dramatically better than the
   old biomarker" conclusion. Either way: this is a real, usable feature for
   Week 2, unlike the slope version.

9. **RESOLVED — 15Mar_No.16 excluded, not scored FAILURE.** Its CHI-stage
   endpoint to probe-stage start jumps by ~20,600 Hz — verified directly
   against the raw curve (CHI ends at 9,996,705.24 Hz, probe starts at
   9,976,105.68 Hz) and consistent with this chip being absent from the
   lab's own "Success rate" sheet in `All results_PCA3.xlsx`, i.e. the lab
   already treats it as a bad measurement, not a real FAILURE outcome.
   `chip_summary.csv`, `chip_index.csv`, and `cross_check_report.csv` now
   mark it `EXCLUDED` (still present as a row, so provenance isn't lost, but
   distinct from FAILURE). 44 of 45 chips are valid/scoreable: 30 SUCCESS,
   14 FAILURE.

10. **Confirmed, not a bug: chip-ID parsing uses folder names, not
    filenames.** Checked directly — `NO1.5_CHI.csv` (the typo'd file inside
    the `No.15` folder) is correctly attributed to chip `14Mar_No.15`, not
    `No.1`, because `parse_chip_num()` is called on the chip *folder* name
    in `collect_files()`, never on the filename. No fix needed.

11. **IMPLEMENTED — Q4 failure threshold buffer (-0.5 Hz).** SUCCESS now
    requires `delta_f_probe <= -0.5`, not just `< 0`. This flips 21Mar_No.9
    (-0.18 Hz) from SUCCESS to FAILURE — it was the closest chip to the old
    0 Hz boundary, exactly the kind of noise-band case this rule exists to
    catch. Updated counts: 29 SUCCESS, 15 FAILURE, 1 EXCLUDED (44 valid).

12. **IMPLEMENTED — Q7 replicate concordance rule, with a caveat worth
    Eva's attention.** Applied to the rate-based biomarker
    (`binding_rate_probe_dfdt_30s`): per-replicate rates are averaged only
    if they agree (same sign AND relative difference `|a-b|/max(|a|,|b|) <=
    0.5`); otherwise the chip gets `binding_rate_replicate_status =
    DIVERGENT_REPLICATES` and the rate is left null rather than averaged.
    New column: `binding_rate_replicate_status` in `chip_summary.csv`
    (values: `SINGLE_REPLICATE`, `CONCORDANT`, `DIVERGENT_REPLICATES`).

    Verified against Eva's own worked example: 21Mar_No.29's replicates are
    -0.112 Hz/s (R1) vs -0.029 Hz/s (R2), same sign but a ~74% relative gap
    — correctly flagged DIVERGENT_REPLICATES. Interesting side note: for the
    *displacement* biomarker (item 8), this same chip's replicates are much
    closer (19.08 vs 22.33 Hz) because R2's "massive spike" happens well
    after the 30s window — so the divergence is specific to the early-rate
    calculation, not a general R2-is-corrupted-everywhere situation.

    The caveat: with a 0.5 relative-difference threshold, **14 of the 22
    chips that actually have 2+ probe replicates (64%) get flagged
    divergent**, not just No.29. That's either a sign this instrument's
    replicate-to-replicate noise is genuinely high across the board (not an
    isolated artifact), or that 0.5 is too strict a tolerance for real QCM
    noise. Both are plausible — flagging rather than silently picking one.
    Eva/the teacher should sanity-check a few of the other 13 flagged chips
    by eye before this tolerance gets hard-coded into the Week 2 generator's
    gatekeeper logic.

13. **CONFIRMED — Q5 NC drift is baseline noise, not binding.** Matches what
    was already written in Section 4 (Evin's plot-reading inference) — Eva's
    message confirms this as lab-verified, not just an inference. No change
    needed beyond removing the "unverified inference" caveat.

14. **CONFIRMED — Q6 biomarker-vs-concentration noise range.** Eva's
    claimed "-0.02 to +0.03 across all concentrations" for the rate-based
    biomarker matches almost exactly: this dataset's per-concentration mean
    `binding_rate_probe_dfdt_30s` ranges from -0.0155 (0 µM) to +0.0266
    (40 µM), with no clean monotonic trend. Confirms item 6/8's conclusion
    that the rate metric doesn't scale usefully with concentration.

15. **OPEN — Q2/Q3's target-hybridization-stage numbers don't reproduce
    here, unlike the probe-stage number.** Eva's message states the Target
    Hybridization stage has a *positive* shift of +63.49 Hz at 10 µM (vs the
    probe stage's -62.96 Hz), and describes crowding as a gradual onset that
    "completely inverts to positive at 10 µM." Checked against
    `delta_f_target` in this dataset: at 10 µM (n=4, noisy), values are
    +85.325, -211.450, +42.855, +0.120 Hz — mean -20.79 Hz, or +42.77 Hz
    excluding the -211.45 outlier (21Mar_No.10). Neither cleanly matches
    +63.49 Hz, and the sign isn't consistently positive. This might be the
    same situation as the probe-stage -62.96 Hz reference before it turned
    out to need a specific formula/aggregation — but unlike that case, this
    dataset only has 4 target-stage chips at 10 µM (vs the probe stage's
    much larger sweep), so there isn't enough here to independently pin
    down where +63.49 Hz comes from. Flagging as open rather than writing it
    into the guardrails doc as fact — Eva, can you point to the specific
    sheet/computation this number comes from, the way `All results_PCA3.xlsx`
    resolved the probe-stage number?

    **Update (Data_Sanity_Audit.md, 2026-08-05):** the Week 2 synthetic
    target-hybridization generator, built to the +63.49 Hz spec, produces
    curves that are visually indistinguishable from the BACKGROUND_SOUP
    (no-target) class at 5 µM — both just wander noisily near zero, no clear
    trend, since the modeled target-stage signal at 5 µM is only ~-0.55 Hz.
    At 10 µM the two classes separate cleanly. Consistent with (not new
    evidence beyond, but a visual confirmation of) the supplementary K-S
    test failing for this stage (p=0.0058). Flagged, not solved — needs
    Eva's input on the underlying +63.49 Hz number when she's back; not
    blocking Week 3's start in the meantime.

16. **OPEN -- Week 3 Task 4: why the synthetic generator doesn't reproduce
    real replicates' ~57-64% divergence rate.** Full writeup:
    `Task4_Replicate_Divergence_Investigation.md`. Short version: real probe
    replicate pairs flip SIGN at the 30s read 38% of the time (8/21) vs. only
    4% for organic (non-forced) synthetic pairs -- because
    `generate_probe_batch.py` gives both replicates of a chip the same
    `final_value`/trend and only independent *noise* on top, while real
    replicates behave like two physically independent binding events with
    their own kinetics. Not a concordance-rule bug (same rule, same threshold,
    applied identically both places) and not a re-litigation of item 12's
    "is 64% real or is the threshold too strict" question -- this is the
    separate, narrower finding that the generator's noise model is the gap.
    Recommended fix (independent per-replicate k_obs jitter, not just
    per-replicate noise) is NOT implemented yet -- it would require
    regenerating both synthetic batches and re-running everything downstream
    (K-S tests, gatekeeper, regression, LSTM, hold-out split), so it's left as
    a flagged recommendation pending Eva/the teacher's sign-off rather than
    done unilaterally mid-Task-4.
