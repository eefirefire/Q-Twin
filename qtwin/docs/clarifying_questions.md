# Clarifying questions on the guardrails draft (Evin's Task 6)

Sent ahead of Week 2. These are the "DRAFT — needs Eva's review" flags from
`Biochemical_Guardrails.docx`, pulled out into one list:

1. **Success-rate mismatch (also Eva's Task 2):** the teacher's number
   (40/57 = 70.18%) vs the TISIIF paper's 70.72%. Which one do we use going
   forward, and does it match the 20/45 = 44.4% SUCCESS rate computed here
   from the raw dataset? (Different denominators — 57 vs 45 chips — so this
   may just be a different/larger historical dataset, but worth confirming
   it isn't a parsing bug on our end.)

2. **Crowding reference number:** the guide cites −62.96 Hz at 10 µM, room
   temperature, but the 14 Mar concentration-sweep chips in this dataset
   average +3.44 Hz at 10 µM (noisy, n=3). Is −62.96 Hz from a different
   measurement (e.g. aggregated/cleaned data, or a different stage than
   "probe")? Which one should the Week 2 synthetic data generator target?

3. **Crowding onset:** is 10 µM the right cutoff, or does crowding start
   showing up as early as 5 µM in this dataset (mean Δf there is only
   weakly negative)?

4. **Failure threshold:** is a hard 0 Hz cutoff right, or should there be a
   small negative buffer (e.g. −0.5 Hz) to account for instrument noise?
   Three chips sit within ~0.15 Hz of zero (No.3, No.9, No.29).

5. **NC curve interpretation:** the draft in Section 4 calls the NC drift
   "thermal/baseline drift, not binding" — that's Evin's inference from the
   plots, not a lab-verified fact. Eva should confirm from lab notes/context.

6. **Biomarker concentration trend:** Section 5(c) predicts binding rate
   should rise with concentration up to ~10 µM then plateau/shrink. This is
   inferred from the crowding model, not yet checked against
   `chip_summary.csv`'s `binding_rate_probe_dfdt_30s` column broken out by
   concentration. Worth a quick look before it goes in the proposal.

7. **Biomarker false positive on a real NC chip:** chip 21Mar_No.29 (a true
   negative control — no target present) has the *second-fastest* early
   binding rate in the entire 45-chip dataset (-0.071 Hz/s), yet its full-run
   probe Δf lands near zero (+0.08 Hz, correctly FAILURE). Looking at the raw
   points, this is driven almost entirely by replicate 1 declining smoothly
   for the first 30s while replicate 2 is flat/noisy with a mid-window step
   artifact — likely sensor noise on a single replicate, not real chemistry.
   This means the 30s binding-rate biomarker can look "success-like" on a
   genuine NC if only one replicate is checked. Eva/the teacher should weigh
   in on whether Week 2's models should require agreement across replicates
   (not just an averaged rate) before trusting the biomarker alone.
