const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType,
} = require("docx");
const fs = require("fs");

const PAGE = { size: { width: 12240, height: 15840 } }; // US Letter

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 250, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    bullet: { level: 0 },
    spacing: { after: 80 },
  });
}
function note(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, color: "555555" })],
    spacing: { after: 150 },
    border: { left: { style: BorderStyle.SINGLE, size: 12, color: "F5A623", space: 8 } },
    indent: { left: 200 },
  });
}
function draftFlag(text) {
  return new Paragraph({
    children: [new TextRun({ text: "DRAFT — needs Eva's review: ", bold: true, color: "B00020" }),
               new TextRun({ text })],
    spacing: { after: 150 },
  });
}

function cell(text, opts = {}) {
  return new TableCell({
    width: { size: opts.width || 2000, type: WidthType.DXA },
    shading: opts.header ? { type: ShadingType.CLEAR, fill: "1F4E79" } : undefined,
    children: [new Paragraph({
      children: [new TextRun({ text, bold: !!opts.header, color: opts.header ? "FFFFFF" : "000000" })],
    })],
  });
}

const colWidths = [1800, 2200, 1800, 1800, 2200];
const table = new Table({
  width: { size: colWidths.reduce((a, b) => a + b, 0), type: WidthType.DXA },
  columnWidths: colWidths,
  rows: [
    new TableRow({
      children: [
        cell("Concentration", { header: true, width: colWidths[0] }),
        cell("Chips (n)", { header: true, width: colWidths[1] }),
        cell("Mean probe Δf (Hz)", { header: true, width: colWidths[2] }),
        cell("Range (Hz)", { header: true, width: colWidths[3] }),
        cell("Note", { header: true, width: colWidths[4] }),
      ],
    }),
    new TableRow({ children: [cell("0 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("+10.40", { width: colWidths[2] }), cell("+6.28 to +15.65", { width: colWidths[3] }), cell("baseline, no probe", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("5 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("-11.58", { width: colWidths[2] }), cell("-26.05 to +7.01", { width: colWidths[3] }), cell("", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("10 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("-62.04", { width: colWidths[2] }), cell("-105.86 to -31.61", { width: colWidths[3] }), cell("peak binding", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("20 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("-55.30", { width: colWidths[2] }), cell("-120.81 to -10.36", { width: colWidths[3] }), cell("above crowding onset", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("40 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("-43.60", { width: colWidths[2] }), cell("-76.77 to -10.07", { width: colWidths[3] }), cell("", { width: colWidths[4] })] }),
  ],
});

const doc = new Document({
  sections: [{
    properties: { page: PAGE },
    children: [
      new Paragraph({ text: "Biochemical Guardrails — Q-Twin", heading: HeadingLevel.TITLE }),
      p("Draft prepared from the 45-chip raw dataset (14/15/20/21 Mar 2026). This is a first pass built directly off the computed numbers — Eva should read through, correct anything that conflicts with lab notes, and own the final judgment calls (flagged below).", { italics: true, color: "555555" }),
      note("Errata (fixed since the first draft): Delta-f for the probe/target stages was originally computed as each stage's own start-to-end drift. That was wrong — cross-checking against the lab's own \"All results_PCA3.xlsx\" (Mean&SE sheet) showed the correct formula is a stage's ENDPOINT frequency minus the PRIOR stage's endpoint frequency (e.g. probe-endpoint minus chi-endpoint). ingest_raw_curves.py has been fixed and this now reproduces the -62.96 Hz / 10 µM reference almost exactly (see Section 2). This also changed which chips are SUCCESS vs FAILURE."),
      note("Update: chip 15Mar_No.16 is now marked EXCLUDED, not FAILURE (~20,600 Hz nonsensical jump between its CHI and probe stages, a lab-flagged bad measurement, absent from the lab's own Success-rate tally too). 44 of 45 chips are valid: 29 SUCCESS, 15 FAILURE (updated again after Eva's Q4 threshold change in Section 3 — was 30/14 under the old 0 Hz cutoff). The kinetic biomarker (Section 5) has also been redefined as early displacement, replacing the original slope version which was confirmed not predictive — see Section 5(c), and a replicate-concordance rule was added for the rate biomarker — see Section 5(d)."),

      h1("1. Concentration bounds"),
      p("Valid range: 0.5–10 µM. Concentrations outside this range are not physically meaningful for this assay and should be excluded from training data for the AI models in Weeks 2+."),
      draftFlag("this range is taken directly from the Week 1 guide (Task 5, item 13). The raw dataset also includes 20 µM, 40 µM and NC chips, used here only to characterize the crowding inversion (Section 2) and negative-control behavior (Section 4) — confirm this is the intended scope, not a contradiction."),

      h1("2. Two separate crowding curves — probe-stage and target-stage"),
      p("Important distinction (per Eva's clarification): 'crowding at 10 µM' is not one curve, it's two different stages behaving differently. Keeping them separate below, since conflating them was the source of the earlier confusion."),

      h2("2a. Probe Immobilization stage — confirmed, matches reference"),
      p("Above roughly 10 µM (probe concentration during the probe-deposition stage), Δf magnitude peaks and then weakens at higher concentrations — surface crowding limits further mass loading. Reference number: −62.96 Hz at 10 µM, room temperature."),
      table,
      note("Table computed from the 14 Mar concentration-sweep chips (probe-endpoint minus chi-endpoint, averaged per chip; n=3 per concentration). The 10 µM mean, -62.04 Hz, matches the -62.96 Hz reference almost exactly (the ~1 Hz residual is well within instrument noise / endpoint-selection rounding). The trend is physically clean: strongly positive at 0 µM (no probe, no mass added), peak binding at 10 µM, then decreasing magnitude at 20-40 µM. It never goes positive again at 20-40 µM in this dataset (magnitude just shrinks) — so for THIS stage, 'crowding-limited plateau' describes it better than 'inversion'. That's a separate statement from 2b below, not a contradiction of it."),
      p("This resolves clarifying_questions.md item 2 — the -62.96 Hz mismatch was a bug in the original Δf computation (see Errata at top: within-stage drift instead of cross-stage endpoint difference), not a data or lab-notes issue. Fixed in ingest_raw_curves.py."),

      h2("2b. Target Hybridization stage — per teacher's clarification, not independently confirmed here"),
      p("Separately, the Target Hybridization stage is described as having a positive +63.49 Hz shift at 10 µM — a real sign inversion, distinct from the probe stage's plateau in 2a. This is a different curve measuring a different thing (target binding to already-immobilized probe, not probe binding to chitosan)."),
      draftFlag("checked against this dataset's delta_f_target at 10 µM (n=4, noisy): +85.325, -211.450, +42.855, +0.120 Hz — mean -20.79 Hz, or +42.77 Hz excluding the -211.45 outlier. Neither cleanly matches +63.49 Hz, and the sign isn't consistently positive. Unlike 2a's probe-stage number, there isn't enough data here (only 4 target-stage chips at 10 µM, vs. the probe stage's larger sweep) to independently confirm this on our own. Not written in as confirmed fact — flagged in clarifying_questions.md item 15. Eva: could you point to the specific sheet/computation +63.49 Hz comes from, the way All results_PCA3.xlsx resolved 2a's number?"),

      h1("3. Chip failure logic — threshold updated per Eva's Q4"),
      p("SUCCESS if probe-stage Δf <= -0.5 Hz. FAILURE otherwise (Δf > -0.5 Hz, or missing probe data). Δf here is probe-endpoint minus chi-endpoint (see Errata). The -0.5 to 0 Hz band is treated as within instrument baseline noise, not a real signal."),
      p(`Across the 44 valid chips (15Mar_No.16 excluded): 29 SUCCESS, 15 FAILURE (65.9% success rate). This moved one chip relative to the old 0 Hz cutoff: 21Mar_No.9 (-0.18 Hz) flips from SUCCESS to FAILURE, since it falls inside the new noise buffer. Mean probe Δf for SUCCESS chips is -78.56 Hz; mean for FAILURE chips is +19.93 Hz.`),
      p("This resolves clarifying_questions.md item 4/11 — Eva confirmed -0.5 Hz as the buffer. Note the 65.9% chip-level rate is a bit further from the teacher's 70.18% (40/57) than the old 0 Hz cutoff's 68.2% was — see item 1, still most likely a chip-count vs trial-count denominator difference, not a real disagreement."),

      h1("4. Real negative-control behavior"),
      p("Chips No.7 and No.29 (21 Mar) are the only genuine 'no target present' runs. Plots: figures/NC_21Mar_No_7.png and figures/NC_21Mar_No_29.png."),
      p("With the corrected Δf, both NC chips are clearly FAILURE and clearly positive: No.7 is +3.29 Hz, No.29 is +22.615 Hz (previously this looked like a near-zero borderline call at +0.08 Hz under the buggy formula — the fix makes the negative-control result much cleaner and more convincing, not less). Draft description of the curves themselves: both NC chips show flat, low-amplitude, monotonic upward drift over their full run (roughly +10-15 Hz over 150-220s) with no sharp steps or large excursions — this looks like slow thermal/baseline drift, not a binding event. By contrast, a genuine hybridization curve (e.g. chip No.2, a SUCCESS chip from the same 21 Mar session) shows a fast initial drop of tens of Hz within the first ~20-30s and sharp step-like jumps later in the run, both absent from the NC curves. Chip No.29's probe-stage replicates are noisier than No.7's, with small dips instead of a smooth drift, but neither shows the large negative excursion characteristic of a SUCCESS chip."),
      p("Update: Eva has confirmed this is lab-verified thermal/buffer baseline drift, not binding (Q5) — the paragraph above is no longer just Evin's inference from the plots."),

      h1("5. The Novel Kinetic Biomarker — REDEFINED as displacement, validated"),
      h2("(a) Time window"),
      p("30 seconds, measured from the start of the probe-stage curve (also checked at 15s/45s/60s, see (c))."),
      h2("(b) Definition — changed from slope to displacement"),
      p("The original definition was the SLOPE of a linear fit to the probe curve's first 30s (binding_rate_probe_dfdt_30s in chip_summary.csv). That version is confirmed NOT predictive of outcome (see prior draft in version history / clarifying_questions.md item 8) — it's a same-stage-only quantity, mechanically disconnected from delta_f_probe, which is a cross-stage quantity (see Errata)."),
      p("Redefined as early DISPLACEMENT: the probe curve's own value AT t=30s (linearly interpolated) minus the CHI-stage endpoint baseline — the same cross-stage baseline delta_f_probe itself uses, just read 30s into the run instead of at the end. New column: early_displacement_30s in chip_summary.csv."),
      h2("(c) Validation"),
      p("Correlation between early_displacement_30s and the corrected delta_f_probe is 0.9997 across the 44 valid chips — this is essentially an early readout of the same signal, which is exactly why it works as a feature (unlike the slope version, which was a genuinely different quantity). Simple threshold classification (predict SUCCESS if displacement < 0) scores 97.5% (39/40 scoreable chips) at the 30s window in this implementation, and holds flat at 97.5% across 15s/30s/45s/60s windows too."),
      note("These numbers were independently re-derived, not copied from Eva's message — the exact figures (97.5% here vs Eva's reported 97.7%/15s, 95.5%/30s, 93-98% range) differ slightly, most likely from a difference in interpolation or accuracy-scoring method between the two implementations. Both independently land on the same conclusion: displacement is a strong, usable feature; slope was not. Worth a quick sync between the two methodologies before Week 2, but not blocking."),
      p("The one thing to watch: chip 21Mar_No.29 (true negative control) still shows an early value consistent with weak binding under the old slope metric, but the corrected full-run Δf (+22.615 Hz) is unambiguous FAILURE, and the new displacement metric agrees with the outcome for this chip. The previously-flagged single-replicate noise issue was specific to the slope calculation, not the displacement one."),

      h2("(d) Replicate concordance rule for the rate biomarker (Q7)"),
      p("Eva's rule, as specified: calculate the early binding rate for Replicate 1 and Replicate 2 independently. If they agree (both show a similar early trend within an acceptable variance), average the two rates to get the final biomarker value. If they diverge (e.g. one dips while the other is completely flat or spiking), do NOT average them -- flag the sample as \"Divergent Replicates / Artifact Detected\" and reject it, rather than passing a corrupted average to the next stage."),
      p("Written as an explicit rule (sits next to Eva's plain-language version above, not a replacement for it):"),
      note("Let a, b = the two replicates' early binding rates. Agree ⇔ sign(a) = sign(b) AND |a-b| / max(|a|,|b|) <= 0.5. If agree: final_rate = (a+b)/2. If not: final_rate = null, status = DIVERGENT_REPLICATES / Artifact Detected."),
      p("Implemented in curve_generator.py's check_replicate_concordance() (Week 2 generator) and ingest_raw_curves.py's concordance() (Week 1 real-data pipeline) -- same formula in both, so real and synthetic data are flagged the same way. The rate-based biomarker (binding_rate_probe_dfdt_30s) is only averaged across replicates if this rule says they agree; otherwise the chip is flagged binding_rate_replicate_status = DIVERGENT_REPLICATES and the rate is left null rather than passing a corrupted average downstream. Column in chip_summary.csv."),
      p("Verified against Eva's own example: chip 21Mar_No.29's replicate rates are -0.112 Hz/s (R1) vs -0.029 Hz/s (R2) -- same sign but a ~74% relative gap -- correctly flagged DIVERGENT_REPLICATES. Worth noting: the displacement metric in (b)/(c) does NOT diverge for this same chip (19.08 vs 22.33 Hz) because R2's spike happens after the 30s window -- the divergence is specific to the early-rate calculation."),
      draftFlag("with this 0.5 tolerance, 14 of the 22 chips that have 2+ probe replicates (64%) get flagged divergent, not just No.29. That's either genuinely high replicate-to-replicate noise across this instrument, or the tolerance is too strict for real QCM data. Eva/the teacher should spot-check a handful of the other 13 flagged chips before this tolerance is hard-coded into Week 2's generator/gatekeeper logic. See clarifying_questions.md item 12."),

      h1("Sources"),
      bullet("Computed values: qtwin/data/chip_summary.csv (45 chips, all 4 sessions)."),
      bullet("Cross-check against Eva's chip_index.csv: qtwin/data/cross_check_report.csv (0 mismatches)."),
      bullet("Plots: qtwin/figures/ (NC_21Mar_No_7.png, NC_21Mar_No_29.png, and grouped condition plots)."),
    ],
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(__dirname + "/Biochemical_Guardrails.docx", buf);
  console.log("wrote Biochemical_Guardrails.docx");
});
