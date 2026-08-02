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
      note("Errata (fixed since the first draft): Delta-f for the probe/target stages was originally computed as each stage's own start-to-end drift. That was wrong — cross-checking against the lab's own \"All results_PCA3.xlsx\" (Mean&SE sheet) showed the correct formula is a stage's ENDPOINT frequency minus the PRIOR stage's endpoint frequency (e.g. probe-endpoint minus chi-endpoint). ingest_raw_curves.py has been fixed and this now reproduces the -62.96 Hz / 10 µM reference almost exactly (see Section 2). All numbers below reflect the corrected values. This also changed which chips are SUCCESS vs FAILURE (30/45 now, was 20/45) and, importantly, weakened the kinetic biomarker's case — see Section 5(c)."),

      h1("1. Concentration bounds"),
      p("Valid range: 0.5–10 µM. Concentrations outside this range are not physically meaningful for this assay and should be excluded from training data for the AI models in Weeks 2+."),
      draftFlag("this range is taken directly from the Week 1 guide (Task 5, item 13). The raw dataset also includes 20 µM, 40 µM and NC chips, used here only to characterize the crowding inversion (Section 2) and negative-control behavior (Section 4) — confirm this is the intended scope, not a contradiction."),

      h1("2. The 10 µM crowding inversion"),
      p("Above roughly 10 µM (probe concentration during the probe-deposition stage), Δf stops getting more negative and starts trending positive — surface crowding limits further mass loading, and non-specific/multilayer effects can push the frequency back up instead of down."),
      p("Reference number from the published paper/summary: −62.96 Hz at 10 µM, room temperature."),
      table,
      note("Table computed from the 14 Mar concentration-sweep chips (probe-endpoint minus chi-endpoint, averaged per chip; n=3 per concentration). The 10 µM mean, -62.04 Hz, now matches the -62.96 Hz reference almost exactly (the ~1 Hz residual is well within instrument noise / endpoint-selection rounding). The concentration trend is also physically clean: strongly positive at 0 µM (no probe, no mass added), peak binding at 10 µM, then decreasing magnitude at 20-40 µM — consistent with the crowding-inversion story, though note it never goes positive again at 20-40 µM in this dataset (magnitude just shrinks), so 'inversion' may be too strong a word for what's really 'crowding-limited plateau'."),
      p("This resolves clarifying_questions.md item 2 — the mismatch was a bug in the original Δf computation (see Errata at top), not a data or lab-notes issue. Fixed in ingest_raw_curves.py."),

      h1("3. Chip failure logic"),
      p("SUCCESS if probe-stage Δf < 0 Hz. FAILURE otherwise (Δf ≥ 0, or missing probe data). Δf here is probe-endpoint minus chi-endpoint (see Errata)."),
      p(`Across the 45-chip dataset: 30 SUCCESS, 15 FAILURE (70.0% success rate). Mean probe Δf for SUCCESS chips is -75.94 Hz (range -372.07 to -0.18); mean for FAILURE chips is +21.36 Hz (range +0.10 to +104.30). The two groups are well separated — the closest chips to the 0 Hz boundary are 21Mar_No.9 (-0.18 Hz, SUCCESS) and 21Mar_No.8 (+0.10 Hz, FAILURE), about 0.3 Hz apart, still supporting 0 Hz as a clean cutoff.`),
      draftFlag("confirm 0 Hz (not some small negative buffer, e.g. -0.5 Hz, to account for instrument noise) is the right cutoff — 21Mar_No.9 and 21Mar_No.8 sit close enough to zero (0.3 Hz apart) that measurement noise could flip either call. Also note the 70.0% chip-level rate is close to, but not identical to, the teacher's 70.18% (40/57) — see clarifying_questions.md item 1, likely a chip-count vs trial-count difference, not a real disagreement (Success rate sheet in All results_PCA3.xlsx counts 57 individual Probe-Chi/Target-Probe trial values, not 45 physical chips)."),

      h1("4. Real negative-control behavior"),
      p("Chips No.7 and No.29 (21 Mar) are the only genuine 'no target present' runs. Plots: figures/NC_21Mar_No_7.png and figures/NC_21Mar_No_29.png."),
      p("With the corrected Δf, both NC chips are clearly FAILURE and clearly positive: No.7 is +3.29 Hz, No.29 is +22.615 Hz (previously this looked like a near-zero borderline call at +0.08 Hz under the buggy formula — the fix makes the negative-control result much cleaner and more convincing, not less). Draft description of the curves themselves: both NC chips show flat, low-amplitude, monotonic upward drift over their full run (roughly +10-15 Hz over 150-220s) with no sharp steps or large excursions — this looks like slow thermal/baseline drift, not a binding event. By contrast, a genuine hybridization curve (e.g. chip No.2, a SUCCESS chip from the same 21 Mar session) shows a fast initial drop of tens of Hz within the first ~20-30s and sharp step-like jumps later in the run, both absent from the NC curves. Chip No.29's probe-stage replicates are noisier than No.7's, with small dips instead of a smooth drift, but neither shows the large negative excursion characteristic of a SUCCESS chip."),
      draftFlag("this paragraph is Evin's read of the plots, standing in for Eva's Task 4 write-up. Eva should look at the actual figures (not just this description) and confirm/adjust — in particular the 'this is thermal drift, not binding' interpretation is an inference, not a lab-verified fact."),

      h1("5. NEW — The Novel Kinetic Biomarker"),
      h2("(a) Time window"),
      p("30 seconds, measured from the start of the probe-stage curve. Computed as the slope of a linear least-squares fit of Resonance_Frequency vs Relative_time over Relative_time ≤ 30s."),
      h2("(b) Why this window avoids the crowding problem"),
      p("The sampling interval is ~0.6s, so a 30s window still gives ~45-50 points — enough for a stable slope estimate. A 60s window was considered but rejected: at the higher probe concentrations in this dataset (20-40 µM), the crowding-driven flattening/reversal described in Section 2 can already be influencing the curve well before 60s, which would contaminate the 'early kinetics' signal with the same surface-saturation effect the biomarker is meant to isolate from. 30s stays safely inside the initial near-linear binding regime for all concentrations tested."),
      h2("(c) Expected behavior — REVISED, weaker than first thought"),
      note("This whole subsection changed once Δf was fixed (see Errata). The original version claimed a clean separation (SUCCESS ≈ -0.0149 Hz/s vs FAILURE ≈ +0.0187 Hz/s). That was measured against SUCCESS/FAILURE labels computed with the buggy same-stage Δf — which is mechanically similar to the biomarker itself (both are 'slope within the probe curve'), so of course they correlated. Against the corrected, lab-formula-matching Δf, the picture is much weaker."),
      p(`With the corrected SUCCESS/FAILURE labels: SUCCESS chips (n=30) have mean early binding rate +0.0061 Hz/s (std 0.035); FAILURE chips (n=15) have mean -0.0024 Hz/s (std 0.034). The sign is now backwards from the original hypothesis and the groups overlap heavily. Correlation between the 30s biomarker and the corrected probe Δf across all 44 scoreable chips is -0.16 — essentially no relationship (down from 0.75 against the old, buggy Δf).`),
      draftFlag("this is the most important open item in the whole document: the kinetic biomarker as currently defined (linear-fit slope of the probe curve's own first 30s) does not reliably predict the (now correctly computed) SUCCESS/FAILURE outcome. Before this goes into Week 2 model training, Eva/the teacher need to decide: (a) is the biomarker concept still worth pursuing with a different definition — e.g. slope of probe-endpoint-relative-to-chi-baseline trajectory, not the probe curve's own internal slope — or (b) should Week 2 drop the kinetic biomarker as a training feature until a version that actually correlates with outcome is found? Right now this feature would likely hurt, not help, a model trained on it."),
      p("The one qualitative pattern that does survive: chip 21Mar_No.29, a true negative control, still shows a fast early binding rate (-0.071 Hz/s, the 2nd-fastest in the dataset) despite a now clearly-positive full-run Δf (+22.615 Hz, unambiguous FAILURE). This is still driven by one noisy replicate declining smoothly for the first 30s while the other replicate is flat with a step artifact — a single-replicate artifact, not real chemistry. It's a smaller-scale example of the same disconnect between the early-window slope and the actual outcome described above."),

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
