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
    new TableRow({ children: [cell("0 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("-0.72", { width: colWidths[2] }), cell("-1.46 to +0.14", { width: colWidths[3] }), cell("baseline, no probe", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("5 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("+1.15", { width: colWidths[2] }), cell("-0.81 to +4.39", { width: colWidths[3] }), cell("", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("10 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("+3.44", { width: colWidths[2] }), cell("+1.94 to +5.19", { width: colWidths[3] }), cell("above crowding onset", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("20 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("+1.11", { width: colWidths[2] }), cell("-1.12 to +5.12", { width: colWidths[3] }), cell("", { width: colWidths[4] })] }),
    new TableRow({ children: [cell("40 µM", { width: colWidths[0] }), cell("3", { width: colWidths[1] }), cell("+2.91", { width: colWidths[2] }), cell("-1.48 to +6.98", { width: colWidths[3] }), cell("", { width: colWidths[4] })] }),
  ],
});

const doc = new Document({
  sections: [{
    properties: { page: PAGE },
    children: [
      new Paragraph({ text: "Biochemical Guardrails — Q-Twin", heading: HeadingLevel.TITLE }),
      p("Draft prepared from the 45-chip raw dataset (14/15/20/21 Mar 2026). This is a first pass built directly off the computed numbers — Eva should read through, correct anything that conflicts with lab notes, and own the final judgment calls (flagged below).", { italics: true, color: "555555" }),

      h1("1. Concentration bounds"),
      p("Valid range: 0.5–10 µM. Concentrations outside this range are not physically meaningful for this assay and should be excluded from training data for the AI models in Weeks 2+."),
      draftFlag("this range is taken directly from the Week 1 guide (Task 5, item 13). The raw dataset also includes 20 µM, 40 µM and NC chips, used here only to characterize the crowding inversion (Section 2) and negative-control behavior (Section 4) — confirm this is the intended scope, not a contradiction."),

      h1("2. The 10 µM crowding inversion"),
      p("Above roughly 10 µM (probe concentration during the probe-deposition stage), Δf stops getting more negative and starts trending positive — surface crowding limits further mass loading, and non-specific/multilayer effects can push the frequency back up instead of down."),
      p("Reference number from the published paper/summary: −62.96 Hz at 10 µM, room temperature."),
      table,
      note("Table computed from the 14 Mar concentration-sweep chips (probe-stage Δf, averaged per replicate then per chip). Sample size is small (n=3 per concentration) and noisy — the trend is directionally consistent with crowding (Δf becomes less negative / goes positive above ~5–10 µM) but does not cleanly reproduce the −62.96 Hz reference point from the published paper. That number likely comes from a cleaner/aggregated measurement elsewhere in the original dataset."),
      draftFlag("reconcile the −62.96 Hz reference against this table with the teacher — same open question as Task 2 in the Week 1 guide (the 70.18% vs 70.72% mismatch). Also confirm whether 'crowding onset' should be pinned at 10 µM or whether 5 µM already shows early signs (mean here is only weakly negative)."),

      h1("3. Chip failure logic"),
      p("SUCCESS if probe-stage Δf < 0 Hz. FAILURE otherwise (Δf ≥ 0, or missing probe data)."),
      p(`Across the 45-chip dataset: 20 SUCCESS, 25 FAILURE. Mean probe Δf for SUCCESS chips is -1.47 Hz (range -5.09 to -0.13); mean for FAILURE chips is +2.91 Hz (range +0.08 to +7.81). The two groups are well separated around zero — the largest gap in the sorted Δf values near the boundary falls between +0.14 Hz and +0.86 Hz, just above the threshold, supporting 0 Hz as a clean, physically-motivated cutoff (negative Δf = added mass = binding).`),
      draftFlag("confirm 0 Hz (not some small negative buffer, e.g. -0.5 Hz, to account for instrument noise) is the right cutoff — a few chips sit close to zero (No.3: +0.14, No.9: -0.13, No.29 NC: +0.08) where measurement noise could flip the call."),

      h1("4. Real negative-control behavior"),
      p("Chips No.7 and No.29 (21 Mar) are the only genuine 'no target present' runs. Plots: figures/NC_21Mar_No_7.png and figures/NC_21Mar_No_29.png."),
      p("Draft description: both NC chips show flat, low-amplitude, monotonic upward drift over their full run (roughly +10–15 Hz over 150–220s) with no sharp steps or large excursions — this looks like slow thermal/baseline drift, not a binding event. By contrast, a genuine hybridization curve (e.g. chip No.2, a SUCCESS chip from the same 21 Mar session) shows a fast initial drop of tens of Hz within the first ~20–30s and sharp step-like jumps later in the run, both absent from the NC curves. Chip No.29's probe-stage replicates are noisier than No.7's, with small dips instead of a smooth drift, but neither shows the large negative excursion characteristic of a SUCCESS chip."),
      draftFlag("this paragraph is Evin's read of the plots, standing in for Eva's Task 4 write-up. Eva should look at the actual figures (not just this description) and confirm/adjust — in particular the 'this is thermal drift, not binding' interpretation is an inference, not a lab-verified fact."),

      h1("5. NEW — The Novel Kinetic Biomarker"),
      h2("(a) Time window"),
      p("30 seconds, measured from the start of the probe-stage curve. Computed as the slope of a linear least-squares fit of Resonance_Frequency vs Relative_time over Relative_time ≤ 30s."),
      h2("(b) Why this window avoids the crowding problem"),
      p("The sampling interval is ~0.6s, so a 30s window still gives ~45-50 points — enough for a stable slope estimate. A 60s window was considered but rejected: at the higher probe concentrations in this dataset (20-40 µM), the crowding-driven flattening/reversal described in Section 2 can already be influencing the curve well before 60s, which would contaminate the 'early kinetics' signal with the same surface-saturation effect the biomarker is meant to isolate from. 30s stays safely inside the initial near-linear binding regime for all concentrations tested."),
      h2("(c) Expected behavior"),
      p("SUCCESS chips: mean binding rate ≈ -0.0149 Hz/s (std 0.032) over the 30s window — i.e., frequency dropping right from the start."),
      p("FAILURE chips: mean binding rate ≈ +0.0187 Hz/s (std 0.030) — flat or drifting upward from the start, consistent with no early binding."),
      p("Across concentrations: expect the magnitude of the early binding rate to increase with concentration up to the ~10 µM crowding onset (more target/probe available to bind faster), then plateau or shrink above that as surface crowding slows the initial adsorption kinetics too, not just the final Δf."),
      draftFlag("the concentration-dependence claim in the last sentence is a hypothesis based on the crowding model in Section 2, not yet directly verified against this dataset's biomarker column (qtwin/data/chip_summary.csv, binding_rate_probe_dfdt_30s) broken out by concentration group — worth a quick check before this goes in the proposal."),
      p("Known limitation: chip 21Mar_No.29, a true negative control, has the second-fastest early binding rate in the whole dataset (-0.071 Hz/s) despite a near-zero full-run Δf (correctly FAILURE). This comes from one noisy replicate declining smoothly for 30s while the other replicate is flat with a step artifact — i.e. a single-replicate false positive on the early-window biomarker alone."),
      draftFlag("should Week 2's models require agreement across replicates before trusting the 30s biomarker, rather than the simple per-chip average used here? See clarifying_questions.md item 7."),

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
