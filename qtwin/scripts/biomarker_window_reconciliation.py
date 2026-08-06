"""
Reconciles the kinetic biomarker's threshold-classification accuracy
(early_displacement_<window>s, predict SUCCESS if <= FAILURE_THRESHOLD_HZ)
across the 15/30/45/60s windows, against the CURRENT, corrected -0.5 Hz
SUCCESS/FAILURE rule.

Why this exists: the number originally written into clarifying_questions.md
item 8 / build_guardrails_doc.js ("97.5% flat across all four windows,
39/40 scoreable chips") was computed with a `predict SUCCESS if
displacement < 0` threshold (see the exact wording in
clarifying_questions.md: "Simple threshold-at-0 classification accuracy") --
i.e. the OLD 0 Hz cutoff, not the corrected -0.5 Hz FAILURE_THRESHOLD_HZ
rule the rest of the project has used since item 11 was implemented. Eva
flagged this exact staleness independently while finalizing the guardrails
doc. This script re-verifies it directly against real data with the correct
threshold, using the SAME baseline logic as ingest_raw_curves.py's
build_chip_summary() (CHI endpoint averaged per replicate, with the
documented No.7/8/9/10 probe-start fallback) -- reimplemented once here
rather than duplicated ad hoc, so this reconciliation itself is
re-runnable and auditable, not a one-off console computation.

Output: printed table + qtwin/models/biomarker_window_reconciliation.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

import constants as C

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

WINDOWS = [15.0, 30.0, 45.0, 60.0]


def endpoint(group: pd.DataFrame) -> float:
    return group.sort_values("Relative_time")["Resonance_Frequency"].iloc[-1]


def startpoint(group: pd.DataFrame) -> float:
    return group.sort_values("Relative_time")["Resonance_Frequency"].iloc[0]


def value_at(group: pd.DataFrame, t: float):
    g = group.sort_values("Relative_time")
    if len(g) < 2 or g["Relative_time"].max() < t:
        return np.nan
    return float(np.interp(t, g["Relative_time"], g["Resonance_Frequency"]))


def compute_chi_baseline(clean: pd.DataFrame) -> pd.Series:
    """Exact mirror of ingest_raw_curves.py's build_chip_summary() baseline
    logic: CHI endpoint per (chip, replicate), averaged per chip; falls back
    to the probe curve's own start point (averaged per replicate) for the
    No.7/8/9/10 chips with no logged CHI file."""
    chi_end_per_rep = (
        clean[clean["stage"] == "CHI"]
        .groupby(["chip_id", "replicate"])
        .apply(endpoint, include_groups=False)
        .reset_index(name="end_f")
    )
    chi_end_per_chip = chi_end_per_rep.groupby("chip_id")["end_f"].mean()

    probe_start_per_rep = (
        clean[clean["stage"] == "probe"]
        .groupby(["chip_id", "replicate"])
        .apply(startpoint, include_groups=False)
        .reset_index(name="start_f")
    )
    probe_start_per_chip = probe_start_per_rep.groupby("chip_id")["start_f"].mean()

    all_chip_ids = clean["chip_id"].unique()
    chi_end_per_chip = chi_end_per_chip.reindex(all_chip_ids)
    return chi_end_per_chip.fillna(probe_start_per_chip.reindex(all_chip_ids))


def main():
    master = pd.read_csv(DATA_DIR / "raw_timeseries_master.csv")
    clean = master[~master["is_error_file"]].copy()
    probe_rows = clean[clean["stage"] == "probe"]

    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"].set_index("chip_id")

    baseline = compute_chi_baseline(clean)

    lines = []
    lines.append("Biomarker threshold-classification accuracy reconciliation")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Ground truth: chip_summary.csv success_or_fail (44 valid chips, "
                  f"{ (valid.success_or_fail=='SUCCESS').sum()} SUCCESS / "
                  f"{(valid.success_or_fail=='FAILURE').sum()} FAILURE), "
                  f"using the current -0.5 Hz FAILURE_THRESHOLD_HZ rule throughout.")
    lines.append("")
    lines.append(f"{'window':>8s}  {'n_scoreable':>11s}  {'old (0 Hz cutoff)':>18s}  {'correct (-0.5 Hz cutoff)':>24s}")

    for window in WINDOWS:
        val_per_rep = (
            probe_rows.groupby(["chip_id", "replicate"])
            .apply(lambda g: value_at(g, window), include_groups=False)
            .reset_index(name="val")
        )
        val_per_chip = val_per_rep.groupby("chip_id")["val"].mean()

        n_old = n_new = correct_old = correct_new = 0
        for chip_id in valid.index:
            if chip_id not in val_per_chip.index or pd.isna(val_per_chip[chip_id]):
                continue
            if chip_id not in baseline.index or pd.isna(baseline[chip_id]):
                continue
            disp = val_per_chip[chip_id] - baseline[chip_id]
            actual = valid.loc[chip_id, "success_or_fail"]

            pred_old = "SUCCESS" if disp < 0 else "FAILURE"
            n_old += 1
            correct_old += int(pred_old == actual)

            pred_new = "SUCCESS" if disp <= C.FAILURE_THRESHOLD_HZ else "FAILURE"
            n_new += 1
            correct_new += int(pred_new == actual)

        acc_old = correct_old / n_old if n_old else float("nan")
        acc_new = correct_new / n_new if n_new else float("nan")
        lines.append(
            f"{int(window):7d}s  {n_new:11d}  "
            f"{acc_old*100:16.1f}% ({correct_old}/{n_old})  "
            f"{acc_new*100:22.1f}% ({correct_new}/{n_new})"
        )

    text = "\n".join(lines) + "\n"
    print(text)
    out_path = MODEL_DIR / "biomarker_window_reconciliation.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
