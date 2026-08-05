"""
Walks qtwin/data/raw, parses the 150 raw QCM time-series CSVs across the
4 lab-session date folders, and produces:

  - raw_timeseries_master.csv : one row per (chip, stage, replicate, timestamp, frequency)
  - chip_summary.csv          : one row per chip, with computed delta-f per stage and
                                 a SUCCESS/FAILURE call based on the probe-stage delta-f

File/folder naming across the 4 sessions is inconsistent (mixed case "CHI"/"chi",
"No."/"NO."/"N0.", space vs underscore, r1/R1/R2 replicate suffixes, the odd
"probe_RT error.csv" file, etc). All of that gets normalized here.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from constants import BIOMARKER_WINDOW_S, EXCLUDED_CHIP_IDS, FAILURE_THRESHOLD_HZ
# BIOMARKER_WINDOW_S, FAILURE_THRESHOLD_HZ, and EXCLUDED_CHIP_IDS used to be
# hardcoded separately in this file (duplicating constants.py, which was
# written afterwards as "the" single source of truth). Consolidated here so
# a future change to any of the three in constants.py can't silently apply
# to Week 2's synthetic generator while this real-data pipeline keeps using
# a stale value -- found on a code review pass before Week 3.
#
# Kinetic biomarker window: see Biochemical_Guardrails.md for the justification.
# 30s is used (not 60s) because at 40 uM the crowding inversion can already start
# bending the curve well before 60s, and the ~0.6s sampling interval gives ~45-50
# points in a 30s window, which is enough for a stable linear-fit slope.

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
OUT_DIR = Path(__file__).resolve().parents[1] / "data"

DATE_FOLDER_RE = re.compile(r"^(\d{1,2}) (\w{3}) (\d{4})$")
CHIP_NUM_RE = re.compile(r"[Nn][O0o]\.?\s*(\d+)", re.IGNORECASE)
CONC_RE = re.compile(r"(\d+(?:\.\d+)?)\s*u[Mm]", re.IGNORECASE)
TEMP_RE = re.compile(r"\b(RT|37\s*C?|60\s*C?)\b", re.IGNORECASE)
REPLICATE_RE = re.compile(r"[_\s][Rr](\d+)\b")
ERROR_RE = re.compile(r"error", re.IGNORECASE)

STAGE_KEYWORDS = ["target", "probe", "chi"]  # order matters: check target/probe before chi


def parse_date_folder(name: str) -> str:
    m = DATE_FOLDER_RE.match(name)
    if not m:
        raise ValueError(f"Unrecognized date folder: {name}")
    day, mon, _year = m.groups()
    return f"{int(day):02d}{mon}"  # e.g. '14Mar' (matches chip_index.csv convention)


def parse_chip_num(name: str) -> str | None:
    m = CHIP_NUM_RE.search(name)
    return m.group(1) if m else None


def parse_concentration(name: str) -> str | None:
    m = CONC_RE.search(name)
    if m:
        return m.group(1)
    if re.search(r"\bNC\b", name, re.IGNORECASE):
        return "NC"
    return None


def parse_temperature(name: str) -> str | None:
    m = TEMP_RE.search(name)
    if not m:
        return None
    val = m.group(1).upper().replace(" ", "")
    if val == "RT":
        return "RT"
    if val.startswith("37"):
        return "37C"
    if val.startswith("60"):
        return "60C"
    return None


def parse_stage(filename: str) -> str | None:
    lower = filename.lower()
    for kw in STAGE_KEYWORDS:
        if kw in lower:
            return "target" if kw == "target" else ("probe" if kw == "probe" else "CHI")
    return None


def parse_replicate(filename: str) -> int:
    m = REPLICATE_RE.search(filename)
    return int(m.group(1)) if m else 1


def collect_files():
    """Return list of dicts with parsed metadata for every raw CSV file."""
    records = []
    for date_dir in sorted(RAW_DIR.iterdir()):
        if not date_dir.is_dir() or not DATE_FOLDER_RE.match(date_dir.name):
            continue
        date_code = parse_date_folder(date_dir.name)

        for csv_path in date_dir.rglob("*.csv"):
            rel_parts = csv_path.relative_to(date_dir).parts
            filename = rel_parts[-1]

            # Concentration/temperature come from the folder structure when present
            # (3-level dates: date/conc-folder/chip-folder/file), else from the
            # chip-folder name itself (21 Mar: date/chip-folder/file).
            if len(rel_parts) == 3:
                conc_folder, chip_folder, _ = rel_parts
                concentration = parse_concentration(conc_folder)
                temperature = parse_temperature(conc_folder) or "RT"
                condition_label = conc_folder
            elif len(rel_parts) == 2:
                chip_folder, _ = rel_parts
                concentration = parse_concentration(chip_folder)
                temperature = parse_temperature(chip_folder) or "RT"
                condition_label = chip_folder
            else:
                continue  # unexpected depth, skip

            chip_num = parse_chip_num(chip_folder)
            stage = parse_stage(filename)
            replicate = parse_replicate(filename)
            is_error = bool(ERROR_RE.search(filename))

            if chip_num is None or stage is None:
                print(f"  [WARN] could not parse metadata from: {csv_path}")
                continue

            chip_id = f"{date_code}_No.{chip_num}"

            records.append(
                {
                    "path": csv_path,
                    "chip_id": chip_id,
                    "date_folder": date_dir.name,
                    "condition_label": condition_label,
                    "concentration_uM": concentration,
                    "temperature": temperature,
                    "stage": stage,
                    "replicate": replicate,
                    "is_error_file": is_error,
                }
            )
    return records


def build_timeseries_master(records):
    frames = []
    for rec in records:
        df = pd.read_csv(rec["path"])
        df = df.rename(columns=lambda c: c.strip())
        if "Relative_time" not in df.columns or "Resonance_Frequency" not in df.columns:
            print(f"  [WARN] unexpected columns in {rec['path']}: {list(df.columns)}")
            continue
        df["chip_id"] = rec["chip_id"]
        df["date_folder"] = rec["date_folder"]
        df["stage"] = rec["stage"]
        df["replicate"] = rec["replicate"]
        df["condition_label"] = rec["condition_label"]
        df["concentration_uM"] = rec["concentration_uM"]
        df["temperature"] = rec["temperature"]
        df["is_error_file"] = rec["is_error_file"]
        df["source_file"] = rec["path"].name
        frames.append(
            df[
                [
                    "chip_id",
                    "date_folder",
                    "stage",
                    "replicate",
                    "condition_label",
                    "concentration_uM",
                    "temperature",
                    "Relative_time",
                    "Resonance_Frequency",
                    "is_error_file",
                    "source_file",
                ]
            ]
        )
    return pd.concat(frames, ignore_index=True)


def binding_rate(group: pd.DataFrame) -> float | None:
    """Slope of Resonance_Frequency vs Relative_time over the first BIOMARKER_WINDOW_S
    seconds (linear least-squares fit), i.e. dDf/dt in Hz/s.

    Superseded as a predictive feature -- see early_displacement() below and
    clarifying_questions.md item 8. Kept in chip_summary.csv for reference /
    comparison, not because it's a good feature."""
    g = group[group["Relative_time"] <= BIOMARKER_WINDOW_S].sort_values("Relative_time")
    if len(g) < 3:
        return None
    slope, _intercept = np.polyfit(g["Relative_time"], g["Resonance_Frequency"], 1)
    return slope


def value_at_time(group: pd.DataFrame, t: float) -> float | None:
    """Linearly-interpolated Resonance_Frequency at Relative_time == t. None if the
    curve doesn't reach t."""
    g = group.sort_values("Relative_time")
    if len(g) < 2 or g["Relative_time"].max() < t:
        return None
    return float(np.interp(t, g["Relative_time"], g["Resonance_Frequency"]))


def build_chip_summary(master: pd.DataFrame) -> pd.DataFrame:
    clean = master[~master["is_error_file"]].copy()

    def endpoint(group: pd.DataFrame) -> float:
        return group.sort_values("Relative_time")["Resonance_Frequency"].iloc[-1]

    def startpoint(group: pd.DataFrame) -> float:
        return group.sort_values("Relative_time")["Resonance_Frequency"].iloc[0]

    # Endpoint (and, for CHI only, startpoint) frequency per (chip, stage, replicate),
    # then averaged across replicates. Matches the lab's own "All results_PCA3.xlsx"
    # methodology: Delta-f for a stage is that stage's endpoint frequency MINUS the
    # PRIOR stage's endpoint frequency (cross-stage), not the stage's own internal
    # drift. Verified against Mean&SE sheet: this reproduces -62.96 Hz at 10 uM RT
    # (see clarifying_questions.md item 2 — this was a bug in the original ingestion,
    # now fixed).
    end_per_rep = (
        clean.groupby(["chip_id", "stage", "replicate"])
        .apply(endpoint, include_groups=False)
        .reset_index(name="end_f")
    )
    end_per_stage = end_per_rep.groupby(["chip_id", "stage"])["end_f"].mean().reset_index()
    end_pivot = end_per_stage.pivot(index="chip_id", columns="stage", values="end_f")

    start_per_rep = (
        clean[clean["stage"] == "CHI"]
        .groupby(["chip_id", "replicate"])
        .apply(startpoint, include_groups=False)
        .reset_index(name="start_f")
    )
    start_per_chip = start_per_rep.groupby("chip_id")["start_f"].mean()

    # Some 21 Mar chips (No.7/8/9/10) have no separate CHI file -- probe was added
    # directly to an already-chitosan-coated chip with no logged chi step. For those,
    # fall back to the probe curve's own start point as the pre-probe baseline (that
    # start point IS the post-chi baseline reading, just not saved as its own file).
    probe_start_per_rep = (
        clean[clean["stage"] == "probe"]
        .groupby(["chip_id", "replicate"])
        .apply(startpoint, include_groups=False)
        .reset_index(name="start_f")
    )
    probe_start_per_chip = probe_start_per_rep.groupby("chip_id")["start_f"].mean()

    pivot = pd.DataFrame(index=end_pivot.index)
    if "CHI" in end_pivot.columns:
        pivot["delta_f_chi"] = end_pivot["CHI"] - start_per_chip.reindex(end_pivot.index)
    if "probe" in end_pivot.columns:
        chi_baseline = end_pivot["CHI"] if "CHI" in end_pivot.columns else pd.Series(index=end_pivot.index, dtype=float)
        baseline = chi_baseline.fillna(probe_start_per_chip.reindex(end_pivot.index))
        pivot["delta_f_probe"] = end_pivot["probe"] - baseline
    if "target" in end_pivot.columns and "probe" in end_pivot.columns:
        pivot["delta_f_target"] = end_pivot["target"] - end_pivot["probe"]
    pivot = pivot.reset_index()

    meta = (
        clean.groupby("chip_id")
        .agg(
            date_folder=("date_folder", "first"),
            condition_label=("condition_label", "first"),
            concentration_uM=("concentration_uM", "first"),
            temperature=("temperature", "first"),
            stages_present=("stage", lambda s: "+".join(sorted(set(s)))),
        )
        .reset_index()
    )

    probe_rows = clean[clean["stage"] == "probe"]
    rate_per_rep = (
        probe_rows.groupby(["chip_id", "replicate"])
        .apply(binding_rate, include_groups=False)
        .reset_index(name="binding_rate_dfdt")
    )

    def concordance(rates: pd.Series) -> pd.Series:
        """Task 5 replicate-concordance rule (per Eva's Q7 spec): average the
        per-replicate early binding rates only if they agree; otherwise flag
        as a divergent-replicate artifact rather than passing a corrupted
        average through. "Agree" = same sign AND relative difference
        |a-b|/max(|a|,|b|) <= 0.5 for every pair. Calibrated against this
        dataset's actual replicate-pair spread (see clarifying_questions.md
        item 11) rather than picked arbitrarily -- chip 21Mar_No.29 (Eva's
        worked example: R1 a normal decline, R2 flat then spiking well after
        the 30s window) sits at ratio 0.74, clearly on the divergent side."""
        vals = rates.dropna().tolist()
        if len(vals) <= 1:
            return pd.Series({"rate": vals[0] if vals else np.nan, "status": "SINGLE_REPLICATE"})
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                a, b = vals[i], vals[j]
                same_sign = (a >= 0) == (b >= 0)
                denom = max(abs(a), abs(b))
                rel_diff = abs(a - b) / denom if denom > 0 else 0.0
                if not same_sign or rel_diff > 0.5:
                    return pd.Series({"rate": np.nan, "status": "DIVERGENT_REPLICATES"})
        return pd.Series({"rate": float(np.mean(vals)), "status": "CONCORDANT"})

    rate_per_chip = (
        rate_per_rep.groupby("chip_id")["binding_rate_dfdt"]
        .apply(concordance)
        .unstack()
        .reset_index()
    )
    rate_per_chip = rate_per_chip.rename(
        columns={
            "rate": f"binding_rate_probe_dfdt_{int(BIOMARKER_WINDOW_S)}s",
            "status": "binding_rate_replicate_status",
        }
    )

    # Redefined kinetic biomarker (validated -- see clarifying_questions.md item 8):
    # early DISPLACEMENT at BIOMARKER_WINDOW_S, i.e. the probe curve's own value at
    # t=30s minus the CHI-stage baseline (same baseline as delta_f_probe, including
    # the No.7/8/9/10 probe-start fallback for chips with no logged CHI file). This
    # replaces binding_rate_probe_dfdt_30s (a same-stage slope, confirmed NOT
    # predictive of the corrected SUCCESS/FAILURE label) with a metric that tracks
    # delta_f_probe almost exactly (corr 0.9997) but 30s into the run instead of at
    # the end -- i.e. an early read of the same signal, not a different one.
    disp_per_rep = (
        probe_rows.groupby(["chip_id", "replicate"])
        .apply(lambda g: value_at_time(g, BIOMARKER_WINDOW_S), include_groups=False)
        .reset_index(name="probe_at_window")
    )
    probe_at_window_per_chip = disp_per_rep.groupby("chip_id")["probe_at_window"].mean()

    displacement = pd.DataFrame(index=end_pivot.index)
    if "CHI" in end_pivot.columns:
        chi_baseline_disp = end_pivot["CHI"].fillna(probe_start_per_chip.reindex(end_pivot.index))
    else:
        chi_baseline_disp = probe_start_per_chip.reindex(end_pivot.index)
    displacement[f"early_displacement_{int(BIOMARKER_WINDOW_S)}s"] = (
        probe_at_window_per_chip.reindex(end_pivot.index) - chi_baseline_disp
    )
    displacement = displacement.reset_index()

    # Week 3 Task 1 needs a replicate-concordance status for early_displacement_30s
    # itself (not just for the rate biomarker above) -- the gatekeeper's secondary
    # feature is supposed to describe the trustworthiness of its primary feature
    # (early_displacement_30s), but until now only the RATE biomarker had a
    # concordance column here (binding_rate_replicate_status), while Week 2's
    # synthetic data computed concordance on DISPLACEMENT instead (see
    # curve_generator.py / generate_probe_batch.py). Different column, different
    # underlying metric -- would have been a real train/validate mismatch for
    # Task 1's gatekeeper if left as-is. Added here so real and synthetic data
    # define this the same way.
    disp_per_rep_baselined = disp_per_rep.copy()
    disp_per_rep_baselined["baseline"] = disp_per_rep_baselined["chip_id"].map(chi_baseline_disp)
    disp_per_rep_baselined["displacement"] = (
        disp_per_rep_baselined["probe_at_window"] - disp_per_rep_baselined["baseline"]
    )
    displacement_concordance = (
        disp_per_rep_baselined.groupby("chip_id")["displacement"]
        .apply(concordance)
        .unstack()
        .reset_index()
    )
    displacement_concordance = displacement_concordance.rename(
        columns={"rate": "_displacement_concordant_mean", "status": "displacement_replicate_status"}
    )
    displacement = displacement.merge(
        displacement_concordance[["chip_id", "displacement_replicate_status"]],
        on="chip_id", how="left",
    )

    summary = meta.merge(pivot, on="chip_id", how="left")
    summary = summary.merge(rate_per_chip, on="chip_id", how="left")
    summary = summary.merge(displacement, on="chip_id", how="left")

    # Threshold per Eva's Q4: SUCCESS requires delta_f_probe <= FAILURE_THRESHOLD_HZ
    # (-0.5 Hz, imported from constants.py). The -0.5 to 0 Hz band is within
    # instrument baseline noise and is now scored FAILURE. See
    # clarifying_questions.md item 4.
    if "delta_f_probe" in summary.columns:
        summary["success_or_fail"] = summary["delta_f_probe"].apply(
            lambda v: "SUCCESS" if pd.notna(v) and v <= FAILURE_THRESHOLD_HZ else "FAILURE"
        )
    else:
        summary["success_or_fail"] = "FAILURE"

    # Lab-flagged bad measurements (imported from constants.py's
    # EXCLUDED_CHIP_IDS -- currently just 15Mar_No.16, ~20,600 Hz nonsensical
    # CHI-to-probe jump, see clarifying_questions.md item "EXCLUDED"),
    # confirmed against the raw curve and consistent with this chip being
    # absent from the lab's own "Success rate" sheet in All results_PCA3.xlsx.
    # Excluded, not scored as FAILURE.
    summary.loc[summary["chip_id"].isin(EXCLUDED_CHIP_IDS), "success_or_fail"] = "EXCLUDED"

    return summary.sort_values("chip_id").reset_index(drop=True)


def main():
    print(f"Scanning {RAW_DIR} ...")
    records = collect_files()
    print(f"  parsed metadata for {len(records)} files")

    master = build_timeseries_master(records)
    master_path = OUT_DIR / "raw_timeseries_master.csv"
    master.to_csv(master_path, index=False)
    print(f"  wrote {master_path} ({len(master)} rows)")

    summary = build_chip_summary(master)
    summary_path = OUT_DIR / "chip_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"  wrote {summary_path} ({len(summary)} chips)")


if __name__ == "__main__":
    main()
