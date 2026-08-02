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

# Kinetic biomarker window: see Biochemical_Guardrails.md for the justification.
# 30s is used (not 60s) because at 40 uM the crowding inversion can already start
# bending the curve well before 60s, and the ~0.6s sampling interval gives ~45-50
# points in a 30s window, which is enough for a stable linear-fit slope.
BIOMARKER_WINDOW_S = 30.0

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
    seconds (linear least-squares fit), i.e. dDf/dt in Hz/s."""
    g = group[group["Relative_time"] <= BIOMARKER_WINDOW_S].sort_values("Relative_time")
    if len(g) < 3:
        return None
    slope, _intercept = np.polyfit(g["Relative_time"], g["Resonance_Frequency"], 1)
    return slope


def build_chip_summary(master: pd.DataFrame) -> pd.DataFrame:
    clean = master[~master["is_error_file"]].copy()

    def delta_f(group: pd.DataFrame) -> float:
        g = group.sort_values("Relative_time")
        return g["Resonance_Frequency"].iloc[-1] - g["Resonance_Frequency"].iloc[0]

    # delta-f per (chip, stage, replicate), then averaged across replicates
    per_rep = (
        clean.groupby(["chip_id", "stage", "replicate"])
        .apply(delta_f, include_groups=False)
        .reset_index(name="delta_f")
    )
    per_stage = per_rep.groupby(["chip_id", "stage"])["delta_f"].mean().reset_index()

    pivot = per_stage.pivot(index="chip_id", columns="stage", values="delta_f")
    pivot.columns = [f"delta_f_{c.lower()}" for c in pivot.columns]
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
    rate_per_chip = (
        rate_per_rep.groupby("chip_id")["binding_rate_dfdt"].mean().reset_index()
    )
    rate_per_chip = rate_per_chip.rename(
        columns={"binding_rate_dfdt": f"binding_rate_probe_dfdt_{int(BIOMARKER_WINDOW_S)}s"}
    )

    summary = meta.merge(pivot, on="chip_id", how="left")
    summary = summary.merge(rate_per_chip, on="chip_id", how="left")

    if "delta_f_probe" in summary.columns:
        summary["success_or_fail"] = summary["delta_f_probe"].apply(
            lambda v: "SUCCESS" if pd.notna(v) and v < 0 else "FAILURE"
        )
    else:
        summary["success_or_fail"] = "FAILURE"

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
