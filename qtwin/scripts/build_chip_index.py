"""
Builds chip_index.csv (Eva's Week 1 Task 3) from the same raw folder/file
metadata that ingest_raw_curves.py parses, plus the computed delta-f/status
from chip_summary.csv.

Run ingest_raw_curves.py first so chip_summary.csv exists.
"""

from pathlib import Path

import pandas as pd

from ingest_raw_curves import collect_files, parse_date_folder, DATE_FOLDER_RE

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

MONTH_NUM = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
             "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}


def date_folder_to_iso(name: str) -> str:
    m = DATE_FOLDER_RE.match(name)
    day, mon, year = m.groups()
    return f"{year}-{MONTH_NUM[mon]:02d}-{int(day):02d}"


def main():
    records = collect_files()

    rows = {}
    for rec in records:
        cid = rec["chip_id"]
        row = rows.setdefault(
            cid,
            {
                "Chip_ID": cid,
                "Date": date_folder_to_iso(rec["date_folder"]),
                "Condition_Label": rec["condition_label"],
                "Concentration_uM": rec["concentration_uM"],
                "Temperature": rec["temperature"],
                "_stages": set(),
            },
        )
        row["_stages"].add(rec["stage"])

    index_df = pd.DataFrame(
        [
            {**{k: v for k, v in row.items() if k != "_stages"},
             "Stage_Files_Present": "+".join(sorted(row["_stages"]))}
            for row in rows.values()
        ]
    )

    summary = pd.read_csv(DATA_DIR / "chip_summary.csv")
    index_df = index_df.merge(
        summary[["chip_id", "success_or_fail"]].rename(
            columns={"chip_id": "Chip_ID", "success_or_fail": "Success_or_Fail"}
        ),
        on="Chip_ID",
        how="left",
    )

    index_df = index_df.sort_values(["Date", "Chip_ID"]).reset_index(drop=True)
    out_path = DATA_DIR / "chip_index.csv"
    index_df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(index_df)} chips)")


if __name__ == "__main__":
    main()
