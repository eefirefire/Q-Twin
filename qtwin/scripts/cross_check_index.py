"""
Compares chip_summary.csv (Evin, computed from raw Delta-f) against
chip_index.csv (Eva, lab-context-driven) and reports any Success/Failure
mismatches.
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    summary = pd.read_csv(DATA_DIR / "chip_summary.csv")
    index = pd.read_csv(DATA_DIR / "chip_index.csv")

    merged = summary[["chip_id", "success_or_fail"]].merge(
        index[["Chip_ID", "Success_or_Fail"]],
        left_on="chip_id",
        right_on="Chip_ID",
        how="outer",
        indicator=True,
    )

    only_in_summary = merged[merged["_merge"] == "left_only"]
    only_in_index = merged[merged["_merge"] == "right_only"]
    both = merged[merged["_merge"] == "both"]
    mismatches = both[both["success_or_fail"] != both["Success_or_Fail"]]

    print(f"Chips in both files: {len(both)}")
    print(f"Only in chip_summary.csv: {len(only_in_summary)}")
    print(f"Only in chip_index.csv: {len(only_in_index)}")
    print(f"Status mismatches: {len(mismatches)}")
    if len(mismatches):
        print(mismatches[["chip_id", "success_or_fail", "Success_or_Fail"]].to_string(index=False))

    report_path = DATA_DIR / "cross_check_report.csv"
    merged.to_csv(report_path, index=False)
    print(f"\nfull report written to {report_path}")


if __name__ == "__main__":
    main()
