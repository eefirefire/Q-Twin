"""
Loader for qtwin/data/holdout_chips.txt -- the blind hold-out set reserved
in Week 2 Task 6 for Weeks 3-4 model validation.

Written because holdout_chips.txt was only ever WRITTEN by
reserve_holdout.py, never read anywhere in the codebase -- nothing
currently stops Week 3 training code from accidentally including these 11
chips. The file also has 5 '#' comment header lines before the actual
tab-separated data, so a naive pd.read_csv() on it breaks or silently
misparses unless comment handling is done correctly. This module is the
one place that parsing logic should live, so Week 3+ code has a ready-made,
correct way to enforce the hold-out rather than each script re-parsing it
(and risking getting the comment-skipping wrong).

Usage:
    from holdout import load_holdout_chip_ids, split_holdout

    holdout_ids = load_holdout_chip_ids()          # -> set of chip_id strings
    train_df, holdout_df = split_holdout(chip_summary_df)  # chip_id column required
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
HOLDOUT_PATH = DATA_DIR / "holdout_chips.txt"


def load_holdout_chip_ids(path: Path = HOLDOUT_PATH) -> set:
    """Returns the set of chip_id strings reserved as the blind hold-out
    set. Skips '#' comment lines and blank lines; each data line is
    tab-separated as 'chip_id<TAB>SUCCESS_or_FAILURE'."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run reserve_holdout.py first (Week 2 Task 6)."
        )
    ids = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            chip_id = line.split("\t")[0]
            ids.add(chip_id)
    return ids


def split_holdout(df: pd.DataFrame, chip_id_col: str = "chip_id",
                   holdout_ids: set = None) -> tuple:
    """Splits a real-chip dataframe into (train_df, holdout_df) based on
    the chip_id column. Raises if chip_id_col isn't in df, so a caller
    can't silently get an unsplit dataframe back from a typo'd column name."""
    if chip_id_col not in df.columns:
        raise KeyError(f"'{chip_id_col}' not in dataframe columns: {list(df.columns)}")
    if holdout_ids is None:
        holdout_ids = load_holdout_chip_ids()
    is_holdout = df[chip_id_col].isin(holdout_ids)
    return df[~is_holdout].copy(), df[is_holdout].copy()


if __name__ == "__main__":
    # Quick self-check against the real files, so this module's own
    # correctness is verified every time it's run standalone, not just
    # trusted at review time.
    ids = load_holdout_chip_ids()
    print(f"Loaded {len(ids)} hold-out chip IDs: {sorted(ids)}")

    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"]
    train, holdout = split_holdout(valid)
    print(f"chip_summary.csv split: {len(train)} train, {len(holdout)} holdout "
          f"(expected {len(valid) - len(ids)} / {len(ids)})")
    assert len(holdout) == len(ids), "split_holdout count doesn't match holdout file -- investigate before Week 3"
    assert set(holdout["chip_id"]) == ids, "split_holdout IDs don't match holdout file exactly"
    print("OK: split_holdout() verified against chip_summary.csv")
