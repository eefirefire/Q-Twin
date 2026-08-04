"""
Week 2 Task 6: Reserve the blind hold-out set.

Randomly selects 9-13 of the 44 valid real chips (20-30%) as a hold-out set
with a reasonable SUCCESS/FAILURE mix, for Weeks 3-4's blind model
validation. Must be excluded from all generator seeding and Week 2-6 work
from this point forward.

Note on leakage (see assemble_batch.py / Task 5 completion notes): the
probe/target generators (Tasks 2-3) were already calibrated against all 44
real chips before this hold-out set existed, matching the Week 2 planning
doc's own Day 1-4 ordering (generators built Days 1-3, hold-out reserved
Day 4). This means the hold-out chips are not fully "unseen" by the
generator's noise/trend calibration, only by explicit per-chip seeding from
this point forward. Worth Eva's attention -- flagged in
Data_Sanity_Audit_draft.md, not silently glossed over.

Output: qtwin/data/holdout_chips.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SEED = 20260811  # Day 4 of Week 2, for reproducibility

N_HOLDOUT = 11  # midpoint of the 9-13 (20-30% of 44) range requested


def main():
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    assert len(valid) == 44, f"expected 44 valid chips, got {len(valid)}"

    rng = np.random.default_rng(SEED)

    # Stratified sample to guarantee a reasonable SUCCESS/FAILURE mix,
    # proportional to the real 29:15 (~66:34%) split rather than leaving it
    # to chance.
    n_success = round(N_HOLDOUT * (29 / 44))
    n_failure = N_HOLDOUT - n_success

    success_ids = valid[valid.success_or_fail == "SUCCESS"]["chip_id"].values
    failure_ids = valid[valid.success_or_fail == "FAILURE"]["chip_id"].values

    holdout_success = rng.choice(success_ids, size=n_success, replace=False)
    holdout_failure = rng.choice(failure_ids, size=n_failure, replace=False)
    holdout = sorted(list(holdout_success) + list(holdout_failure))

    out_path = DATA_DIR / "holdout_chips.txt"
    with open(out_path, "w") as f:
        f.write("# Week 2 Task 6: blind hold-out set for Weeks 3-4 model validation.\n")
        f.write(f"# {len(holdout)} of 44 valid chips ({len(holdout)/44:.1%}), reserved {pd.Timestamp.now().date()}.\n")
        f.write(f"# SUCCESS: {n_success}, FAILURE: {n_failure}\n")
        f.write("# DO NOT use these chip IDs to seed/calibrate any generator or model from this point on.\n")
        f.write("#\n")
        for cid in holdout:
            status = valid.loc[valid.chip_id == cid, "success_or_fail"].iloc[0]
            f.write(f"{cid}\t{status}\n")

    print(f"wrote {out_path} ({len(holdout)} chips: {n_success} SUCCESS, {n_failure} FAILURE)")
    print(holdout)


if __name__ == "__main__":
    main()
