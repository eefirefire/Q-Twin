"""
Week 2 Task 3: Build the target hybridization generator.

Generates two classes of synthetic target-hybridization-stage curves:
  - CLEAN_PCA3_TARGET_HYB: the "real binding" class for this stage, using
    the (UNVERIFIED_AGAINST_LOCAL_DATA -- see curve_generator.py and
    constants.py) weak-negative-to-+63.49Hz-inversion model per Eva's Q2/Q3.
  - BACKGROUND_SOUP: informed by the real NC chips (21Mar_No.7, No.29),
    chaotic/non-Langmuir baseline drift (Eva's Q5, confirmed).

The Week 2 planning doc's Task 3 output line names only
"BACKGROUND_SOUP curves for the target stage" -- but item 7 clearly
specifies a signal model (the weak-to-inversion curve) that needs
representative samples too, so both classes are generated here rather than
only BACKGROUND_SOUP. Flagged in this docstring in case that reading is
wrong and Eva only wanted BACKGROUND_SOUP for this stage.

Output: qtwin/data/target_synthetic_batch.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import constants as C
import curve_generator as cg
from generate_probe_batch import make_forced_divergent_pair  # reuse the same artifact-injection logic

OUT_DIR = Path(__file__).resolve().parents[1] / "data"

N_CLEAN = 100
N_BACKGROUND = 55
N_INTENTIONAL_DIVERGENT = 4

BASE_SEED = 20260809  # different seed than probe batch, still reproducible


def _real_target_concentration_weights():
    cs = pd.read_csv(OUT_DIR / "chip_summary.csv")
    cs = cs[cs.success_or_fail != "EXCLUDED"]
    conc = pd.to_numeric(cs["concentration_uM"], errors="coerce").dropna()
    conc = conc[conc > 0]
    return conc.values


def make_replicate_pair(final_value: float, kind: str, rng: np.random.Generator, duration_s: float):
    reps = []
    for _ in range(2):
        if kind == "CLEAN_PCA3_TARGET_HYB":
            t, y = cg.generate_association_curve(final_value, duration_s, rng)
        else:  # BACKGROUND_SOUP
            t, y = cg.generate_baseline_drift_curve(final_value, duration_s, rng)
        reps.append((t, y))
    return reps


def build_rows(kind: str, n: int, rng: np.random.Generator, concs: np.ndarray, forced_divergent: int = 0):
    rows = []
    gen_kind_for_forced = "CLEAN_PCA3_TARGET" if kind == "CLEAN_PCA3_TARGET_HYB" else "DEFECTIVE_CHIP"
    for i in range(n):
        conc = float(rng.choice(concs))
        duration = cg.sample_duration(rng)
        if kind == "CLEAN_PCA3_TARGET_HYB":
            final_value = cg.sample_target_clean_final_value(conc, rng)
        else:
            final_value = cg.sample_background_soup_final_value(rng)

        force = i < forced_divergent
        if force:
            reps = make_forced_divergent_pair(final_value, gen_kind_for_forced, rng, duration)
        else:
            reps = make_replicate_pair(final_value, kind, rng, duration)

        disp_vals = [cg.compute_early_displacement(t, y) for t, y in reps]
        final_disp, status = cg.check_replicate_concordance(disp_vals[0], disp_vals[1])

        true_endpoint = float(np.mean([y[-1] for _, y in reps]))

        rows.append({
            "synthetic_id": f"SYN_TARGET_{kind}_{i:04d}{'_FORCED_DIVERGENT' if force else ''}",
            "class": kind,
            "stage": "target",
            "concentration_uM": conc,
            "duration_s": duration,
            "replicate_1_early_disp_30s": disp_vals[0],
            "replicate_2_early_disp_30s": disp_vals[1],
            "biomarker_replicate_status": status,
            "early_displacement_30s": final_disp,
            "true_endpoint_delta_f": true_endpoint,
            "intentionally_divergent": force,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(BASE_SEED + args.seed_offset)
    concs = _real_target_concentration_weights()

    rows = []
    rows += build_rows("CLEAN_PCA3_TARGET_HYB", N_CLEAN, rng, concs, forced_divergent=N_INTENTIONAL_DIVERGENT // 2)
    rows += build_rows("BACKGROUND_SOUP", N_BACKGROUND, rng, concs,
                        forced_divergent=N_INTENTIONAL_DIVERGENT - N_INTENTIONAL_DIVERGENT // 2)

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "target_synthetic_batch.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} synthetic curves)")
    print(df["class"].value_counts())
    print(df["biomarker_replicate_status"].value_counts())


if __name__ == "__main__":
    main()
