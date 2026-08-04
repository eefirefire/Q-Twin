"""
Week 2 Task 2: Build the probe immobilization generator.

Generates CLEAN_PCA3_TARGET (SUCCESS) and DEFECTIVE_CHIP (FAILURE) synthetic
probe-stage curves, seeded/calibrated from the real 44-chip dataset (see
curve_generator.py for exactly which real numbers feed which parameters).
Each synthetic chip gets 2 replicates; Task 4's concordance rule (Eva's Q7)
is applied per chip to decide whether the two replicates get averaged into
a final early_displacement_30s value or the chip is flagged
DIVERGENT_REPLICATES / Artifact Detected.

Output: qtwin/data/probe_synthetic_batch.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import constants as C
import curve_generator as cg

OUT_DIR = Path(__file__).resolve().parents[1] / "data"

N_CLEAN = 100
N_DEFECTIVE = 55
N_INTENTIONAL_DIVERGENT = 4  # extra, on top of N_CLEAN/N_DEFECTIVE, forced-divergent examples

BASE_SEED = 20260808  # Week 2 start date, for reproducibility


def _real_concentration_weights():
    """Bootstrap concentrations from the real chip_summary.csv distribution,
    so synthetic concentration mix matches what the real instrument actually
    saw, rather than an arbitrary uniform spread."""
    cs = pd.read_csv(OUT_DIR / "chip_summary.csv")
    cs = cs[cs.success_or_fail != "EXCLUDED"]
    conc = pd.to_numeric(cs["concentration_uM"], errors="coerce").dropna()
    conc = conc[conc > 0]
    return conc.values


def make_replicate_pair(final_value: float, kind: str, rng: np.random.Generator, duration_s: float):
    """Generate two independent replicate curves for one synthetic chip."""
    reps = []
    for _ in range(2):
        if kind == "CLEAN_PCA3_TARGET":
            t, y = cg.generate_association_curve(final_value, duration_s, rng)
        else:  # DEFECTIVE_CHIP
            t, y = cg.generate_baseline_drift_curve(final_value, duration_s, rng)
        reps.append((t, y))
    return reps


def make_forced_divergent_pair(final_value: float, kind: str, rng: np.random.Generator, duration_s: float):
    """Task 4 item 11: intentionally-divergent examples for gatekeeper
    testing. R1 = normal curve. R2 = corrupted -- flat-then-spiking, matching
    the real 21Mar_No.29 artifact pattern this whole rule was built from."""
    if kind == "CLEAN_PCA3_TARGET":
        t1, y1 = cg.generate_association_curve(final_value, duration_s, rng)
    else:
        t1, y1 = cg.generate_baseline_drift_curve(final_value, duration_s, rng)

    t2 = np.arange(0.0, duration_s, cg.SAMPLING_INTERVAL_S)
    flat_frac = rng.uniform(0.4, 0.7)
    flat_n = int(len(t2) * flat_frac)
    y2 = np.zeros_like(t2)
    y2[:flat_n] = rng.normal(0, 0.05, size=flat_n)  # flat, near-zero
    spike_region = t2[flat_n:]
    spike = rng.normal(0, 1.0, size=len(spike_region))
    spike += rng.choice([-1, 1]) * rng.uniform(30, 80, size=len(spike_region))
    y2[flat_n:] = spike
    return [(t1, y1), (t2, y2)]


def build_rows(kind: str, n: int, rng: np.random.Generator, concs: np.ndarray, forced_divergent: int = 0):
    rows = []
    for i in range(n):
        conc = float(rng.choice(concs))
        duration = cg.sample_duration(rng)
        if kind == "CLEAN_PCA3_TARGET":
            final_value = cg.sample_clean_final_value(conc, rng)
        else:
            final_value = cg.sample_defective_final_value(rng)

        force = i < forced_divergent
        if force:
            reps = make_forced_divergent_pair(final_value, kind, rng, duration)
        else:
            reps = make_replicate_pair(final_value, kind, rng, duration)

        disp_vals = [cg.compute_early_displacement(t, y) for t, y in reps]
        final_disp, status = cg.check_replicate_concordance(disp_vals[0], disp_vals[1])

        true_endpoint = float(np.mean([y[-1] for _, y in reps]))
        success_or_fail = "SUCCESS" if true_endpoint <= C.FAILURE_THRESHOLD_HZ else "FAILURE"

        rows.append({
            "synthetic_id": f"SYN_PROBE_{kind}_{i:04d}{'_FORCED_DIVERGENT' if force else ''}",
            "class": kind,
            "stage": "probe",
            "concentration_uM": conc,
            "duration_s": duration,
            "replicate_1_early_disp_30s": disp_vals[0],
            "replicate_2_early_disp_30s": disp_vals[1],
            "biomarker_replicate_status": status,
            "early_displacement_30s": final_disp,
            "true_endpoint_delta_f": true_endpoint,
            "success_or_fail": success_or_fail,
            "intentionally_divergent": force,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-offset", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(BASE_SEED + args.seed_offset)
    concs = _real_concentration_weights()

    rows = []
    rows += build_rows("CLEAN_PCA3_TARGET", N_CLEAN, rng, concs, forced_divergent=N_INTENTIONAL_DIVERGENT // 2)
    rows += build_rows("DEFECTIVE_CHIP", N_DEFECTIVE, rng, concs,
                        forced_divergent=N_INTENTIONAL_DIVERGENT - N_INTENTIONAL_DIVERGENT // 2)

    df = pd.DataFrame(rows)
    out_path = OUT_DIR / "probe_synthetic_batch.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} ({len(df)} synthetic curves)")
    print(df["class"].value_counts())
    print(df["biomarker_replicate_status"].value_counts())
    print(df["success_or_fail"].value_counts())


if __name__ == "__main__":
    main()
