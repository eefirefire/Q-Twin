"""
LSTM improvement technique: single-replicate data augmentation, targeting
the exact root cause diagnosed for the LSTM's blind hold-out drop
(75.8% -> 45.5%).

Root cause (see Known_Limitations_Master.md Stage 2a): the LSTM's
accuracy on SINGLE_REPLICATE real chips is only 56.25% (9/16) even in
the 33-chip validation set, vs. 90-100% on CONCORDANT/DIVERGENT chips --
a weakness already visible there, not just a hold-out artifact.
generate_probe_batch.py's synthetic training sequences are ALWAYS the
AVERAGE of two replicate curves (make_replicate_pair() -> np.mean of two
resampled curves). Averaging two independent noise draws mathematically
reduces point-to-point noise variance by ~2x (Var(mean of 2 iid) =
Var/2). The LSTM has therefore never seen the noise level of a genuine
single real replicate during training -- a real train/test distribution
mismatch. Its default response to an out-of-distribution, noisier input
appears to be classifying it DIVERGENT_REPLICATES (the class already
defined by "looks unusual"), which is exactly the failure pattern
observed.

Technique: generate additional SINGLE-replicate synthetic training
sequences (one curve, not an average of two), each labeled by its own
endpoint vs. the -0.5 Hz threshold -- SUCCESS or FAILURE only, NEVER
DIVERGENT_REPLICATES, matching exactly how real single-replicate chips
are labeled (build_label() maps SINGLE_REPLICATE status straight through
to success_or_fail, since concordance is undefined for n=1). This
directly teaches the model that a noisier, single-curve-shaped input can
legitimately be SUCCESS or FAILURE, not automatically anomalous.

Same class ratio (100 clean : 55 defective) and concentration
distribution as generate_probe_batch.py, same duration model, same base
seed family -- everything held constant except that each augmented
example is ONE replicate's curve, not an average of two.

Output: qtwin/data/probe_single_replicate_sequences.npz
"""

from pathlib import Path

import numpy as np
import pandas as pd

import constants as C
import curve_generator as cg
from generate_probe_batch import N_CLEAN, N_DEFECTIVE, _real_concentration_weights

OUT_DIR = Path(__file__).resolve().parents[1] / "data"
AUGMENT_SEED = 20260907  # testing week -- distinct from the base synthetic seed


def build_single_replicate_batch(rng: np.random.Generator, concs: np.ndarray) -> pd.DataFrame:
    rows = []
    sequences = []
    for kind, n in [("CLEAN_PCA3_TARGET", N_CLEAN), ("DEFECTIVE_CHIP", N_DEFECTIVE)]:
        for i in range(n):
            conc = float(rng.choice(concs))
            duration = cg.sample_duration(rng)
            if kind == "CLEAN_PCA3_TARGET":
                final_value = cg.sample_clean_final_value(conc, rng)
                t, y = cg.generate_association_curve(final_value, duration, rng)
            else:
                final_value = cg.sample_defective_final_value(rng)
                t, y = cg.generate_baseline_drift_curve(final_value, duration, rng)

            true_endpoint = float(y[-1])
            # Matches build_label()'s behavior for real SINGLE_REPLICATE chips
            # exactly: concordance is undefined for n=1, so the label is just
            # the endpoint threshold call -- never DIVERGENT_REPLICATES.
            label = "SUCCESS" if true_endpoint <= C.FAILURE_THRESHOLD_HZ else "FAILURE"

            sequences.append(cg.resample_curve(t, y))
            rows.append({
                "synthetic_id": f"SYN_SINGLE_PROBE_{kind}_{i:04d}",
                "class": kind,
                "concentration_uM": conc,
                "true_endpoint_delta_f": true_endpoint,
                "label": label,
            })
    df = pd.DataFrame(rows)
    return df, np.array(sequences)


def main():
    rng = np.random.default_rng(AUGMENT_SEED)
    concs = _real_concentration_weights()
    df, sequences = build_single_replicate_batch(rng, concs)

    print(f"Generated {len(df)} single-replicate synthetic sequences")
    print(df["label"].value_counts())

    out_path = OUT_DIR / "probe_single_replicate_sequences.npz"
    np.savez(out_path, sequences=sequences, synthetic_id=df["synthetic_id"].values, label=df["label"].values)
    print(f"wrote {out_path} (shape {sequences.shape})")

    df.to_csv(OUT_DIR / "probe_single_replicate_batch.csv", index=False)
    print(f"wrote {OUT_DIR / 'probe_single_replicate_batch.csv'}")


if __name__ == "__main__":
    main()
