"""
Week 3 Task 3: build fixed-length probe-stage sequences for the real
44-chip dataset, resampled the same way as the synthetic sequences
(curve_generator.resample_curve, 45s window / 60 points) so the LSTM can
validate against real data on a like-for-like representation.

Baseline convention MUST match the synthetic curves and delta_f_probe /
early_displacement_30s (see ingest_raw_curves.py): each probe curve is
zeroed against its CHI-stage endpoint (per replicate), i.e. the same
cross-stage baseline used everywhere else in this project -- NOT the
probe curve's own first sample. (An earlier version of this script used
y - y[0], which silently drops any CHI-to-probe jump and produced
sequences on a wildly different scale from the synthetic ones -- caught
via a sanity check comparing sequence min/max/std between the two sets,
see git history.) Falls back to the probe curve's own start point only
for the No.7/8/9/10 chips that have no logged CHI file, same fallback as
ingest_raw_curves.py.

Averages replicates when a chip has 2 (matching how the synthetic
sequences were built: mean of both replicates' resampled curves).
Chips whose probe-stage duration is under the 45s window are dropped
(would require extrapolation past real data) -- checked earlier that
the real minimum is 49.99s, so this should drop ~0 chips in practice,
but the check is left in rather than assumed.

Output: qtwin/data/real_probe_sequences.npz (sequences, chip_id arrays)
"""

from pathlib import Path

import numpy as np
import pandas as pd

import curve_generator as cg

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def main():
    m = pd.read_csv(DATA_DIR / "raw_timeseries_master.csv")
    m = m[~m.is_error_file]
    probe = m[m.stage == "probe"]
    chi = m[m.stage == "CHI"]

    chip_ids = []
    sequences = []
    dropped = []
    for chip_id, g in probe.groupby("chip_id"):
        chi_g = chi[chi.chip_id == chip_id]
        rep_curves = []
        for rep, rg in g.groupby("replicate"):
            rg = rg.sort_values("Relative_time")
            t = rg["Relative_time"].values
            y_raw = rg["Resonance_Frequency"].values

            chi_rep = chi_g[chi_g.replicate == rep].sort_values("Relative_time")
            if len(chi_rep) > 0:
                baseline = chi_rep["Resonance_Frequency"].iloc[-1]  # CHI endpoint, matches delta_f_probe
            else:
                baseline = y_raw[0]  # No.7/8/9/10 fallback, matches ingest_raw_curves.py

            y = y_raw - baseline
            if t.max() < cg.SEQUENCE_WINDOW_S:
                continue
            rep_curves.append(cg.resample_curve(t, y))
        if not rep_curves:
            dropped.append(chip_id)
            continue
        chip_ids.append(chip_id)
        sequences.append(np.mean(rep_curves, axis=0))

    seq_array = np.array(sequences)
    out_path = DATA_DIR / "real_probe_sequences.npz"
    np.savez(out_path, sequences=seq_array, chip_id=np.array(chip_ids))
    print(f"wrote {out_path} (shape {seq_array.shape})")
    if dropped:
        print(f"Dropped {len(dropped)} chips (probe duration < {cg.SEQUENCE_WINDOW_S}s): {dropped}")
    else:
        print("No chips dropped -- all real probe-stage curves reach the 45s window.")


if __name__ == "__main__":
    main()
