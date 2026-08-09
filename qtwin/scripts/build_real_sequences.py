"""
Week 3 Task 3: build fixed-length probe-stage sequences for the real
44-chip dataset, resampled the same way as the synthetic sequences
(curve_generator.resample_curve, 45s window / 60 points) so the LSTM can
validate against real data on a like-for-like representation.

Baseline convention MUST match the synthetic curves and delta_f_probe /
early_displacement_30s (see ingest_raw_curves.py): each probe curve is
zeroed against its CHI-stage endpoint, i.e. the same cross-stage baseline
used everywhere else in this project -- NOT the probe curve's own first
sample. (An earlier version of this script used y - y[0], which silently
drops any CHI-to-probe jump and produced sequences on a wildly different
scale from the synthetic ones -- caught via a sanity check comparing
sequence min/max/std between the two sets, see git history.) Falls back
to the probe curve's own start point only for chips with NO CHI file at
all (the No.7/8/9/10 chips), same fallback as ingest_raw_curves.py.

BUG FOUND AND FIXED (2026-09-12, caught during a full-codebase review):
the baseline was matched PER PROBE REPLICATE NUMBER against a same-
numbered CHI replicate (chi_g[chi_g.replicate == rep]). For 2 real
chips where CHI and probe don't have the same replicate COUNT
(15Mar_No.17: 1 CHI replicate, 2 probe replicates; 21Mar_No.2: 2 CHI
replicates, 3 probe replicates), the probe replicate with no matching
CHI replicate number silently fell into the "no CHI file" fallback
(baseline = its own y[0]) even though CHI data DOES exist for that chip
-- it just doesn't have a same-numbered replicate. This is NOT what
ingest_raw_curves.py actually does (it averages CHI endpoints across
CHI's own replicates per chip, then applies that ONE value to every
probe replicate -- it never pairs specific replicate numbers together).
Impact was large for 15Mar_No.17 (in the official hold-out set):
replicate 2's fallback baseline (its own near-flat start point) produced
a near-zero delta_f (~0.1 Hz) instead of the real ~-138 Hz binding
signal replicate 1 independently showed -- averaging a real signal with
a fake near-zero one roughly halved the stored sequence's effective
magnitude (was -73 to -72 Hz stored; corrected sequence is much closer
to replicate 1's own -142 Hz). Fixed by replicating
ingest_raw_curves.py's actual averaging logic exactly -- verified this
matches the CHI-baseline fix already applied in
biological_plausibility_check.py.

Averages replicates when a chip has 2+ (matching how the synthetic
sequences were built: mean of all replicates' resampled curves).
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


def _endpoint(g: pd.DataFrame) -> float:
    return g.sort_values("Relative_time")["Resonance_Frequency"].iloc[-1]


def main():
    m = pd.read_csv(DATA_DIR / "raw_timeseries_master.csv")
    m = m[~m.is_error_file]
    probe = m[m.stage == "probe"]
    chi = m[m.stage == "CHI"]

    # One CHI baseline per chip_id: average CHI endpoints across CHI's own
    # replicates, applied to every probe replicate -- matches
    # ingest_raw_curves.py exactly (see BUG note above).
    chi_end_per_rep = chi.groupby(["chip_id", "replicate"]).apply(_endpoint, include_groups=False)
    chi_baseline_per_chip = chi_end_per_rep.groupby("chip_id").mean()

    chip_ids = []
    sequences = []
    dropped = []
    for chip_id, g in probe.groupby("chip_id"):
        rep_curves = []
        for rep, rg in g.groupby("replicate"):
            rg = rg.sort_values("Relative_time")
            t = rg["Relative_time"].values
            y_raw = rg["Resonance_Frequency"].values

            if chip_id in chi_baseline_per_chip.index:
                baseline = chi_baseline_per_chip.loc[chip_id]
            else:
                baseline = y_raw[0]  # No CHI file at all (No.7/8/9/10 fallback), matches ingest_raw_curves.py

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
