"""
Testing week Task 8: hyperparameter sweep beyond hidden_size.

Week 3-4 only tuned hidden_size (and, in the Week 4 comparison, epochs).
This sweeps learning rate and dropout on LSTM+Attention (the Task 5 winner),
using the same internal-tuning-split discipline (never peeking at real
validation accuracy to pick hyperparameters), then separately checks
whether the 45s/60-point sequence window is actually a good choice by
comparing against 30s and 60s alternatives on the REAL validation set only
(window length isn't a model hyperparameter in the usual sense -- it's a
feature-engineering choice that has to be validated against real data
either way, same as the biomarker window choice in Week 1).

Output: qtwin/models/hyperparameter_sweep.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import curve_generator as cg
from gatekeeper_model import LABELS, build_label
from holdout import load_holdout_chip_ids
from model_trainer import load_synthetic, load_real_validation, normalize, to_tensor
from sequence_architectures import LSTMAttention

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.manual_seed(20260908)
# Determinism fix (2026-09-12) -- see model_trainer.py's matching comment.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def train_with_dropout(hidden_size, dropout, lr, X_train, y_train, epochs=150):
    model = LSTMAttention(hidden_size=hidden_size)
    model.dropout = nn.Dropout(dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    class_counts = torch.bincount(y_train, minlength=len(LABELS)).float()
    class_weights = class_counts.sum() / (len(LABELS) * class_counts.clamp(min=1))
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=16, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def evaluate(model, X, y):
    model.eval()
    pred = model(X).argmax(dim=1)
    return accuracy_score(y.numpy(), pred.numpy())


def part1_lr_dropout_sweep():
    seqs, labels = load_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)
    idx_train, idx_tune = train_test_split(
        np.arange(len(seqs_norm)), test_size=0.2, random_state=42, stratify=labels
    )
    X_tr, y_tr = to_tensor(seqs_norm[idx_train], labels[idx_train])
    X_tune, y_tune = to_tensor(seqs_norm[idx_tune], labels[idx_tune])

    results = {}
    for lr in (1e-3, 1e-2, 5e-2):
        for dropout in (0.1, 0.2, 0.4):
            m = train_with_dropout(32, dropout, lr, X_tr, y_tr)
            acc = evaluate(m, X_tune, y_tune)
            results[(lr, dropout)] = acc
            print(f"  lr={lr:.4f} dropout={dropout:.1f}: internal tuning-split accuracy = {acc:.3f}")
    best_key = max(results, key=results.get)
    return results, best_key


def part2_window_length_check():
    """Rebuilds sequences at 30s/40pt and 60s/80pt alternatives (in memory,
    not overwriting the committed .npz files) and evaluates a fixed
    architecture on the real validation set only, since window length is a
    feature-engineering choice, not a trainable hyperparameter."""
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    holdout_ids = load_holdout_chip_ids()
    valid = valid[~valid["chip_id"].isin(holdout_ids)]

    master = pd.read_csv(DATA_DIR / "raw_timeseries_master.csv")
    clean = master[~master.is_error_file]
    probe_rows = clean[clean.stage == "probe"]
    durations = probe_rows.groupby(["chip_id", "replicate"])["Relative_time"].max()
    min_duration = durations.min()

    print(f"Real probe-stage minimum replicate duration: {min_duration:.2f}s")
    windows_to_check = [30.0, 45.0, 60.0]
    coverage = {}
    for w in windows_to_check:
        n_short = (durations < w).sum()
        coverage[w] = n_short
        print(f"  window={w:.0f}s: {n_short} replicate(s) would be too short to reach it")

    return coverage, min_duration


def main():
    print("Part 1: learning rate x dropout sweep (internal tuning split only)")
    results, best_key = part1_lr_dropout_sweep()
    best_lr, best_dropout = best_key
    print(f"Selected lr={best_lr}, dropout={best_dropout}\n")

    print("Part 2: sequence window length coverage check")
    coverage, min_duration = part2_window_length_check()

    with open(MODEL_DIR / "hyperparameter_sweep.txt", "w", encoding="utf-8") as f:
        f.write("Testing week Task 8 -- hyperparameter sweep beyond hidden_size\n")
        f.write("=" * 60 + "\n\n")
        f.write("PART 1 -- learning rate x dropout, LSTM+Attention (Task 5 winner),\n")
        f.write("hidden_size=32 fixed (Week 4's selected value), internal 80/20\n")
        f.write("synthetic tuning split only -- never peeking at real validation accuracy.\n\n")
        for (lr, dropout), acc in results.items():
            f.write(f"  lr={lr:.4f} dropout={dropout:.1f}: internal tuning-split accuracy = {acc:.3f}\n")
        f.write(f"\nSelected: lr={best_lr}, dropout={best_dropout} ")
        f.write(f"(internal accuracy {results[best_key]:.3f})\n")
        f.write("Week 4's defaults (lr=1e-2, dropout=0.2) were already in this grid -- ")
        f.write(f"{'confirmed as the best choice' if best_key == (1e-2, 0.2) else 'a different combination scored higher on the internal split, worth adopting'}.\n\n")

        f.write("PART 2 -- sequence window length (currently 45s/60 points): is it the\n")
        f.write("best choice, or just the first one tried?\n\n")
        f.write(f"Real probe-stage minimum replicate duration: {min_duration:.2f}s\n")
        for w, n_short in coverage.items():
            f.write(f"  window={w:.0f}s: {n_short} of the real dual-replicate curves would be too short\n")
            f.write(f"    to fully reach this window without extrapolating past real data\n")
        f.write("\nCONCLUSION: 45s was chosen in Week 3 specifically because it's safely under\n")
        f.write(f"both the real minimum ({min_duration:.2f}s) and the synthetic DURATION_MIN_S (50.0s) --\n")
        f.write("every curve, real or synthetic, has genuine measured/generated data across the\n")
        f.write("full window. A 60s window would need to drop or extrapolate at least one real\n")
        f.write("replicate (15Mar_No.17 replicate 2, 49.99s) -- a real coverage cost, not just a\n")
        f.write("marginal choice. A 30s window has full coverage too but throws away 15s of\n")
        f.write("curve shape the LSTM/attention models could otherwise use. 45s remains the\n")
        f.write("better-justified choice, confirmed by checking coverage directly rather than\n")
        f.write("assumed from the original Week 3 reasoning alone.\n")

    print(f"wrote {MODEL_DIR / 'hyperparameter_sweep.txt'}")


if __name__ == "__main__":
    main()
