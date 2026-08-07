"""
Testing week Task 6: attention layer as a validity check on the biomarker
discovery itself.

If an attention layer independently learns to weight the early part of the
probe curve most heavily, that's the model rediscovering, on its own, the
same "the first 30 seconds matter most" insight the early_displacement_30s
biomarker was built on by hand in Week 1 -- an independent confirmation,
not just a performance number.

Trains LSTMAttention (sequence_architectures.py, same as the Week 4 Task 4
comparison) on all synthetic data, then extracts and plots the learned
attention weights across the 60 timesteps for the real validation chips.

Output: qtwin/models/attention_weights_visualization.png,
qtwin/models/attention_weight_analysis.txt
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import curve_generator as cg
from gatekeeper_model import LABELS
from model_trainer import load_synthetic, load_real_validation, normalize, to_tensor
from sequence_architectures import LSTMAttention

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.manual_seed(20260907)  # testing week


def train(model, X_train, y_train, epochs=150, lr=1e-2):
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
def extract_attention_weights(model, X):
    """Re-runs the attention math from LSTMAttention.forward() but returns
    the weights instead of the classification output."""
    model.eval()
    outputs, _ = model.lstm(X)
    scores = model.attn_score(outputs).squeeze(-1)
    weights = torch.softmax(scores, dim=1)
    return weights.numpy()


def main():
    seqs, labels = load_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)
    X_all, y_all = to_tensor(seqs_norm, labels)

    model = LSTMAttention(hidden_size=32)
    train(model, X_all, y_all)

    real_seqs, real_labels, real_chip_ids, _ = load_real_validation()
    real_norm = normalize(real_seqs, mean, std)
    X_real, y_real = to_tensor(real_norm, real_labels)

    weights = extract_attention_weights(model, X_real)  # (n_chips, 60)
    time_grid = np.linspace(0.0, cg.SEQUENCE_WINDOW_S, cg.SEQUENCE_N_POINTS)

    mean_weights = weights.mean(axis=0)
    early_window_mask = time_grid <= 15.0  # first 15s of the 45s window
    early_weight_share = mean_weights[early_window_mask].sum()
    uniform_share = early_window_mask.sum() / len(time_grid)  # what "no preference" would look like

    print(f"Mean attention weight in first 15s: {early_weight_share:.3f} "
          f"(uniform baseline would be {uniform_share:.3f})")

    fig, axes = plt.subplots(2, 1, figsize=(9, 8))
    axes[0].plot(time_grid, mean_weights, marker="o", markersize=3)
    axes[0].axvline(30.0, color="gray", linestyle="--", alpha=0.6, label="t=30s (biomarker window)")
    axes[0].set_xlabel("Time into probe stage (s)")
    axes[0].set_ylabel("Mean attention weight")
    axes[0].set_title("Mean learned attention weight across the 45s window\n(averaged over all real validation chips)")
    axes[0].legend()

    for i in range(min(8, len(weights))):
        axes[1].plot(time_grid, weights[i], alpha=0.6, label=f"{real_chip_ids[i]}")
    axes[1].axvline(30.0, color="gray", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("Time into probe stage (s)")
    axes[1].set_ylabel("Attention weight")
    axes[1].set_title("Per-chip attention weights (first 8 real validation chips)")
    axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(MODEL_DIR / "attention_weights_visualization.png", dpi=130)
    plt.close(fig)
    print(f"wrote {MODEL_DIR / 'attention_weights_visualization.png'}")

    peak_time = time_grid[np.argmax(mean_weights)]
    concentrates_early = early_weight_share > uniform_share * 1.3  # meaningfully above uniform

    with open(MODEL_DIR / "attention_weight_analysis.txt", "w", encoding="utf-8") as f:
        f.write("Testing week Task 6 -- attention weight visualization\n")
        f.write("=" * 60 + "\n\n")
        f.write("Question: does an attention layer, trained with no knowledge of the\n")
        f.write("hand-derived early_displacement_30s biomarker, independently learn to\n")
        f.write("weight the early part of the probe curve most heavily?\n\n")
        f.write(f"Mean attention weight in the first 15s of the 45s window: {early_weight_share:.3f}\n")
        f.write(f"(a uniform/no-preference attention would give {uniform_share:.3f} to that same span)\n")
        f.write(f"Peak mean attention weight occurs at t={peak_time:.1f}s\n\n")
        if concentrates_early:
            f.write("RESULT: YES -- attention concentrates in the early part of the curve,\n")
            f.write("independently confirming the Week 1 'the first 30 seconds matter most'\n")
            f.write("insight the biomarker was hand-built on. This is a genuinely independent\n")
            f.write("piece of evidence (the model was never told about early_displacement_30s\n")
            f.write("or the 30s window during training), not a restatement of the same claim.\n")
        else:
            f.write("RESULT: NO -- attention does NOT clearly concentrate early; weight is spread\n")
            f.write("across the window (or concentrates elsewhere, e.g. near the end). This is\n")
            f.write("itself a genuinely interesting finding worth investigating further, not a\n")
            f.write("failed replication to hide: it suggests either (a) the model is picking up a\n")
            f.write("different, real signal than the hand-derived biomarker relies on, or (b) with\n")
            f.write("only 155 synthetic training curves, the attention layer hasn't converged to a\n")
            f.write("stable, interpretable pattern yet. See attention_weights_visualization.png\n")
            f.write("for the actual per-chip weight curves before drawing further conclusions.\n")
        f.write("\nHONEST CAVEAT: this is one training run (fixed seed). Given how much the\n")
        f.write("Week 4 tuning sweep showed run-to-run variance on this dataset size, treat this\n")
        f.write("as a suggestive single observation, not a robust, reproduced-across-seeds claim.\n")

    print(f"wrote {MODEL_DIR / 'attention_weight_analysis.txt'}")


if __name__ == "__main__":
    main()
