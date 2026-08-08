"""
Promotes TCN (trained on the single-replicate-augmented 310-sequence set)
to the official Stage 2a model, per explicit decision after the
determinism-bug fix reversed the two prior candidates:

- The single-replicate-augmented PLAIN LSTM (previously official) showed
  ZERO real hold-out improvement over the pre-augmentation baseline once
  training was made properly deterministic (0.545 -> 0.545) and actually
  regressed 33-chip accuracy (0.818 -> 0.576). The originally-reported
  45.5%->72.7% improvement was a CPU-thread-scheduling artifact, not a
  real effect -- see Known_Limitations_Master.md's determinism section.
- Plain LSTM with NO augmentation scores best on 33-chip (0.818) but worst
  on hold-out (0.545) among the candidates -- a wide 33-chip/hold-out gap
  consistent with overfitting to the 33-chip set, not real strength.
- TCN on the augmented set ties for best 33-chip (0.788, matching plain
  LSTM) AND best hold-out (0.727, matching LSTM+Attention) -- best or
  tied-best on both metrics simultaneously, with no interpretability
  complications the way attention weights had (see
  attention_weight_analysis.txt's contrary finding).
- The soft-vote ensemble matches TCN's numbers exactly but adds inference
  complexity (3 models) for zero net gain over TCN alone -- not chosen.

Uses TCN's ALREADY-SELECTED hyperparameters from
augmented_architecture_comparison.py's internal-tuning-split sweep
(channels=8, n_layers=3, epochs=220) rather than re-tuning here --
tuning was already done properly under deterministic training; this
script's only job is producing the final promoted artifact.

Output: overwrites qtwin/models/lstm_probe_stage.pt and lstm_config.json
(now with an "architecture": "TCN" field -- see pipeline_api.py's
load_lstm(), made architecture-aware for this promotion), plus
qtwin/models/tcn_official_metrics.txt documenting the promotion.
"""

from pathlib import Path

import json
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatekeeper_model import LABELS
from model_trainer import load_real_validation, normalize, to_tensor
from model_trainer_augmented import load_augmented_synthetic, load_holdout_validation
from sequence_architectures import TCN

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.manual_seed(20260912)
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)

CHANNELS = 8
N_LAYERS = 3
EPOCHS = 220


def train_tcn(X_train, y_train, epochs, lr=1e-2):
    model = TCN(channels=CHANNELS, n_layers=N_LAYERS)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    class_counts = torch.bincount(y_train, minlength=len(LABELS)).float()
    class_weights = class_counts.sum() / (len(LABELS) * class_counts.clamp(min=1))
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train, y_train), batch_size=16, shuffle=True
    )
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
    return (pred.numpy() == y.numpy()).mean(), pred.numpy()


def run_eval(model, seqs, labels, mean, std, name, cmap, out_stub):
    seqs_norm = normalize(seqs, mean, std)
    X, y = to_tensor(seqs_norm, labels)
    acc, pred_idx = evaluate(model, X, y)
    pred_labels = [LABELS[i] for i in pred_idx]
    report = classification_report(labels, pred_labels, labels=LABELS, zero_division=0)
    cm = confusion_matrix(labels, pred_labels, labels=LABELS)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"TCN (official, promoted 2026-09-12) -- {name} (n={len(labels)})", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / f"{out_stub}.png", dpi=130)
    plt.close(fig)

    return acc, report, cm


def main():
    seqs, labels = load_augmented_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)
    X_all, y_all = to_tensor(seqs_norm, labels)

    print(f"Training TCN (channels={CHANNELS}, n_layers={N_LAYERS}, epochs={EPOCHS}) "
          f"on all {len(seqs)} augmented sequences (already-selected hyperparameters, not re-tuning).")
    final_model = train_tcn(X_all, y_all, epochs=EPOCHS)

    real_seqs, real_labels, _, n_dropped = load_real_validation()
    real_acc, real_report, real_cm = run_eval(
        final_model, real_seqs, real_labels, mean, std,
        "33-chip real validation", "Greens", "tcn_official_confusion_matrix"
    )
    print(f"33-chip validation accuracy: {real_acc:.3f}")
    print(real_report)

    holdout_seqs, holdout_labels, holdout_chip_ids = load_holdout_validation()
    holdout_acc, holdout_report, holdout_cm = run_eval(
        final_model, holdout_seqs, holdout_labels, mean, std,
        "BLIND hold-out", "Purples", "tcn_official_holdout_confusion_matrix"
    )
    print(f"Hold-out accuracy: {holdout_acc:.3f}")
    print(holdout_report)

    # PROMOTION: overwrite the shared official model files. pipeline_api.py's
    # load_lstm() was made architecture-aware for this -- the "architecture"
    # key is what tells it to instantiate TCN instead of SequenceLSTM.
    torch.save(final_model.state_dict(), MODEL_DIR / "lstm_probe_stage.pt")
    with open(MODEL_DIR / "lstm_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "architecture": "TCN",
            "channels": CHANNELS,
            "n_layers": N_LAYERS,
            "epochs": EPOCHS,
            "seq_mean": float(mean),
            "seq_std": float(std),
        }, f, indent=2)
    print(f"PROMOTED: wrote {MODEL_DIR / 'lstm_probe_stage.pt'} and lstm_config.json "
          f"(official model is now TCN, channels={CHANNELS}, n_layers={N_LAYERS})")

    with open(MODEL_DIR / "tcn_official_metrics.txt", "w", encoding="utf-8") as f:
        f.write("TCN promoted to official Stage 2a model (2026-09-12)\n")
        f.write("=" * 60 + "\n\n")
        f.write("PROMOTION CONTEXT: this replaces the single-replicate-augmented plain\n")
        f.write("LSTM, which was found -- after fixing a CPU-thread non-determinism bug\n")
        f.write("affecting every torch training script in this project -- to show ZERO\n")
        f.write("real hold-out improvement over the pre-augmentation baseline (0.545 vs\n")
        f.write("0.545) and to have actually REGRESSED 33-chip accuracy (0.818 -> 0.576).\n")
        f.write("The originally-reported 45.5%->72.7% improvement was an artifact of\n")
        f.write("uncontrolled thread scheduling, not a real effect -- see\n")
        f.write("Known_Limitations_Master.md's determinism-bug section for the full\n")
        f.write("account, including the two intermediate wrong conclusions (\"keep\n")
        f.write("augmented LSTM official\" and \"promote Attention\") this correction\n")
        f.write("retracts.\n\n")
        f.write("DECISION: among all candidates re-tested under proper determinism on\n")
        f.write("the same augmented training set (per augmented_architecture_comparison.txt),\n")
        f.write("TCN tied for best 33-chip accuracy (0.788, matching plain LSTM) AND best\n")
        f.write("hold-out accuracy (0.727, matching LSTM+Attention and the ensemble) --\n")
        f.write("the only candidate best-or-tied-best on BOTH metrics simultaneously in\n")
        f.write("that comparison. Not LSTM+Attention (matched TCN on hold-out but not\n")
        f.write("33-chip in that same comparison, and its attention weights don't actually\n")
        f.write("confirm the intended interpretability story -- see\n")
        f.write("attention_weight_analysis.txt). Not the ensemble (same numbers as TCN\n")
        f.write("alone, 3x the inference cost, zero net gain). Not pre-augmentation plain\n")
        f.write("LSTM (best 33-chip at 0.818, but worst hold-out at 0.545 -- a wide gap\n")
        f.write("consistent with overfitting to the 33-chip set, not real generalization).\n\n")
        f.write(f"Architecture: TCN, channels={CHANNELS}, n_layers={N_LAYERS}, epochs={EPOCHS}\n")
        f.write("(hyperparameters already selected via augmented_architecture_comparison.py's\n")
        f.write("internal 80/20 synthetic tuning split -- not re-tuned here, this script\n")
        f.write("only produces the final promoted artifact.)\n\n")
        f.write("HONEST NOTE ON THE NUMBERS BELOW vs. augmented_architecture_comparison.txt's\n")
        f.write("TCN row (0.788/0.727): this script trains TCN as the ONLY model in a fresh\n")
        f.write("process, while that comparison trained LSTM, then LSTM+Attention, then TCN\n")
        f.write("in sequence within one process -- each earlier model's training consumes the\n")
        f.write("shared torch RNG, so TCN there and TCN here start from a different RNG\n")
        f.write("position even with the same seed and the same determinism fix. Both are\n")
        f.write("individually bit-reproducible (verified: two standalone runs of this exact\n")
        f.write("script produce a bit-identical lstm_probe_stage.pt) -- 'deterministic' means\n")
        f.write("reproducible given the same RNG-consumption history, not identical across\n")
        f.write("scripts with different call sequences. The numbers below are this promoted\n")
        f.write("artifact's actual, real, verified performance -- treat these as authoritative\n")
        f.write("for the deployed model, not the comparison script's in-sequence number.\n\n")
        f.write(f"33-chip validation accuracy: {real_acc:.3f}\n")
        f.write(real_report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{real_cm}\n\n")
        f.write(f"Hold-out accuracy: {holdout_acc:.3f}\n")
        f.write(holdout_report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{holdout_cm}\n\n")
        f.write("HONEST METHODOLOGY NOTE (same as every prior LSTM promotion in this\n")
        f.write("project): reporting hold-out numbers as part of this architecture\n")
        f.write("decision means the hold-out set is no longer fully untouched by any\n")
        f.write("downstream decision. Both TCN and LSTM+Attention were evaluated on it\n")
        f.write("before this choice was made.\n")

    print(f"\nwrote {MODEL_DIR / 'tcn_official_metrics.txt'}")


if __name__ == "__main__":
    main()
