"""
Week 3 Task 3: Stage 2a sequence-model prototype (LSTM).

Unlike the Task 1 gatekeeper (which only saw the single-number
early_displacement_30s biomarker), this model gets the actual probe-stage
curve shape: a 60-point resampled sequence over the first 45s of the
probe stage, built by generate_probe_batch.py (synthetic training set,
probe_synthetic_sequences.npz) and build_real_sequences.py (real
validation set, real_probe_sequences.npz).

Labels reuse the exact same SUCCESS / FAILURE / DIVERGENT_REPLICATES
scheme and build_label() logic as the Task 1 gatekeeper, so results are
directly comparable between the two models on the same validation chips.

One real difference from the gatekeeper: DIVERGENT_REPLICATES rows are
NOT feature-blanked here. The gatekeeper's early_displacement_30s is null
by construction whenever concordance fails (so it leans entirely on the
is_divergent flag for that class). The LSTM instead sees the genuine
average-of-two-replicates curve shape (including the corrupted flat/spike
replicate baked into that average) -- so this model has to learn to
recognize divergence from curve shape itself, not from a hand-fed flag.
That makes the two models meaningfully different, not just a fancier
copy of Task 1.

Does one real tuning iteration (hidden layer size: 16 vs 32 vs 64) using
an internal 80/20 split of the synthetic data, then trains the chosen
architecture on all synthetic data and validates against the real,
non-hold-out 33-chip set.

Output: model_trainer.py (this file), qtwin/models/lstm_confusion_matrix.png,
qtwin/models/lstm_metrics.txt, qtwin/models/lstm_probe_stage.pt
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatekeeper_model import LABELS, build_label
from holdout import load_holdout_chip_ids

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.manual_seed(20260810)
# Determinism fix (2026-09-12, caught during independent review): manual_seed
# alone does not make CPU LSTM training bit-reproducible -- PyTorch's default
# multi-threaded CPU execution has non-deterministic float reduction order,
# which compounds over training into meaningfully different results run to
# run. Verified fix: pinning both of these makes repeated runs bit-identical.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
LABEL_TO_IDX = {label: i for i, label in enumerate(LABELS)}


class SequenceLSTM(nn.Module):
    def __init__(self, hidden_size: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, len(LABELS))

    def forward(self, x):
        # x: (batch, seq_len, 1)
        _, (h_n, _) = self.lstm(x)
        out = self.dropout(h_n[-1])
        return self.fc(out)


def load_synthetic():
    npz = np.load(DATA_DIR / "probe_synthetic_sequences.npz", allow_pickle=True)
    seqs, synth_ids = npz["sequences"], npz["synthetic_id"]
    df = pd.read_csv(DATA_DIR / "probe_synthetic_batch.csv").set_index("synthetic_id")
    df = df.loc[synth_ids]  # enforce identical row order to the .npz array
    labels = df.apply(lambda r: build_label(r["success_or_fail"], r["biomarker_replicate_status"]), axis=1)
    return seqs, labels.values


def load_real_validation():
    npz = np.load(DATA_DIR / "real_probe_sequences.npz", allow_pickle=True)
    seqs, chip_ids = npz["sequences"], npz["chip_id"]
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv").set_index("chip_id")
    holdout_ids = load_holdout_chip_ids()

    keep_mask, labels = [], []
    for cid in chip_ids:
        row = cs.loc[cid]
        if row["success_or_fail"] == "EXCLUDED" or cid in holdout_ids:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        labels.append(build_label(row["success_or_fail"], row["displacement_replicate_status"]))

    keep_mask = np.array(keep_mask)
    return seqs[keep_mask], np.array(labels), chip_ids[keep_mask], int((~keep_mask).sum())


def normalize(seqs: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (seqs - mean) / std


def to_tensor(seqs: np.ndarray, labels: np.ndarray):
    X = torch.tensor(seqs, dtype=torch.float32).unsqueeze(-1)  # (N, 60, 1)
    y = torch.tensor([LABEL_TO_IDX[l] for l in labels], dtype=torch.long)
    return X, y


def train_model(X_train, y_train, hidden_size: int, epochs: int = 60, lr: float = 1e-2):
    model = SequenceLSTM(hidden_size=hidden_size)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # Class-weighted loss (added after the Week 3 Task 4 divergence fix):
    # that fix correctly makes DIVERGENT_REPLICATES realistically common,
    # but as a side effect it shrinks FAILURE down to a small minority of
    # training rows (weak-signal/DEFECTIVE_CHIP curves are the ones most
    # prone to replicate sign-flips under the new jitter, so they get
    # swept into DIVERGENT_REPLICATES disproportionately). Unweighted loss
    # let the model ignore FAILURE almost entirely (0% recall) -- weighting
    # inversely by class frequency, same approach the Task 1 gatekeeper's
    # RandomForestClassifier(class_weight="balanced") already uses, so
    # both models handle the same imbalance the same way.
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
    return accuracy_score(y.numpy(), pred.numpy()), pred.numpy()


def main():
    seqs, labels = load_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)

    # --- Tuning iteration: hidden size, via internal 80/20 synthetic split ---
    idx_train, idx_tune = train_test_split(
        np.arange(len(seqs_norm)), test_size=0.2, random_state=42, stratify=labels
    )
    X_tr, y_tr = to_tensor(seqs_norm[idx_train], labels[idx_train])
    X_tune, y_tune = to_tensor(seqs_norm[idx_tune], labels[idx_tune])

    # Sweeps hidden_size AND epochs together now (added after the Week 3 Task 4
    # divergence fix made this a harder, more imbalanced 3-class problem --
    # class-weighted loss alone wasn't enough, and epochs turned out to matter
    # a lot). Selection stays on the INTERNAL tuning split only, never on real
    # validation accuracy -- picking hyperparameters by peeking at the real
    # validation set would be exactly the kind of leakage this project has
    # avoided everywhere else (hold-out set, K-S tests, etc).
    print("Tuning hidden_size x epochs on an internal 80/20 split of the synthetic data...")
    tuning_results = {}
    for hidden_size in (16, 32, 64):
        for epochs in (80, 150, 220):
            m = train_model(X_tr, y_tr, hidden_size=hidden_size, epochs=epochs)
            acc, _ = evaluate(m, X_tune, y_tune)
            tuning_results[(hidden_size, epochs)] = acc
            print(f"  hidden_size={hidden_size:3d} epochs={epochs:3d}: internal tuning-split accuracy = {acc:.3f}")
    best_hidden, best_epochs = max(tuning_results, key=tuning_results.get)
    print(f"Selected hidden_size={best_hidden}, epochs={best_epochs} (highest internal tuning accuracy).")

    # --- Final model: train on ALL synthetic data with the chosen architecture ---
    X_all, y_all = to_tensor(seqs_norm, labels)
    final_model = train_model(X_all, y_all, hidden_size=best_hidden, epochs=best_epochs)

    # --- Validate against real, non-hold-out chips ---
    real_seqs, real_labels, real_chip_ids, n_dropped = load_real_validation()
    real_seqs_norm = normalize(real_seqs, mean, std)  # reuse synthetic-fit scaling, not real stats
    X_real, y_real = to_tensor(real_seqs_norm, real_labels)

    acc, pred_idx = evaluate(final_model, X_real, y_real)
    pred_labels = [LABELS[i] for i in pred_idx]
    report = classification_report(real_labels, pred_labels, labels=LABELS, zero_division=0)
    print(f"\nReal (non-hold-out) validation accuracy: {acc:.3f}")
    print(report)

    cm = confusion_matrix(real_labels, pred_labels, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Greens", colorbar=False)
    ax.set_title(f"LSTM sequence-model confusion matrix\n(real, non-hold-out validation, hidden_size={best_hidden}, epochs={best_epochs})")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    cm_path = MODEL_DIR / "lstm_confusion_matrix.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"wrote {cm_path}")

    torch.save(final_model.state_dict(), MODEL_DIR / "lstm_probe_stage.pt")

    # Week 3 Task 5 (Streamlit mockup) needs to reload this exact model outside
    # this script -- save the architecture choice and normalization stats
    # (fit on synthetic data only, see `normalize()` call above) alongside the
    # weights so the mockup doesn't have to re-derive or guess either one.
    import json
    with open(MODEL_DIR / "lstm_config.json", "w", encoding="utf-8") as f:
        json.dump({"hidden_size": best_hidden, "epochs": best_epochs, "seq_mean": float(mean), "seq_std": float(std)}, f, indent=2)

    with open(MODEL_DIR / "lstm_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Week 3 Task 3 -- Stage 2a Sequence Model (LSTM) Prototype\n")
        f.write("=" * 60 + "\n\n")
        f.write("This model takes the full probe-stage curve shape (60-point resampled\n")
        f.write("sequence, 0-45s window) rather than the single early_displacement_30s\n")
        f.write("number the Task 1 gatekeeper used. DIVERGENT_REPLICATES rows are NOT\n")
        f.write("feature-blanked the way they are for the gatekeeper -- this model sees\n")
        f.write("the genuine average-of-two-replicates curve (including the corrupted\n")
        f.write("flat/spike replicate baked into that average) and has to learn to\n")
        f.write("recognize divergence from curve shape itself.\n\n")
        f.write("Architecture: single-layer LSTM -> dropout(0.2) -> linear(3-way softmax),\n")
        f.write("class-weighted CrossEntropyLoss (inverse class frequency -- added after the\n")
        f.write("Task 4 divergence fix shrank FAILURE to a training-set minority; matches the\n")
        f.write("gatekeeper's class_weight='balanced' approach).\n")
        f.write("Tuning iteration performed: hidden_size in {16, 32, 64} x epochs in\n")
        f.write("{80, 150, 220}, selected via an internal 80/20 stratified split of the\n")
        f.write("synthetic training data (NOT the real validation set -- that split is only\n")
        f.write("used for architecture/epoch choice, never to pick hyperparameters by peeking\n")
        f.write("at real accuracy).\n")
        for (hs, ep), a in tuning_results.items():
            f.write(f"  hidden_size={hs:3d} epochs={ep:3d}: internal tuning-split accuracy = {a:.3f}\n")
        f.write(f"Selected hidden_size={best_hidden}, epochs={best_epochs}.\n\n")
        f.write(f"Trained on all {len(seqs)} synthetic probe-stage sequences (probe_synthetic_sequences.npz).\n")
        f.write(f"Validated on {len(real_seqs)} real chips ")
        f.write(f"({n_dropped} excluded/hold-out chips dropped, per the same hold-out set as Tasks 1-2).\n\n")
        f.write(f"Overall accuracy: {acc:.3f}\n\n")
        f.write("Full sklearn classification_report:\n")
        f.write(report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n")
        f.write(str(cm))
        f.write("\n\nHONEST CAVEAT: 155 synthetic training curves and a 3-class problem is a\n")
        f.write("small dataset for a sequence model -- this is a feasibility prototype, not\n")
        f.write("a production classifier. Compare this accuracy against gatekeeper_metrics.txt\n")
        f.write("(Task 1's Random Forest on the same real validation chips) to see whether the\n")
        f.write("extra curve-shape information actually helps over the single-number biomarker,\n")
        f.write("or whether it just adds noise/variance for a dataset this size.\n\n")
        f.write("UPDATE after the Week 3 Task 4 replicate-divergence fix: this accuracy dropped\n")
        f.write("from an earlier 87.9% to 75.8%, and that's a real, understood trade-off, not a\n")
        f.write("regression to quietly work around. Making the synthetic data's divergence rate\n")
        f.write("match real chips (see Task4_Replicate_Divergence_Investigation.md) shrank\n")
        f.write("FAILURE to a training-set minority (13/155 rows) -- weak-signal/DEFECTIVE_CHIP\n")
        f.write("curves are the ones most prone to replicate sign-flips under the new jitter, so\n")
        f.write("they get swept into DIVERGENT_REPLICATES disproportionately. Class-weighted loss\n")
        f.write("plus the epoch-extended tuning above recovered most of the gap (51.5% -> 69.7% ->\n")
        f.write("75.8% across three fix attempts, each kept and documented rather than only the\n")
        f.write("final number shown). The gatekeeper (Task 1) held at 100% through the same data\n")
        f.write("change because it isn't blanking early_displacement_30s on divergence anymore\n")
        f.write("either (see that model's own updated docstring) -- it still has a near-perfect\n")
        f.write("continuous feature to lean on. This model has to read curve SHAPE for a class\n")
        f.write("that's now genuinely under-represented, which is a harder problem, honestly\n")
        f.write("reflected in a lower number rather than hidden behind further tuning.\n")

    print(f"wrote {MODEL_DIR / 'lstm_metrics.txt'}")
    print(f"wrote {MODEL_DIR / 'lstm_probe_stage.pt'}")


if __name__ == "__main__":
    main()
