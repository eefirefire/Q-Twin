"""
LSTM improvement: retrain on the ORIGINAL 155 two-replicate-averaged
synthetic sequences PLUS the 155 new single-replicate augmentation
sequences (augment_single_replicate_data.py) -- 310 total training
examples -- to directly address the diagnosed root cause of the LSTM's
weak SINGLE_REPLICATE performance (56.25% on that subclass in the 33-chip
set; see Known_Limitations_Master.md Stage 2a).

Same architecture (SequenceLSTM), same class-weighted loss, same
internal-80/20-synthetic-tuning-split discipline as model_trainer.py --
the ONLY change is what's in the training set. Evaluated on the same 33
real non-hold-out chips AND the hold-out set, reported side by side with
the pre-augmentation numbers for a direct, honest before/after.

HONEST METHODOLOGY NOTE: the root-cause diagnosis (SINGLE_REPLICATE
weakness) was visible in the 33-chip validation set on its own (56.25%
there), not derived from hold-out failures alone -- so this fix isn't
reverse-engineered from hold-out chip identities. That said, once the
hold-out numbers are reported here, this hold-out set can no longer be
treated as untouched by any downstream decision the way it was before --
flagged plainly rather than glossed over.

Output: qtwin/models/lstm_augmented_metrics.txt,
qtwin/models/lstm_augmented_confusion_matrix.png,
qtwin/models/lstm_augmented_holdout_confusion_matrix.png
"""

from pathlib import Path

import numpy as np
import torch
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

from gatekeeper_model import LABELS
from holdout import load_holdout_chip_ids
from model_trainer import (
    SequenceLSTM,
    load_synthetic,
    load_real_validation,
    normalize,
    to_tensor,
    train_model,
    evaluate,
)

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.manual_seed(20260907)
# Determinism fix (2026-09-12) -- see model_trainer.py's matching comment.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def load_augmented_synthetic():
    """Original 155 two-replicate-averaged sequences + 155 new
    single-replicate augmentation sequences = 310 total."""
    orig_seqs, orig_labels = load_synthetic()

    aug = np.load(DATA_DIR / "probe_single_replicate_sequences.npz", allow_pickle=True)
    aug_seqs, aug_labels = aug["sequences"], aug["label"]

    all_seqs = np.concatenate([orig_seqs, aug_seqs], axis=0)
    all_labels = np.concatenate([orig_labels, aug_labels], axis=0)
    return all_seqs, all_labels


def load_holdout_validation():
    """Mirrors load_real_validation() but returns hold-out chips instead
    of non-hold-out ones -- used only for the final, clearly-labeled
    honest comparison below, never for any tuning decision."""
    npz = np.load(DATA_DIR / "real_probe_sequences.npz", allow_pickle=True)
    seqs, chip_ids = npz["sequences"], npz["chip_id"]
    import pandas as pd
    from gatekeeper_model import build_label
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv").set_index("chip_id")
    holdout_ids = load_holdout_chip_ids()

    keep_mask, labels = [], []
    for cid in chip_ids:
        row = cs.loc[cid]
        if row["success_or_fail"] == "EXCLUDED" or cid not in holdout_ids:
            keep_mask.append(False)
            continue
        keep_mask.append(True)
        labels.append(build_label(row["success_or_fail"], row["displacement_replicate_status"]))
    keep_mask = np.array(keep_mask)
    return seqs[keep_mask], np.array(labels), chip_ids[keep_mask]


def run_eval(model, seqs, labels, mean, std, name, cmap, out_stub):
    seqs_norm = normalize(seqs, mean, std)
    X, y = to_tensor(seqs_norm, labels)
    acc, pred_idx = evaluate(model, X, y)
    pred_labels = [LABELS[i] for i in pred_idx]
    report = classification_report(labels, pred_labels, labels=LABELS, zero_division=0)
    cm = confusion_matrix(labels, pred_labels, labels=LABELS)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"LSTM (single-replicate-augmented) -- {name} (n={len(labels)})", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / f"{out_stub}.png", dpi=130)
    plt.close(fig)

    return acc, report, cm


def main():
    seqs, labels = load_augmented_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)

    idx_train, idx_tune = train_test_split(
        np.arange(len(seqs_norm)), test_size=0.2, random_state=42, stratify=labels
    )
    X_tr, y_tr = to_tensor(seqs_norm[idx_train], labels[idx_train])
    X_tune, y_tune = to_tensor(seqs_norm[idx_tune], labels[idx_tune])

    print(f"Training set size: {len(seqs)} (155 original + 155 single-replicate augmentation)")
    print("Tuning hidden_size x epochs on internal 80/20 split (same grid as model_trainer.py)...")
    tuning_results = {}
    for hidden_size in (16, 32, 64):
        for epochs in (80, 150, 220):
            m = train_model(X_tr, y_tr, hidden_size=hidden_size, epochs=epochs)
            acc, _ = evaluate(m, X_tune, y_tune)
            tuning_results[(hidden_size, epochs)] = acc
            print(f"  hidden_size={hidden_size:3d} epochs={epochs:3d}: internal tuning-split accuracy = {acc:.3f}")
    best_hidden, best_epochs = max(tuning_results, key=tuning_results.get)
    print(f"Selected hidden_size={best_hidden}, epochs={best_epochs}")

    X_all, y_all = to_tensor(seqs_norm, labels)
    final_model = train_model(X_all, y_all, hidden_size=best_hidden, epochs=best_epochs)

    real_seqs, real_labels, _, n_dropped = load_real_validation()
    real_acc, real_report, real_cm = run_eval(
        final_model, real_seqs, real_labels, mean, std,
        "33-chip real validation", "Greens", "lstm_augmented_confusion_matrix"
    )
    print(f"\n33-chip validation accuracy: {real_acc:.3f}")
    print(real_report)

    # Promote to the official model paths -- the user asked to apply the
    # improvement, not just report it. lstm_probe_stage.pt/lstm_config.json
    # are read by pipeline_api.load_lstm(), so the Streamlit mockup and any
    # future holdout_validation.py run automatically pick this up.
    import json
    torch.save(final_model.state_dict(), MODEL_DIR / "lstm_probe_stage.pt")
    with open(MODEL_DIR / "lstm_config.json", "w", encoding="utf-8") as f:
        json.dump({"hidden_size": best_hidden, "epochs": best_epochs,
                    "seq_mean": float(mean), "seq_std": float(std)}, f, indent=2)
    print(f"PROMOTED: wrote {MODEL_DIR / 'lstm_probe_stage.pt'} and lstm_config.json "
          f"(official model paths, now the single-replicate-augmented LSTM)")

    holdout_seqs, holdout_labels, holdout_chip_ids = load_holdout_validation()
    holdout_acc, holdout_report, holdout_cm = run_eval(
        final_model, holdout_seqs, holdout_labels, mean, std,
        "BLIND hold-out", "Purples", "lstm_augmented_holdout_confusion_matrix"
    )
    print(f"\nHold-out accuracy: {holdout_acc:.3f}")
    print(holdout_report)

    with open(MODEL_DIR / "lstm_augmented_metrics.txt", "w", encoding="utf-8") as f:
        f.write("LSTM improvement: single-replicate data augmentation\n")
        f.write("=" * 60 + "\n\n")
        f.write("Technique: added 155 single-replicate synthetic training sequences\n")
        f.write("(augment_single_replicate_data.py) alongside the original 155\n")
        f.write("two-replicate-averaged ones, directly targeting the diagnosed root\n")
        f.write("cause of the LSTM's weak SINGLE_REPLICATE performance (56.25% on that\n")
        f.write("subclass in the 33-chip validation set) -- the original training data\n")
        f.write("only ever contained averaged-2-replicate curves, so the model had never\n")
        f.write("seen genuine single-replicate noise statistics.\n\n")
        f.write(f"Training set: {len(seqs)} sequences (310 = 155 original + 155 augmentation).\n")
        f.write("Tuning: same hidden_size x epochs grid, same internal 80/20 synthetic\n")
        f.write("split discipline as model_trainer.py -- never selecting hyperparameters\n")
        f.write("by peeking at real validation accuracy.\n")
        for (hs, ep), acc in tuning_results.items():
            f.write(f"  hidden_size={hs:3d} epochs={ep:3d}: internal tuning-split accuracy = {acc:.3f}\n")
        f.write(f"Selected hidden_size={best_hidden}, epochs={best_epochs}.\n\n")

        f.write("-" * 60 + "\n")
        f.write("BEFORE (official LSTM, lstm_metrics.txt): 33-chip accuracy = 0.758, ")
        f.write("hold-out accuracy = 0.455\n\n")
        f.write("AFTER (this augmented model):\n")
        f.write(f"33-chip validation accuracy: {real_acc:.3f}\n")
        f.write(real_report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{real_cm}\n\n")
        f.write(f"Hold-out accuracy: {holdout_acc:.3f}\n")
        f.write(holdout_report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{holdout_cm}\n\n")

        f.write("-" * 60 + "\n")
        f.write("HONEST METHODOLOGY NOTE: the root-cause diagnosis (SINGLE_REPLICATE\n")
        f.write("weakness) was visible in the 33-chip validation set on its own (56.25%\n")
        f.write("there), independent of hold-out results -- so this fix isn't reverse-\n")
        f.write("engineered from hold-out chip identities. That said, reporting hold-out\n")
        f.write("numbers here means this hold-out set can no longer be treated as fully\n")
        f.write("untouched by any downstream modeling decision the way it was before this\n")
        f.write("change -- flagged plainly, not glossed over. A cleaner future test would\n")
        f.write("need a second, still-untouched hold-out set to confirm this generalizes.\n")

    print(f"\nwrote {MODEL_DIR / 'lstm_augmented_metrics.txt'}")


if __name__ == "__main__":
    main()
