"""
Next improvement attempt after augmented_architecture_comparison.py found
that no single architecture (LSTM/LSTM+Attention/TCN, all trained on the
augmented 310-sequence set) beats the officially-promoted plain LSTM.

Different architectures can still make DIFFERENT mistakes on different
chips even when their overall accuracy ties or loses individually --
ensembling (averaging predicted class probabilities across models) can
sometimes recover a net win from that error diversity even when no single
member does. This tests that directly rather than assuming it works:
retrains the same three architectures (same tuning discipline, same
augmented training set) and evaluates a soft-vote ensemble (mean softmax
probability across all three, argmax of the average) against each
individual model and the officially-promoted baseline, on both the
33-chip set and the blind hold-out set.

Output: qtwin/models/augmented_ensemble_results.txt,
qtwin/models/augmented_ensemble_{real,holdout}_confusion_matrix.png
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatekeeper_model import LABELS
from model_trainer import SequenceLSTM, load_real_validation, normalize, to_tensor
from model_trainer_augmented import load_augmented_synthetic, load_holdout_validation
from sequence_architectures import LSTMAttention, TCN
from augmented_architecture_comparison import ARCHITECTURES, tune_and_train, evaluate

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.manual_seed(20260912)
# Determinism fix (2026-09-12) -- see model_trainer.py's matching comment.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


@torch.no_grad()
def softmax_probs(model, X):
    model.eval()
    return torch.softmax(model(X), dim=1).numpy()


def run_eval(y_pred_idx, real_labels, split_label, cmap, out_stub):
    pred_labels = [LABELS[i] for i in y_pred_idx]
    acc = accuracy_score(real_labels, pred_labels)
    report = classification_report(real_labels, pred_labels, labels=LABELS, zero_division=0, output_dict=True)
    report_text = classification_report(real_labels, pred_labels, labels=LABELS, zero_division=0)
    cm = confusion_matrix(real_labels, pred_labels, labels=LABELS)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"Soft-vote ensemble (augmented) -- {split_label}", fontsize=10)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / f"{out_stub}.png", dpi=130)
    plt.close(fig)

    return acc, report, report_text, cm


def main():
    seqs, labels = load_augmented_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)

    idx_train, idx_tune = train_test_split(
        np.arange(len(seqs_norm)), test_size=0.2, random_state=42, stratify=labels
    )
    X_tr, y_tr = to_tensor(seqs_norm[idx_train], labels[idx_train])
    X_tune, y_tune = to_tensor(seqs_norm[idx_tune], labels[idx_tune])
    X_all, y_all = to_tensor(seqs_norm, labels)

    real_seqs, real_labels, _, n_dropped = load_real_validation()
    real_norm = normalize(real_seqs, mean, std)
    X_real, y_real = to_tensor(real_norm, real_labels)

    holdout_seqs, holdout_labels, holdout_chip_ids = load_holdout_validation()
    holdout_norm = normalize(holdout_seqs, mean, std)
    X_holdout, y_holdout = to_tensor(holdout_norm, holdout_labels)

    models = {}
    individual_results = {}
    for name, spec in ARCHITECTURES.items():
        model, best_params, best_epochs, _ = tune_and_train(name, spec, X_tr, y_tr, X_tune, y_tune, X_all, y_all)
        models[name] = model
        real_acc, real_pred_idx = evaluate(model, X_real, y_real)
        holdout_acc, holdout_pred_idx = evaluate(model, X_holdout, y_holdout)
        individual_results[name] = (real_acc, holdout_acc)
        print(f"{name}: 33chip={real_acc:.3f} holdout={holdout_acc:.3f}")

    # Soft-vote: average softmax probability across all three models, argmax the average.
    real_probs = np.mean([softmax_probs(m, X_real) for m in models.values()], axis=0)
    holdout_probs = np.mean([softmax_probs(m, X_holdout) for m in models.values()], axis=0)
    real_pred_idx = real_probs.argmax(axis=1)
    holdout_pred_idx = holdout_probs.argmax(axis=1)

    real_acc, real_report, real_report_text, real_cm = run_eval(
        real_pred_idx, real_labels, "33-chip real validation", "Greens", "augmented_ensemble_real_confusion_matrix"
    )
    holdout_acc, holdout_report, holdout_report_text, holdout_cm = run_eval(
        holdout_pred_idx, holdout_labels, "BLIND hold-out", "Purples", "augmented_ensemble_holdout_confusion_matrix"
    )

    print(f"\nEnsemble (soft-vote, all 3): 33chip={real_acc:.3f} holdout={holdout_acc:.3f}")
    print(real_report_text)
    print(holdout_report_text)

    official_lstm_33chip, official_lstm_holdout = 0.758, 0.727

    with open(MODEL_DIR / "augmented_ensemble_results.txt", "w", encoding="utf-8") as f:
        f.write("Soft-vote ensemble: LSTM + LSTM+Attention + TCN (all augmented-trained)\n")
        f.write("=" * 74 + "\n\n")
        f.write("Follow-up to augmented_architecture_comparison.txt, which found no single\n")
        f.write("architecture beats the officially-promoted plain LSTM on the augmented\n")
        f.write("310-sequence training set. Different architectures can still make\n")
        f.write("DIFFERENT mistakes on different chips even when tied/losing individually --\n")
        f.write("this tests whether averaging predicted probabilities across all three\n")
        f.write("(soft-vote ensemble, mean softmax then argmax) recovers a net improvement\n")
        f.write("from that error diversity, rather than assuming it would.\n\n")
        f.write(f"Currently-promoted official model: single-replicate-augmented LSTM,\n")
        f.write(f"33-chip accuracy = {official_lstm_33chip:.3f}, hold-out accuracy = {official_lstm_holdout:.3f}\n\n")
        f.write("Individual models in this ensemble (same run, same tuning, feeds the\n")
        f.write("ensemble below):\n")
        for name, (racc, hacc) in individual_results.items():
            f.write(f"  {name:20s}  33chip={racc:.3f}  holdout={hacc:.3f}\n")
        f.write(f"\nENSEMBLE (soft-vote, mean of all 3 softmax outputs):\n")
        f.write(f"  33-chip accuracy: {real_acc:.3f}\n")
        f.write(f"  hold-out accuracy: {holdout_acc:.3f}\n\n")
        f.write("33-chip real validation report:\n")
        f.write(real_report_text)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{real_cm}\n\n")
        f.write("BLIND hold-out report:\n")
        f.write(holdout_report_text)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{holdout_cm}\n\n")

        f.write("HONEST ASSESSMENT:\n")
        best_individual_holdout = max(individual_results.values(), key=lambda r: r[1])[1]
        if holdout_acc > max(official_lstm_holdout, best_individual_holdout):
            f.write(f"The ensemble ({holdout_acc:.3f} hold-out) beats BOTH the officially-promoted\n")
            f.write(f"LSTM ({official_lstm_holdout:.3f}) and every individual member tested here\n")
            f.write(f"(best individual: {best_individual_holdout:.3f}) -- genuine evidence the\n")
            f.write("architectures' errors are diverse enough for averaging to help. Real\n")
            f.write("promotion candidate, though ensembling triples inference cost/complexity\n")
            f.write("for a small hold-out-set gain (n=11) -- worth weighing against the\n")
            f.write("single-model simplicity the current official model has.\n")
        elif holdout_acc >= official_lstm_holdout:
            f.write(f"The ensemble ({holdout_acc:.3f} hold-out) ties or marginally beats the\n")
            f.write(f"officially-promoted LSTM ({official_lstm_holdout:.3f}), but does not clearly\n")
            f.write(f"beat the best individual model tested here ({best_individual_holdout:.3f}) --\n")
            f.write("not a strong enough case to justify tripling model complexity for\n")
            f.write("production. Reported as a real but marginal result, not oversold.\n")
        else:
            f.write(f"The ensemble ({holdout_acc:.3f} hold-out) does NOT beat the officially-\n")
            f.write(f"promoted LSTM ({official_lstm_holdout:.3f}) -- another honest null result.\n")
            f.write("Combined with augmented_architecture_comparison.txt's finding, the\n")
            f.write("simplest model (plain LSTM, single-replicate augmentation, no ensemble)\n")
            f.write("remains the best-supported choice found across this project's full\n")
            f.write("history of attempts. Reported as-is rather than forced into a positive\n")
            f.write("finding.\n")
        f.write("\nn=11 hold-out, n=33 real validation -- both small, treat any single-chip\n")
        f.write("swing as real sampling noise, not a decisive result on its own.\n")

    print(f"\nwrote {MODEL_DIR / 'augmented_ensemble_results.txt'}")


if __name__ == "__main__":
    main()
