"""
Combines the two separate, previously-unmerged fixes sitting in this
project's history:

1. Single-replicate data augmentation (augment_single_replicate_data.py /
   model_trainer_augmented.py) -- fixed the LSTM baseline's blind hold-out
   drop (45.5% -> 72.7%) by adding 155 single-replicate synthetic
   sequences, but was only ever applied to the plain SequenceLSTM
   architecture.

2. LSTM+Attention / TCN architectures (sequence_architectures.py /
   lstm_tcn_comparison.py) -- both beat the plain LSTM baseline (87.9% vs
   75.8%) on DIVERGENT_REPLICATES precision and overall accuracy, but were
   only ever trained on the ORIGINAL 155-sequence (two-replicate-averaged
   only) training set, and were NEVER evaluated against the blind
   hold-out set at all -- lstm_tcn_comparison.py's own text explicitly
   flags "LSTM+Attention is the more natural promotion candidate...
   Deliberately NOT swapping this in... flagged here for explicit
   sign-off", and Known_Limitations_Master.md separately flags that
   neither architecture has been "re-compared against the single-
   replicate-augmented LSTM baseline."

This script closes BOTH open items in one pass: retrains all three
architectures (LSTM baseline, LSTM+Attention, TCN) on the AUGMENTED
310-sequence training set, then evaluates each on BOTH the 33-chip real
validation set AND the blind hold-out set -- the first time any of these
three architectures has been checked against the hold-out set at all.
Same internal-80/20-synthetic-tuning-split discipline as every other
model in this project.

If any architecture beats the currently-promoted single-replicate-
augmented LSTM (75.8% / 72.7%) on hold-out without a real trade-off
elsewhere, it becomes the new promotion candidate.

Output: qtwin/models/augmented_architecture_comparison.txt,
qtwin/models/augmented_{name}_{real,holdout}_confusion_matrix.png
"""

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from gatekeeper_model import LABELS
from model_trainer import SequenceLSTM, load_real_validation, normalize, to_tensor
from model_trainer_augmented import load_augmented_synthetic, load_holdout_validation
from sequence_architectures import LSTMAttention, TCN

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
torch.manual_seed(20260912)
# Determinism fix (2026-09-12) -- see model_trainer.py's matching comment.
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)


def train_generic(model_ctor, model_kwargs, X_train, y_train, epochs, lr=1e-2):
    model = model_ctor(**model_kwargs)
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
    return accuracy_score(y.numpy(), pred.numpy()), pred.numpy()


ARCHITECTURES = {
    "LSTM (baseline)": {
        "ctor": SequenceLSTM,
        "grid": [{"hidden_size": hs} for hs in (16, 32, 64)],
    },
    "LSTM+Attention": {
        "ctor": LSTMAttention,
        "grid": [{"hidden_size": hs} for hs in (16, 32, 64)],
    },
    "TCN": {
        "ctor": TCN,
        "grid": [{"channels": c, "n_layers": nl} for c in (8, 16, 32) for nl in (2, 3)],
    },
}
EPOCHS_GRID = (80, 150, 220)


def tune_and_train(name, spec, X_tr, y_tr, X_tune, y_tune, X_all, y_all):
    print(f"\n=== {name}: tuning on internal 80/20 synthetic split (augmented set) ===")
    results = {}
    for params in spec["grid"]:
        for epochs in EPOCHS_GRID:
            m = train_generic(spec["ctor"], params, X_tr, y_tr, epochs)
            acc, _ = evaluate(m, X_tune, y_tune)
            key = tuple(sorted(params.items())) + (("epochs", epochs),)
            results[key] = acc
            print(f"  {dict(params)} epochs={epochs:3d}: internal tuning-split accuracy = {acc:.3f}")
    best_key = max(results, key=results.get)
    best_params = {k: v for k, v in best_key if k != "epochs"}
    best_epochs = dict(best_key)["epochs"]
    print(f"Selected {best_params}, epochs={best_epochs} (internal accuracy {results[best_key]:.3f})")

    final_model = train_generic(spec["ctor"], best_params, X_all, y_all, best_epochs)
    return final_model, best_params, best_epochs, results


def run_eval(model, X, y, real_labels, name, split_label, cmap, out_stub):
    acc, pred_idx = evaluate(model, X, y)
    pred_labels = [LABELS[i] for i in pred_idx]
    report = classification_report(real_labels, pred_labels, labels=LABELS, zero_division=0, output_dict=True)
    report_text = classification_report(real_labels, pred_labels, labels=LABELS, zero_division=0)
    cm = confusion_matrix(real_labels, pred_labels, labels=LABELS)

    fig, ax = plt.subplots(figsize=(7, 5.8))
    ConfusionMatrixDisplay(cm, display_labels=LABELS).plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(f"{name} (augmented) -- {split_label}", fontsize=10)
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

    print(f"Training set: {len(seqs)} sequences (155 original + 155 single-replicate augmentation)")

    full_reports = {}
    for name, spec in ARCHITECTURES.items():
        slug = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        model, best_params, best_epochs, tuning_results = tune_and_train(
            name, spec, X_tr, y_tr, X_tune, y_tune, X_all, y_all
        )

        real_acc, real_report, real_report_text, real_cm = run_eval(
            model, X_real, y_real, real_labels, name, "33-chip real validation", "Greens",
            f"augmented_{slug}_real_confusion_matrix"
        )
        holdout_acc, holdout_report, holdout_report_text, holdout_cm = run_eval(
            model, X_holdout, y_holdout, holdout_labels, name, "BLIND hold-out", "Purples",
            f"augmented_{slug}_holdout_confusion_matrix"
        )

        print(f"\n{name} (augmented) -- 33-chip accuracy: {real_acc:.3f}, hold-out accuracy: {holdout_acc:.3f}")
        print(real_report_text)
        print(holdout_report_text)

        full_reports[name] = {
            "best_params": best_params, "best_epochs": best_epochs, "tuning_results": tuning_results,
            "real_acc": real_acc, "real_report": real_report, "real_report_text": real_report_text, "real_cm": real_cm,
            "holdout_acc": holdout_acc, "holdout_report": holdout_report, "holdout_report_text": holdout_report_text,
            "holdout_cm": holdout_cm,
        }

    print("\n=== Comparison summary (augmented training set) ===")
    for name, r in full_reports.items():
        div_p = r["real_report"]["DIVERGENT_REPLICATES"]["precision"]
        div_r = r["real_report"]["DIVERGENT_REPLICATES"]["recall"]
        print(f"{name:20s}  33chip={r['real_acc']:.3f}  holdout={r['holdout_acc']:.3f}  "
              f"DIVERGENT_precision={div_p:.3f}  DIVERGENT_recall={div_r:.3f}")

    # UPDATED 2026-09-12: this used to hardcode the augmented plain LSTM's
    # numbers (0.758/0.727) as "the currently-promoted model" -- stale ever
    # since TCN was promoted official (0.727/0.727, see
    # promote_tcn_official.py/tcn_official_metrics.txt -- the ACTUAL
    # deployed TCN artifact's own dedicated-run number, not this script's
    # in-sequence TCN row below, which differs due to RNG-consumption
    # order; see Known_Limitations_Master.md's canonical-number note).
    # Renamed from official_lstm_* since the official model is TCN now,
    # not LSTM -- same category of stale-reference bug as
    # benchmark_comparison.py's/holdout_validation.py's hardcoded
    # constants, caught during a dedicated sweep for this exact pattern.
    official_model_name = "TCN (promoted, standalone run)"
    official_33chip = 0.727
    official_holdout = 0.727

    with open(MODEL_DIR / "augmented_architecture_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Augmented architecture comparison: LSTM vs. LSTM+Attention vs. TCN,\n")
        f.write("all trained on the single-replicate-augmented (310-sequence) training set\n")
        f.write("=" * 78 + "\n\n")
        f.write("Closes two previously-separate open items:\n")
        f.write("1. lstm_tcn_comparison.py flagged LSTM+Attention/TCN as beating the plain\n")
        f.write("   LSTM (87.9% vs 75.8%) but never promoted them, and never trained them on\n")
        f.write("   the single-replicate-augmented data (that fix came later).\n")
        f.write("2. Known_Limitations_Master.md flagged that neither architecture had been\n")
        f.write("   re-compared against the augmented LSTM baseline, or evaluated on the\n")
        f.write("   blind hold-out set AT ALL -- this is the first time either has been.\n\n")
        f.write(f"Officially-promoted model: {official_model_name},\n")
        f.write(f"33-chip accuracy = {official_33chip:.3f}, hold-out accuracy = {official_holdout:.3f}\n")
        f.write("(tcn_official_metrics.txt) -- the baseline every architecture below is\n")
        f.write("compared against.\n\n")
        f.write(f"Training set: {len(seqs)} sequences (155 original two-replicate-averaged +\n")
        f.write("155 single-replicate augmentation). Same internal 80/20 synthetic tuning-\n")
        f.write("split discipline as every other model in this project.\n\n")

        f.write("Summary (33-chip real validation vs. BLIND hold-out):\n")
        for name, r in full_reports.items():
            div_p = r["real_report"]["DIVERGENT_REPLICATES"]["precision"]
            div_r = r["real_report"]["DIVERGENT_REPLICATES"]["recall"]
            f.write(f"  {name:20s}  33chip_acc={r['real_acc']:.3f}  holdout_acc={r['holdout_acc']:.3f}  "
                    f"DIVERGENT_precision={div_p:.3f}  DIVERGENT_recall={div_r:.3f}  "
                    f"(params={r['best_params']}, epochs={r['best_epochs']})\n")
        f.write("\n")

        for name, r in full_reports.items():
            f.write("-" * 78 + "\n")
            f.write(f"{name} (augmented training set)\n")
            f.write(f"Selected hyperparameters: {r['best_params']}, epochs={r['best_epochs']}\n")
            f.write("Internal tuning-split results:\n")
            for key, tacc in r["tuning_results"].items():
                f.write(f"  {dict(key)}: {tacc:.3f}\n")
            f.write(f"\n33-chip real validation accuracy: {r['real_acc']:.3f}\n")
            f.write(r["real_report_text"])
            f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{r['real_cm']}\n\n")
            f.write(f"BLIND hold-out accuracy: {r['holdout_acc']:.3f}\n")
            f.write(r["holdout_report_text"])
            f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{r['holdout_cm']}\n\n")

        best_holdout = max(full_reports.items(), key=lambda kv: kv[1]["holdout_acc"])
        best_33chip = max(full_reports.items(), key=lambda kv: kv[1]["real_acc"])
        f.write("HONEST INTERPRETATION:\n")
        f.write(f"Best hold-out accuracy in THIS comparison: {best_holdout[0]} "
                f"({best_holdout[1]['holdout_acc']:.3f}) vs. the officially-promoted "
                f"{official_model_name}'s {official_holdout:.3f}.\n")
        f.write(f"Best 33-chip accuracy in THIS comparison: {best_33chip[0]} "
                f"({best_33chip[1]['real_acc']:.3f}) vs. the officially-promoted "
                f"{official_model_name}'s {official_33chip:.3f}.\n\n")
        f.write("NOTE on the 'LSTM (baseline)' row: this is a fresh training run within this\n")
        f.write("script's own three-way sweep (same architecture/training-set/tuning as any\n")
        f.write("other plain-LSTM run, different RNG seed/consumption order), not a claim\n")
        f.write("about the officially-promoted model, which is TCN, not LSTM -- see\n")
        f.write("tcn_official_metrics.txt for that artifact's own dedicated numbers.\n\n")

        # BUG FOUND AND FIXED (2026-09-12): comparing raw computed floats
        # (e.g. 8/11 = 0.7272727...) against hardcoded, already-ROUNDED
        # reference literals (0.727) meant a chip-for-chip TIE could read as
        # "beats" purely from floating-point noise below display precision
        # -- caught when this printed "LSTM+Attention beats TCN on BOTH
        # metrics (0.727/0.727 vs 0.727/0.727)," identical-looking numbers
        # claimed as a win. Round both sides to the same 3-decimal display
        # precision before comparing so a genuine tie reads as a tie.
        if (round(best_holdout[1]["holdout_acc"], 3) > round(official_holdout, 3)
                and round(best_holdout[1]["real_acc"], 3) > round(official_33chip, 3)):
            f.write(f"POTENTIAL PROMOTION CANDIDATE: {best_holdout[0]} beats the officially-\n")
            f.write(f"promoted {official_model_name} on BOTH metrics "
                    f"({best_holdout[1]['real_acc']:.3f}/{best_holdout[1]['holdout_acc']:.3f} vs "
                    f"{official_33chip:.3f}/{official_holdout:.3f}) using the SAME augmented\n")
            f.write("training data -- worth a deliberate re-promotion decision, not acted on\n")
            f.write("automatically by rerunning this script.\n")
        else:
            f.write(f"No architecture in this comparison clearly beats the officially-promoted\n")
            f.write(f"{official_model_name} on both metrics simultaneously, given this hold-out\n")
            f.write("set's small n=11 (each chip is ~9% of the reported accuracy) -- ties and\n")
            f.write("single-metric edges are within this project's known run-to-run margin.\n")
        f.write("\nn=11 hold-out and n=33 real validation are both small -- treat every\n")
        f.write("number here as having real sampling noise, especially any single-chip\n")
        f.write("swing on the hold-out set. HONEST METHODOLOGY NOTE (same as\n")
        f.write("model_trainer_augmented.py / lstm_augmented_metrics.txt): reporting\n")
        f.write("hold-out numbers as part of an architecture-selection decision means this\n")
        f.write("hold-out set is no longer fully untouched by downstream choices.\n")

    print(f"\nwrote {MODEL_DIR / 'augmented_architecture_comparison.txt'}")
    return full_reports


if __name__ == "__main__":
    main()
