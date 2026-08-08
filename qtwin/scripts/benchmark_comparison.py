"""
Testing week Task 3: lightweight benchmarking -- is the LSTM's sequence
architecture actually earning its complexity over a standard baseline?

Benchmarks the official LSTM (Task 3/Week 3-4) against a Random Forest
trained on the SAME 60-point curves, just flattened into 60 independent
features instead of processed sequentially. Same synthetic training data,
same 33-chip real validation split, same hold-out exclusion.

Output: qtwin/models/benchmark_comparison.txt
"""

from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from gatekeeper_model import LABELS
from model_trainer import load_synthetic, load_real_validation, normalize

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    seqs, labels = load_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)  # same normalization as the LSTM, for a fair comparison

    real_seqs, real_labels, _, n_dropped = load_real_validation()
    real_norm = normalize(real_seqs, mean, std)

    # Caught during an independent review: an earlier version of this script
    # used a single arbitrary RandomForest config (max_depth=8, n_estimators=200)
    # with no tuning at all, while every OTHER model in this project (including
    # the gatekeeper's own RF) is at least minimally justified. A quick check
    # showed shallower trees do meaningfully better on this flattened
    # representation (depth=3-4 reaches 0.636 vs. the untuned 0.515) -- an
    # undertuned baseline makes "the sequence models win" a weaker, easier
    # claim to challenge than it needs to be. Small grid, same internal-tuning-
    # split discipline used everywhere else in this project (never selecting by
    # peeking at real validation accuracy).
    seqs_tr, seqs_tune, labels_tr, labels_tune = train_test_split(
        seqs_norm, labels, test_size=0.2, random_state=42, stratify=labels
    )
    best_acc, best_params = -1.0, None
    tuning_results = {}
    for depth in (3, 4, 5, 6, 8, None):
        for n_est in (100, 200, 400):
            m = RandomForestClassifier(n_estimators=n_est, max_depth=depth, random_state=42, class_weight="balanced")
            m.fit(seqs_tr, labels_tr)
            acc = accuracy_score(labels_tune, m.predict(seqs_tune))
            tuning_results[(depth, n_est)] = acc
            if acc > best_acc:
                best_acc, best_params = acc, (depth, n_est)
    best_depth, best_n_est = best_params
    print(f"Selected max_depth={best_depth}, n_estimators={best_n_est} (internal tuning-split accuracy {best_acc:.3f})")

    rf = RandomForestClassifier(n_estimators=best_n_est, max_depth=best_depth, random_state=42, class_weight="balanced")
    rf.fit(seqs_norm, labels)  # (N, 60) flattened -- sklearn treats each timestep as an independent feature
    rf_pred = rf.predict(real_norm)
    rf_acc = accuracy_score(real_labels, rf_pred)
    rf_report = classification_report(real_labels, rf_pred, labels=LABELS, zero_division=0)
    rf_cm = confusion_matrix(real_labels, rf_pred, labels=LABELS)

    print(f"Random Forest (flattened 60-point curve): accuracy = {rf_acc:.3f}")
    print(rf_report)

    # Official LSTM numbers (lstm_metrics.txt) and Week 4 comparison numbers
    # (lstm_tcn_comparison.txt), quoted directly rather than re-run here to
    # avoid re-training-noise -- see those files for the exact source runs.
    # UPDATED (2026-09-12) after fixing a CPU-thread non-determinism bug in
    # every torch training script -- these numbers changed from the
    # pre-fix values (0.758 / 0.879) once training became actually
    # reproducible. Attention and TCN no longer tie (0.788 vs 0.727), so
    # they're tracked separately instead of one combined constant.
    official_lstm_acc = 0.818
    attention_tcn_acc = 0.788  # best of LSTM+Attention (0.788) / TCN (0.727)

    with open(MODEL_DIR / "benchmark_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Testing week Task 3 -- lightweight benchmarking\n")
        f.write("=" * 60 + "\n\n")
        f.write("Random Forest on the same 60-point resampled probe curves as the LSTM,\n")
        f.write("just flattened into 60 independent features (no sequential processing) --\n")
        f.write("same synthetic training data, same 33-chip real validation set, same\n")
        f.write("hold-out exclusion, same class-balancing approach as the gatekeeper's RF.\n\n")
        f.write("max_depth x n_estimators tuned via an internal 80/20 synthetic tuning split\n")
        f.write("(an earlier version of this script used a single untuned config and reported\n")
        f.write("0.515 -- caught during independent review that shallower trees do meaningfully\n")
        f.write("better, 0.636 at depth 3-4 with the untuned n_estimators=200, so this baseline\n")
        f.write("was understating its own ceiling and making the comparison easier than it\n")
        f.write("needed to be):\n")
        for (depth, n_est), acc in tuning_results.items():
            f.write(f"  max_depth={depth} n_estimators={n_est}: internal tuning-split accuracy = {acc:.3f}\n")
        f.write(f"Selected max_depth={best_depth}, n_estimators={best_n_est}.\n\n")
        f.write(f"Random Forest (flattened curve, tuned):  accuracy = {rf_acc:.3f}\n")
        f.write(rf_report)
        f.write(f"Confusion matrix: {rf_cm.tolist()}\n\n")
        f.write(f"Official LSTM (lstm_metrics.txt):                    accuracy = {official_lstm_acc:.3f}\n")
        f.write(f"Week 4 LSTM+Attention/TCN (lstm_tcn_comparison.txt): accuracy = {attention_tcn_acc:.3f}\n\n")
        f.write("CONCLUSION: ")
        if rf_acc >= attention_tcn_acc:
            f.write(f"the flattened-feature Random Forest ({rf_acc:.3f}) matches or beats even\n")
            f.write(f"the best sequence model tried ({attention_tcn_acc:.3f}) -- on a dataset this\n")
            f.write("small (155 synthetic curves), the sequential/architectural sophistication of\n")
            f.write("the LSTM/TCN family isn't clearly earning its complexity over a much simpler\n")
            f.write("baseline that just treats each timestep as an independent feature. Worth\n")
            f.write("stating plainly in the proposal rather than presenting the sequence models as\n")
            f.write("an obviously superior architecture choice.\n")
        elif rf_acc >= official_lstm_acc:
            f.write(f"the flattened-feature Random Forest ({rf_acc:.3f}) beats the official LSTM\n")
            f.write(f"({official_lstm_acc:.3f}) but not the tuned LSTM+Attention/TCN comparison\n")
            f.write(f"({attention_tcn_acc:.3f}) -- suggesting the sequential architecture DOES add\n")
            f.write("value, but only once properly tuned (attention/TCN), not in its first-pass form.\n")
        else:
            f.write(f"the sequence models beat the flattened-feature Random Forest baseline\n")
            f.write(f"({rf_acc:.3f}) at every stage -- genuine evidence that treating this as a\n")
            f.write("sequence (not just 60 independent numbers) is earning real value, not just\n")
            f.write("added complexity.\n")

    print(f"wrote {MODEL_DIR / 'benchmark_comparison.txt'}")


if __name__ == "__main__":
    main()
