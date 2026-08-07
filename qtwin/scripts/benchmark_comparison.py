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

from gatekeeper_model import LABELS
from model_trainer import load_synthetic, load_real_validation, normalize

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    seqs, labels = load_synthetic()
    mean, std = seqs.mean(), seqs.std()
    seqs_norm = normalize(seqs, mean, std)  # same normalization as the LSTM, for a fair comparison

    real_seqs, real_labels, _, n_dropped = load_real_validation()
    real_norm = normalize(real_seqs, mean, std)

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight="balanced")
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
    official_lstm_acc = 0.758
    attention_tcn_acc = 0.879

    with open(MODEL_DIR / "benchmark_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Testing week Task 3 -- lightweight benchmarking\n")
        f.write("=" * 60 + "\n\n")
        f.write("Random Forest on the same 60-point resampled probe curves as the LSTM,\n")
        f.write("just flattened into 60 independent features (no sequential processing) --\n")
        f.write("same synthetic training data, same 33-chip real validation set, same\n")
        f.write("hold-out exclusion, same class-balancing approach as the gatekeeper's RF.\n\n")
        f.write(f"Random Forest (flattened curve):  accuracy = {rf_acc:.3f}\n")
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
