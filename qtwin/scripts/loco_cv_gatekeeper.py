"""
Reviewer-anticipated gap: a single ~25% (11/44) hold-out split is a much
weaker generalization claim than Leave-One-Chip-Out Cross-Validation
(LOCO-CV) over ALL real chips -- pull one chip's data out entirely, train
on every other real chip, test on the one held out, repeat until every
chip has been the test case exactly once. This directly answers "can the
AI actually generalize to a brand-new chip it's never seen, not just the
11 chips we happened to reserve."

Deliberately PURE REAL-DATA LOCO-CV, no synthetic data in the training
fold -- this is a different, complementary experiment to the existing
synthetic-trained/real-validated gatekeeper (gatekeeper_model.py), not a
replacement for it. Only 44 valid real chips exist (45 - 1 EXCLUDED), so
each fold trains on 43 chips and tests on 1 -- honestly small, but it's
the standard LOCO-CV definition and it uses every real chip this project
has, not just the ones in one particular hold-out split.

Same 2 features as the official gatekeeper (early_displacement_30s,
is_divergent), same RandomForestClassifier config, so results are
directly comparable to gatekeeper_metrics.txt's synthetic-trained number.

Output: qtwin/models/loco_cv_gatekeeper_results.txt
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from gatekeeper_model import LABELS, build_features, build_label

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def load_all_real_chips():
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    y = valid.apply(
        lambda r: build_label(r["success_or_fail"], r["displacement_replicate_status"]), axis=1
    )
    X = build_features(valid, "early_displacement_30s", "displacement_replicate_status")
    return X.reset_index(drop=True), y.reset_index(drop=True), valid["chip_id"].reset_index(drop=True)


def main():
    X, y, chip_ids = load_all_real_chips()
    n = len(X)
    print(f"LOCO-CV over {n} real chips (pure real-data training, no synthetic).")
    print("Label distribution:\n", y.value_counts())

    y_true_all, y_pred_all, chip_results = [], [], []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test = X.iloc[[i]], y.iloc[i]

        clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)[0]

        y_true_all.append(y_test)
        y_pred_all.append(pred)
        chip_results.append((chip_ids.iloc[i], y_test, pred, y_test == pred))

    n_correct = sum(1 for _, actual, pred, ok in chip_results if ok)
    acc = n_correct / n
    report = classification_report(y_true_all, y_pred_all, labels=LABELS, zero_division=0)
    cm = confusion_matrix(y_true_all, y_pred_all, labels=LABELS)

    print(f"\nLOCO-CV accuracy: {acc:.3f} ({n_correct}/{n})")
    print(report)

    misses = [r for r in chip_results if not r[3]]

    with open(MODEL_DIR / "loco_cv_gatekeeper_results.txt", "w", encoding="utf-8") as f:
        f.write("Leave-One-Chip-Out Cross-Validation (LOCO-CV) -- Gatekeeper (Stage 0)\n")
        f.write("=" * 70 + "\n\n")
        f.write("Answers the reviewer-anticipated question: can the model generalize\n")
        f.write("to a brand-new real chip it has never seen, tested across EVERY real\n")
        f.write(f"chip this project has (n={n}), not just one 11-chip hold-out split.\n\n")
        f.write("Method: for each of the n real chips, train a fresh RandomForest on\n")
        f.write("the OTHER n-1 real chips only (no synthetic data in this experiment --\n")
        f.write("deliberately a different, complementary test to gatekeeper_model.py's\n")
        f.write("synthetic-trained/real-validated approach), then predict the held-out\n")
        f.write("chip. Same features (early_displacement_30s, is_divergent) and RF\n")
        f.write("config as the official gatekeeper, so results are directly comparable.\n\n")
        f.write(f"LOCO-CV accuracy: {acc:.3f} ({n_correct}/{n})\n\n")
        f.write("Full sklearn classification_report:\n")
        f.write(report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{cm}\n\n")
        if misses:
            f.write(f"Misclassified chips ({len(misses)}):\n")
            for chip_id, actual, pred, _ in misses:
                f.write(f"  {chip_id}: actual={actual}, predicted={pred}\n")
        else:
            f.write("No misclassifications -- every real chip correctly predicted when\n")
            f.write("trained on all other real chips.\n")
        f.write("\nHONEST CAVEAT: each fold trains on only 43 chips (very small for a\n")
        f.write("3-class problem), and this is real-to-real generalization only -- it\n")
        f.write("does not by itself validate the synthetic-data training pipeline the\n")
        f.write("official gatekeeper actually uses in production (pipeline_api.py).\n")
        f.write("It answers a different, narrower question than gatekeeper_metrics.txt:\n")
        f.write("not \"does synthetic-trained generalize to real chips\" but \"does the\n")
        f.write("early_displacement_30s + is_divergent feature pair separate these\n")
        f.write("classes at all, on real data alone, across every chip available.\"\n")

    print(f"\nwrote {MODEL_DIR / 'loco_cv_gatekeeper_results.txt'}")


if __name__ == "__main__":
    main()
