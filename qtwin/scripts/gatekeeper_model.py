"""
Week 3 Task 1: Stage 0 Chip Quality Gatekeeper.

Trains a Random Forest on Week 2's synthetic probe-stage data
(probe_synthetic_batch.csv) to predict SUCCESS / FAILURE /
DIVERGENT_REPLICATES from early_displacement_30s (primary feature) plus
replicate-concordance status (secondary feature). Validates against the
33 non-hold-out real chips (44 valid - 11 reserved in Week 2 Task 6 --
explicitly NOT touched here, per this week's plan: "save that for later").

Design note worth being explicit about, UPDATED after the Week 3 Task 4
generator fix: early_displacement_30s used to be NULL exactly when
concordance status was DIVERGENT_REPLICATES for synthetic data. That's no
longer true -- curve_generator.check_replicate_concordance() now always
returns the averaged value (matching real data's own convention, which
never nulled it either; see that function's docstring for why). So
early_displacement_30s is populated for every row now, real or synthetic,
divergent or not. "is_divergent" is still a real, separate feature (not
redundant with the averaged value), but it's no longer the ONLY thing
identifying that class the way it was when the value was blanked out --
worth re-reading feature_importances_ below after this change rather than
assuming the old reasoning still holds.

Output: gatekeeper_model.py (this file), qtwin/models/gatekeeper_confusion_matrix.png,
qtwin/models/gatekeeper_metrics.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from holdout import load_holdout_chip_ids

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"
MODEL_DIR.mkdir(exist_ok=True)

LABELS = ["SUCCESS", "FAILURE", "DIVERGENT_REPLICATES"]


def build_label(success_or_fail: str, concordance_status: str) -> str:
    if concordance_status == "DIVERGENT_REPLICATES":
        return "DIVERGENT_REPLICATES"
    return success_or_fail


def build_features(df: pd.DataFrame, displacement_col: str, status_col: str) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    # Primary feature: early_displacement_30s (see the module docstring's
    # UPDATED note -- no longer NaN on divergence for either real or
    # synthetic data, so fillna here is now just a defensive no-op, not
    # load-bearing imputation).
    feat["early_displacement_30s"] = df[displacement_col].fillna(0.0)
    feat["is_divergent"] = (df[status_col] == "DIVERGENT_REPLICATES").astype(int)
    return feat


def load_training_data():
    probe = pd.read_csv(DATA_DIR / "probe_synthetic_batch.csv")
    y = probe.apply(
        lambda r: build_label(r["success_or_fail"], r["biomarker_replicate_status"]), axis=1
    )
    X = build_features(probe, "early_displacement_30s", "biomarker_replicate_status")
    meta = probe[["synthetic_id", "class", "intentionally_divergent"]].copy()
    return X, y, meta


def load_validation_data():
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    holdout_ids = load_holdout_chip_ids()
    valid = valid[~valid["chip_id"].isin(holdout_ids)]  # explicitly NOT the hold-out set
    y = valid.apply(
        lambda r: build_label(r["success_or_fail"], r["displacement_replicate_status"]), axis=1
    )
    X = build_features(valid, "early_displacement_30s", "displacement_replicate_status")
    meta = valid[["chip_id"]].copy()
    return X, y, meta, len(holdout_ids)


def main():
    X_train, y_train, train_meta = load_training_data()
    X_val, y_val, val_meta, n_holdout_excluded = load_validation_data()

    print(f"Training on {len(X_train)} synthetic probe-stage curves.")
    print(f"Validating on {len(X_val)} real chips ({n_holdout_excluded} hold-out chips excluded, saved for later).")
    print()
    print("Training label distribution:\n", y_train.value_counts())
    print()
    print("Validation label distribution:\n", y_val.value_counts())

    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)

    acc = accuracy_score(y_val, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_val, y_pred, labels=LABELS, zero_division=0
    )
    report = classification_report(y_val, y_pred, labels=LABELS, zero_division=0)

    print(f"\nOverall accuracy on real (non-hold-out) validation set: {acc:.3f}")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_val, y_pred, labels=LABELS)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Gatekeeper confusion matrix\n(real, non-hold-out validation)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    cm_path = MODEL_DIR / "gatekeeper_confusion_matrix.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"wrote {cm_path}")

    # Task 1 item 3: specifically check how well it catches the
    # intentionally-corrupted Week 2 test examples (on the TRAINING data,
    # since those examples only exist in the synthetic batch -- this is a
    # sanity/self-consistency check, not held-out validation).
    forced = train_meta[train_meta["intentionally_divergent"]]
    forced_X = X_train.loc[forced.index]
    forced_pred = clf.predict(forced_X)
    forced_caught = (forced_pred == "DIVERGENT_REPLICATES").sum()
    print(f"\nIntentionally-corrupted Week 2 examples: {forced_caught}/{len(forced)} correctly flagged DIVERGENT_REPLICATES")

    # Honest-reporting check (Task 1 item 4 says "doesn't need to be perfect,
    # needs to be real and reported honestly" -- a perfect score needs a
    # caveat, not just a headline number): how close are the non-divergent
    # validation chips to the -0.5 Hz decision boundary?
    val_full = pd.concat([X_val, y_val.rename("label")], axis=1)
    non_div = val_full[val_full["is_divergent"] == 0]
    near_boundary = non_div[non_div["early_displacement_30s"].between(-2.0, 0.5)]

    with open(MODEL_DIR / "gatekeeper_metrics.txt", "w", encoding="utf-8") as f:
        f.write("Week 3 Task 1 -- Stage 0 Chip Quality Gatekeeper\n")
        f.write("=" * 60 + "\n\n")
        f.write("HONEST CAVEAT on the 100% accuracy below: this isn't a hard\n")
        f.write("classification problem and the perfect score shouldn't be read as\n")
        f.write("evidence of a sophisticated model. Two reasons:\n")
        f.write("  1. DIVERGENT_REPLICATES is constructed directly from the same\n")
        f.write("     concordance status used as the 'is_divergent' feature -- that\n")
        f.write("     class is caught by definition, not learned.\n")
        f.write(f"  2. early_displacement_30s correlates 0.9997 with delta_f_probe\n")
        f.write("     (the quantity that literally defines SUCCESS/FAILURE via the\n")
        f.write("     -0.5 Hz threshold, see Week 1). Only "
                f"{len(near_boundary)} of {len(non_div)} non-divergent\n")
        f.write("     validation chips sit anywhere near that boundary (within 2 Hz),\n")
        f.write("     so a threshold-like classifier separating them cleanly is the\n")
        f.write("     EXPECTED outcome given how strong that biomarker already was,\n")
        f.write("     not a surprising result. Real-world performance on chips closer\n")
        f.write("     to the boundary is genuinely untested here.\n\n")
        f.write(f"Trained on {len(X_train)} synthetic probe-stage curves (probe_synthetic_batch.csv).\n")
        f.write(f"Validated on {len(X_val)} real chips ({n_holdout_excluded} hold-out chips excluded, saved per Task 6).\n\n")
        f.write(f"Overall accuracy: {acc:.3f}\n\n")
        f.write("Per-class precision/recall/F1 (real validation set):\n")
        for label, p, r, f1_, sup in zip(LABELS, precision, recall, f1, support):
            f.write(f"  {label:22s} precision={p:.3f} recall={r:.3f} f1={f1_:.3f} support={sup}\n")
        f.write(f"\nIntentionally-corrupted Week 2 examples caught: {forced_caught}/{len(forced)}\n")
        f.write("\nFull sklearn classification_report:\n")
        f.write(report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n")
        f.write(str(cm))
        f.write("\n")

    print(f"wrote {MODEL_DIR / 'gatekeeper_metrics.txt'}")

    # Feature importance -- honest reporting requires showing whether the
    # model actually leans on the biomarker or just the divergence flag.
    importances = dict(zip(X_train.columns, clf.feature_importances_))
    print(f"\nFeature importances: {importances}")


if __name__ == "__main__":
    main()
