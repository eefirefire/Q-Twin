"""
Testing week Task 1: the real blind hold-out validation.

The 11 hold-out chips (holdout_chips.txt) have been reserved since Week 2
Task 6, explicitly excluded from every training run AND every validation
run reported so far (Tasks 1-3, Week 3-4's gatekeeper/regression/LSTM
metrics, the boundary stress test, the Task 4 comparison). Every accuracy
number reported until now was computed against the 33 non-hold-out chips
-- chips the models' own design choices (biomarker window, threshold
tuning, architecture selection) could still have been indirectly shaped
around during development. This is the first run against chips no design
decision ever saw, in any form.

Reuses the exact trained models from pipeline_api.py (same training code
as gatekeeper_model.py/regression_model.py/model_trainer.py) -- this is
not a new model, just the first time these particular 11 chips are used
for evaluation instead of being structurally excluded.

Output: qtwin/models/holdout_validation_results.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pipeline_api as api
from gatekeeper_model import LABELS, build_label

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    gatekeeper_clf = api.train_gatekeeper()
    regression_models = api.train_regression_models()
    lstm_model, lstm_config = api.load_lstm()
    chip_table = api.load_chip_table()

    holdout = chip_table[chip_table.is_holdout].copy()
    n_holdout = len(holdout)
    print(f"Hold-out set: {n_holdout} chips: {sorted(holdout.chip_id.tolist())}")

    # --- Gatekeeper + LSTM: 3-class SUCCESS/FAILURE/DIVERGENT_REPLICATES ---
    gk_actual, gk_pred = [], []
    lstm_actual, lstm_pred = [], []
    # Task 7's Option A model is only VALIDATED for true concentration <10 uM
    # (its own docstring explains it can't detect this from the reading alone,
    # since noise makes low/high-concentration delta_f values overlap). We DO
    # know ground truth here, unlike at real inference time, so evaluate it the
    # honest way the original Task 7 script did: MAE on the in-scope (<10 uM)
    # subset separately from what happens if the model is naively applied to
    # every hold-out chip regardless of true concentration.
    probe_actual_conc_inscope, probe_pred_conc_inscope = [], []
    probe_actual_conc_all, probe_pred_conc_all = [], []
    target_actual_conc, target_pred_conc = [], []
    per_chip_rows = []

    for _, row in holdout.iterrows():
        result = api.predict_chip(row.chip_id, gatekeeper_clf, regression_models,
                                   lstm_model, lstm_config, chip_table)
        actual_label = result["actual_label"]

        gk_actual.append(actual_label)
        gk_pred.append(result["gatekeeper_pred"])

        if result["lstm_pred"] is not None:
            lstm_actual.append(actual_label)
            lstm_pred.append(result["lstm_pred"])

        conc = row.get("concentration_uM")
        conc_num = pd.to_numeric(conc, errors="coerce")
        if pd.notna(conc_num) and conc_num > 0 and result.get("probe_predicted_uM") is not None:
            probe_actual_conc_all.append(conc_num)
            probe_pred_conc_all.append(result["probe_predicted_uM"])
            if conc_num < 10:
                probe_actual_conc_inscope.append(conc_num)
                probe_pred_conc_inscope.append(result["probe_predicted_uM"])
        if pd.notna(conc_num) and conc_num > 0 and result.get("target_predicted_uM") is not None:
            target_actual_conc.append(conc_num)
            target_pred_conc.append(result["target_predicted_uM"])

        per_chip_rows.append({
            "chip_id": row.chip_id, "actual_label": actual_label,
            "gatekeeper_pred": result["gatekeeper_pred"],
            "gatekeeper_correct": result["gatekeeper_pred"] == actual_label,
            "lstm_pred": result["lstm_pred"],
            "lstm_correct": (result["lstm_pred"] == actual_label) if result["lstm_pred"] else None,
            "probe_pred_uM": result.get("probe_predicted_uM"),
            "actual_conc_uM": conc,
        })

    per_chip_df = pd.DataFrame(per_chip_rows)

    gk_acc = accuracy_score(gk_actual, gk_pred)
    gk_report = classification_report(gk_actual, gk_pred, labels=LABELS, zero_division=0)
    gk_cm = confusion_matrix(gk_actual, gk_pred, labels=LABELS)

    lstm_acc = accuracy_score(lstm_actual, lstm_pred) if lstm_actual else float("nan")
    lstm_report = classification_report(lstm_actual, lstm_pred, labels=LABELS, zero_division=0) if lstm_actual else "n/a"
    lstm_cm = confusion_matrix(lstm_actual, lstm_pred, labels=LABELS) if lstm_actual else None

    probe_mae_inscope = mean_absolute_error(probe_actual_conc_inscope, probe_pred_conc_inscope) if probe_actual_conc_inscope else float("nan")
    probe_mae_all = mean_absolute_error(probe_actual_conc_all, probe_pred_conc_all) if probe_actual_conc_all else float("nan")
    target_mae = mean_absolute_error(target_actual_conc, target_pred_conc) if target_actual_conc else float("nan")

    print(f"\nGatekeeper hold-out accuracy: {gk_acc:.3f} (n={n_holdout})")
    print(gk_report)
    print(f"\nLSTM hold-out accuracy: {lstm_acc:.3f} (n={len(lstm_actual)})")
    print(lstm_report)
    print(f"\nProbe regressor hold-out MAE, in-scope (<10 uM) only: {probe_mae_inscope:.2f} uM (n={len(probe_actual_conc_inscope)})")
    print(f"Probe regressor hold-out MAE, naively applied to ALL chips: {probe_mae_all:.2f} uM (n={len(probe_actual_conc_all)})")
    print(f"Target regressor hold-out MAE: {target_mae:.2f} uM (n={len(target_actual_conc)})")

    # Confusion matrix plots
    fig, ax = plt.subplots(figsize=(7, 5.8))
    ConfusionMatrixDisplay(gk_cm, display_labels=LABELS).plot(ax=ax, cmap="Reds", colorbar=False)
    ax.set_title(f"Gatekeeper -- BLIND hold-out (n={n_holdout})", fontsize=11)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(MODEL_DIR / "holdout_gatekeeper_confusion_matrix.png", dpi=130)
    plt.close(fig)

    if lstm_cm is not None:
        fig, ax = plt.subplots(figsize=(7, 5.8))
        ConfusionMatrixDisplay(lstm_cm, display_labels=LABELS).plot(ax=ax, cmap="Reds", colorbar=False)
        ax.set_title(f"LSTM -- BLIND hold-out (n={len(lstm_actual)})", fontsize=11)
        plt.xticks(rotation=20, ha="right")
        fig.tight_layout()
        fig.savefig(MODEL_DIR / "holdout_lstm_confusion_matrix.png", dpi=130)
        plt.close(fig)

    # Pull the already-reported 33-chip numbers for direct side-by-side comparison.
    # probe_mae is intentionally NOT included here: that 7.09 uM figure was the
    # OLD unscoped model's number. Since pipeline_api.py now uses the Testing
    # week Task 7 fix (Option A, <10 uM scoped), there is no directly comparable
    # 33-chip number for the corrected model yet -- reporting the stale one next
    # to this run's number would misleadingly imply an apples-to-apples
    # comparison. Caught during an independent review; not silently carried over.
    known_33chip = {
        "gatekeeper_acc": 1.000,
        "lstm_acc": 0.758,
        "target_mae": 5.35,
    }

    with open(MODEL_DIR / "holdout_validation_results.txt", "w", encoding="utf-8") as f:
        f.write("Testing week Task 1 -- BLIND hold-out validation results\n")
        f.write("=" * 60 + "\n\n")
        f.write("The first genuinely blind numbers in this project: the 11 chips below\n")
        f.write("were reserved in Week 2 Task 6 and have never been touched by any\n")
        f.write("training run, threshold decision, hyperparameter sweep, or architecture\n")
        f.write("choice -- unlike the 33-chip validation set, which every design decision\n")
        f.write("in Weeks 3-4 was developed and iterated against, even if never directly\n")
        f.write("trained on.\n\n")
        f.write(f"Hold-out chips (n={n_holdout}): {sorted(holdout.chip_id.tolist())}\n\n")

        f.write("-" * 60 + "\n")
        f.write("GATEKEEPER (Task 1)\n")
        f.write(f"Hold-out accuracy: {gk_acc:.3f}  (33-chip validation accuracy: {known_33chip['gatekeeper_acc']:.3f})\n")
        f.write(gk_report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{gk_cm}\n\n")

        f.write("-" * 60 + "\n")
        f.write("LSTM (Task 3, official model)\n")
        f.write(f"Hold-out accuracy: {lstm_acc:.3f}  (33-chip validation accuracy: {known_33chip['lstm_acc']:.3f})\n")
        f.write(lstm_report)
        if lstm_cm is not None:
            f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n{lstm_cm}\n")
        f.write("\n")

        f.write("-" * 60 + "\n")
        f.write("REGRESSION (Task 2, probe stage using the Task 7 Option A fix: trained on <10 uM only)\n")
        f.write(f"In-scope MAE (evaluated only on hold-out chips whose TRUE concentration is <10 uM,\n")
        f.write(f"the honest way to evaluate a model that can't detect its own scope from the reading\n")
        f.write(f"alone): {probe_mae_inscope:.2f} uM (n={len(probe_actual_conc_inscope)})\n")
        f.write(f"Naive MAE (same model applied to ALL {len(probe_actual_conc_all)} hold-out chips regardless of true\n")
        f.write(f"concentration, i.e. what happens if the scope caveat is ignored): {probe_mae_all:.2f} uM\n")
        f.write("No directly comparable 33-chip number exists for this scoped model -- the old\n")
        f.write("7.09 uM figure was the unscoped model's, not an apples-to-apples baseline.\n")
        f.write(f"Target-stage hold-out MAE: {target_mae:.2f} uM (n={len(target_actual_conc)})")
        f.write(f"  (33-chip validation MAE: {known_33chip['target_mae']:.2f} uM)\n\n")

        f.write("-" * 60 + "\n")
        f.write("Per-chip detail:\n")
        f.write(per_chip_df.to_string(index=False))
        f.write("\n\n")

        gap_gk = known_33chip["gatekeeper_acc"] - gk_acc
        gap_lstm = known_33chip["lstm_acc"] - lstm_acc if lstm_actual else float("nan")
        f.write("-" * 60 + "\n")
        f.write("HONEST ASSESSMENT:\n")
        f.write(f"Gatekeeper: {'DROPS' if gap_gk > 0.02 else 'HOLDS'} on genuinely blind data ")
        f.write(f"({known_33chip['gatekeeper_acc']:.3f} -> {gk_acc:.3f}, gap={gap_gk:+.3f}).\n")
        if lstm_actual:
            f.write(f"LSTM: {'DROPS' if gap_lstm > 0.02 else 'HOLDS'} on genuinely blind data ")
            f.write(f"({known_33chip['lstm_acc']:.3f} -> {lstm_acc:.3f}, gap={gap_lstm:+.3f}).\n")
        f.write(f"n=11 is a small hold-out set -- treat any single-chip miss as a large swing\n")
        f.write(f"in the reported percentage (each chip is ~9% of the total), not as strong\n")
        f.write(f"statistical evidence on its own. Reported as-is regardless of which\n")
        f.write(f"direction it points, per this project's standing rule of reporting gaps\n")
        f.write(f"plainly rather than only when they're flattering.\n")

    print(f"\nwrote {MODEL_DIR / 'holdout_validation_results.txt'}")


if __name__ == "__main__":
    main()
