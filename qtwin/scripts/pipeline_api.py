"""
Week 3 Task 5: shared prediction API for the Streamlit mockup.

Wraps the three models already built and validated this week (Task 1
gatekeeper, Task 2 regression, Task 3 LSTM) behind one `predict_chip()`
call so the mockup can run all three against a real chip and show
predicted-vs-actual side by side, without duplicating any of the
training/feature logic those scripts already got right.

Nothing here changes what those scripts do when run standalone -- this
only imports their functions/classes and adds thin re-training +
inference wrappers for interactive use. Retraining sklearn/torch models
at app startup (instead of pickling once) is intentional: every model
here trains in well under a second on this dataset size, and re-fitting
from the same committed CSVs on every app launch means the mockup can
never silently drift from what `python gatekeeper_model.py` /
`regression_model.py` / `model_trainer.py` actually produce.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

import curve_generator as cg
from gatekeeper_model import LABELS, build_features, build_label, load_training_data as load_gatekeeper_training
from holdout import load_holdout_chip_ids
from model_trainer import SequenceLSTM, load_real_validation, load_synthetic, normalize
from regression_model import pick_degree

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def train_gatekeeper() -> RandomForestClassifier:
    X_train, y_train, _ = load_gatekeeper_training()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def train_regression_models():
    """Same fit as regression_model.py's train_and_report(), minus the plot."""
    probe = pd.read_csv(DATA_DIR / "probe_synthetic_batch.csv")
    probe_clean = probe[probe["class"] == "CLEAN_PCA3_TARGET"].dropna(subset=["true_endpoint_delta_f"])
    X_probe = probe_clean[["true_endpoint_delta_f"]].values
    y_probe = probe_clean["concentration_uM"].values
    probe_degree = pick_degree(X_probe, y_probe)
    probe_model = make_pipeline(PolynomialFeatures(probe_degree), LinearRegression())
    probe_model.fit(X_probe, y_probe)

    target = pd.read_csv(DATA_DIR / "target_synthetic_batch.csv")
    target_clean = target[target["class"] == "CLEAN_PCA3_TARGET_HYB"].dropna(subset=["true_endpoint_delta_f"])
    X_target = target_clean[["true_endpoint_delta_f"]].values
    y_target = target_clean["concentration_uM"].values
    target_degree = pick_degree(X_target, y_target)
    target_model = make_pipeline(PolynomialFeatures(target_degree), LinearRegression())
    target_model.fit(X_target, y_target)

    return {
        "probe_model": probe_model, "probe_degree": probe_degree,
        "target_model": target_model, "target_degree": target_degree,
    }


def load_lstm():
    with open(MODEL_DIR / "lstm_config.json", encoding="utf-8") as f:
        config = json.load(f)
    model = SequenceLSTM(hidden_size=config["hidden_size"])
    model.load_state_dict(torch.load(MODEL_DIR / "lstm_probe_stage.pt", map_location="cpu"))
    model.eval()
    return model, config


def load_chip_table() -> pd.DataFrame:
    """All 44 valid (non-EXCLUDED) real chips, tagged with hold-out status
    and whether the LSTM has a usable probe-stage sequence for them."""
    cs = pd.read_csv(DATA_DIR / "chip_summary.csv")
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    holdout_ids = set(load_holdout_chip_ids())
    valid["is_holdout"] = valid["chip_id"].isin(holdout_ids)

    seq_npz = np.load(DATA_DIR / "real_probe_sequences.npz", allow_pickle=True)
    seq_chip_ids = set(seq_npz["chip_id"].tolist())
    valid["has_sequence"] = valid["chip_id"].isin(seq_chip_ids)
    return valid.sort_values("chip_id").reset_index(drop=True)


def _gatekeeper_predict(clf, chip_row: pd.Series) -> str:
    X = build_features(
        pd.DataFrame([chip_row]), "early_displacement_30s", "displacement_replicate_status"
    )
    return clf.predict(X)[0]


def _lstm_predict(model, config, chip_id: str):
    npz = np.load(DATA_DIR / "real_probe_sequences.npz", allow_pickle=True)
    idx = list(npz["chip_id"]).index(chip_id)
    raw_seq = npz["sequences"][idx]
    seq_norm = normalize(raw_seq, config["seq_mean"], config["seq_std"])
    X = torch.tensor(seq_norm, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        pred_idx = model(X).argmax(dim=1).item()
    return LABELS[pred_idx], raw_seq


def _regression_predict(models, chip_row: pd.Series):
    out = {}
    if pd.notna(chip_row.get("delta_f_probe")):
        out["probe_predicted_uM"] = float(models["probe_model"].predict([[chip_row["delta_f_probe"]]])[0])
    if pd.notna(chip_row.get("delta_f_target")):
        out["target_predicted_uM"] = float(models["target_model"].predict([[chip_row["delta_f_target"]]])[0])
    return out


def predict_chip(chip_id: str, gatekeeper_clf, regression_models, lstm_model, lstm_config, chip_table: pd.DataFrame) -> dict:
    row = chip_table.set_index("chip_id").loc[chip_id]
    actual_label = build_label(row["success_or_fail"], row["displacement_replicate_status"])

    result = {
        "chip_id": chip_id,
        "is_holdout": bool(row["is_holdout"]),
        "actual_label": actual_label,
        "actual_concentration_uM": row.get("concentration_uM"),
        "early_displacement_30s": row.get("early_displacement_30s"),
        "delta_f_probe": row.get("delta_f_probe"),
        "delta_f_target": row.get("delta_f_target"),
    }
    result["gatekeeper_pred"] = _gatekeeper_predict(gatekeeper_clf, row)
    result.update(_regression_predict(regression_models, row))
    if row["has_sequence"]:
        lstm_pred, raw_seq = _lstm_predict(lstm_model, lstm_config, chip_id)
        result["lstm_pred"] = lstm_pred
        result["sequence"] = raw_seq
        result["sequence_grid_s"] = np.linspace(0.0, cg.SEQUENCE_WINDOW_S, cg.SEQUENCE_N_POINTS)
    else:
        result["lstm_pred"] = None
        result["sequence"] = None
    return result
