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
from gatekeeper_model import (
    LABELS, build_features, build_label, EXTRA_WINDOW_COLS,
    real_multiwindow_displacement, load_training_data as load_gatekeeper_training,
)
from holdout import load_holdout_chip_ids
from model_trainer import SequenceLSTM, load_real_validation, load_synthetic, normalize
from regression_model import pick_degree
from sequence_architectures import TCN

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

# Target-stage regressor scope (2026-09-12) -- Hook Effect / surface
# crowding onset per Eva's Q3, NOT accuracy optimization. See
# train_regression_models()'s docstring for the full justification.
TARGET_SCOPE_LOW_UM = 0.5
TARGET_SCOPE_HIGH_UM = 5.0


def train_gatekeeper() -> RandomForestClassifier:
    X_train, y_train, _ = load_gatekeeper_training()
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    clf.fit(X_train, y_train)
    return clf


def train_regression_models():
    """Probe stage: Testing week Task 7's adopted fix (Option A), NOT the
    original Week 3/4 unscoped model. This was found missing during an
    independent review -- Task 7's regression_curve_shape_fix.py concluded
    "adopt Option A" (restrict the probe regressor to <10 uM, where the
    concentration-response curve is genuinely monotonic) but that decision
    was never wired in here, so the Streamlit mockup and
    holdout_validation.py were both silently still using the acknowledged-
    flawed unscoped model. probe_model below is now trained ONLY on
    synthetic rows with concentration_uM < 10.

    A first attempt at this fix also tried gating individual predictions by
    whether the real chip's delta_f_probe fell inside the <10 uM subset's
    observed input range, to report "out of scope" instead of silently
    extrapolating for chips that are actually >10 uM. That gate turned out
    to be a second bug, caught in the same review pass: per-chip noise is
    large enough that the <10 uM subset's delta_f range (-367 to +56 Hz)
    is almost as wide as the FULL, unrestricted range (-394 to +56 Hz) --
    30 of 33 real chips fall "inside" it regardless of their true
    concentration, so the gate would have rejected almost nothing. Root
    cause: this is a restatement of the exact same non-monotonicity/noise
    problem Option A exists to work around -- a low true concentration
    with a large noise excursion and a high true concentration can produce
    the same delta_f_probe reading, so the reading alone cannot certify
    which regime a chip is in. Removed the ineffective gate. Instead,
    probe_predicted_uM is always returned, but every caller MUST treat it
    as conditional on "assuming true concentration < 10 uM" -- an
    assumption this model cannot verify from the reading itself, only
    state as a caveat (see PROBE_SCOPE_CAVEAT below)."""
    probe = pd.read_csv(DATA_DIR / "probe_synthetic_batch.csv")
    probe_clean = probe[probe["class"] == "CLEAN_PCA3_TARGET"].dropna(subset=["true_endpoint_delta_f"])
    probe_below_peak = probe_clean[probe_clean["concentration_uM"] < 10]
    X_probe = probe_below_peak[["true_endpoint_delta_f"]].values
    y_probe = probe_below_peak["concentration_uM"].values
    probe_degree = pick_degree(X_probe, y_probe, max_degree=min(5, len(X_probe) - 1))
    probe_model = make_pipeline(PolynomialFeatures(probe_degree), LinearRegression())
    probe_model.fit(X_probe, y_probe)

    target = pd.read_csv(DATA_DIR / "target_synthetic_batch.csv")
    target_clean = target[target["class"] == "CLEAN_PCA3_TARGET_HYB"].dropna(subset=["true_endpoint_delta_f"])

    # Target stage: scoped to 0.5-5 uM (2026-09-12), per Eva's Q3 finding --
    # the target-hybridization curve stays clean and monotonic from 0.5-5 uM,
    # then enters a gradual surface-crowding onset that fully inverts by
    # 10 uM (the Hook Effect; TARGET_ONSET_CONC_UM=5.0,
    # TARGET_INVERT_CONC_UM=10.0 in constants.py already encode this in the
    # generator). This scoping is a Hook Effect / surface crowding onset
    # decision per Eva's Q3, NOT an accuracy-optimization choice -- it
    # exists because outside 0.5-5 uM the delta_f-to-concentration mapping
    # is not even a function (multiple concentrations map to the same
    # reading), the same structural problem Option A solved for the probe
    # stage, at a different, narrower boundary here.
    target_scoped = target_clean[
        (target_clean["concentration_uM"] >= TARGET_SCOPE_LOW_UM)
        & (target_clean["concentration_uM"] <= TARGET_SCOPE_HIGH_UM)
    ]
    X_target = target_scoped[["true_endpoint_delta_f"]].values
    y_target = target_scoped["concentration_uM"].values
    target_degree = pick_degree(X_target, y_target, max_degree=min(5, len(X_target) - 1))
    target_model = make_pipeline(PolynomialFeatures(target_degree), LinearRegression())
    target_model.fit(X_target, y_target)

    return {
        "probe_model": probe_model, "probe_degree": probe_degree,
        "target_model": target_model, "target_degree": target_degree,
    }


def load_lstm():
    """Loads whichever Stage 2a sequence model is currently promoted to
    official (lstm_probe_stage.pt / lstm_config.json). Architecture-aware
    since 2026-09-12's determinism-fix promotion (TCN) -- config["architecture"]
    picks the model class; older configs with no "architecture" key default
    to SequenceLSTM for backward compatibility with configs saved before
    this field existed."""
    with open(MODEL_DIR / "lstm_config.json", encoding="utf-8") as f:
        config = json.load(f)
    architecture = config.get("architecture", "LSTM")
    if architecture == "TCN":
        model = TCN(channels=config["channels"], n_layers=config["n_layers"])
    elif architecture == "LSTM":
        model = SequenceLSTM(hidden_size=config["hidden_size"])
    else:
        raise ValueError(f"unknown architecture {architecture!r} in lstm_config.json")
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
    row_df = pd.DataFrame([chip_row])
    extra = real_multiwindow_displacement([chip_row["chip_id"]])
    row_df = row_df.join(extra, on="chip_id")
    X = build_features(
        row_df, "early_displacement_30s", "displacement_replicate_status",
        extra_window_cols=EXTRA_WINDOW_COLS,
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


PROBE_SCOPE_CAVEAT = (
    "Valid only if the true concentration is below 10 uM (Testing week Task 7's "
    "Option A fix). This cannot be verified from the reading alone -- per-chip "
    "noise makes a low concentration with a large excursion and a high "
    "concentration produce statistically indistinguishable delta_f_probe values "
    "(see pipeline_api.py's train_regression_models() docstring for the "
    "measurement that shows this)."
)

TARGET_SCOPE_CAVEAT = (
    f"Valid only for {TARGET_SCOPE_LOW_UM}-{TARGET_SCOPE_HIGH_UM} uM (Hook Effect / "
    "surface crowding onset per Eva's Q3, not accuracy optimization -- the "
    "target-hybridization curve stays clean and monotonic in this range, then "
    "enters a gradual crowding onset that fully inverts by 10 uM)."
)

TARGET_OUT_OF_RANGE_MESSAGE = "outside validated range"


def _regression_predict(models, chip_row: pd.Series):
    out = {}
    if pd.notna(chip_row.get("delta_f_probe")):
        out["probe_predicted_uM"] = float(models["probe_model"].predict([[chip_row["delta_f_probe"]]])[0])
        out["probe_scope_caveat"] = PROBE_SCOPE_CAVEAT
    if pd.notna(chip_row.get("delta_f_target")):
        # Gated on the model's own OUTPUT prediction (the true concentration
        # isn't known at prediction time, so an input-range gate isn't
        # possible the way it might seem). HONEST CAVEAT, checked directly
        # rather than assumed (see target_regressor_hook_effect_scope.txt):
        # on the only real out-of-scope test available (3 real 10 uM chips),
        # this gate caught 0/3 -- the scoped training subset's own
        # delta_f_target range is noisy enough (-349 to +75 Hz) that a weak
        # linear fit compresses even wildly out-of-range inputs back into
        # the narrow [0.5, 5] uM trained output band. Same underlying
        # failure mode as the probe stage's abandoned input-range gate
        # (per-chip noise makes the scoped subset's range nearly as wide as
        # the unrestricted one). Kept anyway -- it is still the logically
        # correct check and does fire when it can -- but it is not a
        # reliable safeguard at this dataset's noise level; the 0.5-5 uM
        # SCOPE ITSELF (training data restriction) is the real fix.
        raw_pred = float(models["target_model"].predict([[chip_row["delta_f_target"]]])[0])
        if TARGET_SCOPE_LOW_UM <= raw_pred <= TARGET_SCOPE_HIGH_UM:
            out["target_predicted_uM"] = raw_pred
        else:
            out["target_predicted_uM"] = None
            out["target_out_of_range"] = True
            out["target_raw_prediction_uM"] = raw_pred
        out["target_scope_caveat"] = TARGET_SCOPE_CAVEAT
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
