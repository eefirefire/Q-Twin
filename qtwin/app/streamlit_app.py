"""
Week 3 Task 5: Streamlit mockup (stretch goal, per the Week 3 plan --
only built once Tasks 1-4 were done).

Ties together all three Week 3 models -- the Stage 0 gatekeeper (Task 1),
the Stage 2b concentration regressor (Task 2), and the Stage 2a LSTM
prototype (Task 3) -- into one interactive view: pick a real chip, see
what each model predicts, and compare it against what the chip actually
was. This is a MOCKUP, not a production tool: it retrains the sklearn
models and reloads the LSTM from the exact same committed data/weights
Tasks 1-3 already validated (see pipeline_api.py), so nothing about the
"pipeline" story here is new science -- it's a way to look at those three
already-honest results side by side, chip by chip, instead of only as
aggregate accuracy numbers in three separate .txt files.

Run with: streamlit run qtwin/app/streamlit_app.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import pipeline_api as api  # noqa: E402

st.set_page_config(page_title="Q-Twin Week 3 Pipeline Mockup", layout="wide")


@st.cache_resource
def load_everything():
    gatekeeper = api.train_gatekeeper()
    regression = api.train_regression_models()
    lstm_model, lstm_config = api.load_lstm()
    chip_table = api.load_chip_table()
    return gatekeeper, regression, lstm_model, lstm_config, chip_table


st.title("Q-Twin: Week 3 model pipeline mockup")
st.caption(
    "Optional stretch deliverable (Task 5) -- runs the Task 1 gatekeeper, Task 2 regressors, "
    "and Task 3 LSTM against a real chip you pick, and shows all three predictions next to "
    "what the chip actually was. Not a new model or a new result on its own."
)

gatekeeper_clf, regression_models, lstm_model, lstm_config, chip_table = load_everything()

with st.sidebar:
    st.header("Pick a chip")
    show_holdout = st.checkbox(
        "Include hold-out chips",
        value=False,
        help="The 11 chips reserved in Week 2 Task 6 for later validation. Off by default so "
             "this mockup doesn't casually leak hold-out results by default.",
    )
    options_df = chip_table if show_holdout else chip_table[~chip_table.is_holdout]
    chip_id = st.selectbox("Chip ID", options_df["chip_id"].tolist())

    st.divider()
    st.caption(
        "All three models were trained/validated on synthetic + the 33 non-hold-out real "
        "chips in Weeks 2-3. See gatekeeper_metrics.txt, regression_metrics.txt, and "
        "lstm_metrics.txt in qtwin/models/ for the full honest-caveat writeups this mockup "
        "is summarizing, not replacing."
    )

result = api.predict_chip(chip_id, gatekeeper_clf, regression_models, lstm_model, lstm_config, chip_table)

if result["is_holdout"]:
    st.warning(
        f"**{chip_id}** is one of the 11 hold-out chips reserved in Week 2 Task 6. "
        "None of the models below were trained or tuned using this chip -- but they also "
        "haven't been formally evaluated against the hold-out set yet (that's saved for "
        "later per the plan), so treat this single-chip prediction as illustrative, not "
        "as a hold-out evaluation result."
    )

col_actual, col_gate, col_lstm, col_conc = st.columns(4)

with col_actual:
    st.metric("Actual outcome", result["actual_label"])
    conc = result["actual_concentration_uM"]
    st.caption(f"Concentration: {conc} uM" if conc not in (None, "") else "Concentration: NC (no target)")

with col_gate:
    match = result["gatekeeper_pred"] == result["actual_label"]
    st.metric("Gatekeeper (Task 1) predicts", result["gatekeeper_pred"], delta="match" if match else "MISS",
               delta_color="normal" if match else "inverse")
    st.caption("Random Forest on early_displacement_30s + divergence flag")

with col_lstm:
    if result["lstm_pred"] is not None:
        match = result["lstm_pred"] == result["actual_label"]
        st.metric("LSTM (Task 3) predicts", result["lstm_pred"], delta="match" if match else "MISS",
                   delta_color="normal" if match else "inverse")
        st.caption("Sequence model on the full 45s probe curve shape")
    else:
        st.metric("LSTM (Task 3) predicts", "n/a")
        st.caption("No usable probe-stage sequence for this chip")

with col_conc:
    pred_uM = result.get("probe_predicted_uM")
    if pred_uM is not None:
        st.metric("Regressor (Task 2) predicts", f"{pred_uM:.1f} uM")
        st.caption("Only valid if true concentration < 10 uM (Task 7 fix) -- not verifiable from the reading alone")
    else:
        st.metric("Regressor (Task 2) predicts", "n/a")

st.divider()

left, right = st.columns([2, 1])

with left:
    st.subheader("Probe-stage curve (baseline-relative to CHI endpoint)")
    if result["sequence"] is not None:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(result["sequence_grid_s"], result["sequence"], marker="o", markersize=3)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Time into probe stage (s)")
        ax.set_ylabel("Delta-f vs. CHI baseline (Hz)")
        ax.set_title(f"{chip_id}: 45s window fed to the LSTM")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.info("This chip's probe-stage curve was too short to resample into the 45s/60-point window.")

with right:
    st.subheader("Raw numbers")
    st.write(f"**early_displacement_30s:** {result['early_displacement_30s']:.2f} Hz"
             if result["early_displacement_30s"] == result["early_displacement_30s"] else "**early_displacement_30s:** n/a (divergent replicates)")
    st.write(f"**delta_f_probe:** {result['delta_f_probe']:.2f} Hz"
             if result["delta_f_probe"] == result["delta_f_probe"] else "**delta_f_probe:** n/a")
    if result.get("target_predicted_uM") is not None:
        st.write(f"**Target-stage regressor:** {result['target_predicted_uM']:.1f} uM")

st.divider()
st.subheader("Honest caveats (pulled from this week's own metrics files, not restated from memory)")
tab1, tab2, tab3 = st.tabs(["Gatekeeper (Task 1)", "Regression (Task 2)", "LSTM (Task 3)"])
models_dir = Path(__file__).resolve().parents[1] / "models"
with tab1:
    st.text((models_dir / "gatekeeper_metrics.txt").read_text(encoding="utf-8"))
with tab2:
    st.text((models_dir / "regression_metrics.txt").read_text(encoding="utf-8"))
with tab3:
    st.text((models_dir / "lstm_metrics.txt").read_text(encoding="utf-8"))
