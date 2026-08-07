# Q-Twin

Biosensor (QCM/PCA3 assay) data pipeline: real-data ingestion, a validated
synthetic-data generator, and three trained models (Stage 0 gatekeeper,
Stage 2a sequence model, Stage 2b concentration regressor), plus a
Streamlit review mockup. Current through the Testing/Tuning/Grooming week
(inserted between Week 4 and Week 5).

## Setup

```bash
python -m venv qtwin/.venv
source qtwin/.venv/Scripts/activate   # Windows Git Bash
pip install -r qtwin/requirements.txt
```

If pip hits `CERTIFICATE_VERIFY_FAILED` inside the venv, pass the system
certifi bundle explicitly: `pip install --cert "$(python -c 'import certifi;print(certifi.where())')" -r qtwin/requirements.txt`

## Folder structure

- `data/raw/` — the 150 raw CSVs across 4 lab sessions (14/15/20/21 Mar 2026), untouched.
- `data/raw_timeseries_master.csv`, `data/chip_summary.csv`, `data/chip_index.csv`,
  `data/cross_check_report.csv` — real-data pipeline output (see `ingest_raw_curves.py`).
- `data/holdout_chips.txt` — 11 chips reserved as a blind hold-out set (Week 2 Task 6).
- `data/probe_synthetic_batch.csv`, `data/target_synthetic_batch.csv`,
  `data/synthetic_batch_v1.csv`, `data/*_synthetic_sequences.npz`,
  `data/real_probe_sequences.npz` — synthetic training data and resampled
  curve sequences (probe/target stages).
- `data/ks_validation_report.txt` — required K-S tests gating synthetic data before training.
- `models/` — trained model outputs: gatekeeper, regression, LSTM/TCN/attention
  comparisons, boundary stress test, ablation/benchmark/hold-out results —
  every `*_metrics.txt`/`*_results.txt` file has an HONEST CAVEAT section.
- `app/` — `streamlit_app.py` + `pipeline_api.py`, an interactive mockup
  running all three models against any selected real chip.
- `docs/` — planning docs, `clarifying_questions.md` (running log of every
  data/methodology question and its resolution), `Known_Limitations_Master.md`
  (consolidated honest-caveat reference for the proposal), and the
  guardrails doc source (`Biochemical_Guardrails.docx`/`.pdf`).
- `scripts/` — see Pipeline order below.

## Pipeline order

```bash
cd qtwin/scripts

# 1. Real-data ingestion
python ingest_raw_curves.py      # -> raw_timeseries_master.csv, chip_summary.csv
python build_chip_index.py       # -> chip_index.csv
python cross_check_index.py      # -> cross_check_report.csv
python make_sanity_plots.py      # -> figures/
python reserve_holdout.py        # -> holdout_chips.txt (already reserved; re-running is destructive, don't)

# 2. Synthetic data generation + validation
python generate_probe_batch.py
python generate_target_batch.py
python assemble_batch.py         # -> synthetic_batch_v1.csv, ks_validation_report.txt (must pass)
python build_real_sequences.py   # -> real_probe_sequences.npz

# 3. Models (each trains on synthetic data, validates on the 33 real
#    non-hold-out chips -- the 11 hold-out chips are never touched here)
python gatekeeper_model.py       # Stage 0: RandomForestClassifier
python regression_model.py       # Stage 2b: polynomial regression
python model_trainer.py          # Stage 2a: LSTM (official)
python lstm_tcn_comparison.py    # Stage 2a: LSTM+Attention / TCN comparison
python gatekeeper_boundary_stress_test.py

# 4. Testing week: real blind hold-out + ablation/benchmarking
python holdout_validation.py     # first genuinely blind validation (n=11)
python ablation_study.py
python benchmark_comparison.py
python gatekeeper_logistic_baseline.py
python attention_weight_visualization.py
python regression_curve_shape_fix.py
python hyperparameter_sweep.py

# 5. Interactive mockup
streamlit run ../app/streamlit_app.py
```

## Key results (see qtwin/docs/Known_Limitations_Master.md for full caveats)

- Stage 0 gatekeeper: 100% on 33-chip validation, 100% on the blind hold-out
  set, drops to 91.7% near the -0.5 Hz decision boundary specifically.
- Stage 2a LSTM: 75.8% on 33-chip validation, **45.5% on the blind hold-out
  set** (n=11) -- the most important honest finding from the testing week.
  LSTM+Attention/TCN reach 87.9% on 33-chip validation (not yet promoted to
  the official model).
- Stage 2b regressor: probe-stage is structurally ill-posed above 10 µM
  (non-monotonic curve); restricted to <10 µM for a scoped, honest model
  (MAE 2.34 µM) rather than an unscoped one that's silently wrong half the time.

## Open questions for the teacher / Eva

Running log: `docs/clarifying_questions.md`. Consolidated limitations for
the Week 5 proposal: `docs/Known_Limitations_Master.md`.
