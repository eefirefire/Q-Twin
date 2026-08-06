# Q-Twin Week 3 AI pipeline architecture

Per the Week 3 action-items checklist ("ส่งมอบผลลัพธ์โมเดล... และแผนภาพสถาปัตยกรรม AI").
Shows how raw lab data flows through to the three Week 3 models and the
Streamlit mockup. GitHub renders this diagram inline.

```mermaid
flowchart TD
    RAW["Raw QCM chip CSVs\n(qtwin/data/raw/)"] --> INGEST["ingest_raw_curves.py"]
    INGEST --> CHIPSUM["chip_summary.csv\n(44 valid chips: delta_f, early_displacement_30s,\nreplicate concordance status)"]
    INGEST --> RAWMASTER["raw_timeseries_master.csv\n(full real curves)"]

    CHIPSUM --> HOLDOUT["reserve_holdout.py"]
    HOLDOUT --> HOLDOUTFILE["holdout_chips.txt\n(11 chips, reserved, never trained on)"]

    CHIPSUM --> GEN["curve_generator.py\n(shared generation library:\nLangmuir kinetics + OU drift +\nper-replicate kinetic jitter)"]
    GEN --> PROBEBATCH["generate_probe_batch.py"]
    GEN --> TARGETBATCH["generate_target_batch.py"]
    PROBEBATCH --> PROBECSV["probe_synthetic_batch.csv\n+ probe_synthetic_sequences.npz"]
    TARGETBATCH --> TARGETCSV["target_synthetic_batch.csv\n+ target_synthetic_sequences.npz"]
    PROBECSV --> ASSEMBLE["assemble_batch.py\n(K-S validation gate)"]
    TARGETCSV --> ASSEMBLE
    ASSEMBLE --> SYNTHFULL["synthetic_batch_v1.csv"]

    RAWMASTER --> REALSEQ["build_real_sequences.py"]
    REALSEQ --> REALSEQNPZ["real_probe_sequences.npz"]

    PROBECSV --> GATEKEEPER["Task 1: Stage 0 Gatekeeper\ngatekeeper_model.py\nRandomForestClassifier"]
    CHIPSUM --> GATEKEEPER
    HOLDOUTFILE -. excluded from validation .-> GATEKEEPER

    PROBECSV --> REGRESSION["Task 2: Stage 2b Regressor\nregression_model.py\nPolynomialFeatures + LinearRegression"]
    TARGETCSV --> REGRESSION
    CHIPSUM --> REGRESSION
    HOLDOUTFILE -. excluded from validation .-> REGRESSION

    PROBECSV --> LSTM["Task 3: Stage 2a Sequence Model\nmodel_trainer.py\nLSTM (class-weighted, hidden_size/epochs tuned)"]
    REALSEQNPZ --> LSTM
    HOLDOUTFILE -. excluded from validation .-> LSTM

    GATEKEEPER --> APP["Task 5: Streamlit mockup\napp/streamlit_app.py + pipeline_api.py"]
    REGRESSION --> APP
    LSTM --> APP
    CHIPSUM --> APP

    GATEKEEPER --> STRESS["Boundary stress test\ngatekeeper_boundary_stress_test.py\n(same trained classifier, synthetic\nchips near the -0.5 Hz threshold)"]

    subgraph OUTPUTS["Reported outputs (qtwin/models/)"]
        GKMETRICS["gatekeeper_metrics.txt +\nconfusion matrix"]
        REGMETRICS["regression_metrics.txt +\nprobe/target plots"]
        LSTMMETRICS["lstm_metrics.txt +\nconfusion matrix"]
        STRESSMETRICS["gatekeeper_boundary_stress_test.txt +\nconfusion matrix"]
    end
    GATEKEEPER --> GKMETRICS
    REGRESSION --> REGMETRICS
    LSTM --> LSTMMETRICS
    STRESS --> STRESSMETRICS
```

## Notes on what this diagram is (and isn't) claiming

- Every arrow is an actual file dependency in the repo, not a conceptual
  simplification -- trace any box back to the script/file named on it.
- The hold-out set (dashed lines) is structurally excluded from every
  model's training AND validation this week, per Week 2 Task 6's plan --
  it's reserved for a later, not-yet-run evaluation pass.
- `curve_generator.py`'s per-replicate kinetic jitter (feeding both
  synthetic batches) is the Week 3 Task 4 fix -- see
  `Task4_Replicate_Divergence_Investigation.md` for why it exists and how
  it was calibrated.
