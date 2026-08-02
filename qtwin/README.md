# Q-Twin — Week 1

## Setup

```bash
python -m venv qtwin/.venv
source qtwin/.venv/Scripts/activate   # Windows Git Bash
pip install -r qtwin/requirements.txt
```

If pip hits `CERTIFICATE_VERIFY_FAILED` inside the venv, pass the system
certifi bundle explicitly: `pip install --cert "$(python -c 'import certifi;print(certifi.where())')" -r qtwin/requirements.txt`
(only needed once per venv, doesn't affect anything else on the machine).

## Folder structure

- `data/raw/` — the 150 raw CSVs across 4 lab sessions (14/15/20/21 Mar 2026), untouched.
- `data/raw_timeseries_master.csv` — every raw (chip, stage, replicate, timestamp, frequency) row, one table.
- `data/chip_summary.csv` — one row per chip: Δf per stage, SUCCESS/FAILURE, kinetic biomarker.
- `data/chip_index.csv` — Eva's chip index (lab-context view): condition labels, concentration, temperature.
- `data/cross_check_report.csv` — chip_summary vs chip_index status comparison.
- `figures/` — NC chip plots (priority) + grouped condition overlay plots.
- `docs/Biochemical_Guardrails.docx` / `.pdf` — draft guardrails doc, flagged sections need Eva's sign-off.
- `scripts/` — `ingest_raw_curves.py`, `build_chip_index.py`, `cross_check_index.py`, `make_sanity_plots.py`.
- `models/`, `app/` — empty, for Weeks 2+.

## Pipeline order

```bash
cd qtwin/scripts
python ingest_raw_curves.py      # -> raw_timeseries_master.csv, chip_summary.csv
python build_chip_index.py       # -> chip_index.csv
python cross_check_index.py      # -> cross_check_report.csv
python make_sanity_plots.py      # -> figures/
```

## Open questions for the teacher / Eva

See the "DRAFT — needs Eva's review" callouts inside `docs/Biochemical_Guardrails.docx`.
Summarized in `docs/clarifying_questions.md`.
