"""
Week 2 Task 1: single source of truth for the finalized Week 1 guardrails.
All of Week 2's generator code should import from here rather than
hard-coding these numbers separately.

Two flags worth reading before using this file:

  UNVERIFIED_AGAINST_LOCAL_DATA on TARGET_INVERT_HZ -- this number (+63.49 Hz)
  is Eva's/the published paper's figure for the target-hybridization stage at
  10 uM. It does NOT independently reproduce against this project's own
  21 Mar target-stage data (n=4, noisy: +85.325, -211.450, +42.855, +0.120 Hz;
  mean -20.79 Hz, or +42.77 Hz excluding the outlier). Unlike PROBE_PEAK_HZ
  (-62.96 Hz), which was confirmed to ~1 Hz against 45 real chips, this one
  is used on Eva's/the paper's authority alone. See
  qtwin/docs/clarifying_questions.md item 15. If Eva confirms a source for
  this number, update this comment (and the flag below) accordingly.

  FAILURE_THRESHOLD_HZ is -0.5, not 0. The Week 2 planning doc's Task 1 says
  "hard 0 Hz cutoff", but that's stale -- Eva's Q4 (already implemented and
  tested in ingest_raw_curves.py, see clarifying_questions.md item 11)
  explicitly moved this to -0.5 Hz to account for instrument baseline noise.
  Using 0 Hz here would silently regress an already-confirmed fix, so -0.5
  is used instead.
"""

# --- Probe Immobilization stage (confirmed against 45 real chips) ---------
# See Biochemical_Guardrails.pdf Section 2a. Delta-f = probe-endpoint minus
# chi-endpoint. Rises to a peak (most negative) at 10 uM, then WEAKENS
# (magnitude shrinks) at 20-40 uM -- no sign inversion in this dataset.
PROBE_PEAK_CONC_UM = 10.0
PROBE_PEAK_HZ = -62.96  # published/aggregated reference; our own 45-chip
                         # reproduction independently landed at -62.04 Hz
PROBE_ABOVE_PEAK_BEHAVIOR = "weaken, no inversion"

# --- Target Hybridization stage (per Eva/published paper, UNVERIFIED here) -
# See Biochemical_Guardrails.pdf Section 2b. A separate curve from the probe
# stage: gradual onset of dampening from ~5 uM, inverting to a POSITIVE
# shift by 10 uM.
TARGET_ONSET_CONC_UM = 5.0
TARGET_INVERT_CONC_UM = 10.0
TARGET_INVERT_HZ = 63.49
TARGET_INVERT_HZ_UNVERIFIED_AGAINST_LOCAL_DATA = True

# --- Chip failure logic (Eva's Q4, IMPLEMENTED -- see item 11) ------------
# SUCCESS requires delta_f_probe <= FAILURE_THRESHOLD_HZ. The -0.5 to 0 Hz
# band is instrument baseline noise, not a real signal.
FAILURE_THRESHOLD_HZ = -0.5

# --- Negative control drift model (Eva's Q5, CONFIRMED -- see item 13) ----
# NC chips (no target DNA present) show thermal/buffer baseline drift, not
# specific binding. Model as environmental/instrument noise (e.g. slow
# random-walk / Ornstein-Uhlenbeck-style drift), NOT a Langmuir binding
# curve -- there is no receptor-ligand kinetics happening in an NC chip.
NC_DRIFT_MODEL = "thermal_baseline_noise"  # not "langmuir"

# --- Kinetic biomarker (redefined + validated -- see item 8) --------------
BIOMARKER_WINDOW_S = 30.0
BIOMARKER_NAME = "early_displacement_30s"  # NOT the old slope-based metric

# --- Replicate concordance rule (Eva's Q7, IMPLEMENTED -- see item 12) ----
# Two replicates "agree" if same sign AND relative difference
# |a-b|/max(|a|,|b|) <= this tolerance. This exact tolerance flagged 14/22
# real multi-replicate chips (64%) as divergent, not just the No.29 example
# -- Eva/the teacher have not yet confirmed 0.5 is the right number for real
# QCM noise (see item 12). Used as-is here since it's the only value we
# have, but worth revisiting if Eva's async review flags anything odd about
# how many synthetic samples get excluded.
REPLICATE_CONCORDANCE_RELATIVE_TOLERANCE = 0.5

# --- Excluded / bad-measurement chips (see item 9) -------------------------
EXCLUDED_CHIP_IDS = ["15Mar_No.16"]
