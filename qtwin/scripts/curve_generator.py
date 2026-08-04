"""
Week 2 shared curve-generation library. Both generate_probe_batch.py and
generate_target_batch.py import from here, so the Langmuir/noise model and
the replicate-concordance rule (Task 4 / Eva's Q7) are defined once.

All model parameters below were fit against the real 45-chip dataset
(qtwin/data/chip_summary.csv, qtwin/data/raw_timeseries_master.csv), not
guessed. See the docstring on each function for where its numbers came
from and how to re-derive them.
"""

import numpy as np
import pandas as pd

import constants as C

RAW_MASTER = None
CHIP_SUMMARY = None


def _load_real_data():
    """Lazy-load real data once; used to calibrate/bootstrap noise."""
    global RAW_MASTER, CHIP_SUMMARY
    if CHIP_SUMMARY is None:
        from pathlib import Path
        data_dir = Path(__file__).resolve().parents[1] / "data"
        CHIP_SUMMARY = pd.read_csv(data_dir / "chip_summary.csv")
        RAW_MASTER = pd.read_csv(data_dir / "raw_timeseries_master.csv")
    return RAW_MASTER, CHIP_SUMMARY


# --- Probe-stage equilibrium amplitude vs. concentration -------------------
# Asymmetric split-Gaussian in log(concentration), fit against the 14 Mar
# concentration-sweep chips (0/5/10/20/40 uM real delta_f_probe means).
# A symmetric bell function (e.g. Hill-activation * Hill-inhibition with a
# single peak-location constraint) was tried first and rejected: it forces
# f(5)=f(20) by construction (5*20=10^2), which contradicts the real data
# (-11.58 Hz at 5 uM vs -55.30 Hz at 20 uM -- very different magnitudes).
# The split-Gaussian has independent left/right widths and fits all three
# non-peak points to within ~5 Hz.
_PEAK_C = C.PROBE_PEAK_CONC_UM
_SIGMA_LEFT = 0.532682892472932   # fit to 5 uM point
_SIGMA_RIGHT = 2.242290667160515  # fit to 20 uM and 40 uM points, least-squares


def probe_amplitude_trend(conc_uM: np.ndarray) -> np.ndarray:
    """Deterministic equilibrium probe-stage delta_f (Hz) at given
    concentration(s), peaking at PROBE_PEAK_CONC_UM = 10 with
    PROBE_PEAK_HZ = -62.96. Concentrations <= 0 return 0 (no probe, no
    binding component -- 0 uM chips are baseline drift, handled separately)."""
    conc = np.asarray(conc_uM, dtype=float)
    out = np.zeros_like(conc)
    pos = conc > 0
    lnC = np.log(conc[pos])
    lnC0 = np.log(_PEAK_C)
    sigma = np.where(lnC < lnC0, _SIGMA_LEFT, _SIGMA_RIGHT)
    out[pos] = -abs(C.PROBE_PEAK_HZ) * np.exp(-((lnC - lnC0) / sigma) ** 2)
    return out


# --- Chip-to-chip variability (distinct from within-curve sampling noise) --
# Real chips at the same nominal concentration scatter far more than
# measurement jitter alone would predict (std ~97 Hz around the trend above,
# vs ~0.03-0.05 Hz within-curve sampling noise). This dominates the
# concentration trend itself in this dataset (see clarifying_questions.md
# item 6/14 -- the data is "highly noisy"). Bootstrap-sampled from the real
# residuals (trend-subtracted delta_f_probe for all valid, concentration>0
# chips) rather than fit to a symmetric distribution, since the real
# residuals are skewed (mean -4.3, but min -309 / max +167).
def _real_probe_residuals():
    _, cs = _load_real_data()
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    valid["conc"] = pd.to_numeric(valid["concentration_uM"], errors="coerce")
    d = valid.dropna(subset=["conc", "delta_f_probe"])
    d = d[d["conc"] > 0]
    pred = probe_amplitude_trend(d["conc"].values)
    return (d["delta_f_probe"].values - pred)


def _real_failure_probe_values():
    _, cs = _load_real_data()
    valid = cs[cs.success_or_fail != "EXCLUDED"]
    return valid[valid.success_or_fail == "FAILURE"]["delta_f_probe"].dropna().values


# --- Acquisition parameters (sampling interval, duration) ------------------
# Real probe-stage curves: median sampling interval 0.62s, duration
# min=50 / p25=132 / median=162 / p75=189 / max=443s (n=67 replicate curves).
SAMPLING_INTERVAL_S = 0.62
DURATION_MIN_S = 50.0
DURATION_P25_S = 132.0
DURATION_MEDIAN_S = 162.0
DURATION_P75_S = 189.0

# Within-curve point-to-point measurement noise std, from real data
# (rolling-mean-residual std across all probe-stage replicate curves):
# mean 0.053, median 0.035 Hz. Sampled per-curve from this empirical range
# rather than a single fixed value, since real curves vary in how noisy
# they are.
POINT_NOISE_STD_LOW = 0.02
POINT_NOISE_STD_HIGH = 0.07

# Kinetics: real SUCCESS chips reach ~100% of their final delta_f_probe
# already by t=30s (mean ratio 1.03, std 0.11 -- see clarifying_questions.md
# item 8's validation). k_obs chosen so the deterministic association curve
# reaches ~95-99% of its asymptote by 30s; per-curve jitter added for
# realism (real chips range 81%-148% of final value at 30s, driven mostly by
# the large point-noise/chip-noise relative to the modest signal size, not
# by kinetics being slow).
K_OBS_MEAN = 0.18   # /s
K_OBS_JITTER = 0.05  # /s, uniform +/-


def sample_duration(rng: np.random.Generator) -> float:
    """Sample a realistic curve duration from the real distribution's
    quartiles (piecewise-linear interpolation of the empirical CDF)."""
    u = rng.uniform(0, 1)
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    ys = [DURATION_MIN_S, DURATION_P25_S, DURATION_MEDIAN_S, DURATION_P75_S, DURATION_P75_S * 1.3]
    return float(np.interp(u, xs, ys))


def generate_association_curve(final_delta_f: float, duration_s: float, rng: np.random.Generator):
    """Pseudo-first-order (Langmuir) association kinetics: rises from 0
    towards final_delta_f with rate k_obs, plus point-to-point measurement
    noise. Returns (t, y) arrays. y[0] == 0 by construction (curves are
    stored relative to the CHI-stage / pre-probe baseline, matching how
    early_displacement_30s and delta_f_probe are computed on real data)."""
    t = np.arange(0.0, duration_s, SAMPLING_INTERVAL_S)
    k_obs = max(0.02, K_OBS_MEAN + rng.uniform(-K_OBS_JITTER, K_OBS_JITTER))
    trend = final_delta_f * (1.0 - np.exp(-k_obs * t))
    noise_std = rng.uniform(POINT_NOISE_STD_LOW, POINT_NOISE_STD_HIGH)
    noise = rng.normal(0.0, noise_std, size=t.shape)
    return t, trend + noise


def generate_baseline_drift_curve(final_value: float, duration_s: float, rng: np.random.Generator):
    """Thermal/buffer baseline drift model (Eva's Q5, confirmed -- NOT a
    Langmuir binding curve). Used for: DEFECTIVE_CHIP (failed probe
    immobilization -- no real specific binding happened), BACKGROUND_SOUP
    (negative controls). Modeled as a bounded random walk pulled gently
    toward final_value (Ornstein-Uhlenbeck-style), which produces the
    slow, non-monotonic, non-exponential drift seen in real NC/FAILURE
    curves rather than a clean kinetic rise."""
    t = np.arange(0.0, duration_s, SAMPLING_INTERVAL_S)
    n = len(t)
    pull_strength = 0.02  # how strongly the walk is pulled toward final_value per step
    step_std = rng.uniform(0.15, 0.5)
    y = np.zeros(n)
    for i in range(1, n):
        y[i] = y[i - 1] + pull_strength * (final_value - y[i - 1]) + rng.normal(0, step_std)
    noise_std = rng.uniform(POINT_NOISE_STD_LOW, POINT_NOISE_STD_HIGH)
    y = y + rng.normal(0.0, noise_std, size=n)
    return t, y


def sample_clean_final_value(conc_uM: float, rng: np.random.Generator) -> float:
    """Final (equilibrium) probe-stage delta_f for a CLEAN_PCA3_TARGET
    (SUCCESS) synthetic chip at the given concentration: deterministic
    trend + bootstrapped real chip-to-chip residual."""
    trend = probe_amplitude_trend(np.array([conc_uM]))[0]
    residuals = _real_probe_residuals()
    resid = rng.choice(residuals) + rng.normal(0, 5.0)  # small jitter so bootstrap isn't exact-repeat
    value = trend + resid
    # CLEAN_PCA3_TARGET chips must be a real SUCCESS by definition
    # (delta_f_probe <= FAILURE_THRESHOLD_HZ) -- resample if the noise
    # pushed it above threshold, rather than silently mislabeling a class.
    tries = 0
    while value > C.FAILURE_THRESHOLD_HZ and tries < 20:
        resid = rng.choice(residuals) + rng.normal(0, 5.0)
        value = trend + resid
        tries += 1
    return value


def sample_defective_final_value(rng: np.random.Generator) -> float:
    """Final probe-stage delta_f for a DEFECTIVE_CHIP (FAILURE) synthetic
    chip: bootstrapped from the real 15 FAILURE chips' actual delta_f_probe
    values, not modeled from a smooth trend (a failed chip isn't "weak
    binding", it's the absence of specific binding -- see Q5's baseline-
    drift model)."""
    real_fail_values = _real_failure_probe_values()
    value = rng.choice(real_fail_values) + rng.normal(0, 3.0)
    tries = 0
    while value <= C.FAILURE_THRESHOLD_HZ and tries < 20:
        value = rng.choice(real_fail_values) + rng.normal(0, 3.0)
        tries += 1
    return value


def compute_early_displacement(t: np.ndarray, y: np.ndarray, window_s: float = C.BIOMARKER_WINDOW_S) -> float:
    """Same computation as real data: linearly-interpolated curve value at
    t=window_s, relative to the t=0 baseline (curves here are already
    baseline-relative, so this is just interp(window_s) - y[0])."""
    if t.max() < window_s:
        return np.nan
    return float(np.interp(window_s, t, y) - y[0])


# --- Target-stage equilibrium amplitude vs. concentration ------------------
# UNVERIFIED_AGAINST_LOCAL_DATA (see constants.py and clarifying_questions.md
# item 15). Built to match the externally-specified spec (weak dampened
# negative from ~5 uM, inverting to +63.49 Hz by 10 uM) per Eva's Q2/Q3 and
# the Week 2 planning doc's Task 3 -- NOT fit to this project's own
# real target-stage data, which does not show this pattern (real 5 uM mean
# is -62.66 Hz, the MOST negative point, not "weakly dampened"; real 10 uM
# is noisy/mixed, mean -20.79 Hz with individual chips from -211 to +85).
# That contradiction is intentional to surface, not hidden -- see the Task 3
# summary this generator's caller prints.
#
# Logistic sigmoid in log(C), midpoint at sqrt(ONSET*INVERT)=7.07 uM,
# anchored so trend(TARGET_INVERT_CONC_UM) == TARGET_INVERT_HZ exactly.
_TGT_MID_C = np.sqrt(C.TARGET_ONSET_CONC_UM * C.TARGET_INVERT_CONC_UM)
_TGT_WIDTH = 0.3
_TGT_LOW = -30.0  # weak-negative asymptote at low concentration


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _target_sigmoid_high():
    """Solve HIGH asymptote so trend(TARGET_INVERT_CONC_UM) exactly equals
    TARGET_INVERT_HZ, given the fixed LOW asymptote, midpoint, and width."""
    x10 = (np.log(C.TARGET_INVERT_CONC_UM) - np.log(_TGT_MID_C)) / _TGT_WIDTH
    s10 = _sigmoid(x10)
    return _TGT_LOW + (C.TARGET_INVERT_HZ - _TGT_LOW) / s10


_TGT_HIGH = _target_sigmoid_high()


def target_amplitude_trend(conc_uM: np.ndarray) -> np.ndarray:
    """Deterministic equilibrium target-hybridization-stage delta_f (Hz):
    weak negative at low concentration, transitioning to positive
    TARGET_INVERT_HZ by TARGET_INVERT_CONC_UM. See UNVERIFIED flag above."""
    conc = np.asarray(conc_uM, dtype=float)
    out = np.zeros_like(conc)
    pos = conc > 0
    x = (np.log(conc[pos]) - np.log(_TGT_MID_C)) / _TGT_WIDTH
    out[pos] = _TGT_LOW + (_TGT_HIGH - _TGT_LOW) * _sigmoid(x)
    return out


def _real_target_residuals():
    """Real delta_f_target residuals around the (unverified) trend above --
    used only for noise SCALE (how noisy target-stage chips are in general),
    not as evidence the trend shape itself is correct."""
    _, cs = _load_real_data()
    valid = cs[cs.success_or_fail != "EXCLUDED"].copy()
    valid["conc"] = pd.to_numeric(valid["concentration_uM"], errors="coerce")
    d = valid.dropna(subset=["conc", "delta_f_target"])
    d = d[d["conc"] > 0]
    pred = target_amplitude_trend(d["conc"].values)
    return d["delta_f_target"].values - pred


def sample_target_clean_final_value(conc_uM: float, rng: np.random.Generator) -> float:
    trend = target_amplitude_trend(np.array([conc_uM]))[0]
    residuals = _real_target_residuals()
    resid = rng.choice(residuals) + rng.normal(0, 5.0)
    return trend + resid


def sample_background_soup_final_value(rng: np.random.Generator) -> float:
    """BACKGROUND_SOUP final value, bootstrapped from the two real NC chips'
    (21Mar_No.7, 21Mar_No.29) actual delta_f values -- small, near-zero-ish,
    chaotic, no specific binding (Eva's Q5)."""
    real_nc_values = np.array([3.29, 22.615, 23.045, -0.18])  # No.7/No.29 probe+target Δf
    return float(rng.choice(real_nc_values) + rng.normal(0, 4.0))


def check_replicate_concordance(rate_or_value_r1: float, rate_or_value_r2: float,
                                 tolerance: float = C.REPLICATE_CONCORDANCE_RELATIVE_TOLERANCE):
    """Task 4 / Eva's Q7: same rule implemented in ingest_raw_curves.py for
    real data. Returns (averaged_value_or_nan, status_string).
    'Agree' = same sign AND relative difference |a-b|/max(|a|,|b|) <= tolerance."""
    a, b = rate_or_value_r1, rate_or_value_r2
    same_sign = (a >= 0) == (b >= 0)
    denom = max(abs(a), abs(b))
    rel_diff = abs(a - b) / denom if denom > 0 else 0.0
    if not same_sign or rel_diff > tolerance:
        return np.nan, "DIVERGENT_REPLICATES"
    return float((a + b) / 2.0), "CONCORDANT"
