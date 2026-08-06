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
    binding component -- 0 uM chips are baseline drift, handled separately).

    NOT A BUG, just worth noting so nobody trips over it later: this
    predicts ~0 Hz at 0 uM, but real 0 uM chips (14 Mar sweep) average
    +10.4 Hz. That gap is intentional -- 0 uM chips have no probe binding
    signal at all, so their real behavior is baseline/thermal drift, which
    is generated separately via generate_baseline_drift_curve() /
    sample_defective_final_value(), not through this trend function."""
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
# already by t=30s (mean ratio 1.03, std 0.11, range 0.81-1.48 -- see
# clarifying_questions.md item 8's validation).
#
# CORRECTED (caught on a later review pass -- see git history): an earlier
# version of this comment attributed that 0.81-1.48 spread mostly to
# point-noise/chip-noise. That explanation doesn't survive testing: this
# generator's point-noise and chip-to-chip noise are both calibrated
# against real data at comparable magnitude, yet a pure monotonic
# 1-exp(-kt) rise with only a narrow k_obs jitter produced synthetic
# ratios of 0.97-1.00 with std 0.005 -- 20x less spread than real, and
# structurally INCAPABLE of ever exceeding 1.0 (a monotonic rise can't
# read higher at t=30s than at its own later, further-converged endpoint).
# Real data hitting ratio > 1 means some real chips' signal partially
# relaxes back after an early peak -- a secondary, slower dynamic on top
# of the fast initial binding, not just noise. Modeled below as a slow
# MEAN-REVERTING (Ornstein-Uhlenbeck) drift term superimposed on the
# deterministic rise. A first attempt used a free (non-mean-reverting)
# random walk -- rejected after testing: its variance grows unboundedly
# with curve duration, which for longer/weaker-signal synthetic curves
# occasionally let the drift dominate and cross zero, producing nonsense
# ratios (tested std=0.414, range -2.1 to +2.4, vs. real std=0.112, range
# 0.81-1.48). The OU form's variance saturates instead of growing forever,
# which avoids the worst of that blow-up. step_std and pull were swept
# numerically against a simulation using typical-magnitude final values,
# landing close to the real 0.81-1.48 / std-0.112 target in that sweep.
#
# KNOWN REMAINING GAP: against the actual generator (not the simplified
# sweep), the ratio distribution is still visibly wider than real data
# (observed std ~0.34, one outlier ratio ~4.1). Root cause identified, not
# hand-waved: sample_clean_final_value() occasionally produces a
# final_delta_f very close to FAILURE_THRESHOLD_HZ (a chip that "barely"
# qualifies as SUCCESS), and the 30s/final ratio is mathematically
# ill-conditioned when dividing by a near-zero denominator -- a fixed-size
# drift perturbation is a huge relative swing for a small final value.
# This is a limitation of the ratio statistic itself for near-threshold
# chips, not something further drift-parameter tuning fixes cleanly. Both
# of Task 5's REQUIRED K-S tests (on the marginal Delta-f and biomarker
# distributions, which don't involve this ratio) still pass -- see
# qtwin/figures/week2_review/summary.txt for the honest writeup.
K_OBS_MEAN = 0.18   # /s
K_OBS_JITTER = 0.05  # /s, uniform +/-
SECONDARY_DRIFT_STEP_STD = 0.6   # Hz per sampling step
SECONDARY_DRIFT_PULL = 0.03      # mean-reversion strength per step

# Week 3 Task 4 fix (see Task4_Replicate_Divergence_Investigation.md): real
# replicate pairs behave like two physically independent immobilization
# spots, not two noisy readings of one shared binding event. Before this,
# both replicates of a synthetic chip were generated from the SAME
# final_delta_f target, diverging only through independent per-timestep
# noise/drift -- which flips sign at the 30s read only 4% of the time,
# vs. 38% for real replicate pairs (see the investigation doc for the
# full evidence). REPLICATE_KINETIC_JITTER_STD gives each replicate its
# own independent target value (chip_final_value + N(0, this_std)) before
# generating its curve, so replicates can diverge kinetically, not just
# through noise. Calibrated numerically (see calibrate_replicate_jitter.py,
# not committed -- a one-off sweep script) against two targets
# simultaneously: (a) probe-stage sign-flip rate at t=30s should land near
# the real 38.1% (8/21), and (b) the two REQUIRED K-S tests in
# assemble_batch.py must still pass, since jitter also widens the
# true_endpoint_delta_f population (each chip's value is still the mean of
# its 2 jittered replicates, so the chip-level population mean is
# unbiased, but its variance grows slightly).
REPLICATE_KINETIC_JITTER_STD = 50.0  # Hz -- calibrated via a two-stage sweep
# against the probe stage, checking BOTH required K-S tests every time
# (not just divergence rate): (1) an initial 10-110 Hz sweep against only
# the endpoint K-S test found ~68 Hz matched the real sign-flip/divergence
# rate best, but that pass ALSO exposed a pre-existing, previously-masked
# inconsistency -- synthetic early_displacement_30s was NaN'd out on
# DIVERGENT_REPLICATES while real data always averages regardless of
# concordance (see check_replicate_concordance()'s docstring above), so
# rising divergence rate was shrinking the synthetic biomarker K-S sample
# into an increasingly biased "happened to agree" subset and failing that
# test outright (p<0.002) at every nonzero jitter tried. Fixed that
# inconsistency first (check_replicate_concordance now always returns the
# mean, matching real data's own convention), then re-swept 42-58 Hz with
# 5 seeds each, checking both required tests' worst-case p-value: 50 Hz
# lands divergence rate at 57.2% (real: 57.1-63.6% across the two
# biomarker definitions in the investigation doc) with comfortable K-S
# margins (p_endpoint >= 0.13, p_biomarker >= 0.26 across seeds).


def jitter_replicate_final_value(final_value: float, rng: np.random.Generator) -> float:
    """Independent per-replicate deviation from a chip's shared target
    value -- see REPLICATE_KINETIC_JITTER_STD above for why this exists."""
    return final_value + rng.normal(0.0, REPLICATE_KINETIC_JITTER_STD)


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
    n = len(t)
    k_obs = max(0.02, K_OBS_MEAN + rng.uniform(-K_OBS_JITTER, K_OBS_JITTER))
    trend = final_delta_f * (1.0 - np.exp(-k_obs * t))
    # Secondary slow drift (see corrected comment above SECONDARY_DRIFT_STEP_STD):
    # mean-reverting (OU) random walk superimposed on the fast rise, so the
    # curve's actual endpoint can end up either further from, or closer to,
    # zero than the 30s reading -- reproducing real chips' occasional >100%
    # "reached by 30s" ratio, which a pure monotonic rise cannot produce.
    drift = np.zeros(n)
    drift_steps = rng.normal(0.0, SECONDARY_DRIFT_STEP_STD, size=n)
    for i in range(1, n):
        drift[i] = drift[i - 1] * (1.0 - SECONDARY_DRIFT_PULL) + drift_steps[i]
    noise_std = rng.uniform(POINT_NOISE_STD_LOW, POINT_NOISE_STD_HIGH)
    noise = rng.normal(0.0, noise_std, size=t.shape)
    return t, trend + drift + noise


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
    if value > C.FAILURE_THRESHOLD_HZ:
        # 20 retries exhausted without landing in SUCCESS range (didn't
        # happen in any observed run, but guarantee correctness rather than
        # silently mislabel a CLEAN_PCA3_TARGET chip as FAILURE-range).
        value = C.FAILURE_THRESHOLD_HZ - abs(rng.normal(1.0, 0.5))
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
    if value <= C.FAILURE_THRESHOLD_HZ:
        # Guarantee correctness rather than silently mislabel (see the
        # matching comment in sample_clean_final_value above).
        value = C.FAILURE_THRESHOLD_HZ + abs(rng.normal(1.0, 0.5))
    return value


def compute_early_displacement(t: np.ndarray, y: np.ndarray, window_s: float = C.BIOMARKER_WINDOW_S) -> float:
    """Same computation as real data: linearly-interpolated curve value at
    t=window_s, relative to the t=0 baseline (curves here are already
    baseline-relative, so this is just interp(window_s) - y[0])."""
    if t.max() < window_s:
        return np.nan
    return float(np.interp(window_s, t, y) - y[0])


# Week 3 Task 3 (LSTM): fixed-length resampling window. 45s chosen because
# it's safely under BOTH the real probe-stage minimum duration (49.99s,
# raw_timeseries_master.csv) and the synthetic DURATION_MIN_S (50.0) --
# every curve, real or synthetic, has real data across this whole window,
# so no curve needs extrapolation past what was actually measured/generated.
SEQUENCE_WINDOW_S = 45.0
SEQUENCE_N_POINTS = 60  # ~0.75s spacing over the 45s window


def resample_curve(t: np.ndarray, y: np.ndarray, window_s: float = SEQUENCE_WINDOW_S,
                    n_points: int = SEQUENCE_N_POINTS) -> np.ndarray:
    """Linearly interpolate (t, y) onto a fixed grid of n_points over
    [0, window_s], so curves of different native duration/sampling all
    become fixed-length vectors an LSTM can batch. Returns y-values only
    (baseline-relative, so y[0] ~= 0); the time grid itself is implicit
    and identical for every curve. Requires t.max() >= window_s -- callers
    should filter out anything shorter first rather than silently
    extrapolate past real/generated data."""
    if t.max() < window_s:
        raise ValueError(f"curve duration {t.max():.1f}s shorter than window {window_s}s -- cannot resample without extrapolating")
    grid = np.linspace(0.0, window_s, n_points)
    return np.interp(grid, t, y)


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
    real data. Returns (averaged_value, status_string).
    'Agree' = same sign AND relative difference |a-b|/max(|a|,|b|) <= tolerance.

    NOTE (Week 3 Task 4 follow-up fix): this used to return NaN for the
    value on DIVERGENT_REPLICATES. That silently diverged from real data's
    own convention -- ingest_raw_curves.py's early_displacement_30s always
    averages both replicates regardless of concordance status, using
    displacement_replicate_status only as a separate caveat flag, never as
    a reason to null the value. The mismatch was invisible while synthetic
    divergence was rare (~12%), but once REPLICATE_KINETIC_JITTER_STD made
    divergence realistically common (~60%+), it meant the synthetic
    early_displacement_30s POPULATION being K-S-tested against real was a
    shrinking, biased "replicates happened to agree" subsample instead of
    the full population real data reports -- which broke the required
    biomarker K-S test even though the underlying divergence-rate fix was
    correct. Fixed by always returning the mean here too, matching real
    data's convention exactly; status_string (and downstream is_divergent
    flags) still carry the concordance information separately."""
    a, b = rate_or_value_r1, rate_or_value_r2
    same_sign = (a >= 0) == (b >= 0)
    denom = max(abs(a), abs(b))
    rel_diff = abs(a - b) / denom if denom > 0 else 0.0
    status = "DIVERGENT_REPLICATES" if (not same_sign or rel_diff > tolerance) else "CONCORDANT"
    return float((a + b) / 2.0), status
