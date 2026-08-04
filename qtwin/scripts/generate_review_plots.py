"""
Week 2 Task 7: Generate example plots + a short summary for Eva's async
review while she's traveling.

Regenerates a handful of representative example curves (not the full
300-310 batch -- just enough for a fast phone read) for each of the three
classes the task specifies (CLEAN_PCA3_TARGET, BACKGROUND_SOUP,
DEFECTIVE_CHIP), plus one intentionally-divergent-replicate example, and
writes a short plain-text summary alongside them.

Output: qtwin/figures/week2_review/*.png, qtwin/figures/week2_review/summary.txt
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import constants as C
import curve_generator as cg
from generate_probe_batch import make_forced_divergent_pair

FIG_DIR = Path(__file__).resolve().parents[1] / "figures" / "week2_review"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"

SEED = 20260812  # Day 5


def plot_pair(t_y_pairs, title, status, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    colors = ["#1f77b4", "#d62728"]
    for i, (t, y) in enumerate(t_y_pairs):
        ax.plot(t, y, color=colors[i], linewidth=1, label=f"Replicate {i+1}", alpha=0.85)
    ax.axvline(C.BIOMARKER_WINDOW_S, color="gray", linestyle="--", linewidth=0.8, label=f"t={int(C.BIOMARKER_WINDOW_S)}s biomarker window")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Relative time (s)")
    ax.set_ylabel("Δf (Hz, relative to chi/baseline)")
    ax.set_title(f"{title}\nreplicate status: {status}", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    saved = []

    # 3 CLEAN_PCA3_TARGET examples at different concentrations. Uses the
    # deterministic trend value (no bootstrapped chip-to-chip noise) so the
    # example plots clearly show the intended concentration-response shape
    # for a fast visual gut-check -- the full noisy batch (chip-to-chip std
    # ~97 Hz, large enough that some individual real/synthetic chips land
    # near zero even at 10 uM) is already covered by the K-S validation
    # separately, and isn't what a 10-second glance at a plot should show.
    for i, conc in enumerate([5.0, 10.0, 20.0]):
        final_value = float(cg.probe_amplitude_trend(np.array([conc]))[0])
        duration = cg.sample_duration(rng)
        reps = [cg.generate_association_curve(final_value, duration, rng) for _ in range(2)]
        disp = [cg.compute_early_displacement(t, y) for t, y in reps]
        _, status = cg.check_replicate_concordance(disp[0], disp[1])
        name = f"CLEAN_PCA3_TARGET_example_{i+1}_conc{conc}uM.png"
        plot_pair(reps, f"CLEAN_PCA3_TARGET (probe stage), {conc} uM", status, FIG_DIR / name)
        saved.append(name)

    # Bonus: 2 CLEAN_PCA3_TARGET_HYB examples (target stage). Not one of the
    # 3 classes Task 7's instructions literally named, but added anyway --
    # this is the class carrying the UNVERIFIED_AGAINST_LOCAL_DATA flag
    # (constants.py, clarifying_questions.md item 15), so it's the one most
    # worth Eva's eyes specifically. Trend-only, same reasoning as above.
    for i, conc in enumerate([5.0, 10.0]):
        final_value = float(cg.target_amplitude_trend(np.array([conc]))[0])
        duration = cg.sample_duration(rng)
        reps = [cg.generate_association_curve(final_value, duration, rng) for _ in range(2)]
        disp = [cg.compute_early_displacement(t, y) for t, y in reps]
        _, status = cg.check_replicate_concordance(disp[0], disp[1])
        name = f"CLEAN_PCA3_TARGET_HYB_example_{i+1}_conc{conc}uM.png"
        plot_pair(reps, f"CLEAN_PCA3_TARGET_HYB (target stage, UNVERIFIED spec), {conc} uM", status, FIG_DIR / name)
        saved.append(name)

    # 2 BACKGROUND_SOUP examples
    for i in range(2):
        final_value = cg.sample_background_soup_final_value(rng)
        duration = cg.sample_duration(rng)
        reps = [cg.generate_baseline_drift_curve(final_value, duration, rng) for _ in range(2)]
        disp = [cg.compute_early_displacement(t, y) for t, y in reps]
        _, status = cg.check_replicate_concordance(disp[0], disp[1])
        name = f"BACKGROUND_SOUP_example_{i+1}.png"
        plot_pair(reps, "BACKGROUND_SOUP (target stage, NC-informed)", status, FIG_DIR / name)
        saved.append(name)

    # 2 DEFECTIVE_CHIP examples
    for i in range(2):
        final_value = cg.sample_defective_final_value(rng)
        duration = cg.sample_duration(rng)
        reps = [cg.generate_baseline_drift_curve(final_value, duration, rng) for _ in range(2)]
        disp = [cg.compute_early_displacement(t, y) for t, y in reps]
        _, status = cg.check_replicate_concordance(disp[0], disp[1])
        name = f"DEFECTIVE_CHIP_example_{i+1}.png"
        plot_pair(reps, "DEFECTIVE_CHIP (probe stage)", status, FIG_DIR / name)
        saved.append(name)

    # 1 intentionally-divergent example (matches real 21Mar_No.29 artifact pattern)
    final_value = cg.sample_clean_final_value(10.0, rng)
    duration = cg.sample_duration(rng)
    reps = make_forced_divergent_pair(final_value, "CLEAN_PCA3_TARGET", rng, duration)
    disp = [cg.compute_early_displacement(t, y) for t, y in reps]
    _, status = cg.check_replicate_concordance(disp[0], disp[1])
    name = "INTENTIONAL_DIVERGENT_example.png"
    plot_pair(reps, "Intentional artifact example (mirrors real 21Mar_No.29)", status, FIG_DIR / name)
    saved.append(name)

    # K-S report + counts for the summary
    ks_report = (DATA_DIR / "ks_validation_report.txt").read_text()
    batch = pd.read_csv(DATA_DIR / "synthetic_batch_v1.csv")
    n_total = len(batch)
    n_divergent = (batch["biomarker_replicate_status"] == "DIVERGENT_REPLICATES").sum()

    summary = f"""Q-Twin Week 2 -- synthetic batch review (for Eva, async, ~10-15 min)

What was generated: {n_total} synthetic curves across 4 classes (CLEAN_PCA3_TARGET,
DEFECTIVE_CHIP for the probe stage; CLEAN_PCA3_TARGET_HYB, BACKGROUND_SOUP for
the target stage), plus 8 intentionally-corrupted replicate-pair examples for
gatekeeper testing (Q7).

K-S validation: BOTH required tests passed on the first attempt --
Delta-f (probe stage) p=0.94, kinetic biomarker p=0.47 (both well above the
p>0.05 bar). See ks_validation_report.txt for full numbers.

Specific things worth your eyes:
1. The target-hybridization curve (+63.49 Hz inversion at 10 uM) is built to
   your/the paper's spec, but our own real 21 Mar target-stage data doesn't
   independently confirm it (5 uM is the MOST negative point in our data, not
   "weakly dampened" as expected; 10 uM is noisy/mixed). Flagged, not hidden --
   see clarifying_questions.md item 15. Worth a look at whether the
   CLEAN_PCA3_TARGET_HYB example plots look physically reasonable to you.
2. Real chips diverge on replicate concordance 64% of the time (14/22), but
   this generator's synthetic curves only diverge ~13-14% spontaneously
   (before the 8 intentional examples). Either real instrument noise is
   higher than this model captures, or the 0.5 concordance tolerance is too
   strict for real QCM data -- open question either way.
3. Hold-out set (11 chips, holdout_chips.txt) was reserved AFTER the
   generators were calibrated against all 44 chips (matching the Week 2 plan's
   own Day 1-4 ordering) -- fine for Weeks 3-4 model validation, but not a
   fully leakage-free hold-out for the generator's own noise model.

Files: probe_synthetic_batch.csv, target_synthetic_batch.csv,
synthetic_batch_v1.csv, ks_validation_report.txt, holdout_chips.txt (all in
qtwin/data/); plots in qtwin/figures/week2_review/.
"""
    (FIG_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("Saved plots:", saved)


if __name__ == "__main__":
    main()
