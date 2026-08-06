"""
Week 3 follow-up (Eva's review feedback): stress-test the Task 1
gatekeeper specifically near the SUCCESS/FAILURE decision boundary
(-0.5 Hz).

The honest caveat already written into gatekeeper_metrics.txt flagged
that 0 of 26 non-divergent real validation chips sit within 2 Hz of that
boundary -- meaning the 100% accuracy never actually tested the hard
case. This script builds a small synthetic batch of chips whose TRUE
final delta_f is deliberately drawn close to the threshold (+/- 3 Hz,
i.e. roughly the real instrument-noise band identified in
clarifying_questions.md item 4) and checks whether the gatekeeper still
holds up there, using the exact same feature/training pipeline as
gatekeeper_model.py (via pipeline_api.train_gatekeeper()) -- not a
separately-trained model, so this is a genuine test of the SAME
classifier Task 1 shipped, not a new one.

Output: qtwin/models/gatekeeper_boundary_stress_test.txt,
qtwin/models/gatekeeper_boundary_confusion_matrix.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import constants as C
import curve_generator as cg
import pipeline_api as api
from gatekeeper_model import LABELS, build_features

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

N_BOUNDARY_CHIPS = 60
BOUNDARY_HALF_WIDTH_HZ = 3.0  # +/- around FAILURE_THRESHOLD_HZ (-0.5 Hz)
SEED = 20260806


def build_boundary_batch(rng: np.random.Generator) -> pd.DataFrame:
    """Deliberately does NOT apply cg.jitter_replicate_final_value() here.
    That jitter (std=50 Hz, calibrated in Task 4 to match real replicate
    DIVERGENCE rates) would push a +/-3 Hz boundary target to wherever --
    completely defeating the point of a boundary-proximity test. Divergence
    behavior near the boundary is a real and interesting question too, but
    it's a DIFFERENT failure axis from "is the true signal close to the
    threshold," already covered by Task 4's investigation; conflating the
    two here would make it impossible to tell which effect caused any given
    misclassification. Both replicates share the exact boundary-adjacent
    target, diverging only through each curve's own generation noise --
    isolating the boundary question on its own."""
    rows = []
    lo = C.FAILURE_THRESHOLD_HZ - BOUNDARY_HALF_WIDTH_HZ
    hi = C.FAILURE_THRESHOLD_HZ + BOUNDARY_HALF_WIDTH_HZ
    for i in range(N_BOUNDARY_CHIPS):
        final_value = float(rng.uniform(lo, hi))
        duration = cg.sample_duration(rng)
        reps = []
        for _ in range(2):
            t, y = cg.generate_association_curve(final_value, duration, rng)
            reps.append((t, y))
        disp_vals = [cg.compute_early_displacement(t, y) for t, y in reps]
        final_disp, status = cg.check_replicate_concordance(disp_vals[0], disp_vals[1])
        true_endpoint = float(np.mean([y[-1] for _, y in reps]))
        actual_label = "DIVERGENT_REPLICATES" if status == "DIVERGENT_REPLICATES" else (
            "SUCCESS" if true_endpoint <= C.FAILURE_THRESHOLD_HZ else "FAILURE"
        )
        rows.append({
            "boundary_id": f"BOUNDARY_{i:03d}",
            "true_final_value_target": final_value,
            "true_endpoint_delta_f": true_endpoint,
            "distance_from_threshold_hz": abs(true_endpoint - C.FAILURE_THRESHOLD_HZ),
            "early_displacement_30s": final_disp,
            "biomarker_replicate_status": status,
            "actual_label": actual_label,
        })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(SEED)
    df = build_boundary_batch(rng)

    clf = api.train_gatekeeper()
    X = build_features(df, "early_displacement_30s", "biomarker_replicate_status")
    df["predicted_label"] = clf.predict(X)

    acc = accuracy_score(df["actual_label"], df["predicted_label"])
    report = classification_report(df["actual_label"], df["predicted_label"], labels=LABELS, zero_division=0)
    cm = confusion_matrix(df["actual_label"], df["predicted_label"], labels=LABELS)

    print(f"Boundary stress test: {N_BOUNDARY_CHIPS} synthetic chips, true final delta_f drawn from "
          f"[{C.FAILURE_THRESHOLD_HZ - BOUNDARY_HALF_WIDTH_HZ:.1f}, {C.FAILURE_THRESHOLD_HZ + BOUNDARY_HALF_WIDTH_HZ:.1f}] Hz "
          f"(threshold +/- {BOUNDARY_HALF_WIDTH_HZ} Hz)")
    print(f"Accuracy: {acc:.3f}")
    print(report)

    # Break down accuracy by distance-from-threshold band, among
    # non-divergent chips (that's what "near the boundary" is actually
    # asking about -- divergent chips are a different failure mode).
    non_div = df[df.actual_label != "DIVERGENT_REPLICATES"].copy()
    non_div["correct"] = non_div["actual_label"] == non_div["predicted_label"]
    bands = [(0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0)]
    band_rows = []
    for band_lo, band_hi in bands:
        band = non_div[(non_div.distance_from_threshold_hz >= band_lo) & (non_div.distance_from_threshold_hz < band_hi)]
        if len(band) == 0:
            continue
        band_acc = band["correct"].mean()
        band_rows.append((band_lo, band_hi, len(band), band_acc))
        print(f"  |true_delta_f - threshold| in [{band_lo},{band_hi}) Hz: n={len(band)}, accuracy={band_acc:.3f}")

    fig, ax = plt.subplots(figsize=(7.5, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS)
    disp.plot(ax=ax, cmap="Oranges", colorbar=False)
    ax.set_title(f"Gatekeeper boundary stress test\n(synthetic, true delta_f within {BOUNDARY_HALF_WIDTH_HZ} Hz of threshold)")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    cm_path = MODEL_DIR / "gatekeeper_boundary_confusion_matrix.png"
    fig.savefig(cm_path, dpi=130)
    plt.close(fig)
    print(f"wrote {cm_path}")

    with open(MODEL_DIR / "gatekeeper_boundary_stress_test.txt", "w", encoding="utf-8") as f:
        f.write("Week 3 follow-up -- Gatekeeper boundary stress test\n")
        f.write("=" * 60 + "\n\n")
        f.write("Motivation: gatekeeper_metrics.txt's honest caveat noted that 0 of 26\n")
        f.write("non-divergent real validation chips sit within 2 Hz of the -0.5 Hz\n")
        f.write("SUCCESS/FAILURE threshold, so the Task 1 gatekeeper's 100% accuracy never\n")
        f.write("actually tested the hard, near-boundary case. This script builds\n")
        f.write(f"{N_BOUNDARY_CHIPS} synthetic chips with TRUE final delta_f drawn uniformly from\n")
        f.write(f"[{C.FAILURE_THRESHOLD_HZ - BOUNDARY_HALF_WIDTH_HZ:.1f}, {C.FAILURE_THRESHOLD_HZ + BOUNDARY_HALF_WIDTH_HZ:.1f}] Hz ")
        f.write(f"(threshold +/- {BOUNDARY_HALF_WIDTH_HZ} Hz) and runs\n")
        f.write("the SAME trained gatekeeper (pipeline_api.train_gatekeeper(), identical\n")
        f.write("features/training data to gatekeeper_model.py) against them.\n\n")
        f.write(f"Overall accuracy on this near-boundary batch: {acc:.3f}\n\n")
        f.write("Accuracy by distance from threshold (non-divergent chips only --\n")
        f.write("divergent chips are a separate failure mode, not a boundary-proximity one):\n")
        for band_lo, band_hi, n, band_acc in band_rows:
            f.write(f"  |true_delta_f - threshold| in [{band_lo},{band_hi}) Hz: n={n}, accuracy={band_acc:.3f}\n")
        f.write("\nFull sklearn classification_report:\n")
        f.write(report)
        f.write(f"\nConfusion matrix (rows=actual, cols=predicted), labels={LABELS}:\n")
        f.write(str(cm))
        f.write("\n\nINTERPRETATION: ")
        if acc >= 0.95:
            f.write("the gatekeeper held up even in this deliberately hard band -- the earlier\n")
            f.write("100% now reads as genuinely trustworthy near the boundary too, not just an\n")
            f.write("artifact of real validation chips all being far from it.\n")
        else:
            f.write("accuracy drops meaningfully near the boundary -- this is the honest answer\n")
            f.write("the earlier caveat predicted might exist. The gatekeeper's real-world 100%\n")
            f.write("should be read as 'validated on the chips this dataset actually has', not\n")
            f.write("'reliable arbitrarily close to -0.5 Hz' -- worth flagging to Eva/the teacher\n")
            f.write("as a genuine performance boundary, not a bug to silently patch away.\n")

    print(f"wrote {MODEL_DIR / 'gatekeeper_boundary_stress_test.txt'}")


if __name__ == "__main__":
    main()
