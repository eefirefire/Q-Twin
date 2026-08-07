"""
Testing week Task 4: does the Random Forest earn its complexity?

The gatekeeper's 100% accuracy is suspicious by its own honest caveat --
early_displacement_30s correlates 0.9997 with the label-defining quantity.
Trains plain logistic regression on the exact same features/split as the
Random Forest and compares directly.

Output: qtwin/models/gatekeeper_logistic_comparison.txt
"""

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import gatekeeper_model as gk
from gatekeeper_model import LABELS

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"


def main():
    X_train, y_train, _ = gk.load_training_data()
    X_val, y_val, _, n_holdout_excluded = gk.load_validation_data()

    from sklearn.ensemble import RandomForestClassifier
    rf_clf = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, class_weight="balanced")
    rf_clf.fit(X_train, y_train)
    rf_pred = rf_clf.predict(X_val)
    rf_acc = accuracy_score(y_val, rf_pred)

    lr_clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr_clf.fit(X_train, y_train)
    lr_pred = lr_clf.predict(X_val)
    lr_acc = accuracy_score(y_val, lr_pred)

    rf_report = classification_report(y_val, rf_pred, labels=LABELS, zero_division=0)
    lr_report = classification_report(y_val, lr_pred, labels=LABELS, zero_division=0)
    rf_cm = confusion_matrix(y_val, rf_pred, labels=LABELS)
    lr_cm = confusion_matrix(y_val, lr_pred, labels=LABELS)

    print(f"Random Forest accuracy: {rf_acc:.3f}")
    print(f"Logistic Regression accuracy: {lr_acc:.3f}")

    match = rf_acc == lr_acc

    with open(MODEL_DIR / "gatekeeper_logistic_comparison.txt", "w", encoding="utf-8") as f:
        f.write("Testing week Task 4 -- RF vs. Logistic Regression baseline\n")
        f.write("=" * 60 + "\n\n")
        f.write("Same features (early_displacement_30s + is_divergent), same train/val split.\n\n")
        f.write(f"Random Forest:        accuracy = {rf_acc:.3f}\n")
        f.write(rf_report)
        f.write(f"Confusion matrix: {rf_cm.tolist()}\n\n")
        f.write(f"Logistic Regression:  accuracy = {lr_acc:.3f}\n")
        f.write(lr_report)
        f.write(f"Confusion matrix: {lr_cm.tolist()}\n\n")
        f.write("CONCLUSION: ")
        if match:
            f.write("Logistic regression matches Random Forest exactly. The simpler model is\n")
            f.write("the better result to report -- the relationship between\n")
            f.write("early_displacement_30s/is_divergent and the label is linearly separable\n")
            f.write("(consistent with the 0.9997 correlation already documented), so Random\n")
            f.write("Forest's complexity isn't earning anything here. Recommend citing logistic\n")
            f.write("regression as the gatekeeper's simplest sufficient model in the proposal,\n")
            f.write("while keeping the RF implementation since it costs nothing extra to run.\n")
        elif rf_acc > lr_acc:
            f.write(f"Random Forest outperforms logistic regression ({rf_acc:.3f} vs {lr_acc:.3f}).\n")
            f.write("That's evidence the early_displacement_30s/is_divergent relationship to the\n")
            f.write("label isn't purely linear -- worth citing as support for keeping RF, not\n")
            f.write("just defaulting to it.\n")
        else:
            f.write(f"Logistic regression outperforms Random Forest ({lr_acc:.3f} vs {rf_acc:.3f}) --\n")
            f.write("unexpected; worth double-checking RF hyperparameters before trusting this.\n")

    print(f"wrote {MODEL_DIR / 'gatekeeper_logistic_comparison.txt'}")


if __name__ == "__main__":
    main()
