"""Phase 8 - Model evaluation on the held-out TEST set.

The test set is used exactly once, here, after the model has been chosen and
tuned with cross-validation on the training data.
"""
import json
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

from src.config import EVALUATION_PATH, FIGURES_DIR

CLASS_NAMES = ["FAKE", "REAL"]  # label_id 0 and 1


def wilson_interval(correct, n, z=1.96):
    """95% confidence interval for an accuracy measured on n test articles."""
    if n == 0:
        return (0.0, 0.0)
    p = correct / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def reliability_from(accuracy, p_value, baseline):
    """Plain-language verdict used by the API and the web app."""
    if not (p_value < 0.05 and accuracy > baseline):
        return {"level": "no_signal", "reliable": False,
                "message": ("On the held-out test data this model is NOT statistically better than always guessing "
                            "the majority class. Its predictions should not be trusted.")}
    if accuracy < 0.80:
        return {"level": "limited", "reliable": True,
                "message": "The model is better than chance on the test data, but it still makes many mistakes."}
    return {"level": "good", "reliable": True,
            "message": "The model performs well on the test data. It can still be wrong on news that differs from its training data."}


def evaluate_on_test(model, vectorizer, train_df, test_df):
    X_test = vectorizer.transform(test_df["news"])
    y_test = test_df["label_id"].to_numpy()
    y_pred = model.predict(X_test)
    positive_col = list(model.classes_).index(1)
    proba_real = model.predict_proba(X_test)[:, positive_col]

    accuracy = accuracy_score(y_test, y_pred)
    n, correct = len(y_test), int((y_pred == y_test).sum())
    majority_id = int(train_df["label_id"].value_counts().idxmax())
    baseline = float((y_test == majority_id).mean())
    p_value = float(binomtest(correct, n, p=baseline, alternative="greater").pvalue)
    low, high = wilson_interval(correct, n)
    train_acc = float(accuracy_score(train_df["label_id"], model.predict(vectorizer.transform(train_df["news"]))))
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in matrix.ravel())

    report = {
        "test_size": n,
        "accuracy": float(accuracy),
        "accuracy_95ci": [round(low, 4), round(high, 4)],
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "positive_class": "REAL",
        "macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba_real)),
        "train_accuracy": train_acc,
        "overfit_gap_train_minus_test": float(train_acc - accuracy),
        "majority_baseline_accuracy": baseline,
        "p_value_vs_baseline": p_value,
        "better_than_baseline": bool(p_value < 0.05 and accuracy > baseline),
        "confusion_matrix": {"labels": CLASS_NAMES, "matrix": matrix.tolist(),
                             "TN_fake_as_fake": tn, "FP_fake_as_real": fp, "FN_real_as_fake": fn, "TP_real_as_real": tp},
        "classification_report": classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "reliability": reliability_from(accuracy, p_value, baseline),
    }
    return report, y_pred, proba_real


def plot_confusion_matrix(matrix, path=None):
    path = path or FIGURES_DIR / "confusion_matrix.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_title("Confusion matrix (test set)")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center",
                    color="white" if matrix[i, j] > matrix.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_evaluation(report, path=EVALUATION_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
