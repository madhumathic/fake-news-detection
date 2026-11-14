"""Phase 6 - Model selection: fair comparison of candidate algorithms.

Fairness rules
* Every model gets the SAME training data, the SAME stratified folds and the SAME
  TF-IDF settings. The TF-IDF vectorizer sits INSIDE each pipeline, so in every
  fold it is fitted on that fold's training part only (no leakage).
* Only the TRAINING set is used. The test set is untouched until Phase 8.
* A majority-class baseline is included. A model that cannot beat it has learned nothing.
* The winner is the model with the highest mean cross-validated F1.
"""
import time

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.config import COMPARISON_PATH, CV_FOLDS, RANDOM_STATE
from src.features.tfidf_features import build_vectorizer

SCORING = {  # zero_division=0: the majority baseline never predicts the minority class
    "accuracy": make_scorer(accuracy_score),
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": make_scorer(recall_score, zero_division=0),
    "f1": make_scorer(f1_score, zero_division=0),
}


def get_candidates(include_random_forest=False):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(random_state=RANDOM_STATE),
    }
    if include_random_forest:
        models["Random Forest"] = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    return models


def make_pipeline(clf, **tfidf_overrides):
    return Pipeline([("tfidf", build_vectorizer(**tfidf_overrides)), ("clf", clf)])


def compare_models(train_df, include_random_forest=False, folds=CV_FOLDS, verbose=True):
    """Cross-validate every candidate on the training set. Returns (table, best_name)."""
    X, y = train_df["news"], train_df["label_id"]
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    candidates = {"Majority-class baseline": DummyClassifier(strategy="most_frequent"),
                  **get_candidates(include_random_forest)}
    rows = []
    for name, clf in candidates.items():
        start = time.time()
        res = cross_validate(make_pipeline(clf), X, y, cv=cv, scoring=SCORING,
                             return_train_score=True, n_jobs=-1)
        row = {"model": name, "is_baseline": name.startswith("Majority")}
        for m in SCORING:
            row[f"cv_{m}_mean"] = round(float(res[f"test_{m}"].mean()), 4)
            row[f"cv_{m}_std"] = round(float(res[f"test_{m}"].std()), 4)
        row["train_accuracy_mean"] = round(float(res["train_accuracy"].mean()), 4)
        row["overfit_gap"] = round(row["train_accuracy_mean"] - row["cv_accuracy_mean"], 4)
        row["fit_seconds"] = round(time.time() - start, 1)
        rows.append(row)
        if verbose:
            print(f"  {name:<26} CV accuracy {row['cv_accuracy_mean']:.4f} (+/- {row['cv_accuracy_std']:.4f})"
                  f"  F1 {row['cv_f1_mean']:.4f}  train acc {row['train_accuracy_mean']:.4f}")
    table = pd.DataFrame(rows)
    real = table[~table["is_baseline"]].sort_values("cv_f1_mean", ascending=False)
    best_name = real.iloc[0]["model"]
    baseline_acc = float(table[table["is_baseline"]]["cv_accuracy_mean"].iloc[0])
    best_row = real.iloc[0]
    table["selected"] = table["model"] == best_name
    table["beats_baseline_by_2_std"] = (table["cv_accuracy_mean"] - baseline_acc) > 2 * table["cv_accuracy_std"]
    table = pd.concat([table[table["selected"]], table[~table["selected"]].sort_values("cv_f1_mean", ascending=False)])
    return table.reset_index(drop=True), best_name


def save_comparison(table, path=COMPARISON_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
