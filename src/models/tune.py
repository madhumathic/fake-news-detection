"""Phase 7 - Hyperparameter tuning (cross-validation on the TRAINING set only)."""
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src.config import CV_FOLDS, RANDOM_STATE, TUNING_PATH
from src.models.compare import get_candidates, make_pipeline

# Small, explainable grids. TF-IDF n-gram range is tuned for every model.
PARAM_GRIDS = {
    "Logistic Regression": {"clf__C": [0.1, 1, 10]},
    "Multinomial Naive Bayes": {"clf__alpha": [0.1, 0.5, 1.0]},
    "Linear SVM": {"clf__C": [0.1, 1, 10]},
    "Random Forest": {"clf__n_estimators": [100, 200]},
}
NGRAMS = [(1, 1), (1, 2)]


def tune_model(model_name, train_df, folds=CV_FOLDS, verbose=True):
    """Grid-search the chosen model. Returns (best_params, best_cv_f1, results_df)."""
    clf = get_candidates(include_random_forest=True)[model_name]
    grid = {"tfidf__ngram_range": NGRAMS, **PARAM_GRIDS[model_name]}
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(make_pipeline(clf), grid, scoring=make_scorer(f1_score, zero_division=0), cv=cv, n_jobs=-1, refit=False)
    search.fit(train_df["news"], train_df["label_id"])
    results = pd.DataFrame(search.cv_results_)
    keep = ["params", "mean_test_score", "std_test_score", "rank_test_score"]
    results = results[keep].rename(columns={"mean_test_score": "cv_f1_mean", "std_test_score": "cv_f1_std"})
    results["params"] = results["params"].astype(str)
    results = results.sort_values("rank_test_score").reset_index(drop=True)
    best = search.best_params_
    if verbose:
        print(f"  Best parameters: {best}  (CV F1 = {search.best_score_:.4f})")
    return best, float(search.best_score_), results


def save_tuning(results, path=TUNING_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(path, index=False)
