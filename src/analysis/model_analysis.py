"""Phase 8 - extra model diagnostics: influential terms and a label-shuffle sanity check."""
import numpy as np
from sklearn.metrics import accuracy_score

from src.config import RANDOM_STATE
from src.models.train import train_final_model


def _linear_weights(model):
    """Return a per-feature weight vector (positive = pushes towards REAL), or None."""
    base = model
    if hasattr(model, "calibrated_classifiers_"):           # calibrated LinearSVC
        base = model.calibrated_classifiers_[0].estimator
    if hasattr(base, "coef_"):
        return np.ravel(base.coef_)
    if hasattr(base, "feature_log_prob_"):                  # Naive Bayes
        return base.feature_log_prob_[1] - base.feature_log_prob_[0]
    return None


def top_terms(model, vectorizer, n=15):
    weights = _linear_weights(model)
    if weights is None:
        return None
    terms = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(weights)
    return {"towards_FAKE": terms[order[:n]].tolist(), "towards_REAL": terms[order[::-1][:n]].tolist()}


def label_shuffle_check(model_name, best_params, train_df, test_df, repeats=3):
    """Train the same model on SHUFFLED labels.

    If a model trained on random labels scores about the same on the test set as the
    real model, the real model has not found any genuine pattern.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    scores = []
    for _ in range(repeats):
        shuffled = train_df.copy()
        shuffled["label_id"] = rng.permutation(shuffled["label_id"].to_numpy())
        model, vec = train_final_model(model_name, best_params, shuffled)
        scores.append(float(accuracy_score(test_df["label_id"], model.predict(vec.transform(test_df["news"])))))
    return {"repeats": repeats, "test_accuracy_with_shuffled_labels": [round(s, 4) for s in scores],
            "mean": round(float(np.mean(scores)), 4)}
