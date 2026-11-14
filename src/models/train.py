"""Phase 7 - Final model training and saving.

The vectorizer is fitted on the training set only. The classifier is trained on
the resulting features. Both are saved separately (as the project requires).
"""
import json
from datetime import datetime

import joblib
import sklearn
from sklearn.calibration import CalibratedClassifierCV

from src.config import (LABEL_MAPPING, LABEL_MAPPING_PATH, METADATA_PATH, MODEL_PATH, MODELS_DIR,
                        RANDOM_STATE, TFIDF_PARAMS, VECTORIZER_PATH)
from src.features.tfidf_features import fit_vectorizer, save_vectorizer
from src.models.compare import get_candidates


def train_final_model(model_name, best_params, train_df):
    """Fit vectorizer + classifier with the tuned parameters. Returns (model, vectorizer)."""
    tfidf_over = {k.split("__", 1)[1]: v for k, v in best_params.items() if k.startswith("tfidf__")}
    clf_params = {k.split("__", 1)[1]: v for k, v in best_params.items() if k.startswith("clf__")}
    vectorizer, X_train = fit_vectorizer(train_df["news"], **tfidf_over)
    clf = get_candidates(include_random_forest=True)[model_name].set_params(**clf_params)
    if not hasattr(clf, "predict_proba"):  # LinearSVC has no probabilities -> calibrate
        clf = CalibratedClassifierCV(clf, cv=5)
    clf.fit(X_train, train_df["label_id"])
    return clf, vectorizer


def save_artifacts(model, vectorizer, model_name, best_params, comparison_note, train_df, test_df):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    save_vectorizer(vectorizer)
    with open(LABEL_MAPPING_PATH, "w", encoding="utf-8") as fh:
        json.dump({"label_to_id": LABEL_MAPPING, "id_to_label": {str(v): k for k, v in LABEL_MAPPING.items()}}, fh, indent=2)
    metadata = {
        "model_name": model_name,
        "model_class": type(model).__name__,
        "best_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in best_params.items()},
        "tfidf_params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in {**TFIDF_PARAMS, **{
            k.split("__", 1)[1]: v for k, v in best_params.items() if k.startswith("tfidf__")}}.items()},
        "features_used": "title + text only (metadata columns such as source, author, category and date are NOT used)",
        "train_size": int(len(train_df)),
        "test_size": int(len(test_df)),
        "random_state": RANDOM_STATE,
        "selection_note": comparison_note,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)
    return metadata
