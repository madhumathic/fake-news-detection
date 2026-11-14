"""Phase 6/7 - TF-IDF feature extraction."""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import TFIDF_PARAMS, VECTORIZER_PATH


def build_vectorizer(**overrides):
    """TfidfVectorizer with the project defaults (overridable, e.g. ngram_range)."""
    params = {**TFIDF_PARAMS, **overrides}
    if isinstance(params.get("ngram_range"), list):
        params["ngram_range"] = tuple(params["ngram_range"])
    return TfidfVectorizer(**params)


def fit_vectorizer(train_texts, **overrides):
    """Fit on TRAINING text only (the test set must never influence the vocabulary)."""
    vec = build_vectorizer(**overrides)
    matrix = vec.fit_transform(train_texts)
    if hasattr(vec, "stop_words_"):
        vec.stop_words_ = None  # introspection-only attribute that bloats the pickle
    return vec, matrix


def save_vectorizer(vec, path=VECTORIZER_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vec, path)


def load_vectorizer(path=VECTORIZER_PATH):
    return joblib.load(path)
