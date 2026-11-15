
"""Phase 9 - Prediction with the saved model."""

import json

import joblib
import numpy as np

from src.config import (
    LABEL_MAPPING_PATH,
    MIN_WORDS,
    MODEL_PATH,
    VECTORIZER_PATH,
)
from src.data.text_cleaning import clean_text


class ModelNotFoundError(Exception):
    """The trained model files do not exist."""


class Predictor:
    """Loads the trained model and predicts single articles."""

    def __init__(
        self,
        model_path=MODEL_PATH,
        vectorizer_path=VECTORIZER_PATH,
        mapping_path=LABEL_MAPPING_PATH,
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.mapping_path = mapping_path

        self.model = None
        self.vectorizer = None

        self.id_to_label = {0: "FAKE", 1: "REAL"}

    @property
    def is_loaded(self):
        return self.model is not None and self.vectorizer is not None

    def files_exist(self):
        return (
            self.model_path.exists()
            and self.vectorizer_path.exists()
        )

    def load(self):
        if self.is_loaded:
            return self

        if not self.files_exist():
            raise ModelNotFoundError(
                "Model files not found. Train the model first."
            )

        try:
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)

            if self.mapping_path.exists():
                with open(self.mapping_path, encoding="utf-8") as fh:
                    mapping = json.load(fh)

                self.id_to_label = {
                    int(k): v
                    for k, v in mapping["id_to_label"].items()
                }

        except Exception as exc:
            self.model = None
            self.vectorizer = None

            raise ModelNotFoundError(
                f"Model files could not be loaded: {type(exc).__name__}"
            ) from exc

        return self

    def predict(self, text):
        """Predict whether an article is REAL or FAKE."""

        self.load()

        cleaned = clean_text(text)

        if len(cleaned.split()) < MIN_WORDS:
            raise ValueError(
                f"Text too short: at least {MIN_WORDS} words are needed."
            )

        features = self.vectorizer.transform([cleaned])
        probabilities = self.model.predict_proba(features)[0]

        best = int(np.argmax(probabilities))
        class_id = int(self.model.classes_[best])

        return {
            "label": self.id_to_label[class_id],
            "confidence": round(
                float(probabilities[best]) * 100, 2
            ),
            "probabilities": {
                self.id_to_label[int(c)]: round(float(p) * 100, 2)
                for c, p in zip(
                    self.model.classes_,
                    probabilities,
                )
            },
            "words": len(cleaned.split()),
        }
