
"""Glue between the API and the saved ML model."""

import sys
from pathlib import Path

# Add the main project folder to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas.prediction import validate_prediction_payload
from app.services import analysis_service, history_service
from src.models.predict import Predictor

_predictor = Predictor()


def get_predictor():
    return _predictor


def model_status():
    return {
        "files_present": _predictor.files_exist(),
        "loaded": _predictor.is_loaded,
    }


def predict_and_store(payload):
    """Validate, predict, save to history, and return the API response."""

    news = validate_prediction_payload(payload)

    result = _predictor.predict(news)

    prediction = (
        "REAL NEWS" if result["label"] == "REAL" else "FAKE NEWS"
    )

    entry_id = history_service.add_entry(
        news,
        prediction,
        result["confidence"],
    )

    return {
        "id": entry_id,
        "prediction": prediction,
        "label": result["label"],
        "confidence": result["confidence"],
        "probabilities": result["probabilities"],
        "words": result["words"],
        "reliability": analysis_service.get_reliability(),
        "disclaimer": (
            "This prediction is based on patterns learned from the "
            "training dataset. It is not a definitive fact-check."
        ),
    }
