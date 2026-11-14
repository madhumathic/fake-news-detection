from flask import Blueprint, jsonify, request

from backend.app.database.database import db_path
from backend.app.schemas.prediction import ApiError
from backend.app.services import prediction_service

bp = Blueprint("prediction", __name__, url_prefix="/api")


@bp.get("/health")
def health():
    status = prediction_service.model_status()
    return jsonify({"status": "ok" if status["files_present"] else "model_missing",
                    "model_files_present": status["files_present"], "database": str(db_path().name)})


@bp.post("/predict")
def predict():
    payload = request.get_json(silent=True)
    if payload is None:
        raise ApiError("invalid_json", "The request body must be valid JSON such as {\"news\": \"...\"}.", 400)
    return jsonify(prediction_service.predict_and_store(payload))
