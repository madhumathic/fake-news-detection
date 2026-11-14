from flask import Blueprint, jsonify, request

from app.schemas.prediction import ApiError
from app.services import history_service

bp = Blueprint("history", __name__, url_prefix="/api")


@bp.get("/history")
def get_history():
    try:
        limit = int(request.args.get("limit", 100))
        offset = int(request.args.get("offset", 0))
    except ValueError:
        raise ApiError("invalid_query", "limit and offset must be whole numbers.", 400)
    items, total = history_service.list_entries(limit, offset)
    return jsonify({"items": items, "total": total})


@bp.delete("/history")
def clear_history():
    return jsonify({"deleted": history_service.clear_entries()})
