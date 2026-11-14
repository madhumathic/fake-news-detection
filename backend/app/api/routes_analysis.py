from flask import Blueprint, abort, jsonify, send_file

from backend.app.services import analysis_service

bp = Blueprint("analysis", __name__, url_prefix="/api")


@bp.get("/model/performance")
def model_performance():
    return jsonify(analysis_service.get_performance())


@bp.get("/dataset/summary")
def dataset_summary():
    return jsonify(analysis_service.get_dataset_summary())


@bp.get("/analysis/figures/<name>")
def figure(name):
    path = analysis_service.get_figure_path(name)
    if path is None:
        abort(404)
    return send_file(path, mimetype="image/png")
