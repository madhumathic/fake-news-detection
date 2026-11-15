"""Flask application factory.

Run from the PROJECT ROOT:   python -m app.main
API:  http://127.0.0.1:5000/api/health
If frontend/dist exists (after `npm run build`) the same server also serves the web app.
"""
import os
from pathlib import Path
import sys

CURRENT_FILE = Path(__file__).resolve()

PROJECT_ROOT = next(
    (p for p in CURRENT_FILE.parents if (p / "src" / "config.py").is_file()),
    CURRENT_FILE.parents[1]
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from flask import Flask, jsonify, request, send_from_directory
from werkzeug.exceptions import HTTPException

from app.api import routes_analysis, routes_history, routes_prediction
from app.database.init_db import init_db
from app.schemas.prediction import ApiError
from src.config import ROOT
from src.models.predict import ModelNotFoundError

FRONTEND_DIST = ROOT / "frontend" / "dist"
DEFAULT_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"


def error_response(code, message, status):
    return jsonify({"error": {"code": code, "message": message}}), status


def create_app():
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024  # 1 MB request body limit
    allowed = {o.strip() for o in os.environ.get("CORS_ORIGINS", DEFAULT_ORIGINS).split(",") if o.strip()}

    for module in (routes_prediction, routes_history, routes_analysis):
        app.register_blueprint(module.bp)

    @app.after_request
    def add_cors_headers(response):
        origin = request.headers.get("Origin")
        if origin and (origin in allowed or "*" in allowed):
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.errorhandler(ApiError)
    def handle_api_error(exc):
        return error_response(exc.code, exc.message, exc.status)

    @app.errorhandler(ModelNotFoundError)
    def handle_model_missing(exc):
        return error_response("model_unavailable", str(exc), 503)

    @app.errorhandler(ValueError)
    def handle_value_error(exc):
        return error_response("invalid_input", str(exc), 400)

    @app.errorhandler(HTTPException)
    def handle_http_error(exc):
        # unknown URLs outside /api: serve the single-page app when it has been built
        if exc.code == 404 and not request.path.startswith("/api") and (FRONTEND_DIST / "index.html").exists():
            return send_from_directory(FRONTEND_DIST, "index.html")
        codes = {400: "bad_request", 404: "not_found", 405: "method_not_allowed", 413: "input_too_large"}
        return error_response(codes.get(exc.code, "http_error"), exc.description, exc.code or 500)

    @app.errorhandler(Exception)
    def handle_unexpected(exc):
        app.logger.exception("Unexpected error")
        return error_response("server_error", "An unexpected error occurred. Please try again.", 500)

    if (FRONTEND_DIST / "index.html").exists():
        @app.get("/")
        def serve_index():
            return send_from_directory(FRONTEND_DIST, "index.html")

        @app.get("/assets/<path:filename>")
        def serve_assets(filename):
            return send_from_directory(FRONTEND_DIST / "assets", filename)

    init_db()
    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
