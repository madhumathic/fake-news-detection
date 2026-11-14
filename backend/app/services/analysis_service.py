"""Reads the reports written by the ML pipeline and prepares them for the API."""
import json
import re

import pandas as pd

from src.config import (COMPARISON_PATH, DATA_REPORT_PATH, EDA_SUMMARY_PATH, ERROR_ANALYSIS_PATH, EVALUATION_PATH,
                        FIGURES_DIR, METADATA_PATH, TUNING_PATH)

FIGURE_NAME_RE = re.compile(r"^[A-Za-z0-9_\-]+\.png$")


def read_json(path):
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def read_csv_records(path):
    if not path.exists():
        return []
    try:
        return json.loads(pd.read_csv(path).to_json(orient="records"))
    except Exception:
        return []


def get_reliability():
    """Verdict about the saved model, derived from the held-out test evaluation."""
    evaluation = read_json(EVALUATION_PATH)
    if not evaluation:
        return {"level": "unknown", "reliable": False,
                "message": "The model has not been evaluated yet. Run: python scripts/evaluate_model.py"}
    return evaluation["reliability"]


def get_performance():
    evaluation = read_json(EVALUATION_PATH)
    return {
        "metadata": read_json(METADATA_PATH),
        "evaluation": evaluation,
        "comparison": read_csv_records(COMPARISON_PATH),
        "tuning": read_csv_records(TUNING_PATH),
        "error_analysis": read_json(ERROR_ANALYSIS_PATH),
        "reliability": get_reliability(),
        "available": evaluation is not None,
    }


def get_dataset_summary():
    report = read_json(DATA_REPORT_PATH)
    return {"data_report": report, "eda": read_json(EDA_SUMMARY_PATH), "available": report is not None,
            "figures": sorted(p.name for p in FIGURES_DIR.glob("*.png")) if FIGURES_DIR.exists() else []}


def get_figure_path(name):
    """Safe lookup of a figure file (no path traversal)."""
    if not FIGURE_NAME_RE.match(name):
        return None
    path = FIGURES_DIR / name
    return path if path.is_file() else None
