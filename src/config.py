"""Central configuration: every path and constant used by the project.

All paths are relative to the project root, so the project works on any
computer (no hard-coded C:\\Users\\... paths).
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
SPLIT_DIR = PROCESSED_DIR / "train_test_split"
CLEAN_PATH = PROCESSED_DIR / "cleaned_data.csv"
TRAIN_PATH = SPLIT_DIR / "train.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "best_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
LABEL_MAPPING_PATH = MODELS_DIR / "label_mapping.json"

REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
PHASES_DIR = REPORTS_DIR / "phases"
DATA_REPORT_PATH = REPORTS_DIR / "data_report.json"
EDA_SUMMARY_PATH = REPORTS_DIR / "eda_summary.json"
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"
TUNING_PATH = REPORTS_DIR / "tuning_results.csv"
EVALUATION_PATH = REPORTS_DIR / "evaluation_report.json"
ERROR_ANALYSIS_PATH = REPORTS_DIR / "error_analysis.json"
FINAL_REPORT_PATH = REPORTS_DIR / "final_report.md"

DB_PATH = ROOT / "database" / "history.db"

# Reproducibility
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Preprocessing / input limits
MIN_WORDS = 10              # articles with fewer words are removed / rejected
MAX_INPUT_CHARS = 50_000    # largest article accepted by the API

# TF-IDF settings (defaults; the tuning step may change ngram_range)
TFIDF_PARAMS = {
    "stop_words": "english",
    "max_df": 0.7,
    "max_features": 50_000,
    "ngram_range": (1, 2),
}

# Internal class names. 0 = FAKE, 1 = REAL (also saved to models/label_mapping.json)
LABEL_MAPPING = {"FAKE": 0, "REAL": 1}
