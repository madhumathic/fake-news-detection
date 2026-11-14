"""Phases 3-4: load the raw dataset, validate it, clean it and split it into train/test.

Run from the project root:   python scripts/prepare_data.py
Creates: reports/data_report.json, data/processed/cleaned_data.csv,
         data/processed/train_test_split/train.csv and test.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

import json

from src.config import CLEAN_PATH, DATA_REPORT_PATH, SPLIT_DIR, TEST_PATH, TRAIN_PATH
from src.data.load_data import DatasetError, load_raw_dataset
from src.data.preprocess import preprocess_dataframe, split_dataset
from src.data.validate_data import validate_dataframe


def main():
    print("=" * 66 + "\n Data preparation\n" + "=" * 66)
    raw, info = load_raw_dataset()
    print(f"[1/4] Loaded {len(raw):,} records from: {', '.join(f['file'] for f in info['files'])}")

    print("[2/4] Validating (missing values, duplicates, imbalance, leakage checks) ...")
    report = validate_dataframe(raw, info)
    DATA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[3/4] Cleaning ...")
    clean, steps = preprocess_dataframe(raw)
    report["cleaning_steps"] = steps
    with open(DATA_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)

    print("[4/4] Stratified 80/20 split (random_state=42) ...")
    train, test = split_dataset(clean)
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("-" * 66)
    print(f" Records before cleaning : {steps['rows_in']:,}")
    for key in ("removed_unlabelled", "removed_missing_text", "removed_empty_text", "removed_too_short",
                "removed_conflicting_labels", "removed_duplicates"):
        print(f"   {key.replace('_', ' '):<30}{steps[key]:>8,}")
    print(f" Records after cleaning  : {steps['rows_out']:,}   {clean['label'].value_counts().to_dict()}")
    print(f" Train / test            : {len(train):,} / {len(test):,}")
    if report.get("text_signal_warning"):
        sig = report["word_label_signal"]
        print("-" * 66)
        print(" WARNING: the text shows almost no link to the label.")
        print(f"   {sig['significant_words_p_lt_0.05']} of {sig['vocabulary_tested']} words are statistically linked to the label;")
        print(f"   about {sig['expected_by_chance']:.0f} would be expected from pure chance. See reports/data_report.json.")
    print("-" * 66 + "\n Next:  python scripts/run_eda.py\n" + "=" * 66)


if __name__ == "__main__":
    try:
        main()
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
