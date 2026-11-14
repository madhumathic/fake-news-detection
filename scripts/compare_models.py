"""Phase 6: compare candidate models with cross-validation. Run: python scripts/compare_models.py [--with-rf]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

import argparse

from src.config import TRAIN_PATH
from src.data.load_data import DatasetError, load_processed
from src.models.compare import compare_models, save_comparison


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-rf", action="store_true", help="also compare Random Forest (slower)")
    args = parser.parse_args()
    train = load_processed(TRAIN_PATH, "Run first: python scripts/prepare_data.py")
    print(f"Comparing models with 5-fold cross-validation on {len(train):,} training articles ...")
    table, best = compare_models(train, include_random_forest=args.with_rf)
    save_comparison(table)
    print(f"\nHighest mean CV F1 among real models: {best}. Table saved to reports/model_comparison.csv")


if __name__ == "__main__":
    try:
        main()
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
