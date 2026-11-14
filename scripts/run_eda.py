"""Phase 5: exploratory data analysis. Run: python scripts/run_eda.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

from src.analysis.eda import run_eda, save_eda_summary
from src.config import CLEAN_PATH, FIGURES_DIR
from src.data.load_data import DatasetError, load_processed


def main():
    df = load_processed(CLEAN_PATH, "Run first: python scripts/prepare_data.py")
    summary = run_eda(df)
    save_eda_summary(summary)
    print(f"EDA finished: {summary['records']:,} articles analysed. Figures saved in {FIGURES_DIR.relative_to(FIGURES_DIR.parents[1])}/")
    print("Next:  python scripts/train_models.py")


if __name__ == "__main__":
    try:
        main()
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
