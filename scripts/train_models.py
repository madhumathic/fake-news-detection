"""Phases 6-7: compare models, tune the best one, train the final model and save it.

Run: python scripts/train_models.py [--with-rf] [--skip-tuning]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

import argparse

from src.config import TEST_PATH, TRAIN_PATH
from src.data.load_data import DatasetError, load_processed
from src.models.compare import compare_models, save_comparison
from src.models.train import save_artifacts, train_final_model
from src.models.tune import save_tuning, tune_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-rf", action="store_true", help="also compare Random Forest (slower)")
    parser.add_argument("--skip-tuning", action="store_true", help="train with default hyperparameters")
    args = parser.parse_args()

    train = load_processed(TRAIN_PATH, "Run first: python scripts/prepare_data.py")
    test = load_processed(TEST_PATH, "Run first: python scripts/prepare_data.py")

    print("=" * 66 + "\n Model selection and training\n" + "=" * 66)
    print(f"[1/3] Comparing models (5-fold CV on {len(train):,} training articles) ...")
    table, best_name = compare_models(train, include_random_forest=args.with_rf)
    save_comparison(table)
    baseline = table[table["is_baseline"]].iloc[0]
    best = table[table["selected"]].iloc[0]
    if best["beats_baseline_by_2_std"]:
        note = f"{best_name} was selected (highest mean CV F1 = {best['cv_f1_mean']:.4f}) and beats the majority baseline."
    else:
        note = (f"{best_name} was selected (highest mean CV F1 = {best['cv_f1_mean']:.4f}) but NO model beats the "
                f"majority-class baseline (CV accuracy {baseline['cv_accuracy_mean']:.4f}) by more than 2 standard deviations.")
    print(f"  Selected: {best_name}\n  {note}")

    if args.skip_tuning:
        params = {}
    else:
        print(f"[2/3] Tuning {best_name} (grid search, 5-fold CV, training data only) ...")
        params, _, results = tune_model(best_name, train)
        save_tuning(results)

    print("[3/3] Training the final model and saving it ...")
    model, vectorizer = train_final_model(best_name, params, train)
    meta = save_artifacts(model, vectorizer, best_name, params, note, train, test)
    print("  Saved: models/best_model.pkl, models/tfidf_vectorizer.pkl, models/model_metadata.json, models/label_mapping.json")
    print("=" * 66 + f"\n Next:  python scripts/evaluate_model.py\n" + "=" * 66)


if __name__ == "__main__":
    try:
        main()
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
