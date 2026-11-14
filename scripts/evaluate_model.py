"""Phase 8: evaluate the saved model on the unseen test set. Run: python scripts/evaluate_model.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

import json

import joblib

from src.analysis.error_analysis import analyze_errors, save_error_analysis
from src.analysis.model_analysis import label_shuffle_check, top_terms
from src.config import METADATA_PATH, MODEL_PATH, TEST_PATH, TRAIN_PATH, VECTORIZER_PATH
from src.data.load_data import DatasetError, load_processed
from src.models.evaluate import evaluate_on_test, plot_confusion_matrix, save_evaluation


def main():
    if not (MODEL_PATH.exists() and VECTORIZER_PATH.exists()):
        raise DatasetError("Model files not found. Run first: python scripts/train_models.py")
    train = load_processed(TRAIN_PATH, "Run first: python scripts/prepare_data.py")
    test = load_processed(TEST_PATH, "Run first: python scripts/prepare_data.py")
    model, vectorizer = joblib.load(MODEL_PATH), joblib.load(VECTORIZER_PATH)
    with open(METADATA_PATH, encoding="utf-8") as fh:
        meta = json.load(fh)

    print("=" * 66 + f"\n Evaluation of: {meta['model_name']}\n" + "=" * 66)
    report, y_pred, proba_real = evaluate_on_test(model, vectorizer, train, test)
    report["model_name"] = meta["model_name"]
    report["top_terms"] = top_terms(model, vectorizer)
    print("Running label-shuffle sanity check (3 extra trainings) ...")
    report["label_shuffle_check"] = label_shuffle_check(meta["model_name"], meta["best_params"], train, test)
    save_evaluation(report)
    plot_confusion_matrix(report["confusion_matrix"]["matrix"])
    save_error_analysis(analyze_errors(test, y_pred, proba_real))

    cm = report["confusion_matrix"]
    lo, hi = report["accuracy_95ci"]
    print(f"\n Test articles          : {report['test_size']:,}")
    print(f" Accuracy               : {report['accuracy']:.4f}   (95% CI {lo:.3f} - {hi:.3f})")
    print(f" Precision / Recall / F1: {report['precision']:.4f} / {report['recall']:.4f} / {report['f1_score']:.4f}   (REAL = positive class)")
    print(f" ROC-AUC                : {report['roc_auc']:.4f}   (0.5 = random guessing)")
    print(f" Train accuracy         : {report['train_accuracy']:.4f}   (gap to test: {report['overfit_gap_train_minus_test']:.4f})")
    print(f" Majority baseline      : {report['majority_baseline_accuracy']:.4f}   p-value of model vs baseline: {report['p_value_vs_baseline']:.4f}")
    print(f" Shuffled-label models  : {report['label_shuffle_check']['test_accuracy_with_shuffled_labels']}")
    print(f" Confusion matrix       : TN={cm['TN_fake_as_fake']}  FP={cm['FP_fake_as_real']}  FN={cm['FN_real_as_fake']}  TP={cm['TP_real_as_real']}")
    print(f"\n Verdict: {report['reliability']['message']}")
    print("\n Saved: reports/evaluation_report.json, reports/error_analysis.json, reports/figures/confusion_matrix.png")
    print(" Next:  python scripts/generate_reports.py   then   python -m backend.app.main")


if __name__ == "__main__":
    try:
        main()
    except DatasetError as exc:
        print(f"\nERROR: {exc}")
        sys.exit(1)
