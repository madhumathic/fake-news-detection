import unittest

from src.models.compare import compare_models
from src.models.evaluate import evaluate_on_test, reliability_from, wilson_interval
from src.models.train import train_final_model
from src.models.tune import tune_model
from src.data.preprocess import split_dataset
from tests.helpers import make_clean_frame


class ModelPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.test = split_dataset(make_clean_frame(60))

    def test_comparison_includes_required_models_and_a_baseline(self):
        table, best = compare_models(self.train, folds=3, verbose=False)
        names = set(table["model"])
        self.assertTrue({"Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"} <= names)
        self.assertTrue(table["is_baseline"].any())
        self.assertNotIn("Majority", best)
        self.assertEqual(int(table["selected"].sum()), 1)

    def test_tuning_returns_parameters_for_the_chosen_model(self):
        params, score, results = tune_model("Logistic Regression", self.train, folds=3, verbose=False)
        self.assertIn("clf__C", params)
        self.assertIn("tfidf__ngram_range", params)
        self.assertGreater(score, 0.5)
        self.assertGreater(len(results), 1)

    def test_each_model_type_can_be_trained_and_predicts_probabilities(self):
        for name in ("Logistic Regression", "Multinomial Naive Bayes", "Linear SVM"):
            model, vec = train_final_model(name, {}, self.train)
            proba = model.predict_proba(vec.transform(self.test["news"][:3]))
            self.assertEqual(proba.shape, (3, 2), name)
            self.assertAlmostEqual(float(proba[0].sum()), 1.0, places=5)

    def test_evaluation_report_has_all_metrics(self):
        model, vec = train_final_model("Logistic Regression", {}, self.train)
        report, y_pred, proba = evaluate_on_test(model, vec, self.train, self.test)
        for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc", "confusion_matrix",
                    "classification_report", "majority_baseline_accuracy", "p_value_vs_baseline", "reliability"):
            self.assertIn(key, report)
        self.assertGreater(report["accuracy"], 0.9)  # the synthetic corpus is easy by design
        self.assertEqual(sum(sum(r) for r in report["confusion_matrix"]["matrix"]), len(self.test))

    def test_statistics_helpers(self):
        low, high = wilson_interval(97, 100)
        self.assertLess(low, 0.97)
        self.assertGreater(high, 0.97)
        self.assertFalse(reliability_from(0.52, 0.4, 0.51)["reliable"])   # no better than chance
        self.assertEqual(reliability_from(0.97, 0.0, 0.51)["level"], "good")


if __name__ == "__main__":
    unittest.main()
