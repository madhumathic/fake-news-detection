import json
import unittest
from pathlib import Path

from src.config import LABEL_MAPPING_PATH, MODEL_PATH, VECTORIZER_PATH
from src.models.predict import ModelNotFoundError, Predictor

ARTICLE = ("The finance committee said on Tuesday that the new budget bill would be reviewed by lawmakers next month, "
           "according to several officials who spoke to reporters about the plan and its expected cost.")


class PredictorWithoutModelTests(unittest.TestCase):
    def test_missing_model_files_raise_clear_error(self):
        predictor = Predictor(Path("nope/model.pkl"), Path("nope/vec.pkl"))
        with self.assertRaises(ModelNotFoundError) as ctx:
            predictor.predict(ARTICLE)
        self.assertIn("train", str(ctx.exception).lower())


@unittest.skipUnless(MODEL_PATH.exists() and VECTORIZER_PATH.exists(), "trained model not found (run scripts/train_models.py)")
class PredictorWithSavedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = Predictor().load()

    def test_prediction_structure(self):
        result = self.predictor.predict(ARTICLE)
        self.assertIn(result["label"], {"REAL", "FAKE"})
        self.assertTrue(50.0 <= result["confidence"] <= 100.0)
        self.assertAlmostEqual(sum(result["probabilities"].values()), 100.0, delta=0.1)

    def test_too_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            self.predictor.predict("only a few words")

    def test_prediction_is_deterministic(self):
        self.assertEqual(self.predictor.predict(ARTICLE), self.predictor.predict(ARTICLE))

    def test_label_mapping_file_matches_model_classes(self):
        mapping = json.loads(LABEL_MAPPING_PATH.read_text(encoding="utf-8"))
        self.assertEqual(mapping["label_to_id"], {"FAKE": 0, "REAL": 1})
        self.assertEqual(sorted(int(c) for c in self.predictor.model.classes_), [0, 1])

    def test_agency_tag_does_not_change_prediction(self):
        tagged = "LONDON (Reuters) - " + ARTICLE
        self.assertEqual(self.predictor.predict(tagged)["label"], self.predictor.predict(ARTICLE)["label"])


if __name__ == "__main__":
    unittest.main()
