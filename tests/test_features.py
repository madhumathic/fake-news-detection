import tempfile
import unittest
from pathlib import Path

from src.features.tfidf_features import build_vectorizer, fit_vectorizer, load_vectorizer, save_vectorizer


class TfidfTests(unittest.TestCase):
    TRAIN = ["the senate passed the budget today", "officials announced a new budget plan",
             "shocking secret exposed by anonymous insiders", "viral hoax about a secret scandal"]

    def test_default_parameters(self):
        vec = build_vectorizer()
        self.assertEqual(vec.max_df, 0.7)
        self.assertEqual(vec.max_features, 50000)
        self.assertEqual(vec.ngram_range, (1, 2))
        self.assertEqual(vec.stop_words, "english")

    def test_ngram_range_can_be_overridden_from_a_saved_list(self):
        self.assertEqual(build_vectorizer(ngram_range=[1, 1]).ngram_range, (1, 1))

    def test_vectorizer_learns_only_from_training_text(self):
        vec, matrix = fit_vectorizer(self.TRAIN, ngram_range=(1, 1))
        self.assertEqual(matrix.shape[0], len(self.TRAIN))
        self.assertNotIn("zebra", vec.vocabulary_)
        test_matrix = vec.transform(["a zebra walked past the budget"])
        self.assertEqual(len(vec.vocabulary_), test_matrix.shape[1])      # vocabulary unchanged by test text
        self.assertNotIn("zebra", vec.vocabulary_)

    def test_save_and_load_roundtrip(self):
        vec, _ = fit_vectorizer(self.TRAIN, ngram_range=(1, 1))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "v.pkl"
            save_vectorizer(vec, path)
            loaded = load_vectorizer(path)
        self.assertEqual(vec.vocabulary_, loaded.vocabulary_)
        self.assertEqual((vec.transform(["budget plan"]) != loaded.transform(["budget plan"])).nnz, 0)


if __name__ == "__main__":
    unittest.main()
