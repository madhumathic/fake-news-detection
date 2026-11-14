import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR
from src.data.load_data import DatasetError, detect_columns, find_raw_files, infer_label_mapping, load_raw_dataset
from src.data.validate_data import validate_dataframe
from tests.helpers import make_articles


class DatasetLoadingTests(unittest.TestCase):
    def test_empty_folder_gives_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(DatasetError) as ctx:
                load_raw_dataset(Path(tmp))
        self.assertIn("No CSV file found", str(ctx.exception))

    def test_single_file_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            make_articles(5).to_csv(Path(tmp) / "data.csv", index=False)
            df, info = load_raw_dataset(Path(tmp))
        self.assertEqual(len(df), 10)
        self.assertFalse(info["combined"])

    def test_two_files_are_combined_and_labelled_from_file_names(self):
        base = make_articles(5).drop(columns=["label"])
        with tempfile.TemporaryDirectory() as tmp:
            base.to_csv(Path(tmp) / "True.csv", index=False)
            base.to_csv(Path(tmp) / "Fake.csv", index=False)
            df, info = load_raw_dataset(Path(tmp))
        self.assertTrue(info["combined"])
        self.assertEqual(set(df["label"]), {"REAL", "FAKE"})

    def test_column_detection(self):
        cols = detect_columns(make_articles(3))
        self.assertEqual((cols["text"], cols["title"], cols["label"]), ("text", "title", "label"))

    def test_label_mapping_numeric_and_text(self):
        self.assertEqual(infer_label_mapping([0, 1, 0]), {0: "FAKE", 1: "REAL"})
        self.assertEqual(infer_label_mapping(["fake", "real"]), {"fake": "FAKE", "real": "REAL"})

    def test_unknown_label_value_is_rejected(self):
        with self.assertRaises(DatasetError):
            infer_label_mapping(["fake", "satire"])

    def test_actual_dataset_is_present(self):
        if not find_raw_files(RAW_DIR):
            self.skipTest("no dataset in data/raw")
        df, _ = load_raw_dataset()
        self.assertGreater(len(df), 100)


class ValidationTests(unittest.TestCase):
    def test_report_contains_required_sections(self):
        df = make_articles(30)
        df.loc[0, "text"] = None
        df = pd.concat([df, df.iloc[[1]]])  # one duplicate row
        report = validate_dataframe(df)
        for key in ("records", "columns", "missing_values", "duplicates", "class_distribution",
                    "imbalance_ratio_majority_to_minority", "leakage_checks", "word_label_signal"):
            self.assertIn(key, report)
        self.assertEqual(report["missing_values"]["text"], 1)
        self.assertEqual(report["duplicates"]["full_row"], 1)

    def test_perfectly_predictive_metadata_is_flagged_as_leakage(self):
        report = validate_dataframe(make_articles(60))
        subject = next(c for c in report["leakage_checks"] if c["column"] == "subject")
        self.assertTrue(subject["suspicious"])


if __name__ == "__main__":
    unittest.main()
