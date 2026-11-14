import unittest

import pandas as pd

from src.data.preprocess import clean_text, preprocess_dataframe, split_dataset
from tests.helpers import make_articles


class CleanTextTests(unittest.TestCase):
    def test_removes_html_and_urls(self):
        out = clean_text("<p>Hello <b>world</b> see https://example.com/a?b=1 and www.site.org now</p>")
        self.assertEqual(out, "Hello world see and now")

    def test_decodes_entities_and_collapses_whitespace(self):
        self.assertEqual(clean_text("  A &amp; B \n\n  C\t D "), "A & B C D")

    def test_removes_agency_tag(self):
        out = clean_text("WASHINGTON (Reuters) - The senate voted. Reuters reported it.")
        self.assertNotIn("Reuters", out)
        self.assertTrue(out.startswith("The senate voted."))

    def test_missing_values_become_empty_string(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(float("nan")), "")


class PreprocessDataFrameTests(unittest.TestCase):
    def build(self):
        df = make_articles(30)
        extra = [
            {"title": "t", "text": None, "subject": "x", "date": "d", "label": "FAKE"},              # missing text
            {"title": "t", "text": "   ", "subject": "x", "date": "d", "label": "FAKE"},             # blank
            {"title": "t", "text": "far too short", "subject": "x", "date": "d", "label": "REAL"},   # short
            df.iloc[0].to_dict(),                                                                    # duplicate
        ]
        conflict = df.iloc[1].to_dict()
        conflict["label"] = "REAL" if conflict["label"] == "FAKE" else "FAKE"                        # same text, other label
        return pd.concat([df, pd.DataFrame(extra + [conflict])], ignore_index=True)

    def test_each_cleaning_step_is_applied(self):
        clean, steps = preprocess_dataframe(self.build())
        self.assertEqual(steps["removed_missing_text"], 1)
        self.assertEqual(steps["removed_empty_text"], 1)
        self.assertEqual(steps["removed_too_short"], 1)
        self.assertEqual(steps["removed_conflicting_labels"], 2)
        self.assertEqual(steps["removed_duplicates"], 1)
        self.assertEqual(len(clean), steps["rows_out"])
        self.assertEqual(clean["news"].duplicated().sum(), 0)

    def test_title_text_label_stay_aligned(self):
        clean, _ = preprocess_dataframe(make_articles(30))
        fake = clean[clean["label"] == "FAKE"]
        self.assertTrue(fake["title"].str.contains("fake").all())
        self.assertTrue((clean["news"] == (clean["title"] + " " + clean["text"]).str.strip()).all())

    def test_label_ids_follow_mapping(self):
        clean, _ = preprocess_dataframe(make_articles(10))
        self.assertTrue((clean[clean["label"] == "FAKE"]["label_id"] == 0).all())
        self.assertTrue((clean[clean["label"] == "REAL"]["label_id"] == 1).all())

    def test_split_is_stratified_and_disjoint(self):
        clean, _ = preprocess_dataframe(make_articles(50))
        train, test = split_dataset(clean)
        self.assertEqual(len(train) + len(test), len(clean))
        self.assertEqual(set(train["news"]) & set(test["news"]), set())
        self.assertAlmostEqual(train["label_id"].mean(), test["label_id"].mean(), delta=0.05)


if __name__ == "__main__":
    unittest.main()
