"""Phase 4 - Data preparation: text cleaning and dataset cleaning.

clean_text() is the ONE cleaning function used both for the training data and
for every article typed into the website, so training and prediction always see
identically processed text.
"""
import html
import re

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import MIN_WORDS, RANDOM_STATE, TEST_SIZE
from src.data.load_data import DatasetError, detect_columns, infer_label_mapping

HTML_TAG_RE = re.compile(r"</?[A-Za-z!][^>]*>")
URL_RE = re.compile(r"(?:https?://|www\.)\S+|\bpic\.twitter\.com/\S+", re.IGNORECASE)
# "WASHINGTON (Reuters) - " style opening of agency stories
DATELINE_RE = re.compile(r"(?:[A-Z][A-Z .,'/\-]{0,40})?\(Reuters\)\s*[-\u2013\u2014:]*\s*")
SOURCE_RE = re.compile(r"\bReuters\b", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value):
    """Remove HTML tags, decode entities, remove URLs and the news-agency tag, collapse whitespace.

    Why remove the agency tag? In this dataset almost every REAL article contains
    "CITY (Reuters) -". A model could learn that single word instead of the writing
    style of real vs. fake news, and then fail on real articles from other publishers.
    """
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return ""
    text = HTML_TAG_RE.sub(" ", str(value))
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = DATELINE_RE.sub(" ", text)
    text = SOURCE_RE.sub(" ", text)
    return WHITESPACE_RE.sub(" ", text).strip()


def word_count(text):
    return len(str(text).split())


def preprocess_dataframe(df, min_words=MIN_WORDS, seed=RANDOM_STATE):
    """Clean the raw dataframe. Returns (clean_df, steps).

    Rows are processed as whole rows, so title / text / label never get misaligned.
    `steps` records how many rows each step removed (used in the report).
    Metadata columns (source, author, category, date ...) are kept for EDA but
    are NOT used as model features.
    """
    cols = detect_columns(df)
    if cols["text"] is None or cols["label"] is None:
        raise DatasetError("Could not find a text column and a label column in the dataset.")

    work = df.copy()
    steps = {"rows_in": int(len(work))}

    mapping = infer_label_mapping(work[cols["label"]])
    work = work[work[cols["label"]].isin(mapping)].copy()
    work["label"] = work[cols["label"]].map(mapping)
    steps["removed_unlabelled"] = steps["rows_in"] - len(work)

    n = len(work)
    work = work.dropna(subset=[cols["text"]])
    steps["removed_missing_text"] = n - len(work)

    work["text_clean"] = work[cols["text"]].map(clean_text)
    if cols["title"]:
        work["title_clean"] = work[cols["title"]].map(clean_text)
    else:
        work["title_clean"] = ""

    n = len(work)
    work = work[work["text_clean"] != ""]
    steps["removed_empty_text"] = n - len(work)

    n = len(work)
    work = work[work["text_clean"].map(word_count) >= min_words]
    steps["removed_too_short"] = n - len(work)

    work["news"] = (work["title_clean"] + " " + work["text_clean"]).str.strip()

    n = len(work)  # same article labelled both ways -> contradictory, drop every copy
    n_labels = work.groupby("news")["label"].nunique()
    work = work[~work["news"].isin(n_labels[n_labels > 1].index)]
    steps["removed_conflicting_labels"] = n - len(work)

    n = len(work)
    work = work.drop_duplicates(subset=["news"]).drop_duplicates(subset=["text_clean"])
    steps["removed_duplicates"] = n - len(work)

    if work.empty or work["label"].nunique() < 2:
        raise DatasetError("After cleaning, one or both classes have no articles left.")

    keep_meta = [c for c in work.columns
                 if c not in {cols["text"], cols["title"], cols["label"], "text_clean", "title_clean", "news", "label"}]
    out = pd.DataFrame({
        "title": work["title_clean"],
        "text": work["text_clean"],
        "news": work["news"],
        "label": work["label"],
    })
    out["label_id"] = out["label"].map({"FAKE": 0, "REAL": 1})
    for c in keep_meta:
        out[c] = work[c].values
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    steps["rows_out"] = int(len(out))
    steps["label_mapping_used"] = {str(k): v for k, v in mapping.items()}
    steps["columns_detected"] = cols
    return out, steps


def split_dataset(df, test_size=TEST_SIZE, seed=RANDOM_STATE):
    """Stratified train/test split. The test set is never used for training or tuning."""
    return train_test_split(df, test_size=test_size, random_state=seed, stratify=df["label_id"])
