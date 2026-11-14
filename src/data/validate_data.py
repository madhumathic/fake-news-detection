"""Phase 3/4 - Dataset validation and leakage checks.

validate_dataframe() inspects the RAW dataset and returns a report (dict) that
is saved to reports/data_report.json. It answers: how many records, which
columns, label distribution, missing values, duplicates, imbalance, irrelevant
columns and possible data leakage.
"""
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import chi2

from src.data.load_data import detect_columns, infer_label_mapping


def _is_text_dtype(series):
    return series.dtype == object or str(series.dtype).startswith("str")


def cramers_v(a, b):
    """Association (0..1) between two categorical series, and its p-value."""
    table = pd.crosstab(a, b)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0, 1.0
    chi2_val, p, _, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    v = float(np.sqrt(chi2_val / (n * (min(table.shape) - 1)))) if n else 0.0
    return v, float(p)


AGENCY_TAG_RE = re.compile(r"\((?:Reuters|AP|AFP|Associated Press)\)")


def date_shape(value):
    """'Dec 31, 2017' -> 'a 9, 9' ; 'December 31, 2017' -> 'A 9, 9' (format fingerprint)."""
    s = re.sub(r"\d+", "9", str(value))
    return re.sub(r"[A-Za-z]+", lambda m: "A" if len(m.group()) > 3 else "a", s)


def word_signal_check(texts, y, min_df=5, alpha=0.05):
    """How many individual words are statistically linked to the label?

    With pure noise about `alpha` (5%) of words look significant by chance alone.
    If the observed share is close to that, the text carries no label signal.
    """
    vec = CountVectorizer(min_df=min_df, binary=True)
    matrix = vec.fit_transform(texts)
    _, p_values = chi2(matrix, y)
    p_values = np.nan_to_num(p_values, nan=1.0)
    n_words = int(len(p_values))
    significant = int((p_values < alpha).sum())
    expected = alpha * n_words
    return {
        "vocabulary_tested": n_words,
        "significant_words_p_lt_0.05": significant,
        "expected_by_chance": round(expected, 1),
        "ratio_observed_to_chance": round(significant / expected, 2) if expected else None,
        "smallest_p_value": float(p_values.min()) if n_words else None,
    }


def validate_dataframe(df, source_info=None, max_categories=50):
    df = df.reset_index(drop=True)  # tolerate duplicate / unordered index labels
    cols = detect_columns(df)
    text_col, title_col, label_col = cols["text"], cols["title"], cols["label"]
    report = {
        "files": (source_info or {}).get("files", []),
        "records": int(len(df)),
        "n_columns": int(df.shape[1]),
        "columns": {str(c): str(df[c].dtype) for c in df.columns},
        "detected_columns": cols,
    }
    report["missing_values"] = {str(c): int(v) for c, v in df.isna().sum().items()}

    dup = {"full_row": int(df.duplicated().sum())}
    if text_col:
        dup["text"] = int(df[text_col].duplicated().sum())
    if title_col:
        dup["title"] = int(df[title_col].duplicated().sum())
    if title_col and text_col:
        dup["title_and_text"] = int(df.duplicated([title_col, text_col]).sum())
    report["duplicates"] = dup

    if text_col:
        s = df[text_col].fillna("").astype(str).str.strip()
        report["empty_text_records"] = int((s == "").sum())
        wc = s.str.split().str.len()
        report["words_per_article"] = {k: round(float(v), 2) for k, v in wc.describe().items()}
    if title_col:
        report["empty_title_records"] = int((df[title_col].fillna("").astype(str).str.strip() == "").sum())

    if not label_col:
        report["error"] = "No label column detected"
        return report

    raw_counts = df[label_col].value_counts(dropna=False)
    report["label_column"] = label_col
    report["raw_label_values"] = {str(k): int(v) for k, v in raw_counts.items()}
    mapping = infer_label_mapping(df[label_col])
    report["label_mapping"] = {str(k): v for k, v in mapping.items()}
    labels = df[label_col].map(mapping)
    counts = labels.value_counts()
    report["class_distribution"] = {k: int(v) for k, v in counts.items()}
    report["imbalance_ratio_majority_to_minority"] = round(float(counts.max() / counts.min()), 3)
    report["is_balanced"] = bool(counts.max() / counts.min() < 1.5)

    # ---- irrelevant columns & leakage ---------------------------------------
    irrelevant, leakage = [], []
    y = (labels == "REAL").astype(int)
    for c in df.columns:
        if c in (text_col, title_col, label_col):
            continue
        s = df[c]
        nunique = int(s.nunique(dropna=True))
        ratio = nunique / max(len(df), 1)
        entry = {"column": str(c), "unique_values": nunique, "missing": int(s.isna().sum())}
        if ratio > 0.5:
            entry["reason"] = "almost unique per row (identifier-like), not a useful feature"
            irrelevant.append(entry)
        elif c == cols["date"]:
            entry["reason"] = "metadata (a date); not used as a feature"
            irrelevant.append(entry)
        if 1 < nunique <= max_categories:
            v, p = cramers_v(s.fillna("(missing)"), y)
            entry_leak = {"column": str(c), "association_with_label_cramers_v": round(v, 4),
                          "p_value": round(p, 4), "suspicious": bool(v > 0.1 and p < 0.01)}
            if nunique <= 12:  # small tables are kept in the report as evidence
                table = pd.crosstab(s.fillna("(missing)"), labels)
                entry_leak["label_counts_by_value"] = {str(k): {str(c2): int(x) for c2, x in row.items()} for k, row in table.iterrows()}
            leakage.append(entry_leak)
        # is a value being MISSING related to the label?
        if s.isna().any():
            v, p = cramers_v(s.isna(), y)
            leakage.append({"column": f"{c} (is missing)", "association_with_label_cramers_v": round(v, 4),
                            "p_value": round(p, 4), "suspicious": bool(v > 0.1 and p < 0.01)})
    report["irrelevant_columns"] = irrelevant
    report["leakage_checks"] = leakage

    if text_col:
        wc = df[text_col].fillna("").astype(str).str.split().str.len()
        a, b = wc[y == 0], wc[y == 1]
        pooled = float(np.sqrt((a.var() + b.var()) / 2)) or 1.0
        report["length_leakage_check"] = {
            "mean_words_FAKE": round(float(a.mean()), 2), "mean_words_REAL": round(float(b.mean()), 2),
            "effect_size_cohens_d": round(float((a.mean() - b.mean()) / pooled), 4),
        }
    if cols["date"]:
        shape = df[cols["date"]].map(date_shape)
        v, p = cramers_v(shape, y)
        report["date_format_check"] = {"association_with_label_cramers_v": round(v, 4), "p_value": round(p, 4),
                                       "suspicious": bool(v > 0.1 and p < 0.01),
                                       "note": "writing style of the date (abbreviated vs full month) differs between classes"}
    if text_col:
        has_tag = df[text_col].fillna("").astype(str).str.contains(AGENCY_TAG_RE)
        report["agency_tag_check"] = {
            "share_of_articles_with_agency_tag_by_label": {
                "FAKE": round(float(has_tag[y == 0].mean()), 4), "REAL": round(float(has_tag[y == 1].mean()), 4)},
            "suspicious": bool(abs(has_tag[y == 0].mean() - has_tag[y == 1].mean()) > 0.2),
            "action": "agency tags such as '(Reuters)' are removed by clean_text() in training and prediction"}
    if title_col and text_col:
        sample = df[[title_col, text_col]].dropna().head(2000)
        inside = np.mean([str(t).lower().strip(" .") in str(x).lower() for t, x in zip(sample[title_col], sample[text_col])])
        report["title_contained_in_text_share"] = round(float(inside), 4)

    if text_col:
        news = (df[title_col].fillna("") + " " if title_col else "") + df[text_col].fillna("")
        report["word_label_signal"] = word_signal_check(news, y)
        ratio = report["word_label_signal"]["ratio_observed_to_chance"]
        report["text_signal_warning"] = bool(ratio is not None and ratio < 1.5)
    return report
