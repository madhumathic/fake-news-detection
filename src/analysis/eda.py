"""Phase 5 - Exploratory data analysis: figures + summary statistics."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from src.config import EDA_SUMMARY_PATH, FIGURES_DIR

FAKE_COLOR, REAL_COLOR = "#C2312D", "#0F7B4F"
COLORS = {"FAKE": FAKE_COLOR, "REAL": REAL_COLOR}
CATEGORY_NAMES = ("subject", "category", "topic", "section")
SOURCE_NAMES = ("source", "publisher", "outlet", "domain")


def _find(df, names):
    lower = {c.lower(): c for c in df.columns}
    return next((lower[n] for n in names if n in lower), None)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def _grouped_bars(df, column, title, name):
    table = pd.crosstab(df[column].fillna("(missing)"), df["label"])
    table = table.loc[table.sum(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(table))
    width = 0.4
    for i, label in enumerate(["FAKE", "REAL"]):
        if label in table:
            ax.bar(x + (i - 0.5) * width, table[label], width, label=label, color=COLORS[label])
    ax.set_xticks(x, table.index, rotation=30, ha="right")
    ax.set_ylabel("Articles")
    ax.set_title(title)
    ax.legend()
    _save(fig, name)
    return {str(k): {str(c): int(v) for c, v in row.items()} for k, row in table.iterrows()}


def run_eda(df):
    """df = cleaned dataset. Saves figures to reports/figures and returns the summary dict."""
    summary = {"records": int(len(df))}
    counts = df["label"].value_counts()
    summary["class_counts"] = {k: int(v) for k, v in counts.items()}

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=[COLORS[l] for l in counts.index])
    ax.bar_label(bars, labels=[f"{v:,}\n({v / counts.sum():.1%})" for v in counts.values])
    ax.set_ylabel("Articles")
    ax.set_title("Class distribution")
    ax.set_ylim(0, counts.max() * 1.18)
    _save(fig, "class_distribution.png")

    df = df.copy()
    df["words"] = df["text"].str.split().str.len()
    summary["words_per_article_by_label"] = {
        lab: {k: round(float(v), 2) for k, v in g["words"].describe().items()} for lab, g in df.groupby("label")}
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(df["words"].min(), df["words"].max(), 30)
    for lab in ["FAKE", "REAL"]:
        ax.hist(df.loc[df["label"] == lab, "words"], bins=bins, alpha=0.55, label=lab, color=COLORS[lab])
    ax.set_xlabel("Words per article")
    ax.set_ylabel("Articles")
    ax.set_title("Text length distribution")
    ax.legend()
    _save(fig, "text_length_distribution.png")

    cat = _find(df, CATEGORY_NAMES)
    if cat:
        summary["category_column"] = cat
        summary["category_by_label"] = _grouped_bars(df, cat, f"{cat.capitalize()} distribution by label", "subject_distribution.png")
    src = _find(df, SOURCE_NAMES)
    if src:
        summary["source_column"] = src
        summary["source_by_label"] = _grouped_bars(df, src, f"{src.capitalize()} distribution by label", "source_distribution.png")

    vec = CountVectorizer(stop_words="english", max_features=2000)
    matrix = vec.fit_transform(df["news"])
    terms = np.array(vec.get_feature_names_out())
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True)
    summary["top_words"] = {}
    for ax, lab in zip(axes, ["FAKE", "REAL"]):
        freq = np.asarray(matrix[(df["label"] == lab).to_numpy()].sum(axis=0)).ravel()
        top = np.argsort(freq)[::-1][:12]
        ax.barh(terms[top][::-1], freq[top][::-1], color=COLORS[lab])
        ax.set_title(f"Most frequent words: {lab}")
        summary["top_words"][lab] = terms[top].tolist()
    _save(fig, "top_words.png")

    summary["vocabulary_size"] = int(len(set(" ".join(df["news"]).lower().split())))
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if len(dates):
            summary["date_range"] = [str(dates.min().date()), str(dates.max().date())]
    return summary


def save_eda_summary(summary, path=EDA_SUMMARY_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
