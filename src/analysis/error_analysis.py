"""Phase 8 - error analysis: where and how does the model go wrong?"""
import json

import numpy as np
import pandas as pd

from src.config import ERROR_ANALYSIS_PATH

SKIP = {"title", "text", "news", "label", "label_id", "author", "date", "source_file"}


def analyze_errors(test_df, y_pred, proba_real, max_categories=20, n_examples=5):
    df = test_df.reset_index(drop=True).copy()
    df["pred"] = y_pred
    df["wrong"] = df["pred"] != df["label_id"]
    df["conf"] = np.where(df["pred"] == 1, proba_real, 1 - proba_real) * 100
    df["words"] = df["text"].astype(str).str.split().str.len()

    result = {
        "total": int(len(df)), "errors": int(df["wrong"].sum()), "error_rate": round(float(df["wrong"].mean()), 4),
        "false_positives_fake_called_real": int(((df["label_id"] == 0) & (df["pred"] == 1)).sum()),
        "false_negatives_real_called_fake": int(((df["label_id"] == 1) & (df["pred"] == 0)).sum()),
        "mean_confidence_when_correct": round(float(df.loc[~df["wrong"], "conf"].mean()), 2),
        "mean_confidence_when_wrong": round(float(df.loc[df["wrong"], "conf"].mean()), 2) if df["wrong"].any() else None,
    }
    df["length_bucket"] = pd.qcut(df["words"], 3, labels=["short", "medium", "long"], duplicates="drop")
    result["error_rate_by_length"] = {str(k): round(float(v), 4) for k, v in df.groupby("length_bucket", observed=True)["wrong"].mean().items()}

    by_col = {}
    for col in df.columns:
        if col in SKIP or col in {"pred", "wrong", "conf", "words", "length_bucket"}:
            continue
        if 1 < df[col].nunique() <= max_categories:
            by_col[col] = {str(k): round(float(v), 4) for k, v in df.groupby(col, observed=True)["wrong"].mean().items()}
    result["error_rate_by_metadata"] = by_col

    wrong = df[df["wrong"]].sort_values("conf", ascending=False).head(n_examples)
    result["most_confident_mistakes"] = [
        {"actual": "REAL" if r.label_id == 1 else "FAKE", "predicted": "REAL" if r.pred == 1 else "FAKE",
         "confidence": round(float(r.conf), 2), "title": str(r.title)[:120], "excerpt": str(r.text)[:200]}
        for r in wrong.itertuples()]
    return result


def save_error_analysis(result, path=ERROR_ANALYSIS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
