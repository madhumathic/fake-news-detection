"""Phase 3 - Data acquisition: load the raw dataset and detect its structure.

Nothing here assumes column names or label values. The structure is detected
from the data itself, and a clear error is raised when it cannot be understood.
"""
from pathlib import Path

import pandas as pd

from src.config import RAW_DIR

TEXT_NAMES = ("text", "content", "body", "article", "news", "statement")
TITLE_NAMES = ("title", "headline", "heading")
LABEL_NAMES = ("label", "labels", "target", "class", "truth", "is_fake", "fake", "real")
DATE_NAMES = ("date", "published", "timestamp", "time")

FAKE_TOKENS = {"fake", "false", "unreliable", "hoax", "misleading", "0", "f"}
REAL_TOKENS = {"real", "true", "reliable", "genuine", "legit", "1", "t"}


class DatasetError(Exception):
    """Raised when the dataset is missing or its structure is not understood."""


def find_raw_files(raw_dir=RAW_DIR):
    return sorted(Path(raw_dir).glob("*.csv"))


def read_csv_safe(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1")
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        raise DatasetError(f"{Path(path).name} could not be read as CSV: {exc}") from exc


def detect_columns(df):
    """Return {'text','title','label','date'} -> actual column name (or None)."""
    lower = {c.lower().strip(): c for c in df.columns}

    def first(names):
        for n in names:
            if n in lower:
                return lower[n]
        return None

    label = first(LABEL_NAMES)
    text = first(TEXT_NAMES)
    if text is None:  # fall back to the text column with the longest average length
        obj = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype).startswith("str")]
        obj = [c for c in obj if c != label]
        if obj:
            text = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    title = first(TITLE_NAMES)
    if title == text:
        title = None
    return {"text": text, "title": title, "label": label, "date": first(DATE_NAMES)}


def infer_label_mapping(values):
    """Map every distinct raw label value to 'FAKE' or 'REAL'."""
    mapping = {}
    for raw in pd.Series(values).dropna().unique():
        key = str(raw).strip().lower()
        if key in FAKE_TOKENS:
            mapping[raw] = "FAKE"
        elif key in REAL_TOKENS:
            mapping[raw] = "REAL"
        else:
            raise DatasetError(
                f"Cannot tell whether label value {raw!r} means FAKE or REAL. "
                "Add it to FAKE_TOKENS or REAL_TOKENS in src/data/load_data.py."
            )
    if set(mapping.values()) != {"FAKE", "REAL"}:
        raise DatasetError(f"Expected both FAKE and REAL labels but found: {sorted(set(mapping.values()))}")
    return mapping


def _label_from_filename(path):
    name = Path(path).stem.lower()
    if any(t in name for t in ("fake", "false")):
        return "FAKE"
    if any(t in name for t in ("true", "real")):
        return "REAL"
    return None


def load_raw_dataset(raw_dir=RAW_DIR):
    """Load data/raw/*.csv. Returns (DataFrame, info dict).

    * One file with a label column -> used as is.
    * Several files -> concatenated. Files without a label column get their label
      from the file name (e.g. True.csv / Fake.csv).
    """
    files = find_raw_files(raw_dir)
    if not files:
        raise DatasetError(
            f"No CSV file found in {raw_dir}. Copy your dataset (for example "
            "fake_news_dataset.csv) into data/raw/ and try again."
        )
    frames, info_files = [], []
    for path in files:
        df = read_csv_safe(path)
        cols = detect_columns(df)
        if cols["label"] is None:
            guessed = _label_from_filename(path)
            if guessed is None:
                raise DatasetError(f"{path.name} has no label column and its name does not say FAKE or REAL.")
            df["label"] = guessed
        if len(files) > 1:
            df["source_file"] = path.name  # only needed to tell combined files apart
        frames.append(df)
        info_files.append({"file": path.name, "records": int(len(df)), "columns": list(map(str, df.columns))})
    data = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]
    return data, {"files": info_files, "combined": len(frames) > 1}


def load_processed(path, hint):
    """Load a processed CSV (cleaned data / train / test) or explain which step to run first."""
    path = Path(path)
    if not path.exists():
        raise DatasetError(f"{path.name} was not found. {hint}")
    return pd.read_csv(path)
