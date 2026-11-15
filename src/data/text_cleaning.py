
"""Lightweight text cleaning for training and prediction."""

import html
import math
import re

HTML_TAG_RE = re.compile(r"</?[A-Za-z!][^>]*>")
URL_RE = re.compile(
    r"(?:https?://|www\.)\S+|\bpic\.twitter\.com/\S+",
    re.IGNORECASE,
)
DATELINE_RE = re.compile(
    r"(?:[A-Z][A-Z .,'/\-]{0,40})?\(Reuters\)\s*[-–—:]\s*"
)
SOURCE_RE = re.compile(r"\bReuters\b", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(value):
    """Clean article text consistently."""

    if value is None:
        return ""

    if isinstance(value, float) and math.isnan(value):
        return ""

    text = str(value)
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = DATELINE_RE.sub(" ", text)
    text = SOURCE_RE.sub(" ", text)

    return WHITESPACE_RE.sub(" ", text).strip()
