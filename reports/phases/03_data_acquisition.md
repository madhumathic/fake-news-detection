# Phase 3 - Data Acquisition

## Source
The dataset was supplied by the project owner (a Kaggle fake-news dataset) as `news_dataset.csv`. It is the **only** dataset
used. Its columns (`title`, `text`, `subject`, `date`, `label`) match the layout of the ISOT Fake News Dataset, but this
file is a **6,000-record sample**, not the full dataset. The code does not assume any file or column names:
they are detected from the data (`src/data/load_data.py`).

## Structure
* Records: **6,000**    Columns: **5**
* Detected: text column `text`, title column `title`, label column `label`, date column `date`

| Column | Data type | Missing values |
|---|---|---|
| title | str | 0 |
| text | str | 0 |
| subject | str | 0 |
| date | str | 0 |
| label | int64 | 0 |

## Labels
Raw label values: `0` = 3,000, `1` = 3,000.
The values are numeric, so the direction is not written in the data. The mapping used is
**0 -> FAKE, 1 -> REAL**. Evidence: the label perfectly follows the `subject`
column, where subjects typical of mainstream wire reporting (`politicsNews`, `worldnews`) only occur with label 1:

| subject | label 0 (FAKE) | label 1 (REAL) |
|---|---|---|
| Government News | 185 | 0 |
| Middle-east | 100 | 0 |
| News | 1175 | 0 |
| US_News | 97 | 0 |
| left-news | 573 | 0 |
| politics | 870 | 0 |
| politicsNews | 0 | 1564 |
| worldnews | 0 | 1436 |

## Class distribution
**FAKE**: 3,000, **REAL**: 3,000. Majority-to-minority ratio: 1.0 (balanced: True).

## Quality checks (raw data)
* Missing values: 0
* Duplicates: 5 identical rows, 108 repeated title+text pairs, 170 repeated texts
* Empty article text: 66 (all of them are labelled FAKE)
* Words per article: mean 404.07, median 359.0, max 8,135

## Irrelevant columns
* `date`: metadata (a date); not used as a feature

## Potential data leakage (important)
| Column checked | Association with label (Cramer's V, 0-1) | Leakage? |
|---|---|---|
| subject | 1.0 | YES - not used |

* `date` format: the style of the date (abbreviated vs full month name) separates the classes (V = 1.0).
* Agency tag such as "(Reuters)": present in 98.9% of REAL articles but only 0.2% of FAKE articles.
* Word-level check: 7,151 of 16,242 words are statistically linked to the label (8.81x the number expected by chance), so the text itself carries real signal.

**Decision:** only the combined `title + text` is used as a feature. `subject`, `date` and the other metadata columns are
never given to the model, and the agency tag is removed from the text.
