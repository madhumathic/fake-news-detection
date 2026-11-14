# Phase 4 - Data Preparation

Implemented in `src/data/load_data.py`, `validate_data.py` and `preprocess.py`; run with `python scripts/prepare_data.py`.

## Steps and what each removed
| Step | Records removed |
|---|---|
| Unlabelled records | 0 |
| Missing article text | 0 |
| Empty article text (after cleaning) | 75 |
| Too short (fewer than 10 words) | 18 |
| Same article with conflicting labels | 0 |
| Duplicate articles | 106 |

Records before: **6,000**. Records after: **5,801**.

## Text cleaning (`clean_text`)
1. Remove HTML tags and decode HTML entities.
2. Remove URLs (including `pic.twitter.com` links).
3. Remove the news-agency dateline and name (for example `WASHINGTON (Reuters) -`), to prevent shortcut learning.
4. Collapse repeated whitespace and strip leading/trailing spaces.

The same function is applied to every article typed into the website, so training and prediction see identical text.

## Other decisions
* `news = title + " " + text` is the model input; rows are processed whole, so title, text and label stay aligned.
* Labels are converted to FAKE=0 and REAL=1 (`models/label_mapping.json`).
* Metadata columns are kept in the cleaned file for EDA but are never used as features.
* The data is shuffled with `random_state=42` and split **80% / 20%** with stratification:
  **4,640 training** and **1,161 test** records. The test set is not used for training or tuning.
* Output: `data/processed/cleaned_data.csv`, `data/processed/train_test_split/train.csv`, `test.csv`.
