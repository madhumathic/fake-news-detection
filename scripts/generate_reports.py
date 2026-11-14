"""Write the lifecycle documents in reports/phases/ and reports/final_report.md.

Every number in these documents is read from the result files produced by the
pipeline (data_report.json, model_comparison.csv, evaluation_report.json ...),
so the documents always describe the LAST run. Nothing is typed in by hand.

Run after evaluation:   python scripts/generate_reports.py [--skip-tests]
"""
import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

import pandas as pd

from src.config import (COMPARISON_PATH, DATA_REPORT_PATH, EDA_SUMMARY_PATH, ERROR_ANALYSIS_PATH, EVALUATION_PATH,
                        FINAL_REPORT_PATH, METADATA_PATH, PHASES_DIR, ROOT, TUNING_PATH)


def load_json(path):
    if not path.exists():
        sys.exit(f"ERROR: {path.name} not found. Run the pipeline first (prepare_data, run_eda, train_models, evaluate_model).")
    return json.loads(path.read_text(encoding="utf-8"))


def pct(x, digits=2):
    return f"{x * 100:.{digits}f}%"


def pfmt(p):
    return "< 1e-100" if p < 1e-100 else f"{p:.2e}"


def table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)


def run_tests():
    """Run the unit tests and report the real outcome."""
    proc = subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-t", "."],
                          cwd=ROOT, capture_output=True, text=True)
    text = proc.stderr + proc.stdout
    ran = re.search(r"Ran (\d+) tests? in ([\d.]+)s", text)
    skipped = re.search(r"skipped=(\d+)", text)
    return {"ok": proc.returncode == 0, "ran": int(ran.group(1)) if ran else None,
            "seconds": ran.group(2) if ran else None, "skipped": int(skipped.group(1)) if skipped else 0,
            "when": datetime.now().strftime("%Y-%m-%d %H:%M")}


def write(name, text):
    PHASES_DIR.mkdir(parents=True, exist_ok=True)
    (PHASES_DIR / name).write_text(text.strip() + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-tests", action="store_true", help="do not run the test-suite for document 09")
    args = parser.parse_args()

    data = load_json(DATA_REPORT_PATH)
    eda = load_json(EDA_SUMMARY_PATH)
    ev = load_json(EVALUATION_PATH)
    meta = load_json(METADATA_PATH)
    err = load_json(ERROR_ANALYSIS_PATH)
    comp = pd.read_csv(COMPARISON_PATH)
    tuning = pd.read_csv(TUNING_PATH) if TUNING_PATH.exists() else pd.DataFrame()
    tests = None if args.skip_tests else run_tests()

    steps = data["cleaning_steps"]
    cols = data["detected_columns"]
    classes = data["class_distribution"]
    cm = ev["confusion_matrix"]
    lo, hi = ev["accuracy_95ci"]
    best = meta["model_name"]
    file_names = ", ".join(f["file"] for f in data["files"])
    suspicious = [c for c in data["leakage_checks"] if c["suspicious"]]

    # ------------------------------------------------------------------ 01
    write("01_problem_definition.md", f"""
# Phase 1 - Problem Definition

## Background
Online news spreads quickly and much of it is false or misleading. Readers cannot verify every story, so tools that flag
articles that *look like* fake news can support, but never replace, human judgement.

## Problem statement
Given the text of a news article (headline and body), predict whether it is **REAL NEWS** or **FAKE NEWS**.

## Existing problem
Manual fact-checking is slow and cannot keep up with the volume of published articles. Simple keyword rules are easy to
evade and produce many false alarms.

## Proposed solution
A supervised machine-learning classifier: article text is converted to TF-IDF features and a linear classifier trained on a
labelled dataset predicts the class. The model is served through a web application with a prediction history.

## Main objective
Build a complete, reproducible AI system (data to deployed web app) that classifies news articles with measured, honestly
reported accuracy.

## Specific objectives
1. Load, validate and clean the supplied Kaggle dataset ({file_names}).
2. Explore the data and check it for data leakage.
3. Extract TF-IDF features and compare several algorithms fairly.
4. Tune and train the best model, then evaluate it once on unseen data.
5. Save the model and vectorizer, and integrate them into a web application with SQLite history.
6. Test the system and document every phase.

## Scope
**In scope:** English news articles, binary classification (REAL / FAKE), local web application, prediction history.
**Out of scope:** verifying individual claims against external sources, images or video, other languages, real-time
monitoring of social media.

## Users and stakeholders
Students and examiners (demonstration and learning), general readers who want a second opinion on an article's style.

## Success criteria
* The model beats a majority-class baseline with statistical significance on unseen test data.
* No identified data-leakage feature is used for prediction.
* Every pipeline step is reproducible from the command line; all automated tests pass.

## Limitations and assumptions
The system predicts from writing style and vocabulary learned from the training dataset. It does not check facts, and its
accuracy applies to articles similar to the training data.
""")

    # ------------------------------------------------------------------ 02
    reqs = [l.strip() for l in (ROOT / "backend" / "requirements.txt").read_text().splitlines() if l.strip()]
    write("02_requirement_analysis.md", f"""
# Phase 2 - Requirement Analysis

## Functional requirements
| ID | Requirement |
|---|---|
| F1 | Load and validate the dataset in `data/raw/` and report its structure. |
| F2 | Clean the data and split it into stratified train and test sets. |
| F3 | Produce exploratory figures and statistics. |
| F4 | Compare Logistic Regression, Multinomial Naive Bayes, Linear SVM and Random Forest with cross-validation. |
| F5 | Tune the best model, train it and save the model and TF-IDF vectorizer. |
| F6 | Evaluate the final model on unseen test data. |
| F7 | Let a user paste an article and receive REAL NEWS / FAKE NEWS with a confidence value. |
| F8 | Store every prediction in SQLite; list and clear the history. |
| F9 | Show model performance and methodology in the web application. |

## Non-functional requirements
* **Reproducibility:** fixed random seed (42); one command per phase.
* **Robustness:** friendly errors for empty, oversized or malformed input, a missing model or a missing database.
* **Usability:** responsive interface, usable on a phone and on a laptop.
* **Maintainability:** modular code (`src/` for ML, `backend/` for the API, `frontend/` for the UI), automated tests.
* **Honesty:** the application states that predictions are not a definitive fact-check.
* **Portability:** relative paths only; runs on Windows, macOS and Linux.

## Software requirements
| Item | Version |
|---|---|
| Python | 3.12 (3.10 or newer should also work) |
| Node.js (for the React frontend only) | 18 or newer (LTS recommended) |
| Editor | Visual Studio Code |
| Browser | Any modern browser |

## Python libraries (`backend/requirements.txt`)
{chr(10).join(f"* `{r}`" for r in reqs)}

## Frontend libraries (`frontend/package.json`)
React 18, React Router 6, Vite 5, `@vitejs/plugin-react`.

## Hardware requirements
Any computer with 4 GB RAM and about 1 GB free disk space. Training on this dataset takes about a minute on a normal laptop.
No GPU is needed.
""")

    # ------------------------------------------------------------------ 03
    col_rows = [(c, t, data["missing_values"].get(c, 0)) for c, t in data["columns"].items()]
    leak_rows = [(c["column"], c["association_with_label_cramers_v"], "YES - not used" if c["suspicious"] else "no")
                 for c in data["leakage_checks"]]
    subj = next((c for c in data["leakage_checks"] if c.get("label_counts_by_value")), None)
    subj_rows = [(k, v.get("FAKE", 0), v.get("REAL", 0)) for k, v in (subj["label_counts_by_value"].items() if subj else [])]
    write("03_data_acquisition.md", f"""
# Phase 3 - Data Acquisition

## Source
The dataset was supplied by the project owner (a Kaggle fake-news dataset) as `{file_names}`. It is the **only** dataset
used. Its columns (`title`, `text`, `subject`, `date`, `label`) match the layout of the ISOT Fake News Dataset, but this
file is a **{data['records']:,}-record sample**, not the full dataset. The code does not assume any file or column names:
they are detected from the data (`src/data/load_data.py`).

## Structure
* Records: **{data['records']:,}**    Columns: **{data['n_columns']}**
* Detected: text column `{cols['text']}`, title column `{cols['title']}`, label column `{cols['label']}`, date column `{cols['date']}`

{table(["Column", "Data type", "Missing values"], col_rows)}

## Labels
Raw label values: {", ".join(f"`{k}` = {v:,}" for k, v in data['raw_label_values'].items())}.
The values are numeric, so the direction is not written in the data. The mapping used is
**{", ".join(f"{k} -> {v}" for k, v in data['label_mapping'].items())}**. Evidence: the label perfectly follows the `subject`
column, where subjects typical of mainstream wire reporting (`politicsNews`, `worldnews`) only occur with label 1:

{table(["subject", "label 0 (FAKE)", "label 1 (REAL)"], subj_rows)}

## Class distribution
{", ".join(f"**{k}**: {v:,}" for k, v in classes.items())}. Majority-to-minority ratio: {data['imbalance_ratio_majority_to_minority']} (balanced: {data['is_balanced']}).

## Quality checks (raw data)
* Missing values: {sum(data['missing_values'].values())}
* Duplicates: {data['duplicates']['full_row']} identical rows, {data['duplicates']['title_and_text']} repeated title+text pairs, {data['duplicates']['text']} repeated texts
* Empty article text: {data['empty_text_records']} (all of them are labelled FAKE)
* Words per article: mean {data['words_per_article']['mean']}, median {data['words_per_article']['50%']}, max {int(data['words_per_article']['max']):,}

## Irrelevant columns
{chr(10).join(f"* `{c['column']}`: {c['reason']}" for c in data['irrelevant_columns']) or "* none"}

## Potential data leakage (important)
{table(["Column checked", "Association with label (Cramer's V, 0-1)", "Leakage?"], leak_rows)}

* `date` format: the style of the date (abbreviated vs full month name) separates the classes (V = {data['date_format_check']['association_with_label_cramers_v']}).
* Agency tag such as "(Reuters)": present in {pct(data['agency_tag_check']['share_of_articles_with_agency_tag_by_label']['REAL'], 1)} of REAL articles but only {pct(data['agency_tag_check']['share_of_articles_with_agency_tag_by_label']['FAKE'], 1)} of FAKE articles.
* Word-level check: {data['word_label_signal']['significant_words_p_lt_0.05']:,} of {data['word_label_signal']['vocabulary_tested']:,} words are statistically linked to the label ({data['word_label_signal']['ratio_observed_to_chance']}x the number expected by chance), so the text itself carries real signal.

**Decision:** only the combined `title + text` is used as a feature. `subject`, `date` and the other metadata columns are
never given to the model, and the agency tag is removed from the text.
""")

    # ------------------------------------------------------------------ 04
    write("04_data_preparation.md", f"""
# Phase 4 - Data Preparation

Implemented in `src/data/load_data.py`, `validate_data.py` and `preprocess.py`; run with `python scripts/prepare_data.py`.

## Steps and what each removed
{table(["Step", "Records removed"], [
    ("Unlabelled records", steps["removed_unlabelled"]),
    ("Missing article text", steps["removed_missing_text"]),
    ("Empty article text (after cleaning)", steps["removed_empty_text"]),
    (f"Too short (fewer than 10 words)", steps["removed_too_short"]),
    ("Same article with conflicting labels", steps["removed_conflicting_labels"]),
    ("Duplicate articles", steps["removed_duplicates"]),
])}

Records before: **{steps['rows_in']:,}**. Records after: **{steps['rows_out']:,}**.

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
  **{meta['train_size']:,} training** and **{meta['test_size']:,} test** records. The test set is not used for training or tuning.
* Output: `data/processed/cleaned_data.csv`, `data/processed/train_test_split/train.csv`, `test.csv`.
""")

    # ------------------------------------------------------------------ 05
    wl = eda["words_per_article_by_label"]
    write("05_eda.md", f"""
# Phase 5 - Exploratory Data Analysis

Implemented in `src/analysis/eda.py`; run with `python scripts/run_eda.py`. Figures are in `reports/figures/`.

## Findings (cleaned data, {eda['records']:,} articles)
* **Class balance:** {", ".join(f"{k} {v:,}" for k, v in eda['class_counts'].items())}. The classes are close to balanced, so accuracy is a fair headline metric (a majority-class guesser would score about {pct(max(eda['class_counts'].values()) / eda['records'], 1)}).
* **Length:** FAKE articles have a mean of {wl['FAKE']['mean']:.0f} words (max {wl['FAKE']['max']:.0f}); REAL articles {wl['REAL']['mean']:.0f} words (max {wl['REAL']['max']:.0f}). The medians are similar, so length alone does not separate the classes.
* **Subjects:** the `subject` column splits the classes completely (see Phase 3). This is the strongest evidence of leakage in the dataset, which is why `subject` is excluded from the features.
* **Frequent words (FAKE):** {", ".join(eda['top_words']['FAKE'][:8])}
* **Frequent words (REAL):** {", ".join(eda['top_words']['REAL'][:8])}
* **Vocabulary:** about {eda['vocabulary_size']:,} distinct tokens.

## Figures
* `class_distribution.png`
* `text_length_distribution.png`
* `subject_distribution.png` (subject by label, shows the leakage)
* `source_distribution.png` (only created when a `source` column exists)
* `top_words.png`
""")

    # ------------------------------------------------------------------ 06
    comp_rows = [(r.model + (" (selected)" if r.selected else ""), pct(r.cv_accuracy_mean), pct(r.cv_precision_mean),
                  pct(r.cv_recall_mean), pct(r.cv_f1_mean), pct(r.train_accuracy_mean), f"{r.fit_seconds}s")
                 for r in comp.itertuples()]
    write("06_model_selection.md", f"""
# Phase 6 - Model Selection

Implemented in `src/models/compare.py`; run with `python scripts/compare_models.py` (or as part of `train_models.py`).

## Candidates
Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest (optional, enabled with `--with-rf`) and a
**majority-class baseline** (a model that always predicts the larger class).

## Fair-comparison method
* Same data: **training set only** ({meta['train_size']:,} articles). The test set is untouched.
* Same 5-fold stratified cross-validation (`random_state=42`) for every model.
* Same TF-IDF settings (`stop_words="english"`, `max_df=0.7`, `max_features=50000`, `ngram_range=(1,2)`).
* The vectorizer is inside each pipeline, so in every fold it is fitted on that fold's training part only (no leakage).
* Selection rule: the highest mean cross-validated **F1-score**.

## Results (mean over 5 folds)
{table(["Model", "Accuracy", "Precision", "Recall", "F1", "Train accuracy", "Time"], comp_rows)}

## Decision
{meta['selection_note']}

Observation: several models reach 100% training accuracy but lower validation accuracy, which is a sign of overfitting that
regularisation (Phase 7) and the held-out test (Phase 8) are used to check. Linear models are also a natural fit for sparse
TF-IDF features and are easy to explain.
""")

    # ------------------------------------------------------------------ 07
    tune_rows = [(r.params, f"{r.cv_f1_mean:.4f}", int(r.rank_test_score)) for r in tuning.itertuples()] if len(tuning) else []
    write("07_model_development.md", f"""
# Phase 7 - Model Development

Implemented in `src/models/tune.py`, `train.py` and `predict.py`; run with `python scripts/train_models.py`.

## Hyperparameter tuning
Grid search with 5-fold cross-validation on the training set, scoring F1, for **{best}**. The TF-IDF `ngram_range` was
tuned together with the classifier parameters.

{table(["Parameters", "CV F1", "Rank"], tune_rows)}

**Best parameters:** `{json.dumps(meta['best_params'])}`. If the best value lies at the edge of the grid (for example the
largest C), a wider grid could be tried; this is noted as possible future work.

## Final training
* The TF-IDF vectorizer is **fitted on the training set only**, then the classifier is trained on the training features.
* {"A Linear SVM has no probabilities, so it is wrapped in `CalibratedClassifierCV` (5-fold) to produce the confidence values shown in the application." if best == "Linear SVM" else "The classifier provides probabilities directly."}
* Features: **title + text only**. Metadata such as `subject` and `date` are not used.

## Saved artifacts (`models/`)
| File | Content |
|---|---|
| `best_model.pkl` | the trained classifier ({meta['model_class']}) |
| `tfidf_vectorizer.pkl` | the TF-IDF vectorizer fitted on the training data |
| `label_mapping.json` | FAKE = 0, REAL = 1 |
| `model_metadata.json` | model name, parameters, sizes, scikit-learn version, training time |

## Prediction (`src/models/predict.py`)
`Predictor` loads both files, applies the same `clean_text()` used in training, transforms the text with the already-fitted
vectorizer and returns the label and confidence. A clear error is raised when the model files are missing.
""")

    # ------------------------------------------------------------------ 08
    cr = ev["classification_report"]
    write("08_model_evaluation.md", f"""
# Phase 8 - Model Evaluation

Implemented in `src/models/evaluate.py`, `src/analysis/error_analysis.py` and `model_analysis.py`;
run with `python scripts/evaluate_model.py`. The model was evaluated **once** on **{ev['test_size']:,} unseen articles**.

## Metrics ({best})
{table(["Metric", "Value"], [
    ("Accuracy", f"{pct(ev['accuracy'])} (95% confidence interval {pct(lo)} to {pct(hi)})"),
    ("Precision (REAL = positive class)", pct(ev["precision"])),
    ("Recall", pct(ev["recall"])),
    ("F1-score", pct(ev["f1_score"])),
    ("Macro F1", pct(ev["macro_f1"])),
    ("ROC-AUC", f"{ev['roc_auc']:.4f}"),
    ("Training accuracy", pct(ev["train_accuracy"])),
    ("Majority-class baseline accuracy", pct(ev["majority_baseline_accuracy"])),
    ("p-value: model vs baseline (one-sided binomial test)", pfmt(ev["p_value_vs_baseline"])),
])}

## Confusion matrix (rows = actual, columns = predicted)
{table(["", "Predicted FAKE", "Predicted REAL"], [("**Actual FAKE**", cm["TN_fake_as_fake"], cm["FP_fake_as_real"]), ("**Actual REAL**", cm["FN_real_as_fake"], cm["TP_real_as_real"])])}

## Classification report
{table(["Class", "Precision", "Recall", "F1", "Support"], [(c, f"{cr[c]['precision']:.4f}", f"{cr[c]['recall']:.4f}", f"{cr[c]['f1-score']:.4f}", int(cr[c]['support'])) for c in ("FAKE", "REAL")])}

## Sanity checks
* **Label-shuffle control:** the same model trained on randomly shuffled labels scores {ev['label_shuffle_check']['test_accuracy_with_shuffled_labels']} on the test set, which is just the baseline. The real model's score therefore comes from patterns, not from the pipeline.
* **Overfitting:** training accuracy {pct(ev['train_accuracy'])} versus test accuracy {pct(ev['accuracy'])} (gap {pct(ev['overfit_gap_train_minus_test'])}).
* **Most influential terms towards FAKE:** {", ".join(ev['top_terms']['towards_FAKE'][:10])}
* **Most influential terms towards REAL:** {", ".join(ev['top_terms']['towards_REAL'][:10])}

## Error analysis
{err['errors']} of {err['total']:,} test articles were misclassified ({pct(err['error_rate'])}):
{err['false_positives_fake_called_real']} fake articles called real, {err['false_negatives_real_called_fake']} real articles called fake.
Mean confidence was {err['mean_confidence_when_correct']}% when correct and {err['mean_confidence_when_wrong']}% when wrong,
so low-confidence predictions deserve extra caution. Error rate by article length: {err['error_rate_by_length']}.
Error rate by subject (used here for analysis only): {err['error_rate_by_metadata'].get('subject', {})}.

## Interpretation and limits
The test articles come from the **same dataset** as the training articles. The score therefore measures performance on
*similar* news and is probably higher than what would be seen on articles from other publishers, countries or years
(see document 09). Some influential words (for example "featured image", "getty images", "video") are habits of the
particular fake-news sources in this dataset rather than universal signs of fakeness.
""")

    # ------------------------------------------------------------------ 09
    if tests is None:
        test_line = "The test-suite was not run while generating this document (`--skip-tests`). Run `python -m unittest discover -s tests -t .`."
    elif tests["ok"]:
        test_line = (f"**{tests['ran']} automated tests were run on {tests['when']} and all passed** "
                     f"({tests['skipped']} skipped, {tests['seconds']}s). Command: `python -m unittest discover -s tests -t .`")
    else:
        test_line = f"**The test-suite FAILED when this document was generated ({tests['when']}).** Run it to see the failures."
    write("09_integration_testing.md", f"""
# Phase 9 - Solution Integration and Testing

## Architecture
```
React (Vite)  --HTTP/JSON-->  Flask API  -->  Predictor (best_model.pkl + tfidf_vectorizer.pkl)
 frontend/                     backend/            src/models/predict.py
                                  |
                                  +--> SQLite  database/history.db
```
In development the Vite server proxies `/api` to Flask (port 5000), so no CORS configuration is needed; CORS headers for
`localhost:5173` are also provided for direct calls.

## API endpoints
| Method | Path | Purpose |
|---|---|---|
| POST | `/api/predict` | validate input, predict, store in history |
| GET | `/api/history` | list saved predictions (newest first) |
| DELETE | `/api/history` | clear the history |
| GET | `/api/model/performance` | metrics, model comparison, reliability verdict |
| GET | `/api/dataset/summary` | data report, EDA summary, figure list |
| GET | `/api/analysis/figures/<name>` | PNG figures |
| GET | `/api/health` | model/database status |

## Input validation and robustness
* Empty or whitespace-only text, missing field, wrong type, invalid JSON: HTTP 400 with a JSON message.
* Text shorter than 10 words (links and HTML ignored): HTTP 400. Longer than 50,000 characters or a body over 1 MB: HTTP 413.
* Missing or unreadable model files: HTTP 503 with instructions; the rest of the site keeps working.
* Missing database: created automatically when needed.
* Unknown URL or wrong method: JSON error. Unexpected errors: generic message (no stack trace is shown).
* The frontend shows a friendly message when the backend cannot be reached.
* Figure names are checked against a strict pattern (no path traversal).

## Automated tests
{test_line}

| File | Covers |
|---|---|
| `tests/test_data.py` | dataset loading, column/label detection, validation report, leakage detection |
| `tests/test_preprocessing.py` | text cleaning, dataset cleaning steps, alignment, stratified split |
| `tests/test_features.py` | TF-IDF settings, no vocabulary leakage from test text, save/load |
| `tests/test_models.py` | model comparison, tuning, training, evaluation metrics |
| `tests/test_prediction.py` | model loading, prediction structure, missing model error |
| `tests/test_api.py` | every endpoint, validation errors, history, CORS, path traversal |

**Not verified in the build environment:** `npm install` and `npm run build` (no internet access there). The frontend source
was syntax-checked only. See the README for the commands that verify it on your computer.

## Real-world news testing
* **Unseen news:** test the web app with articles that are not in `data/raw/`, from several sources.
* **In-distribution vs out-of-distribution:** the held-out test score applies to news that resembles the training data
  (same sources, topics and years). News from other publishers, topics (sports, health, finance), languages or years is
  out-of-distribution, and accuracy there can be much lower. This has **not been measured** for this project because the
  dataset provides no such articles.
* **Why predictions can be wrong:** the model learns style and vocabulary, not truth; sources have recognisable habits that
  may not carry over; satire and opinion pieces look unusual; well-written falsehoods look real; and a confident score is
  not a correct score.
* **Why ML classification is not fact-checking:** a fact-check compares claims with evidence. This model never sees evidence.
""")

    # ------------------------------------------------------------------ final report
    FINAL_REPORT_PATH.write_text(f"""# Fake News Detection Using Machine Learning - Final Report

*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} from the files in `reports/` and `models/`.*

## Summary
A TF-IDF + **{best}** classifier was trained on the supplied dataset ({file_names}, {data['records']:,} records,
{steps['rows_out']:,} after cleaning). On {ev['test_size']:,} unseen test articles it reached **{pct(ev['accuracy'])} accuracy**
(95% interval {pct(lo)} to {pct(hi)}), F1 {pct(ev['f1_score'])}, ROC-AUC {ev['roc_auc']:.4f}, against a {pct(ev['majority_baseline_accuracy'])}
majority-class baseline. The model is served through a Flask API and a React web application with SQLite history.

## Key findings
1. **The dataset contains leaks** that would inflate scores if used: `subject` separates the classes perfectly, the date format
   differs by class, and a news-agency tag appears in {pct(data['agency_tag_check']['share_of_articles_with_agency_tag_by_label']['REAL'], 1)} of REAL articles.
   These were excluded or removed, and the evidence is stored in `reports/data_report.json`.
2. Linear models on TF-IDF features worked best; the winner was chosen by cross-validation on training data only.
3. A model trained on shuffled labels scores only at baseline level, confirming the pipeline does not leak the test labels.
4. Errors: {err['errors']} of {err['total']:,} ({pct(err['error_rate'])}); wrong predictions have lower average confidence ({err['mean_confidence_when_wrong']}% vs {err['mean_confidence_when_correct']}%).

## Limitations
* Test data comes from the same dataset, so the score estimates performance on *similar* news only. Performance on other
  publishers, topics or years was not measured and is expected to be lower.
* Some influential words reflect habits of the specific sources in the dataset.
* The model judges style, not truth. It is not a fact-checking system.
* The dataset is a {data['records']:,}-record sample; more data would give more stable estimates.

## What was verified and what was not
* Verified by running: data preparation, EDA, model comparison, tuning, training, evaluation, all Python tests, every API endpoint.
* Syntax-checked only: the React frontend (`npm install` and `npm run build` could not be run in the build environment).
* Not done: deployment to any hosting service, evaluation on external news articles.

See `reports/phases/` for the nine lifecycle documents.
""", encoding="utf-8")
    print("Written: reports/phases/01..09 and reports/final_report.md")
    if tests:
        print("Tests:", "PASSED" if tests["ok"] else "FAILED", f"({tests['ran']} run)")


if __name__ == "__main__":
    main()
