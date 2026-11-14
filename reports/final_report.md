# Fake News Detection Using Machine Learning - Final Report

*Generated 2026-10-09 09:35 from the files in `reports/` and `models/`.*

## Summary
A TF-IDF + **Linear SVM** classifier was trained on the supplied dataset (news_dataset.csv, 6,000 records,
5,801 after cleaning). On 1,161 unseen test articles it reached **97.42% accuracy**
(95% interval 96.34% to 98.18%), F1 97.51%, ROC-AUC 0.9969, against a 51.59%
majority-class baseline. The model is served through a Flask API and a React web application with SQLite history.

## Key findings
1. **The dataset contains leaks** that would inflate scores if used: `subject` separates the classes perfectly, the date format
   differs by class, and a news-agency tag appears in 98.9% of REAL articles.
   These were excluded or removed, and the evidence is stored in `reports/data_report.json`.
2. Linear models on TF-IDF features worked best; the winner was chosen by cross-validation on training data only.
3. A model trained on shuffled labels scores only at baseline level, confirming the pipeline does not leak the test labels.
4. Errors: 30 of 1,161 (2.58%); wrong predictions have lower average confidence (74.94% vs 97.81%).

## Limitations
* Test data comes from the same dataset, so the score estimates performance on *similar* news only. Performance on other
  publishers, topics or years was not measured and is expected to be lower.
* Some influential words reflect habits of the specific sources in the dataset.
* The model judges style, not truth. It is not a fact-checking system.
* The dataset is a 6,000-record sample; more data would give more stable estimates.

## What was verified and what was not
* Verified by running: data preparation, EDA, model comparison, tuning, training, evaluation, all Python tests, every API endpoint.
* Syntax-checked only: the React frontend (`npm install` and `npm run build` could not be run in the build environment).
* Not done: deployment to any hosting service, evaluation on external news articles.

See `reports/phases/` for the nine lifecycle documents.
