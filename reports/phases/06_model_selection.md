# Phase 6 - Model Selection

Implemented in `src/models/compare.py`; run with `python scripts/compare_models.py` (or as part of `train_models.py`).

## Candidates
Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest (optional, enabled with `--with-rf`) and a
**majority-class baseline** (a model that always predicts the larger class).

## Fair-comparison method
* Same data: **training set only** (4,640 articles). The test set is untouched.
* Same 5-fold stratified cross-validation (`random_state=42`) for every model.
* Same TF-IDF settings (`stop_words="english"`, `max_df=0.7`, `max_features=50000`, `ngram_range=(1,2)`).
* The vectorizer is inside each pipeline, so in every fold it is fitted on that fold's training part only (no leakage).
* Selection rule: the highest mean cross-validated **F1-score**.

## Results (mean over 5 folds)
| Model | Accuracy | Precision | Recall | F1 | Train accuracy | Time |
|---|---|---|---|---|---|---|
| Linear SVM (selected) | 96.57% | 95.86% | 97.58% | 96.71% | 100.00% | 8.0s |
| Logistic Regression | 95.15% | 94.23% | 96.54% | 95.36% | 98.27% | 9.1s |
| Multinomial Naive Bayes | 92.95% | 91.83% | 94.82% | 93.29% | 96.09% | 8.0s |
| Majority-class baseline | 51.64% | 51.64% | 100.00% | 68.11% | 51.64% | 10.5s |

## Decision
Linear SVM was selected (highest mean CV F1 = 0.9671) and beats the majority baseline.

Observation: several models reach 100% training accuracy but lower validation accuracy, which is a sign of overfitting that
regularisation (Phase 7) and the held-out test (Phase 8) are used to check. Linear models are also a natural fit for sparse
TF-IDF features and are easy to explain.
