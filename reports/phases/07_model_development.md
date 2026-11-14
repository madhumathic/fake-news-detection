# Phase 7 - Model Development

Implemented in `src/models/tune.py`, `train.py` and `predict.py`; run with `python scripts/train_models.py`.

## Hyperparameter tuning
Grid search with 5-fold cross-validation on the training set, scoring F1, for **Linear SVM**. The TF-IDF `ngram_range` was
tuned together with the classifier parameters.

| Parameters | CV F1 | Rank |
|---|---|---|
| {'clf__C': 10, 'tfidf__ngram_range': (1, 2)} | 0.9710 | 1 |
| {'clf__C': 1, 'tfidf__ngram_range': (1, 2)} | 0.9671 | 2 |
| {'clf__C': 10, 'tfidf__ngram_range': (1, 1)} | 0.9615 | 3 |
| {'clf__C': 1, 'tfidf__ngram_range': (1, 1)} | 0.9606 | 4 |
| {'clf__C': 0.1, 'tfidf__ngram_range': (1, 2)} | 0.9549 | 5 |
| {'clf__C': 0.1, 'tfidf__ngram_range': (1, 1)} | 0.9494 | 6 |

**Best parameters:** `{"clf__C": 10, "tfidf__ngram_range": [1, 2]}`. If the best value lies at the edge of the grid (for example the
largest C), a wider grid could be tried; this is noted as possible future work.

## Final training
* The TF-IDF vectorizer is **fitted on the training set only**, then the classifier is trained on the training features.
* A Linear SVM has no probabilities, so it is wrapped in `CalibratedClassifierCV` (5-fold) to produce the confidence values shown in the application.
* Features: **title + text only**. Metadata such as `subject` and `date` are not used.

## Saved artifacts (`models/`)
| File | Content |
|---|---|
| `best_model.pkl` | the trained classifier (CalibratedClassifierCV) |
| `tfidf_vectorizer.pkl` | the TF-IDF vectorizer fitted on the training data |
| `label_mapping.json` | FAKE = 0, REAL = 1 |
| `model_metadata.json` | model name, parameters, sizes, scikit-learn version, training time |

## Prediction (`src/models/predict.py`)
`Predictor` loads both files, applies the same `clean_text()` used in training, transforms the text with the already-fitted
vectorizer and returns the label and confidence. A clear error is raised when the model files are missing.
