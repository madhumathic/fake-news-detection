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
* `Flask>=3.0,<4.0`
* `pandas>=2.2,<4.0`
* `numpy>=1.26,<3.0`
* `scipy>=1.11,<2.0`
* `scikit-learn>=1.4,<2.0`
* `joblib>=1.3,<2.0`
* `matplotlib>=3.8,<4.0`

## Frontend libraries (`frontend/package.json`)
React 18, React Router 6, Vite 5, `@vitejs/plugin-react`.

## Hardware requirements
Any computer with 4 GB RAM and about 1 GB free disk space. Training on this dataset takes about a minute on a normal laptop.
No GPU is needed.
