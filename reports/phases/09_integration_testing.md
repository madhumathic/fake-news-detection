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
**46 automated tests were run on 2026-10-09 09:35 and all passed** (0 skipped, 6.262s). Command: `python -m unittest discover -s tests -t .`

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
