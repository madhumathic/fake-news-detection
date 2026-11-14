# Fake News Detection Using Machine Learning

A complete AI/ML project that predicts whether a news article is **REAL NEWS** or **FAKE NEWS**. It follows the
9-phase AI project lifecycle: data, cleaning, exploratory analysis, model comparison, tuning, evaluation, a Flask API
and a React web application with SQLite prediction history.

> **Important.** The prediction is based on patterns (vocabulary and writing style) learned from the training dataset.
> The system does not check facts and is **not** a definitive fact-checker. Its measured accuracy applies to news that
> resembles the training data.

## 1. Features
* Dataset inspection, validation and **data-leakage checks** (`reports/data_report.json`)
* Cleaning pipeline and stratified 80/20 train/test split (fixed seed)
* Exploratory data analysis with figures
* TF-IDF features and a fair comparison of Logistic Regression, Naive Bayes, Linear SVM and Random Forest (plus a baseline)
* Hyperparameter tuning, final evaluation on unseen data, error analysis and sanity checks
* Flask REST API with input validation and friendly errors
* React + Vite interface: Home, History, Model Performance, Methodology, About
* SQLite prediction history (list and clear)
* 46 automated tests

## 2. Architecture
```
Dataset (data/raw) -> validate -> clean -> split -> EDA -> TF-IDF -> compare -> tune -> train -> evaluate
                                                                                       |
                                                                  models/best_model.pkl + tfidf_vectorizer.pkl
                                                                                       |
React (frontend/)  <--JSON-->  Flask API (backend/)  -->  Predictor (src/models/predict.py)
                                      |
                                      +--> SQLite (database/history.db)
```

## 3. Project structure
```
fake-news-detection/
├── backend/            Flask API: app/main.py, api/ (routes), services/, schemas/, database/, requirements.txt
├── frontend/           React + Vite: src/components, src/pages, src/services/api.js, package.json
├── src/                ML code: data/ (load, validate, preprocess), features/, models/ (compare, tune, train, evaluate, predict), analysis/
├── scripts/            One command per phase (prepare_data, run_eda, train_models, compare_models, evaluate_model, build_database, generate_reports)
├── data/               raw/ (dataset), processed/ (created), external/ (optional)
├── models/             best_model.pkl, tfidf_vectorizer.pkl, model_metadata.json, label_mapping.json
├── reports/            figures/, phases/ (9 lifecycle documents), data_report.json, model_comparison.csv, evaluation_report.json, final_report.md
├── notebooks/          8 notebooks that walk through the phases
├── tests/              6 test files
├── database/           history.db (created automatically)
└── README.md, LICENSE, .gitignore
```

## 4. AI lifecycle -> files
| Phase | Files |
|---|---|
| 1 Problem definition | `reports/phases/01_problem_definition.md` |
| 2 Requirement analysis | `reports/phases/02_requirement_analysis.md`, `backend/requirements.txt`, `frontend/package.json` |
| 3 Data acquisition | `data/raw/`, `src/data/load_data.py`, `src/data/validate_data.py`, `reports/data_report.json` |
| 4 Data preparation | `src/data/preprocess.py`, `scripts/prepare_data.py`, `data/processed/` |
| 5 EDA | `src/analysis/eda.py`, `scripts/run_eda.py`, `reports/figures/` |
| 6 Model selection | `src/features/tfidf_features.py`, `src/models/compare.py`, `scripts/compare_models.py`, `reports/model_comparison.csv` |
| 7 Model development | `src/models/tune.py`, `train.py`, `predict.py`, `scripts/train_models.py`, `models/` |
| 8 Model evaluation | `src/models/evaluate.py`, `src/analysis/*`, `scripts/evaluate_model.py`, `reports/evaluation_report.json` |
| 9 Integration and testing | `backend/`, `frontend/`, `tests/`, `reports/phases/09_integration_testing.md` |

## 5. Requirements
* **Python 3.12** (3.10+ should work) - https://www.python.org/downloads/ (tick **Add python.exe to PATH**)
* **Node.js 18+ (LTS)** - https://nodejs.org/ (only needed for the React frontend)
* **Visual Studio Code** - https://code.visualstudio.com/

Check them in a terminal: `python --version`, `node --version`, `npm --version`.

## 6. Running the project on Windows (step by step)

### Step 1 - Extract and open in VS Code
Right-click `fake-news-detection.zip` > **Extract All...**, then open the folder in VS Code
(**File > Open Folder...**, or from a terminal inside the folder: `code .`).
Open the VS Code terminal with **Ctrl + `** (Terminal > New Terminal). All commands below are run from the project root
(the folder that contains `README.md`).

### Step 2 - Create and activate a virtual environment
```
python -m venv .venv
```
Activate it - use **one** of these, depending on your terminal:

| Terminal | Command |
|---|---|
| PowerShell | `.venv\Scripts\Activate.ps1` |
| Command Prompt | `.venv\Scripts\activate.bat` |

The prompt should now start with `(.venv)`. If PowerShell says *running scripts is disabled*, run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` and activate again (or use Command Prompt).
In VS Code choose the interpreter: **Ctrl+Shift+P > Python: Select Interpreter > .venv**.

### Step 3 - Install the Python packages
```
python -m pip install --upgrade pip
pip install -r backend/requirements.txt
```

### Step 4 - Dataset
The dataset is already in `data/raw/news_dataset.csv`. Keep **only one dataset** in that folder (see `data/raw/README.md`).
To use a different labelled CSV, replace the file; columns and labels are detected automatically.

### Step 5 - Run the ML pipeline
The ZIP already contains the trained model and the reports from the author's run, so you can skip to Step 7 if you only want
to run the app. To reproduce everything yourself:
```
python scripts/prepare_data.py
python scripts/run_eda.py
python scripts/train_models.py
python scripts/evaluate_model.py
python scripts/generate_reports.py
```
| Command | What it does | Creates |
|---|---|---|
| `prepare_data.py` | validates, cleans and splits the data | `reports/data_report.json`, `data/processed/*` |
| `run_eda.py` | exploratory analysis | `reports/figures/*.png`, `reports/eda_summary.json` |
| `train_models.py` | compares models, tunes the best, trains and saves it | `reports/model_comparison.csv`, `reports/tuning_results.csv`, `models/*` |
| `evaluate_model.py` | tests the final model on unseen data, error analysis | `reports/evaluation_report.json`, `reports/error_analysis.json`, `reports/figures/confusion_matrix.png` |
| `generate_reports.py` | writes the nine phase documents and the final report | `reports/phases/*.md`, `reports/final_report.md` |

Options: `python scripts/train_models.py --with-rf` also compares Random Forest (slower); `--skip-tuning` trains with default
parameters; `python scripts/compare_models.py` runs only the comparison. The whole pipeline takes a few minutes.

### Step 6 - Run the tests
```
python -m unittest discover -s tests -t . -v
```
(`pip install pytest` and then `pytest` also works.) Tests that need the trained model are skipped if it does not exist.

### Step 7 - Start the backend (terminal 1)
```
python -m backend.app.main
```
Check it: open http://127.0.0.1:5000/api/health in a browser. The database `database/history.db` is created automatically.

### Step 8 - Start the frontend (terminal 2)
Open a **second** terminal (the first keeps the backend running):
```
cd frontend
npm install
npm run dev
```

### Step 9 - Open the application
**http://localhost:5173** (the Vite server forwards `/api` requests to the backend on port 5000).

### Optional - one server for everything
```
cd frontend
npm run build
cd ..
python -m backend.app.main
```
After the build, Flask also serves the React app, so http://127.0.0.1:5000 shows the whole application.

## 7. API reference
| Method | URL | Body / notes |
|---|---|---|
| POST | `/api/predict` | `{"news": "article text"}` returns prediction, confidence, reliability verdict |
| GET | `/api/history?limit=100` | newest first |
| DELETE | `/api/history` | clears history |
| GET | `/api/model/performance` | metrics, comparison table, reliability |
| GET | `/api/dataset/summary` | data report, EDA summary, figure list |
| GET | `/api/analysis/figures/<name>.png` | figures |
| GET | `/api/health` | status |

Errors are JSON: `{"error": {"code": "empty_input", "message": "..."}}`. Example (PowerShell):
```
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:5000/api/predict -ContentType "application/json" -Body '{"news":"Paste at least ten words of article text here so that the model can analyse the writing."}'
```

## 8. Configuration (optional environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `PORT` | 5000 | backend port |
| `CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | comma-separated browser origins allowed to call the API |
| `FND_DB_PATH` | `database/history.db` | database file (the tests use a temporary one) |
| `VITE_API_URL` | empty | backend address for the frontend when it is hosted separately (see `frontend/.env.example`) |

## 9. Testing with real-world news
1. Copy articles from several news websites (not from the dataset) and paste them into the app.
2. Compare the result with what you know. Note wrong or low-confidence results.
3. Remember: the held-out test score measures *in-distribution* news (same sources, topics and years as the training data).
   Articles from other publishers, topics, languages or years are *out-of-distribution* and accuracy there is likely lower.
   That was **not measured** here. Predictions can also be wrong because the model learns style, not truth.
4. Save interesting articles in `data/external/` for your report.

## 10. Deployment (not performed - for your information)
This project was **not deployed** to any hosting service. Suggested options:

**A. One server (simplest).** Build the frontend (`npm run build`), copy the whole project (including `models/` and `frontend/dist`)
to a host that runs Python, install `backend/requirements.txt` plus a production server and start it from the project root:
* Windows: `pip install waitress` then `waitress-serve --listen=0.0.0.0:8000 backend.app.main:app`
* Linux: `pip install gunicorn` then `gunicorn -w 2 -b 0.0.0.0:8000 backend.app.main:app`

**B. Separate frontend and backend.**
1. Backend on a Python host (for example Render, Railway or PythonAnywhere): same command as above; set
   `CORS_ORIGINS=https://your-frontend-address`.
2. Frontend on a static host (for example Netlify or Vercel): set `VITE_API_URL=https://your-backend-address`, build with
   `npm run build` and publish `frontend/dist`.

Notes: SQLite history is a file, so on hosts with temporary disks it can be lost on restart; use a persistent volume or a
hosted database for a real deployment. Never run with `FLASK_DEBUG=1` in production. The model files (about 4 MB) must be
deployed with the code. Retrain on the host's scikit-learn version if loading the `.pkl` files fails.

## 11. Troubleshooting
| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` or `'backend'` | Run commands from the **project root**, and start the API with `python -m backend.app.main` (not `python main.py`). |
| `ModuleNotFoundError: sklearn / flask / pandas` | Activate `.venv` and run `pip install -r backend/requirements.txt`. |
| `pip` errors building packages | Use Python 3.12 (64-bit), run `python -m pip install --upgrade pip`, retry. |
| PowerShell: *running scripts is disabled* | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or use Command Prompt. |
| `No CSV file found in data/raw` | Put the dataset in `data/raw/` (see Step 4). |
| `clean_news` / `train.csv was not found` | Run `python scripts/prepare_data.py` first. |
| Website says the model is unavailable / HTTP 503 | Run `python scripts/train_models.py`, then restart the backend. |
| `npm` is not recognised | Install Node.js LTS and reopen VS Code. |
| `npm install` fails (network / proxy) | Check your internet connection or proxy; delete `frontend/node_modules` and `package-lock.json`, retry. |
| Browser shows "Cannot reach the server" | Start the backend (Step 7) and check http://127.0.0.1:5000/api/health. |
| CORS error in the browser console | Use http://localhost:5173 (with `npm run dev`) or set `CORS_ORIGINS` to the exact address you use. |
| Port 5000 or 5173 already in use | Set `PORT=5001` (backend, and change the proxy target in `frontend/vite.config.js`) or stop the other program. |
| Model fails to load after upgrading scikit-learn | Re-run `python scripts/train_models.py`. |
| `history.db` locked or corrupted | Stop the backend, run `python scripts/build_database.py --reset`. |

## 12. Notebooks
`notebooks/01...08` walk through the phases using the same `src/` code. Install Jupyter first (`pip install notebook`), then run
`jupyter notebook` from the project root. Run the notebooks in order after the pipeline scripts (they read the files the scripts create).

## 13. Limitations
* Evaluated on data from the same dataset as training; real-world accuracy on other sources is unknown and likely lower.
* The dataset is a 6,000-record sample and contains leaking metadata (excluded) and source-specific habits that the model can pick up.
* English only; very short text is rejected; satire and opinion are not handled specially.
* Scores are estimates with uncertainty (see the confidence interval in `reports/evaluation_report.json`).

## 14. License
MIT - see `LICENSE`.
