# data/raw - the dataset

This folder holds the project's dataset:

| File | Records | Columns |
|---|---|---|
| `news_dataset.csv` | 6,000 | `title`, `text`, `subject`, `date`, `label` (0 = FAKE, 1 = REAL) |

The structure matches the ISOT fake-news layout. The code detects columns and label values from the data itself
(`src/data/load_data.py`), so another labelled CSV can be used by replacing the file.

## Important
* **Keep exactly ONE dataset in this folder.** Every `*.csv` file here is loaded, and several files are concatenated.
  (Two files named like `True.csv` and `Fake.csv` are also supported; each is labelled from its file name.)
* `subject`, `date` and similar metadata leak the label in this dataset and are **not** used as features (see
  `reports/phases/03_data_acquisition.md`).
* An earlier upload named `fake_news_dataset.csv` (20,000 rows) was analysed and **not used**: its text is made of about
  970 randomly combined words and every model scored about 50% on it (the same as guessing), so it contains no learnable
  pattern. Do not place it here.
