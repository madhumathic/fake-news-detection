# Phase 5 - Exploratory Data Analysis

Implemented in `src/analysis/eda.py`; run with `python scripts/run_eda.py`. Figures are in `reports/figures/`.

## Findings (cleaned data, 5,801 articles)
* **Class balance:** REAL 2,995, FAKE 2,806. The classes are close to balanced, so accuracy is a fair headline metric (a majority-class guesser would score about 51.6%).
* **Length:** FAKE articles have a mean of 436 words (max 8134); REAL articles 381 words (max 1944). The medians are similar, so length alone does not separate the classes.
* **Subjects:** the `subject` column splits the classes completely (see Phase 3). This is the strongest evidence of leakage in the dataset, which is why `subject` is excluded from the features.
* **Frequent words (FAKE):** trump, said, president, people, just, clinton, obama, like
* **Frequent words (REAL):** said, trump, president, state, government, house, states, new
* **Vocabulary:** about 118,496 distinct tokens.

## Figures
* `class_distribution.png`
* `text_length_distribution.png`
* `subject_distribution.png` (subject by label, shows the leakage)
* `source_distribution.png` (only created when a `source` column exists)
* `top_words.png`
