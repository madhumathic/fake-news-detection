"""Small synthetic data used ONLY by unit tests (never used to train the real model)."""
import random

import pandas as pd

FAKE_WORDS = "shocking exposed secret hoax outrage scandal viral traitor conspiracy banned truth wake".split()
REAL_WORDS = "ministry statement officials committee announced budget spokesman said parliament report according".split()
FILLER = "the of and to in a for on with at by from city people week country new".split()


def make_articles(n_per_class=40, seed=0):
    """Tiny separable corpus with columns title, text, subject, date, label (raw layout)."""
    rng = random.Random(seed)
    rows = []
    for label, vocab, subject in (("FAKE", FAKE_WORDS, "News"), ("REAL", REAL_WORDS, "politicsNews")):
        for i in range(n_per_class):
            body = " ".join(rng.choice(vocab + FILLER * 2) for _ in range(40)) + f" item {label.lower()} {i}"
            rows.append({"title": f"{rng.choice(vocab)} {rng.choice(vocab)} headline {label.lower()} {i}",
                         "text": body, "subject": subject, "date": "March 1, 2017", "label": label})
    return pd.DataFrame(rows)


def make_clean_frame(n_per_class=40, seed=0):
    """Same corpus in the CLEANED layout (news, label, label_id) used by the model code."""
    from src.data.preprocess import preprocess_dataframe
    clean, _ = preprocess_dataframe(make_articles(n_per_class, seed))
    return clean
