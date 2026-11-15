
"""SQLite access for the prediction history (database/history.db)."""

import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path

# Add the main project folder to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    news_text   TEXT    NOT NULL,
    prediction  TEXT    NOT NULL,
    confidence  REAL    NOT NULL,
    created_at  TEXT    NOT NULL
)
"""


def db_path():
    """Database file. Can be overridden with FND_DB_PATH."""
    return Path(os.environ.get("FND_DB_PATH", DB_PATH))


@contextmanager
def connection():
    """Open a connection and create the table if needed."""

    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    try:
        conn.execute(SCHEMA)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
