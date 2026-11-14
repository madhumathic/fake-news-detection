"""SQLite access for the prediction history (database/history.db)."""
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

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
    """Database file. Can be overridden with the FND_DB_PATH environment variable (used by tests)."""
    return Path(os.environ.get("FND_DB_PATH", DB_PATH))


@contextmanager
def connection():
    """Open a connection, create the table if the database is new or was deleted, commit on success."""
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
