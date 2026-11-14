"""Create (or reset) the SQLite history database. Run: python scripts/build_database.py [--reset]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # project root -> "import src" works

import sys

from app.database.init_db import init_db, reset_db
from src.config import DB_PATH


def main():
    if "--reset" in sys.argv:
        reset_db()
        print(f"Database reset: {DB_PATH}")
    else:
        init_db()
        print(f"Database ready: {DB_PATH}")


if __name__ == "__main__":
    main()
