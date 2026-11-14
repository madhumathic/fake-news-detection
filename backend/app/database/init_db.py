"""Create (or reset) the history database.  python -m app.database.init_db"""
from app.database.database import connection, db_path


def init_db():
    with connection():
        pass  # connection() creates the table when it does not exist
    return db_path()


def reset_db():
    with connection() as conn:
        conn.execute("DROP TABLE IF EXISTS history")
    return init_db()


if __name__ == "__main__":
    print(f"Database ready: {init_db()}")
