"""Store, list and clear prediction history."""
from datetime import datetime

from app.database.database import connection

PREVIEW_CHARS = 240


def add_entry(news_text, prediction, confidence):
    with connection() as conn:
        cur = conn.execute(
            "INSERT INTO history (news_text, prediction, confidence, created_at) VALUES (?, ?, ?, ?)",
            (news_text, prediction, round(float(confidence), 2), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        return cur.lastrowid


def list_entries(limit=100, offset=0):
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    with connection() as conn:
        rows = conn.execute(
            "SELECT id, news_text, prediction, confidence, created_at FROM history ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset)).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
    items = [{
        "id": r["id"],
        "news": r["news_text"],
        "preview": r["news_text"][:PREVIEW_CHARS] + ("..." if len(r["news_text"]) > PREVIEW_CHARS else ""),
        "prediction": r["prediction"],
        "confidence": r["confidence"],
        "created_at": r["created_at"],
    } for r in rows]
    return items, total


def clear_entries():
    with connection() as conn:
        deleted = conn.execute("SELECT COUNT(*) FROM history").fetchone()[0]
        conn.execute("DELETE FROM history")
    return deleted
