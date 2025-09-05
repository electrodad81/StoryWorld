"""SQLite backend for exploration progress and events."""
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = str(BASE_DIR / "story.db")

__all__ = [
    "init_db",
    "save_event",
    "save_snapshot",
    "load_snapshot",
    "delete_snapshot",
]

# Maps snapshot kinds to their backing tables
_SNAPSHOT_TABLES = {
    "world": "world_progress",
    "romance": "romance_progress",
}

@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        yield conn
    finally:
        conn.close()

def init_db():
    with _connect() as conn:
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS world_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         TEXT NOT NULL,
          history         TEXT NOT NULL,
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS romance_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         TEXT NOT NULL,
          history         TEXT NOT NULL,
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS explore_events(
          id      INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          ts      TEXT NOT NULL DEFAULT (datetime('now')),
          kind    TEXT NOT NULL,
          payload TEXT
        );
        """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS world_progress_updated_at_idx ON world_progress(updated_at);"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS romance_progress_updated_at_idx ON romance_progress(updated_at);"
        )
        conn.commit()

def _count_decisions(history_list) -> int:
    try:
        return sum(
            1
            for m in history_list
            if isinstance(m, dict) and m.get("role") == "user" and m.get("content")
        )
    except Exception:
        return 0

def save_event(user_id: str, kind: str, payload: dict | None = None):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO explore_events(user_id, kind, payload) VALUES (?, ?, ?)",
            (user_id, kind, json.dumps(payload or {})),
        )
        conn.commit()

def _table(kind: str) -> str:
    if kind == "world":
        return "world_progress"
    if kind == "romance":
        return "romance_progress"
    raise ValueError("kind must be 'world' or 'romance'")

def save_snapshot(kind: str, user_id: str, scene, choices, history):
    table = _table(kind)
    decisions_count = _count_decisions(history)
    with _connect() as conn:
        conn.execute(
            f"""
        INSERT INTO {table}(user_id, scene, choices, history, decisions_count, updated_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(user_id) DO UPDATE SET
          scene=excluded.scene,
          choices=excluded.choices,
          history=excluded.history,
          decisions_count=excluded.decisions_count,
          updated_at=excluded.updated_at
        """,
            (
                user_id,
                scene,
                json.dumps(choices),
                json.dumps(history),
                decisions_count,
            ),
        )
        conn.commit()

def load_snapshot(kind: str, user_id: str):
    table = _table(kind)
    with _connect() as conn:
        cur = conn.execute(
            f"""
            SELECT scene, choices, history, decisions_count
            FROM {table} WHERE user_id=?
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        scene, choices_s, history_s, decisions_count = row
        try:
            choices = json.loads(choices_s)
        except Exception:
            choices = []
        try:
            history = json.loads(history_s)
        except Exception:
            history = []
        return {
            "scene": scene,
            "choices": choices,
            "history": history,
            "decisions_count": decisions_count,
        }


def delete_snapshot(kind: str, user_id: str) -> None:
    table = _table(kind)
    with _connect() as conn:
        conn.execute(f"DELETE FROM {table} WHERE user_id=?", (user_id,))
        conn.commit()