# data/sqlite_store.py
from __future__ import annotations
import os, json, sqlite3
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = os.path.join(os.getcwd(), "story.db")

@contextmanager
def _connect():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()

def _column_exists(conn, table, column) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table});")
    return any(row[1] == column for row in cur.fetchall())

_SCHEMA = """
CREATE TABLE IF NOT EXISTS story_progress (
  user_id TEXT PRIMARY KEY,
  scene   TEXT NOT NULL,
  choices TEXT NOT NULL,
  history TEXT NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

def _conn():
    return sqlite3.connect(DB_PATH)

def init_db():
    with _connect() as conn:
        # Main table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS story_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         TEXT NOT NULL,     -- JSON string
          history         TEXT NOT NULL,     -- JSON string
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
          username        TEXT
        );
        """)
        # Backfill columns if missing
        if not _column_exists(conn, "story_progress", "decisions_count"):
            conn.execute("ALTER TABLE story_progress ADD COLUMN decisions_count INTEGER NOT NULL DEFAULT 0;")
        if not _column_exists(conn, "story_progress", "username"):
            conn.execute("ALTER TABLE story_progress ADD COLUMN username TEXT;")

        # Visit log table
        conn.execute("""
        CREATE TABLE IF NOT EXISTS story_visits(
          id           INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id      TEXT NOT NULL,
          visited_at   TEXT NOT NULL DEFAULT (datetime('now')),
          scene        TEXT NOT NULL,
          choice_text  TEXT,
          choice_index INTEGER
        );
        """)
        conn.commit()

def _count_decisions(history_list) -> int:
    try:
        return sum(1 for m in history_list if isinstance(m, dict) and m.get("role") == "user" and m.get("content"))
    except Exception:
        return 0

def save_snapshot(user_id, scene, choices, history, username=None):
    decisions_count = _count_decisions(history)
    with _connect() as conn:
        conn.execute("""
        INSERT INTO story_progress(user_id, scene, choices, history, decisions_count, updated_at, username)
        VALUES (?, ?, ?, ?, ?, datetime('now'), ?)
        ON CONFLICT(user_id) DO UPDATE SET
            scene=excluded.scene,
            choices=excluded.choices,
            history=excluded.history,
            decisions_count=excluded.decisions_count,
            username=COALESCE(excluded.username, story_progress.username),
            updated_at=excluded.updated_at
        """, (user_id, scene, json.dumps(choices), json.dumps(history), decisions_count, username))
        conn.commit()

def load_snapshot(user_id):
    with _connect() as conn:
        cur = conn.execute(
            "SELECT scene, choices, history, decisions_count, username FROM story_progress WHERE user_id=?",
            (user_id,)
        )
        row = cur.fetchone()
        if not row:
            return None
        scene, choices_s, history_s, decisions_count, username = row
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
            "username": username,
        }

def delete_snapshot(user_id):
    with _connect() as conn:
        conn.execute("DELETE FROM story_progress WHERE user_id=?", (user_id,))
        conn.commit()

def has_snapshot(user_id) -> bool:
    with _connect() as conn:
        cur = conn.execute("SELECT 1 FROM story_progress WHERE user_id=?", (user_id,))
        return cur.fetchone() is not None

def save_visit(user_id: str, scene: str, choice_text: str | None, choice_index: int | None):
    with _connect() as conn:
        conn.execute("""
        INSERT INTO story_visits(user_id, scene, choice_text, choice_index)
        VALUES (?, ?, ?, ?)
        """, (user_id, scene, choice_text, choice_index))
        conn.commit()
