# data/sqlite_store.py

from __future__ import annotations
import os, json, sqlite3
from pathlib import Path
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
DB_PATH = str(BASE_DIR / "story.db")  # stable location

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
        CREATE TABLE IF NOT EXISTS story_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         TEXT NOT NULL,     -- JSON string
          history         TEXT NOT NULL,     -- JSON string
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TEXT NOT NULL DEFAULT (datetime('now')),
          username        TEXT,
          gender          TEXT,
          archetype       TEXT,
          last_illustration_url TEXT
        );
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS story_visits(
          id           INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id      TEXT NOT NULL,
          visited_at   TEXT NOT NULL DEFAULT (datetime('now')),
          scene        TEXT NOT NULL,
          choice_text  TEXT,
          choice_index INTEGER
        );
        """
        )
        conn.execute(
            """
        CREATE TABLE IF NOT EXISTS story_events(
          id      INTEGER PRIMARY KEY AUTOINCREMENT,
          user_id TEXT NOT NULL,
          ts      TEXT NOT NULL DEFAULT (datetime('now')),
          kind    TEXT NOT NULL,
          payload TEXT
        );
        """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS story_visits_user_id_idx ON story_visits(user_id);")
        conn.execute("CREATE INDEX IF NOT EXISTS story_progress_updated_at_idx ON story_progress(updated_at);")
        conn.commit()


def _count_decisions(history_list) -> int:
    try:
        return sum(1 for m in history_list if isinstance(m, dict) and m.get("role") == "user" and m.get("content"))
    except Exception:
        return 0

def store_name() -> str:
    return "sqlite"


# Optional: lightweight connectivity test
def ping() -> bool:
    with _connect() as conn:
        conn.execute("SELECT 1;")
        return True

def save_event(user_id: str, kind: str, payload: dict | None = None):
    with _connect() as conn:
        conn.execute(
            "INSERT INTO story_events(user_id, kind, payload) VALUES (?, ?, ?)",
            (user_id, kind, json.dumps(payload or {})),
        )
        conn.commit()


def save_snapshot(
    user_id,
    scene,
    choices,
    history,
    username=None,
    gender=None,
    archetype=None,
    last_illustration_url=None,  # NEW
):
    decisions_count = _count_decisions(history)
    with _connect() as conn:
        conn.execute(
            """
        INSERT INTO story_progress(
          user_id, scene, choices, history, decisions_count, updated_at,
          username, gender, archetype, last_illustration_url
        )
        VALUES (?, ?, ?, ?, ?, datetime('now'), ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
          scene=excluded.scene,
          choices=excluded.choices,
          history=excluded.history,
          decisions_count=excluded.decisions_count,
          username=COALESCE(excluded.username, story_progress.username),
          gender=COALESCE(excluded.gender, story_progress.gender),
          archetype=COALESCE(excluded.archetype, story_progress.archetype),
          last_illustration_url=COALESCE(excluded.last_illustration_url, story_progress.last_illustration_url),
          updated_at=excluded.updated_at
        """,
            (
                user_id,
                scene,
                json.dumps(choices),
                json.dumps(history),
                decisions_count,
                username,
                gender,
                archetype,
                last_illustration_url,
            ),
        )
        conn.commit()


def load_snapshot(user_id):
    with _connect() as conn:
        cur = conn.execute(
            """
            SELECT scene, choices, history, decisions_count, username, gender, archetype, last_illustration_url
            FROM story_progress WHERE user_id=?
            """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        scene, choices_s, history_s, decisions_count, username, gender, archetype, last_illustration_url = row
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
            "gender": gender,
            "archetype": archetype,
            "last_illustration_url": last_illustration_url,  # NEW
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
        conn.execute(
            """
        INSERT INTO story_visits(user_id, scene, choice_text, choice_index)
        VALUES (?, ?, ?, ?)
        """,
            (user_id, scene, choice_text, choice_index),
        )
        conn.commit()