# data/sqlite_store.py
from __future__ import annotations
import json, sqlite3
from pathlib import Path
from typing import Optional, Dict, Any

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "storyworld.db"

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

def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)

def save_snapshot(user_id: str, scene: str, choices: list[str], history: list[Dict[str, Any]]) -> None:
    with _conn() as con:
        con.execute(
            """
            INSERT INTO story_progress (user_id, scene, choices, history)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
              scene=excluded.scene,
              choices=excluded.choices,
              history=excluded.history,
              updated_at=CURRENT_TIMESTAMP;
            """,
            (user_id, scene, json.dumps(choices), json.dumps(history)),
        )

def load_snapshot(user_id: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        cur = con.execute(
            "SELECT scene, choices, history FROM story_progress WHERE user_id = ?",
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        scene, choices, history = row
        return {
            "scene": scene,
            "choices": json.loads(choices),
            "history": json.loads(history),
        }

def delete_snapshot(user_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM story_progress WHERE user_id = ?", (user_id,))

def has_snapshot(user_id: str) -> bool:
    with _conn() as con:
        cur = con.execute("SELECT 1 FROM story_progress WHERE user_id = ? LIMIT 1", (user_id,))
        return cur.fetchone() is not None
