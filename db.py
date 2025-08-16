# db.py
from __future__ import annotations
import os, json, sqlite3
from pathlib import Path
from typing import Optional, Tuple, List

# Streamlit optional at import time
try:
    import streamlit as st
except Exception:
    st = None

# Postgres client is optional unless DATABASE_URL is set
try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None

BASE_DIR = Path(__file__).parent
SQLITE_PATH = BASE_DIR / "storyworld.db"


class DB:
    def __init__(self):
        # Prefer Streamlit secrets → env var
        dsn = None
        if st is not None:
            dsn = st.secrets.get("DATABASE_URL", None)
        if not dsn:
            dsn = os.environ.get("DATABASE_URL")

        self.backend = "sqlite"
        self.pg = None
        self.sq = None

        if dsn and psycopg2 is not None:
            self.backend = "postgres"
            self.pg = psycopg2.connect(
                dsn,
                connect_timeout=10,
                keepalives=1, keepalives_idle=30, keepalives_interval=10, keepalives_count=5,
            )
            self.pg.autocommit = True
            self.pg_cursor = psycopg2.extras.RealDictCursor
        else:
            self.sq = sqlite3.connect(str(SQLITE_PATH))
            self.sq.row_factory = sqlite3.Row

    def ensure_schema(self):
        if self.backend == "postgres":
            with self.pg.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS story_progress (
                        user_id   TEXT PRIMARY KEY,
                        scene     TEXT,
                        choices   JSONB,
                        history   JSONB,
                        updated_at TIMESTAMPTZ DEFAULT now()
                    );
                """)
                cur.execute("""
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                          SELECT 1 FROM pg_trigger WHERE tgname = 'sp_set_updated_at'
                        ) THEN
                          CREATE OR REPLACE FUNCTION sp_touch() RETURNS TRIGGER AS $$
                          BEGIN NEW.updated_at = now(); RETURN NEW; END; $$ LANGUAGE plpgsql;
                          CREATE TRIGGER sp_set_updated_at BEFORE UPDATE ON story_progress
                          FOR EACH ROW EXECUTE PROCEDURE sp_touch();
                        END IF;
                    END$$;
                """)
        else:
            self.sq.execute("""
                CREATE TABLE IF NOT EXISTS story_progress (
                    user_id   TEXT PRIMARY KEY,
                    scene     TEXT,
                    choices   TEXT,   -- JSON (text) in SQLite
                    history   TEXT,   -- JSON (text) in SQLite
                    updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                );
            """)
            self.sq.commit()

    def load_progress(self, user_id: str) -> Optional[Tuple[Optional[str], Optional[List[str]], Optional[list]]]:
        if not user_id:
            return None
        if self.backend == "postgres":
            with self.pg.cursor(cursor_factory=self.pg_cursor) as cur:
                cur.execute("SELECT scene, choices, history FROM story_progress WHERE user_id=%s;", (user_id,))
                row = cur.fetchone()
                if not row:
                    return None
                return row["scene"], (row["choices"] or []), (row["history"] or [])
        else:
            cur = self.sq.execute("SELECT scene, choices, history FROM story_progress WHERE user_id=?;", (user_id,))
            r = cur.fetchone()
            if not r:
                return None
            scene = r["scene"]
            try:
                choices = json.loads(r["choices"] or "[]")
                history = json.loads(r["history"] or "[]")
            except Exception:
                choices, history = [], []
            return scene, choices, history

    def save_progress(self, user_id: str, scene: str, choices: List[str], history: list) -> None:
        if self.backend == "postgres":
            with self.pg.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO story_progress (user_id, scene, choices, history)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                      scene=EXCLUDED.scene,
                      choices=EXCLUDED.choices,
                      history=EXCLUDED.history,
                      updated_at=now();
                    """,
                    (user_id, scene, json.dumps(choices), json.dumps(history)),
                )
        else:
            self.sq.execute(
                """
                INSERT INTO story_progress (user_id, scene, choices, history, updated_at)
                VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                ON CONFLICT(user_id) DO UPDATE SET
                  scene=excluded.scene,
                  choices=excluded.choices,
                  history=excluded.history,
                  updated_at=strftime('%Y-%m-%dT%H:%M:%fZ','now');
                """,
                (user_id, scene, json.dumps(choices), json.dumps(history)),
            )
            self.sq.commit()

    def has_progress(self, user_id: str) -> bool:
        if not user_id:
            return False
        if self.backend == "postgres":
            with self.pg.cursor() as cur:
                cur.execute("SELECT 1 FROM story_progress WHERE user_id=%s LIMIT 1;", (user_id,))
                return cur.fetchone() is not None
        else:
            cur = self.sq.execute("SELECT 1 FROM story_progress WHERE user_id=? LIMIT 1;", (user_id,))
            return cur.fetchone() is not None

    def delete_progress(self, user_id: str) -> None:
        if not user_id:
            return
        if self.backend == "postgres":
            with self.pg.cursor() as cur:
                cur.execute("DELETE FROM story_progress WHERE user_id=%s;", (user_id,))
        else:
            self.sq.execute("DELETE FROM story_progress WHERE user_id=?;", (user_id,))
            self.sq.commit()


# module-level singleton
_singleton: Optional[DB] = None
def get_db() -> DB:
    global _singleton
    if _singleton is None:
        _singleton = DB()
        _singleton.ensure_schema()
    return _singleton
