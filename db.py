# db.py
from __future__ import annotations

import os
import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional, Tuple, List, Any, Dict

# Streamlit is optional (tests / non-Streamlit contexts)
try:
    import streamlit as st
except Exception:
    st = None

# Postgres client is optional (only needed if DATABASE_URL is set)
try:
    import psycopg2
    import psycopg2.extras
except Exception:
    psycopg2 = None

BASE_DIR = Path(__file__).parent
SQLITE_PATH = BASE_DIR / "storyworld.db"


class DB:
    """
    Minimal DB helper:
      - Postgres (Neon) if DATABASE_URL provided (prefer st.secrets, else env)
      - SQLite fallback for local dev
    """
    def __init__(self):
        # Prefer Streamlit secrets, then environment
        dsn = None
        if st is not None:
            try:
                dsn = st.secrets.get("DATABASE_URL", None)
            except Exception:
                dsn = None
        if not dsn:
            dsn = os.environ.get("DATABASE_URL")

        self.backend = "sqlite"
        self._pg_conn = None
        self._sq_conn = None
        self._lock = None  # used for SQLite thread-safety

        if dsn and psycopg2 is not None:
            # Postgres (Neon)
            self.backend = "postgres"
            # Keep-alives help with serverless pools
            self._pg_conn = psycopg2.connect(
                dsn,
                connect_timeout=10,
                keepalives=1,
                keepalives_idle=30,
                keepalives_interval=10,
                keepalives_count=5,
            )
            self._pg_conn.autocommit = True
            self._pg_cur_factory = psycopg2.extras.RealDictCursor
        else:
            # SQLite fallback (thread-safe)
            self._sq_conn = sqlite3.connect(
                str(SQLITE_PATH),
                check_same_thread=False,  # allow access across Streamlit threads
            )
            self._sq_conn.row_factory = sqlite3.Row
            self._lock = threading.Lock()

    # ---------- Schema ----------
    def ensure_schema(self):
        """
        Creates the story_progress table if missing.
        Columns:
          user_id (PK), scene, choices, history, created_at, updated_at
        """
        if self.backend == "postgres":
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS story_progress (
                        user_id    TEXT PRIMARY KEY,
                        scene      TEXT,
                        choices    JSONB,
                        history    JSONB,
                        created_at TIMESTAMPTZ DEFAULT now(),
                        updated_at TIMESTAMPTZ DEFAULT now()
                    );
                    """
                )
                # Trigger to auto-touch updated_at on UPDATE
                cur.execute(
                    """
                    DO $$
                    BEGIN
                        IF NOT EXISTS (
                            SELECT 1
                            FROM pg_trigger
                            WHERE tgname = 'story_progress_set_updated_at'
                        ) THEN
                            CREATE OR REPLACE FUNCTION sp_set_updated_at()
                            RETURNS TRIGGER AS $$
                            BEGIN
                                NEW.updated_at = now();
                                RETURN NEW;
                            END;
                            $$ LANGUAGE plpgsql;

                            CREATE TRIGGER story_progress_set_updated_at
                            BEFORE UPDATE ON story_progress
                            FOR EACH ROW
                            EXECUTE PROCEDURE sp_set_updated_at();
                        END IF;
                    END$$;
                    """
                )
        else:
            with self._lock:
                self._sq_conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS story_progress (
                        user_id    TEXT PRIMARY KEY,
                        scene      TEXT,
                        choices    TEXT,   -- JSON as text in SQLite
                        history    TEXT,   -- JSON as text in SQLite
                        created_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                        updated_at TEXT DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    );
                    """
                )
                self._sq_conn.commit()

    # ---------- Load / Save ----------
    def load_progress(self, user_id: str) -> Optional[Tuple[Optional[str], List[str], list]]:
        """
        Returns (scene, choices, history) or None
        """
        if not user_id:
            return None

        if self.backend == "postgres":
            with self._pg_conn.cursor(cursor_factory=self._pg_cur_factory) as cur:
                cur.execute(
                    "SELECT scene, choices, history FROM story_progress WHERE user_id = %s;",
                    (user_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                return (
                    row.get("scene"),
                    row.get("choices") or [],
                    row.get("history") or [],
                )
        else:
            with self._lock:
                cur = self._sq_conn.execute(
                    "SELECT scene, choices, history FROM story_progress WHERE user_id = ?;",
                    (user_id,),
                )
                r = cur.fetchone()
            if not r:
                return None
            scene = r["scene"]
            try:
                choices = json.loads(r["choices"] or "[]")
            except Exception:
                choices = []
            try:
                history = json.loads(r["history"] or "[]")
            except Exception:
                history = []
            return scene, choices, history

    def save_progress(self, user_id: str, scene: str, choices: List[str], history: list) -> None:
        """
        Upsert the row for user_id with latest scene/choices/history.
        """
        if not user_id:
            return

        if self.backend == "postgres":
            with self._pg_conn.cursor() as cur:
                # Use psycopg2.extras.Json to ensure proper JSONB binding
                J = psycopg2.extras.Json
                cur.execute(
                    """
                    INSERT INTO story_progress (user_id, scene, choices, history)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT (user_id) DO UPDATE SET
                        scene   = EXCLUDED.scene,
                        choices = EXCLUDED.choices,
                        history = EXCLUDED.history;
                    """,
                    (user_id, scene, J(choices), J(history)),
                )
        else:
            payload_choices = json.dumps(choices)
            payload_history = json.dumps(history)
            with self._lock:
                self._sq_conn.execute(
                    """
                    INSERT INTO story_progress (user_id, scene, choices, history, updated_at)
                    VALUES (?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ','now'))
                    ON CONFLICT(user_id) DO UPDATE SET
                        scene    = excluded.scene,
                        choices  = excluded.choices,
                        history  = excluded.history,
                        updated_at = strftime('%Y-%m-%dT%H:%M:%fZ','now');
                    """,
                    (user_id, scene, payload_choices, payload_history),
                )
                self._sq_conn.commit()

    def has_progress(self, user_id: str) -> bool:
        if not user_id:
            return False

        if self.backend == "postgres":
            with self._pg_conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM story_progress WHERE user_id = %s LIMIT 1;",
                    (user_id,),
                )
                return cur.fetchone() is not None
        else:
            with self._lock:
                cur = self._sq_conn.execute(
                    "SELECT 1 FROM story_progress WHERE user_id = ? LIMIT 1;",
                    (user_id,),
                )
                return cur.fetchone() is not None

    def delete_progress(self, user_id: str) -> None:
        if not user_id:
            return

        if self.backend == "postgres":
            with self._pg_conn.cursor() as cur:
                cur.execute("DELETE FROM story_progress WHERE user_id = %s;", (user_id,))
        else:
            with self._lock:
                self._sq_conn.execute(
                    "DELETE FROM story_progress WHERE user_id = ?;",
                    (user_id,),
                )
                self._sq_conn.commit()


# -------- get_db() with Streamlit caching (or module singleton) --------

def _create_and_init_db() -> DB:
    db = DB()
    db.ensure_schema()
    return db

if st is not None:
    # Cached per Streamlit session; survives reruns and avoids reconnect churn
    @st.cache_resource(show_spinner=False)
    def get_db() -> DB:
        return _create_and_init_db()
else:
    _DB_SINGLETON: Optional[DB] = None

    def get_db() -> DB:
        global _DB_SINGLETON
        if _DB_SINGLETON is None:
            _DB_SINGLETON = _create_and_init_db()
        return _DB_SINGLETON
