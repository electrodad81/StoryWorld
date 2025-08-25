# data/neon_store.py
from __future__ import annotations
import os, json, psycopg2
from typing import Optional, Dict, Any
import streamlit as st
from psycopg2.extras import register_default_jsonb
from contextlib import contextmanager

DSN = os.getenv("DATABASE_URL")

@contextmanager
def _connect():
    conn = psycopg2.connect(
        DSN,
        connect_timeout=5,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    try:
        yield conn
    finally:
        conn.close()

def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL not set.")
    return url

def _get_conn():
    con = psycopg2.connect(_db_url())
    # ensure JSONB returns native Python types
    register_default_jsonb(con, loads=json.loads)
    return con

_SCHEMA = """
CREATE TABLE IF NOT EXISTS public.story_progress (
  user_id   TEXT PRIMARY KEY,
  scene     TEXT NOT NULL,
  choices   JSONB NOT NULL,
  history   JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

def init_db():
    if not DSN:
        return
    with _connect() as conn, conn.cursor() as cur:
        # Main progress table
        cur.execute("""
        CREATE TABLE IF NOT EXISTS public.story_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         JSONB NOT NULL,
          history         JSONB NOT NULL,
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """)
        
        # Ensure username column exists for user tracking
        cur.execute("""
        ALTER TABLE public.story_progress
        ADD COLUMN IF NOT EXISTS username TEXT
        """)    

        # inside init_db(), after creating story_progress:
        cur.execute("""
        ALTER TABLE story_progress
        ADD COLUMN IF NOT EXISTS last_illustration_url TEXT
        """)

        # Backfill for older deployments
        cur.execute("""
          ALTER TABLE public.story_progress
          ADD COLUMN IF NOT EXISTS decisions_count INTEGER NOT NULL DEFAULT 0;
        """)

        # Visit log table (every screen the user has seen)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS public.story_visits(
        id           BIGSERIAL PRIMARY KEY,
        user_id      TEXT NOT NULL,
        visited_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
        scene        TEXT NOT NULL,
        choice_text  TEXT,
        choice_index INTEGER
        );
        """)

        # events table (you likely already have this)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS public.story_events(
        id      BIGSERIAL PRIMARY KEY,
        user_id TEXT NOT NULL,
        ts      TIMESTAMPTZ NOT NULL DEFAULT now(),
        kind    TEXT NOT NULL,
        payload JSONB
        );
        """)

        # --- in init_db(): add columns if missing (idempotent) ---
        cur.execute("ALTER TABLE public.story_progress ADD COLUMN IF NOT EXISTS username TEXT;")
        cur.execute("ALTER TABLE public.story_progress ADD COLUMN IF NOT EXISTS gender TEXT;")
        cur.execute("ALTER TABLE public.story_progress ADD COLUMN IF NOT EXISTS archetype TEXT;")

        conn.commit()

def _count_decisions(history) -> int:
    try:
        return sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user" and m.get("content"))
    except Exception:
        return 0

# Save an event to the event log
def save_event(user_id: str, kind: str, payload: dict | None = None):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.story_events(user_id, kind, payload) VALUES (%s, %s, %s::jsonb)",
            (user_id, kind, json.dumps(payload or {})),
        )
        conn.commit()

# Snapshot the user's current state
# --- replace/ensure this exact signature & body ---
def save_snapshot(user_id, scene, choices, history, username=None, gender=None, archetype=None):
    decisions_count = _count_decisions(history)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO public.story_progress(user_id, scene, choices, history, decisions_count, username, gender, archetype)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET
                scene=EXCLUDED.scene,
                choices=EXCLUDED.choices,
                history=EXCLUDED.history,
                decisions_count=EXCLUDED.decisions_count,
                username=COALESCE(EXCLUDED.username, public.story_progress.username),
                gender=COALESCE(EXCLUDED.gender, public.story_progress.gender),
                archetype=COALESCE(EXCLUDED.archetype, public.story_progress.archetype),
                updated_at=now();
        """, (user_id, scene, json.dumps(choices), json.dumps(history),
              decisions_count, username, gender, archetype))
        conn.commit()


# --- update load_snapshot to return the new fields ---
def load_snapshot(user_id):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT scene, choices, history, decisions_count, username, gender, archetype
            FROM public.story_progress WHERE user_id=%s
        """, (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        scene, choices, history, decisions_count, username, gender, archetype = row
        return {
            "scene": scene,
            "choices": choices,
            "history": history,
            "decisions_count": decisions_count,
            "username": username,
            "gender": gender,
            "archetype": archetype,
        }

    
def delete_snapshot(user_id):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM public.story_progress WHERE user_id=%s", (user_id,))
        conn.commit()

def has_snapshot(user_id) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM public.story_progress WHERE user_id=%s", (user_id,))
        return cur.fetchone() is not None
    
def save_visit(user_id: str, scene: str, choice_text: str | None, choice_index: int | None):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("""
            INSERT INTO public.story_visits(user_id, scene, choice_text, choice_index)
            VALUES (%s, %s, %s, %s)
        """, (user_id, scene, choice_text, choice_index))
        conn.commit()
