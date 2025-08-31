# data/neon_store.py

from __future__ import annotations
import os, json, psycopg2
from typing import Optional, Dict, Any
import streamlit as st
from psycopg2.extras import register_default_jsonb
from contextlib import contextmanager


def _db_url() -> str:
    url = os.getenv("DATABASE_URL") or (st.secrets.get("DATABASE_URL") if hasattr(st, "secrets") else None)
    if not url:
        raise RuntimeError("DATABASE_URL not set.")
    return url

@contextmanager
def _connect():
    conn = psycopg2.connect(
        _db_url(),
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


def init_db():
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.story_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         JSONB NOT NULL,
          history         JSONB NOT NULL,
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
          username        TEXT,
          gender          TEXT,
          archetype       TEXT,
          last_illustration_url TEXT
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.story_visits(
          id BIGSERIAL PRIMARY KEY,
          user_id TEXT NOT NULL,
          visited_at TIMESTAMPTZ NOT NULL DEFAULT now(),
          scene TEXT NOT NULL,
          choice_text TEXT,
          choice_index INTEGER
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.story_events(
          id BIGSERIAL PRIMARY KEY,
          user_id TEXT NOT NULL,
          ts TIMESTAMPTZ NOT NULL DEFAULT now(),
          kind TEXT NOT NULL,
          payload JSONB
        );
        """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS story_visits_user_id_idx ON public.story_visits(user_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS story_progress_updated_at_idx ON public.story_progress(updated_at);")
        conn.commit()


def _count_decisions(history) -> int:
    try:
        return sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user" and m.get("content"))
    except Exception:
        return 0


def save_event(user_id: str, kind: str, payload: dict | None = None):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.story_events(user_id, kind, payload) VALUES (%s, %s, %s::jsonb)",
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
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.story_progress(
              user_id, scene, choices, history, decisions_count,
              username, gender, archetype, last_illustration_url
            )
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s)
            ON CONFLICT (user_id)
            DO UPDATE SET
              scene=EXCLUDED.scene,
              choices=EXCLUDED.choices,
              history=EXCLUDED.history,
              decisions_count=EXCLUDED.decisions_count,
              username=COALESCE(EXCLUDED.username, public.story_progress.username),
              gender=COALESCE(EXCLUDED.gender, public.story_progress.gender),
              archetype=COALESCE(EXCLUDED.archetype, public.story_progress.archetype),
              last_illustration_url=COALESCE(EXCLUDED.last_illustration_url, public.story_progress.last_illustration_url),
              updated_at=now();
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
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT scene, choices, history, decisions_count, username, gender, archetype, last_illustration_url
            FROM public.story_progress WHERE user_id=%s
        """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        scene, choices, history, decisions_count, username, gender, archetype, last_illustration_url = row
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
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM public.story_progress WHERE user_id=%s", (user_id,))
        conn.commit()


def has_snapshot(user_id) -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1 FROM public.story_progress WHERE user_id=%s", (user_id,))
        return cur.fetchone() is not None


def save_visit(user_id: str, scene: str, choice_text: str | None, choice_index: int | None):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO public.story_visits(user_id, scene, choice_text, choice_index)
            VALUES (%s, %s, %s, %s)
        """,
            (user_id, scene, choice_text, choice_index),
        )
        conn.commit()