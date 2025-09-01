"""Neon (Postgres) backend for exploration progress and events."""
from __future__ import annotations
import os, json, psycopg2
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
        CREATE TABLE IF NOT EXISTS public.world_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         JSONB NOT NULL,
          history         JSONB NOT NULL,
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.romance_progress(
          user_id         TEXT PRIMARY KEY,
          scene           TEXT NOT NULL,
          choices         JSONB NOT NULL,
          history         JSONB NOT NULL,
          decisions_count INTEGER NOT NULL DEFAULT 0,
          updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.explore_events(
          id BIGSERIAL PRIMARY KEY,
          user_id TEXT NOT NULL,
          ts TIMESTAMPTZ NOT NULL DEFAULT now(),
          kind TEXT NOT NULL,
          payload JSONB
        );
        """
        )
        cur.execute("CREATE INDEX IF NOT EXISTS world_progress_updated_at_idx ON public.world_progress(updated_at);")
        cur.execute("CREATE INDEX IF NOT EXISTS romance_progress_updated_at_idx ON public.romance_progress(updated_at);")
        conn.commit()

def _count_decisions(history) -> int:
    try:
        return sum(1 for m in history if isinstance(m, dict) and m.get("role") == "user" and m.get("content"))
    except Exception:
        return 0

def save_event(user_id: str, kind: str, payload: dict | None = None):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "INSERT INTO public.explore_events(user_id, kind, payload) VALUES (%s, %s, %s::jsonb)",
            (user_id, kind, json.dumps(payload or {})),
        )
        conn.commit()

def _table(kind: str) -> str:
    if kind == "world":
        return "public.world_progress"
    if kind == "romance":
        return "public.romance_progress"
    raise ValueError("kind must be 'world' or 'romance'")

def save_snapshot(kind: str, user_id: str, scene, choices, history):
    table = _table(kind)
    decisions_count = _count_decisions(history)
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            INSERT INTO {table}(user_id, scene, choices, history, decisions_count)
            VALUES (%s, %s, %s::jsonb, %s::jsonb, %s)
            ON CONFLICT (user_id) DO UPDATE SET
              scene=EXCLUDED.scene,
              choices=EXCLUDED.choices,
              history=EXCLUDED.history,
              decisions_count=EXCLUDED.decisions_count,
              updated_at=now();
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
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT scene, choices, history, decisions_count
            FROM {table} WHERE user_id=%s
        """,
            (user_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        scene, choices, history, decisions_count = row
        return {
            "scene": scene,
            "choices": choices,
            "history": history,
            "decisions_count": decisions_count,
        }