# data/neon_store.py
from __future__ import annotations
import os, json, psycopg2
from typing import Optional, Dict, Any
import streamlit as st
from psycopg2.extras import register_default_jsonb

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

def init_db() -> None:
    with _get_conn() as con:
        with con.cursor() as cur:
            cur.execute(_SCHEMA)

def save_snapshot(user_id: str, scene: str, choices: list[str], history: list[Dict[str, Any]]) -> None:
    with _get_conn() as con:
        with con.cursor() as cur:
            cur.execute(
                """
                INSERT INTO public.story_progress (user_id, scene, choices, history)
                VALUES (%s, %s, %s::jsonb, %s::jsonb)
                ON CONFLICT (user_id) DO UPDATE SET
                  scene = EXCLUDED.scene,
                  choices = EXCLUDED.choices,
                  history = EXCLUDED.history,
                  updated_at = now();
                """,
                (user_id, scene, json.dumps(choices), json.dumps(history)),
            )

def load_snapshot(user_id: str) -> Optional[Dict[str, Any]]:
    with _get_conn() as con:
        with con.cursor() as cur:
            cur.execute(
                "SELECT scene, choices, history FROM public.story_progress WHERE user_id = %s",
                (user_id,),
            )
            row = cur.fetchone()
            if not row:
                return None
            scene, choices, history = row
            # choices/history already parsed by register_default_jsonb
            return {"scene": scene, "choices": choices, "history": history}

def delete_snapshot(user_id: str) -> None:
    with _get_conn() as con:
        with con.cursor() as cur:
            cur.execute("DELETE FROM public.story_progress WHERE user_id = %s", (user_id,))

def has_snapshot(user_id: str) -> bool:
    with _get_conn() as con:
        with con.cursor() as cur:
            cur.execute("SELECT 1 FROM public.story_progress WHERE user_id = %s LIMIT 1", (user_id,))
            return cur.fetchone() is not None
