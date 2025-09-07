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
        # Ensure JSONB fields are returned as native Python objects
        register_default_jsonb(conn)
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

def store_name() -> str:
    return "neon"


# Optional: lightweight connectivity test
def ping() -> bool:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1;")
        return True

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

        conn.commit()


# ---------------------------------------------------------------------------
# Explore mode helpers
# ---------------------------------------------------------------------------


def ensure_explore_schema():
    """Create exploration tables if they do not exist."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.world_locations(
            id   TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.world_npcs(
            id     TEXT PRIMARY KEY,
            loc_id TEXT NOT NULL,
            name   TEXT NOT NULL,
            traits JSONB NOT NULL DEFAULT '{}',
            state  JSONB NOT NULL DEFAULT '{}'
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.world_objects(
            id     TEXT PRIMARY KEY,
            loc_id TEXT NOT NULL,
            name   TEXT NOT NULL,
            state  JSONB NOT NULL DEFAULT '{}'
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.world_exits(
            src_id    TEXT NOT NULL,
            direction TEXT NOT NULL,
            dst_id    TEXT NOT NULL,
            locked    INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (src_id, direction)
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.player_world(
            player_id TEXT PRIMARY KEY,
            loc_id    TEXT NOT NULL,
            flags     JSONB NOT NULL DEFAULT '[]',
            inventory JSONB NOT NULL DEFAULT '{}',
            danger    INTEGER NOT NULL DEFAULT 0
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.world_items(
            id    TEXT PRIMARY KEY,
            name  TEXT NOT NULL,
            traits JSONB NOT NULL DEFAULT '{}',
            state  JSONB NOT NULL DEFAULT '{}'
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.location_items(
            loc_id  TEXT NOT NULL,
            item_id TEXT NOT NULL,
            qty     INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (loc_id, item_id)
        );
        """
        )
        cur.execute(
            """
        CREATE TABLE IF NOT EXISTS public.romance_cooldowns(
            player_id TEXT NOT NULL,
            npc_id    TEXT NOT NULL,
            turns     INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (player_id, npc_id)
        );
        """
        )
        conn.commit()


def seed_minimal_world():
    """Seed a tiny world graph if the tables are empty."""
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM public.world_locations")
        if cur.fetchone()[0] == 0:
            cur.execute(
                "INSERT INTO public.world_locations(id,name) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                ("crossroads", "Crossroads"),
            )
            cur.execute(
                "INSERT INTO public.world_locations(id,name) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                ("watch-tower", "Watch Tower"),
            )
            cur.execute(
                "INSERT INTO public.world_locations(id,name) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                ("brook", "Brook"),
            )
            exits = [
                ("crossroads", "N", "watch-tower"),
                ("watch-tower", "S", "crossroads"),
                ("crossroads", "E", "brook"),
                ("brook", "W", "crossroads"),
            ]
            for src, d, dst in exits:
                cur.execute(
                    """
            INSERT INTO public.world_exits(src_id, direction, dst_id, locked)
            VALUES (%s,%s,%s,0)
            ON CONFLICT (src_id, direction) DO NOTHING
                    """,
                    (src, d, dst),
                )
            cur.execute(
                "INSERT INTO public.world_items(id,name) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                ("lantern", "Rusted Lantern"),
            )
            cur.execute(
                "INSERT INTO public.world_items(id,name) VALUES (%s,%s) ON CONFLICT (id) DO NOTHING",
                ("coin", "Ancient Coin"),
            )
            cur.execute(
                """
            INSERT INTO public.location_items(loc_id,item_id,qty)
            VALUES (%s,%s,1)
            ON CONFLICT (loc_id,item_id) DO NOTHING
                    """,
                ("crossroads", "lantern"),
            )
            cur.execute(
                """
            INSERT INTO public.location_items(loc_id,item_id,qty)
            VALUES (%s,%s,1)
            ON CONFLICT (loc_id,item_id) DO NOTHING
                    """,
                ("watch-tower", "coin"),
            )
        cur.execute("SELECT COUNT(*) FROM public.world_npcs")
        if cur.fetchone()[0] == 0:
            cur.execute(
                """
            INSERT INTO public.world_npcs(id, loc_id, name)
            VALUES (%s,%s,%s)
            ON CONFLICT (id) DO NOTHING
                """,
                ("sentinel", "watch-tower", "Lonely Sentinel"),
            )
        conn.commit()


def list_location_items(loc_id: str) -> list[dict]:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
        SELECT i.id, i.name, li.qty, i.traits, i.state
        FROM public.location_items li
        JOIN public.world_items i ON li.item_id=i.id
        WHERE li.loc_id=%s
        """,
            (loc_id,),
        )
        rows = cur.fetchall()
    return [
        {
            "id": r[0],
            "name": r[1],
            "qty": r[2],
            "traits": r[3] or {},
            "state": r[4] or {},
        }
        for r in rows
    ]


def list_player_inventory(player_id: str) -> dict:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT inventory FROM public.player_world WHERE player_id=%s",
            (player_id,),
        )
        row = cur.fetchone()
    return row[0] if row and row[0] else {}


def pickup_item(player_id: str, loc_id: str, item_id: str, qty: int = 1) -> dict:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("BEGIN")
        cur.execute(
            "SELECT qty FROM public.location_items WHERE loc_id=%s AND item_id=%s FOR UPDATE",
            (loc_id, item_id),
        )
        row = cur.fetchone()
        if not row or row[0] < qty:
            conn.rollback()
            return {"inventory_delta": {}}
        remaining = row[0] - qty
        if remaining:
            cur.execute(
                "UPDATE public.location_items SET qty=%s WHERE loc_id=%s AND item_id=%s",
                (remaining, loc_id, item_id),
            )
        else:
            cur.execute(
                "DELETE FROM public.location_items WHERE loc_id=%s AND item_id=%s",
                (loc_id, item_id),
            )
        cur.execute(
            "SELECT inventory FROM public.player_world WHERE player_id=%s FOR UPDATE",
            (player_id,),
        )
        prow = cur.fetchone()
        inv = prow[0] if prow and prow[0] else {}
        inv[item_id] = inv.get(item_id, 0) + qty
        cur.execute(
            "UPDATE public.player_world SET inventory=%s::jsonb WHERE player_id=%s",
            (json.dumps(inv), player_id),
        )
        conn.commit()
    return {"inventory_delta": {item_id: qty}}


def drop_item(player_id: str, loc_id: str, item_id: str, qty: int = 1) -> dict:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("BEGIN")
        cur.execute(
            "SELECT inventory FROM public.player_world WHERE player_id=%s FOR UPDATE",
            (player_id,),
        )
        row = cur.fetchone()
        inv = row[0] if row and row[0] else {}
        if inv.get(item_id, 0) < qty:
            conn.rollback()
            return {"inventory_delta": {}}
        inv[item_id] -= qty
        if inv[item_id] <= 0:
            inv.pop(item_id, None)
        cur.execute(
            "UPDATE public.player_world SET inventory=%s::jsonb WHERE player_id=%s",
            (json.dumps(inv), player_id),
        )
        cur.execute(
            "SELECT qty FROM public.location_items WHERE loc_id=%s AND item_id=%s FOR UPDATE",
            (loc_id, item_id),
        )
        row = cur.fetchone()
        if row:
            cur.execute(
                "UPDATE public.location_items SET qty=%s WHERE loc_id=%s AND item_id=%s",
                (row[0] + qty, loc_id, item_id),
            )
        else:
            cur.execute(
                """
            INSERT INTO public.location_items(loc_id,item_id,qty)
            VALUES (%s,%s,%s)
            ON CONFLICT (loc_id,item_id) DO NOTHING
                    """,
                (loc_id, item_id, qty),
            )
        conn.commit()
    return {"inventory_delta": {item_id: -qty}}


def use_item(player_id: str, item_id: str) -> dict:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute("BEGIN")
        cur.execute(
            "SELECT inventory FROM public.player_world WHERE player_id=%s FOR UPDATE",
            (player_id,),
        )
        row = cur.fetchone()
        inv = row[0] if row and row[0] else {}
        if inv.get(item_id, 0) <= 0:
            conn.rollback()
            return {"inventory_delta": {}}
        inv[item_id] -= 1
        if inv[item_id] <= 0:
            inv.pop(item_id, None)
        cur.execute(
            "UPDATE public.player_world SET inventory=%s::jsonb WHERE player_id=%s",
            (json.dumps(inv), player_id),
        )
        conn.commit()
    return {"prose": f"You use the {item_id}.", "inventory_delta": {item_id: -1}}


def get_romance_cooldown(player_id: str, npc_id: str) -> int:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT turns FROM public.romance_cooldowns WHERE player_id=%s AND npc_id=%s",
            (player_id, npc_id),
        )
        row = cur.fetchone()
    return int(row[0]) if row else 0


def set_romance_cooldown(player_id: str, npc_id: str, turns: int) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            """
        INSERT INTO public.romance_cooldowns(player_id,npc_id,turns)
        VALUES (%s,%s,%s)
        ON CONFLICT (player_id,npc_id) DO UPDATE SET turns=EXCLUDED.turns
            """,
            (player_id, npc_id, turns),
        )
        conn.commit()


def get_player_world_state(player_id: str) -> dict:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT loc_id, flags, inventory, danger FROM public.player_world WHERE player_id=%s",
            (player_id,),
        )
        row = cur.fetchone()
        if not row:
            cur.execute(
                "INSERT INTO public.player_world(player_id, loc_id) VALUES (%s, 'crossroads') ON CONFLICT DO NOTHING",
                (player_id,),
            )
            conn.commit()
            return {
                "loc_id": "crossroads",
                "flags": [],
                "inventory": {},
                "danger": 0,
            }
    loc_id, flags, inventory, danger = row
    return {
        "loc_id": loc_id,
        "flags": flags or [],
        "inventory": inventory or {},
        "danger": int(danger),
    }


def get_location(loc_id: str) -> dict:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name FROM public.world_locations WHERE id=%s",
            (loc_id,),
        )
        row = cur.fetchone()
    if not row:
        return {"id": loc_id, "name": loc_id}
    return {"id": row[0], "name": row[1]}


def visible_interactables(loc_id: str):
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id, name, traits, state FROM public.world_npcs WHERE loc_id=%s",
            (loc_id,),
        )
        npcs = [
            {
                "id": r[0],
                "name": r[1],
                "traits": r[2] or {},
                "state": r[3] or {},
            }
            for r in cur.fetchall()
        ]
        cur.execute(
            "SELECT id, name, state FROM public.world_objects WHERE loc_id=%s",
            (loc_id,),
        )
        objects = [
            {"id": r[0], "name": r[1], "state": r[2] or {}} for r in cur.fetchall()
        ]
        cur.execute(
            "SELECT direction, dst_id, locked FROM public.world_exits WHERE src_id=%s",
            (loc_id,),
        )
        exits = [
            {"direction": r[0], "dst_id": r[1], "locked": int(r[2])} for r in cur.fetchall()
        ]
    return npcs, objects, exits


def move(player_id: str, new_loc: str) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE public.player_world SET loc_id=%s WHERE player_id=%s",
            (new_loc, player_id),
        )
        conn.commit()


def set_flag(player_id: str, flag: str, value: bool) -> None:
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT flags FROM public.player_world WHERE player_id=%s FOR UPDATE",
            (player_id,),
        )
        row = cur.fetchone()
        flags = set(row[0] or []) if row else set()
        if value:
            flags.add(flag)
        else:
            flags.discard(flag)
        cur.execute(
            "UPDATE public.player_world SET flags=%s::jsonb WHERE player_id=%s",
            (json.dumps(sorted(flags)), player_id),
        )
        conn.commit()