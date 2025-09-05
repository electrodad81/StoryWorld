# data/sqlite_store.py

from __future__ import annotations
import os, json, sqlite3, time, threading
from pathlib import Path
from contextlib import contextmanager

BASE_DIR = Path(__file__).resolve().parents[1]  # repo root
DB_PATH = str(BASE_DIR / "story.db")  # stable location

_DB_MUTEX = threading.Lock()

def _exec_retry(conn, sql: str, params=()):
    for attempt in range(5):
        try:
            return conn.execute(sql, params)
        except sqlite3.OperationalError as e:
            if "locked" in str(e).lower() and attempt < 4:
                time.sleep(0.05 * (attempt + 1))
                continue
            raise

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


# ---------------------------------------------------------------------------
# Exploration v2 helpers
# ---------------------------------------------------------------------------

def ensure_explore_schema():
    """Create exploration-related tables if missing."""
    with _connect() as conn:
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS world_locations(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS world_npcs(
            id TEXT PRIMARY KEY,
            loc_id TEXT NOT NULL,
            name TEXT NOT NULL,
            traits TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT '{}'
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS world_objects(
            id TEXT PRIMARY KEY,
            loc_id TEXT NOT NULL,
            name TEXT NOT NULL,
            state TEXT NOT NULL DEFAULT '{}'
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS world_exits(
            src_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            dst_id TEXT NOT NULL,
            locked INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (src_id, direction)
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS player_world(
            player_id TEXT PRIMARY KEY,
            loc_id TEXT NOT NULL,
            flags TEXT NOT NULL DEFAULT '[]',
            inventory TEXT NOT NULL DEFAULT '{}',
            danger INTEGER NOT NULL DEFAULT 0
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS world_items(
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            traits TEXT NOT NULL DEFAULT '{}',
            state TEXT NOT NULL DEFAULT '{}'
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS location_items(
            loc_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            qty INTEGER NOT NULL DEFAULT 1,
            PRIMARY KEY (loc_id, item_id)
        );
        """,
        )
        _exec_retry(
            conn,
            """
        CREATE TABLE IF NOT EXISTS romance_cooldowns(
            player_id TEXT NOT NULL,
            npc_id TEXT NOT NULL,
            turns INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY (player_id, npc_id)
        );
        """,
        )
        conn.commit()


def seed_minimal_world():
    """Seed a tiny world graph if empty."""
    with _connect() as conn:
        cur = _exec_retry(conn, "SELECT COUNT(*) FROM world_locations")
        count = cur.fetchone()[0]
        if not count:
            _exec_retry(conn, "INSERT INTO world_locations(id,name) VALUES ('crossroads','Crossroads')")
            _exec_retry(conn, "INSERT INTO world_locations(id,name) VALUES ('watch-tower','Watch Tower')")
            _exec_retry(conn, "INSERT INTO world_locations(id,name) VALUES ('brook','Brook')")
            exits = [
                ("crossroads","N","watch-tower"),
                ("watch-tower","S","crossroads"),
                ("crossroads","E","brook"),
                ("brook","W","crossroads"),
            ]
            for src, d, dst in exits:
                _exec_retry(
                    conn,
                    "INSERT INTO world_exits(src_id,direction,dst_id,locked) VALUES (?,?,?,0)",
                    (src, d, dst),
                )
            _exec_retry(
                conn,
                "INSERT INTO world_items(id,name) VALUES ('lantern','Rusted Lantern')",
            )
            _exec_retry(
                conn,
                "INSERT INTO world_items(id,name) VALUES ('coin','Ancient Coin')",
            )
            _exec_retry(
                conn,
                "INSERT INTO location_items(loc_id,item_id,qty) VALUES ('crossroads','lantern',1)"
            )
            _exec_retry(
                conn,
                "INSERT INTO location_items(loc_id,item_id,qty) VALUES ('watch-tower','coin',1)"
            )
        cur = _exec_retry(conn, "SELECT COUNT(*) FROM world_npcs")
        if not cur.fetchone()[0]:
            _exec_retry(
                conn,
                "INSERT INTO world_npcs(id,loc_id,name) VALUES ('sentinel','watch-tower','Lonely Sentinel')",
            )
        conn.commit()


def list_location_items(loc_id: str) -> list[dict]:
    with _connect() as conn:
        cur = _exec_retry(
            conn,
            """
        SELECT i.id, i.name, li.qty, i.traits, i.state
        FROM location_items li JOIN world_items i ON li.item_id=i.id
        WHERE li.loc_id=?
        """,
            (loc_id,),
        )
        rows = cur.fetchall()
        return [
            {
                "id": r[0],
                "name": r[1],
                "qty": r[2],
                "traits": json.loads(r[3] or "{}"),
                "state": json.loads(r[4] or "{}"),
            }
            for r in rows
        ]


def list_player_inventory(player_id: str) -> dict:
    with _connect() as conn:
        cur = _exec_retry(
            conn, "SELECT inventory FROM player_world WHERE player_id=?", (player_id,)
        )
        row = cur.fetchone()
        if not row:
            return {}
        try:
            return json.loads(row[0] or "{}")
        except Exception:
            return {}


def _update_player_inventory(conn, player_id: str, inv: dict) -> None:
    _exec_retry(
        conn,
        "UPDATE player_world SET inventory=? WHERE player_id=?",
        (json.dumps(inv), player_id),
    )


def pickup_item(player_id: str, loc_id: str, item_id: str, qty: int = 1) -> dict:
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        cur = _exec_retry(
            conn,
            "SELECT qty FROM location_items WHERE loc_id=? AND item_id=?",
            (loc_id, item_id),
        )
        row = cur.fetchone()
        if not row or row[0] < qty:
            conn.rollback()
            return {"inventory_delta": {}}
        remaining = row[0] - qty
        if remaining:
            _exec_retry(
                conn,
                "UPDATE location_items SET qty=? WHERE loc_id=? AND item_id=?",
                (remaining, loc_id, item_id),
            )
        else:
            _exec_retry(
                conn,
                "DELETE FROM location_items WHERE loc_id=? AND item_id=?",
                (loc_id, item_id),
            )
        inv = list_player_inventory(player_id)
        inv[item_id] = inv.get(item_id, 0) + qty
        _update_player_inventory(conn, player_id, inv)
        conn.commit()
        return {"inventory_delta": {item_id: qty}}


def drop_item(player_id: str, loc_id: str, item_id: str, qty: int = 1) -> dict:
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        inv = list_player_inventory(player_id)
        if inv.get(item_id, 0) < qty:
            conn.rollback()
            return {"inventory_delta": {}}
        inv[item_id] -= qty
        if inv[item_id] <= 0:
            inv.pop(item_id, None)
        _update_player_inventory(conn, player_id, inv)
        cur = _exec_retry(
            conn,
            "SELECT qty FROM location_items WHERE loc_id=? AND item_id=?",
            (loc_id, item_id),
        )
        row = cur.fetchone()
        if row:
            _exec_retry(
                conn,
                "UPDATE location_items SET qty=? WHERE loc_id=? AND item_id=?",
                (row[0] + qty, loc_id, item_id),
            )
        else:
            _exec_retry(
                conn,
                "INSERT INTO location_items(loc_id,item_id,qty) VALUES (?,?,?)",
                (loc_id, item_id, qty),
            )
        conn.commit()
        return {"inventory_delta": {item_id: -qty}}


def use_item(player_id: str, item_id: str) -> dict:
    """Simple hook removing one item and returning a generic outcome."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        inv = list_player_inventory(player_id)
        if inv.get(item_id, 0) <= 0:
            conn.rollback()
            return {"inventory_delta": {}}
        inv[item_id] -= 1
        if inv[item_id] <= 0:
            inv.pop(item_id, None)
        _update_player_inventory(conn, player_id, inv)
        conn.commit()
    return {
        "prose": f"You use the {item_id}.",
        "inventory_delta": {item_id: -1},
    }


def get_romance_cooldown(player_id: str, npc_id: str) -> int:
    with _connect() as conn:
        cur = _exec_retry(
            conn,
            "SELECT turns FROM romance_cooldowns WHERE player_id=? AND npc_id=?",
            (player_id, npc_id),
        )
        row = cur.fetchone()
        return int(row[0]) if row else 0


def set_romance_cooldown(player_id: str, npc_id: str, turns: int) -> None:
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        _exec_retry(
            conn,
            """
        INSERT INTO romance_cooldowns(player_id,npc_id,turns)
        VALUES (?,?,?)
        ON CONFLICT(player_id,npc_id) DO UPDATE SET turns=excluded.turns
        """,
            (player_id, npc_id, turns),
        )
        conn.commit()

        conn.commit()


def get_player_world_state(player_id: str) -> dict:
    """Return the player's world state, creating a default row if missing."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        cur = _exec_retry(
            conn,
            "SELECT loc_id, flags, inventory, danger FROM player_world WHERE player_id=?",
            (player_id,),
        )
        row = cur.fetchone()
        if not row:
            _exec_retry(
                conn,
                "INSERT INTO player_world(player_id, loc_id) VALUES (?, 'crossroads')",
                (player_id,),
            )
            loc_id, flags_s, inv_s, danger = "crossroads", "[]", "{}", 0
        else:
            loc_id, flags_s, inv_s, danger = row
        conn.commit()
    try:
        flags = json.loads(flags_s or "[]")
    except Exception:
        flags = []
    try:
        inventory = json.loads(inv_s or "{}")
    except Exception:
        inventory = {}
    return {
        "loc_id": loc_id,
        "flags": flags,
        "inventory": inventory,
        "danger": int(danger),
    }


def get_location(loc_id: str) -> dict:
    with _connect() as conn:
        cur = _exec_retry(
            conn, "SELECT id, name FROM world_locations WHERE id=?", (loc_id,)
        )
        row = cur.fetchone()
    if not row:
        return {"id": loc_id, "name": loc_id}
    return {"id": row[0], "name": row[1]}


def visible_interactables(loc_id: str):
    """Return NPCs, objects, and exits visible from a location."""
    with _connect() as conn:
        cur = _exec_retry(
            conn,
            "SELECT id, name, traits, state FROM world_npcs WHERE loc_id=?",
            (loc_id,),
        )
        npcs = [
            {
                "id": r[0],
                "name": r[1],
                "traits": json.loads(r[2] or "{}"),
                "state": json.loads(r[3] or "{}"),
            }
            for r in cur.fetchall()
        ]
        cur = _exec_retry(
            conn,
            "SELECT id, name, state FROM world_objects WHERE loc_id=?",
            (loc_id,),
        )
        objects = [
            {
                "id": r[0],
                "name": r[1],
                "state": json.loads(r[2] or "{}"),
            }
            for r in cur.fetchall()
        ]
        cur = _exec_retry(
            conn,
            "SELECT direction, dst_id, locked FROM world_exits WHERE src_id=?",
            (loc_id,),
        )
        exits = [
            {"direction": r[0], "dst_id": r[1], "locked": int(r[2])}
            for r in cur.fetchall()
        ]
    return npcs, objects, exits


def move(player_id: str, new_loc: str) -> None:
    """Move the player to a new location."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        _exec_retry(
            conn,
            "UPDATE player_world SET loc_id=? WHERE player_id=?",
            (new_loc, player_id),
        )
        conn.commit()


def set_flag(player_id: str, flag: str, value: bool) -> None:
    """Set or clear a boolean flag on the player."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        cur = _exec_retry(
            conn, "SELECT flags FROM player_world WHERE player_id=?", (player_id,)
        )
        row = cur.fetchone()
        flags = set()
        if row:
            try:
                flags = set(json.loads(row[0] or "[]"))
            except Exception:
                flags = set()
        if value:
            flags.add(flag)
        else:
            flags.discard(flag)
        _exec_retry(
            conn,
            "UPDATE player_world SET flags=? WHERE player_id=?",
            (json.dumps(sorted(flags)), player_id),
        )
        conn.commit()


def get_player_world_state(player_id: str) -> dict:
    """Return the player's world state, creating a default row if missing."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        cur = _exec_retry(
            conn,
            "SELECT loc_id, flags, inventory, danger FROM player_world WHERE player_id=?",
            (player_id,),
        )
        row = cur.fetchone()
        if not row:
            _exec_retry(
                conn,
                "INSERT INTO player_world(player_id, loc_id) VALUES (?, 'crossroads')",
                (player_id,),
            )
            loc_id, flags_s, inv_s, danger = "crossroads", "[]", "{}", 0
        else:
            loc_id, flags_s, inv_s, danger = row
        conn.commit()
    try:
        flags = json.loads(flags_s or "[]")
    except Exception:
        flags = []
    try:
        inventory = json.loads(inv_s or "{}")
    except Exception:
        inventory = {}
    return {
        "loc_id": loc_id,
        "flags": flags,
        "inventory": inventory,
        "danger": int(danger),
    }


def get_location(loc_id: str) -> dict:
    with _connect() as conn:
        cur = _exec_retry(
            conn, "SELECT id, name FROM world_locations WHERE id=?", (loc_id,)
        )
        row = cur.fetchone()
    if not row:
        return {"id": loc_id, "name": loc_id}
    return {"id": row[0], "name": row[1]}


def visible_interactables(loc_id: str):
    """Return NPCs, objects, and exits visible from a location."""
    with _connect() as conn:
        cur = _exec_retry(
            conn,
            "SELECT id, name, traits, state FROM world_npcs WHERE loc_id=?",
            (loc_id,),
        )
        npcs = [
            {
                "id": r[0],
                "name": r[1],
                "traits": json.loads(r[2] or "{}"),
                "state": json.loads(r[3] or "{}"),
            }
            for r in cur.fetchall()
        ]
        cur = _exec_retry(
            conn,
            "SELECT id, name, state FROM world_objects WHERE loc_id=?",
            (loc_id,),
        )
        objects = [
            {
                "id": r[0],
                "name": r[1],
                "state": json.loads(r[2] or "{}"),
            }
            for r in cur.fetchall()
        ]
        cur = _exec_retry(
            conn,
            "SELECT direction, dst_id, locked FROM world_exits WHERE src_id=?",
            (loc_id,),
        )
        exits = [
            {"direction": r[0], "dst_id": r[1], "locked": int(r[2])}
            for r in cur.fetchall()
        ]
    return npcs, objects, exits


def move(player_id: str, new_loc: str) -> None:
    """Move the player to a new location."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        _exec_retry(
            conn,
            "UPDATE player_world SET loc_id=? WHERE player_id=?",
            (new_loc, player_id),
        )
        conn.commit()


def set_flag(player_id: str, flag: str, value: bool) -> None:
    """Set or clear a boolean flag on the player."""
    with _connect() as conn, _DB_MUTEX:
        _exec_retry(conn, "BEGIN IMMEDIATE;")
        cur = _exec_retry(
            conn, "SELECT flags FROM player_world WHERE player_id=?", (player_id,)
        )
        row = cur.fetchone()
        flags = set()
        if row:
            try:
                flags = set(json.loads(row[0] or "[]"))
            except Exception:
                flags = set()
        if value:
            flags.add(flag)
        else:
            flags.discard(flag)
        _exec_retry(
            conn,
            "UPDATE player_world SET flags=? WHERE player_id=?",
            (json.dumps(sorted(flags)), player_id),
        )
        conn.commit()