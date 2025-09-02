# data/store.py
from __future__ import annotations
import os, importlib, logging
import streamlit as st

# Prefer secrets, fall back to env
_DB_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
_STORE_NAME = "sqlite"

# ---- Default to SQLite (and export helpers)
from data import sqlite_store as _sqlite_store  # type: ignore

# Default assignments (SQLite)
init_db = _sqlite_store.init_db
save_snapshot = _sqlite_store.save_snapshot
load_snapshot = _sqlite_store.load_snapshot
delete_snapshot = _sqlite_store.delete_snapshot
has_snapshot = _sqlite_store.has_snapshot
save_visit = _sqlite_store.save_visit
save_event = _sqlite_store.save_event
ensure_explore_schema = _sqlite_store.ensure_explore_schema
seed_minimal_world = _sqlite_store.seed_minimal_world
list_location_items = _sqlite_store.list_location_items
list_player_inventory = _sqlite_store.list_player_inventory
pickup_item = _sqlite_store.pickup_item
drop_item = _sqlite_store.drop_item
use_item = _sqlite_store.use_item
get_romance_cooldown = _sqlite_store.get_romance_cooldown
_STORE_NAME = "sqlite"
ensure_explore_schema = _sqlite_store.ensure_explore_schema
seed_minimal_world = _sqlite_store.seed_minimal_world
list_location_items = _sqlite_store.list_location_items
list_player_inventory = _sqlite_store.list_player_inventory
pickup_item = _sqlite_store.pickup_item
drop_item = _sqlite_store.drop_item
use_item = _sqlite_store.use_item
get_romance_cooldown = _sqlite_store.get_romance_cooldown
set_romance_cooldown = _sqlite_store.set_romance_cooldown

# ---- Prefer Neon if DATABASE_URL is present and module imports cleanly
if _DB_URL:
    try:
        neon = importlib.import_module("data.neon_store")
        init_db = neon.init_db
        save_snapshot = neon.save_snapshot
        load_snapshot = neon.load_snapshot
        delete_snapshot = neon.delete_snapshot
        has_snapshot = neon.has_snapshot
        save_visit = neon.save_visit
        save_event = neon.save_event
        _STORE_NAME = "neon"
    except Exception as e:
        logging.warning(
            "DATABASE_URL is set but Neon backend not available (%s). Falling back to SQLite.",
            e,
        )

def store_name() -> str:
    return _STORE_NAME
