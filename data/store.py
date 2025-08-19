# data/store.py
from __future__ import annotations
import os, importlib, logging
import streamlit as st

# Prefer secrets, fall back to env
_DB_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
_STORE_NAME = "sqlite"

# ---- Default to SQLite (and export save_visit)
from data.sqlite_store import (  # type: ignore
    init_db as _init_db_sqlite,
    save_snapshot as _save_sqlite,
    load_snapshot as _load_sqlite,
    delete_snapshot as _delete_sqlite,
    has_snapshot as _has_sqlite,
    save_visit as _save_visit_sqlite,
)

# Default assignments (SQLite)
init_db = _init_db_sqlite
save_snapshot = _save_sqlite
load_snapshot = _load_sqlite
delete_snapshot = _delete_sqlite
has_snapshot = _has_sqlite
save_visit = _save_visit_sqlite
from data.sqlite_store import save_event as _save_event_sqlite
save_event = _save_event_sqlite

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
