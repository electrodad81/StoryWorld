# data/store.py
from __future__ import annotations
import os
import importlib
import logging

import streamlit as st

_DB_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
_STORE_NAME = "sqlite"  # default

# default to sqlite first
from data.sqlite_store import (  # type: ignore
    init_db as _init_db_sqlite,
    save_snapshot as _save_sqlite,
    load_snapshot as _load_sqlite,
    delete_snapshot as _delete_sqlite,
    has_snapshot as _has_sqlite,
)

# attempt neon only if DATABASE_URL is present AND the module imports
if _DB_URL:
    try:
        neon = importlib.import_module("data.neon_store")
        init_db = neon.init_db
        save_snapshot = neon.save_snapshot
        load_snapshot = neon.load_snapshot
        delete_snapshot = neon.delete_snapshot
        has_snapshot = neon.has_snapshot
        _STORE_NAME = "neon"
    except Exception as e:
        logging.warning(
            "DATABASE_URL is set but Neon backend not available (%s). Falling back to SQLite.",
            e,
        )
        init_db = _init_db_sqlite
        save_snapshot = _save_sqlite
        load_snapshot = _load_sqlite
        delete_snapshot = _delete_sqlite
        has_snapshot = _has_sqlite
else:
    init_db = _init_db_sqlite
    save_snapshot = _save_sqlite
    load_snapshot = _load_sqlite
    delete_snapshot = _delete_sqlite
    has_snapshot = _has_sqlite

def store_name() -> str:
    return _STORE_NAME
