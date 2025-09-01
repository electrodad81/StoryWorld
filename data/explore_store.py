"""Runtime selection between SQLite and Neon for exploration storage."""
from __future__ import annotations
import os, importlib, logging

try:  # Streamlit secrets are optional outside of the app runtime
    import streamlit as st
    _DB_URL = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
except Exception:
    st = None  # type: ignore
    _DB_URL = os.getenv("DATABASE_URL")

_STORE_NAME = "sqlite"

# other_script.py (inside data/)
# --- bootstrap so running this file directly also works ---
if __name__ == "__main__" and (__package__ is None or __package__ == ""):
    import os, sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "data"
# ----------------------------------------------------------
# Always ship with the SQLite implementation; import relative to this package
from .explore_sqlite_store import (
    init_db as _init_db_sqlite,
    save_snapshot as _save_sqlite,
    load_snapshot as _load_sqlite,
    save_event as _save_event_sqlite,
)

init_db = _init_db_sqlite
save_snapshot = _save_sqlite
load_snapshot = _load_sqlite
save_event = _save_event_sqlite

# If DATABASE_URL is present attempt to switch to the Neon backend dynamically
if _DB_URL:
    try:
        neon = importlib.import_module(".explore_neon_store", __name__)
        init_db = neon.init_db  # type: ignore
        save_snapshot = neon.save_snapshot  # type: ignore
        load_snapshot = neon.load_snapshot  # type: ignore
        save_event = neon.save_event  # type: ignore
        _STORE_NAME = "neon"
    except Exception as e:
        logging.warning(
            "DATABASE_URL is set but Neon backend not available (%s). Falling back to SQLite.",
            e,
        )


def store_name() -> str:
    return _STORE_NAME