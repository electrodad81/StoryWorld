# data/store/__init__.py
from __future__ import annotations
import os
import streamlit as st

def _as_bool(x, default=False):
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).strip().lower() in {"1","true","t","yes","y","on"}

def _get_secret(name, default=None):
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.getenv(name, default)

def _persist_enabled() -> bool:
    # Hard override first
    val = _get_secret("persist_enabled")
    if val is not None:
        return _as_bool(val, True)
    # Env override
    val = os.getenv("PERSIST_ENABLED")
    if val is not None:
        return _as_bool(val, True)
    # Default: enabled
    return True

def _which_backend() -> str:
    # Secrets or env choose explicitly
    choice = _get_secret("persist_backend") or os.getenv("PERSIST_BACKEND")
    if choice:
        return str(choice).strip().lower()  # "neon" | "sqlite" | "none"
    # Implicit default: Neon if DSN set, else SQLite
    return "neon" if os.getenv("DATABASE_URL") else "sqlite"

ENABLED = _persist_enabled()
BACKEND = _which_backend()

if not ENABLED or BACKEND in {"none", "off", "noop"}:
    # No-op persistence: everything becomes a stub
    def init_db(): pass
    def save_snapshot(*a, **k): return None
    def load_snapshot(*a, **k): return None
    def delete_snapshot(*a, **k): return None
    def has_snapshot(*a, **k): return False
    def save_visit(*a, **k): return None
    def save_event(*a, **k): return None

else:
    if BACKEND == "sqlite":
        from .sqlite_store import *  # noqa: F401,F403
    else:
        # Default to Neon when unknown
        from .neon_store import *  # noqa: F401,F403
