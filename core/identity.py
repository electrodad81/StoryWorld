# core/identity.py
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

_LS_KEY = "browser_id_v1"

import uuid

def ensure_browser_id(store_key: str = "storyworld_browser_id") -> str:
    """Return a stable per-browser id using localStorage; fall back to a session-sticky id."""
    # Already resolved this run?
    bid = st.session_state.get("browser_id")
    if bid:
        return bid

    # Try localStorage via streamlit-js-eval (if available)
    try:
        from streamlit_js_eval import streamlit_js_eval

        # read existing
        existing = streamlit_js_eval(js_expressions=f"localStorage.getItem('{store_key}')", key="read_bid")
        if isinstance(existing, str) and existing.strip():
            bid = existing.strip()
        else:
            # create, store, then read back
            bid = "brw-" + uuid.uuid4().hex
            streamlit_js_eval(
                js_expressions=f"localStorage.setItem('{store_key}', '{bid}')",
                key="write_bid"
            )
    except Exception:
        # Fallback: keep a session-sticky id (won't span different browsers, but handles no-JS cases)
        bid = st.session_state.get("_fallback_browser_id")
        if not bid:
            bid = "brw-" + uuid.uuid4().hex
            st.session_state["_fallback_browser_id"] = bid

    st.session_state["browser_id"] = bid
    return bid

def clear_browser_id_and_reload():
    """Delete the id from localStorage and reload the page."""
    streamlit_js_eval(
        js_expressions=f"""
            (function(){{
                window.localStorage.removeItem({_LS_KEY!r});
                window.location.reload();
            }})();
        """,
        key="clear_browser_id",
    )
