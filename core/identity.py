# core/identity.py
import streamlit as st
from streamlit_js_eval import streamlit_js_eval

_LS_KEY = "browser_id_v1"

def ensure_browser_id() -> str:
    """
    Get a persistent browser id from localStorage (create if missing).
    Returns the id and caches it in st.session_state['browser_id'].
    """
    bid = streamlit_js_eval(
        js_expressions=f"""
            (function(){{
                const KEY = {_LS_KEY!r};
                let id = window.localStorage.getItem(KEY);
                if (!id) {{
                    const buf = new Uint8Array(16);
                    (window.crypto || window.msCrypto).getRandomValues(buf);
                    id = Array.from(buf, x => x.toString(16).padStart(2,'0')).join('');
                    window.localStorage.setItem(KEY, id);
                }}
                return id;
            }})();
        """,
        key="get_browser_id",
    )
    if not bid:
        import uuid
        bid = st.session_state.get("browser_id") or uuid.uuid4().hex
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
