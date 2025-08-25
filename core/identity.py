# core/identity.py
import streamlit as st
from streamlit_js_eval import streamlit_js_eval
from typing import Optional

_LS_KEY = "browser_id_v1"

import uuid

# core/identity.py
import uuid
import streamlit as st

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None

_COOKIE = "storyworld_pid"
_LS_KEY = "storyworld_browser_id"

def _pin_pid_in_url(pid: str) -> None:
    if not pid:
        return
    # Prefer JS replaceState (no reload)
    if streamlit_js_eval:
        try:
            streamlit_js_eval(
                js_expressions=f"""
                    (() => {{
                        const url = new URL(window.location.href);
                        url.searchParams.set('pid', '{pid}');
                        window.history.replaceState({{}}, '', url.toString());
                        return url.toString();
                    }})()
                """,
                key=f"pin_pid_{pid}",
            )
            return
        except Exception:
            pass
    # Fallback to Streamlit API
    try:
        if hasattr(st, "query_params"):
            st.query_params.update({"pid": pid})
        else:
            cur = st.experimental_get_query_params()
            cur["pid"] = pid
            st.experimental_set_query_params(**cur)
    except Exception:
        pass

def _read_cookie(name: str) -> Optional[str]:
    try:
        from streamlit_js_eval import streamlit_js_eval
        raw = streamlit_js_eval(js_expressions="document.cookie", key="read_cookie")
        if isinstance(raw, str):
            parts = [p.strip() for p in raw.split(";")]
            for p in parts:
                if p.startswith(name + "="):
                    return p[len(name) + 1:]
    except Exception:
        pass
    return None

def _write_cookie(name: str, value: str, days: int = 365) -> None:
    try:
        from streamlit_js_eval import streamlit_js_eval
        streamlit_js_eval(
            js_expressions=(
                f"document.cookie = '{name}={value};path=/;max-age={days*24*3600}';"
            ),
            key="write_cookie",
        )
    except Exception:
        pass

def ensure_browser_id(store_key: str = _LS_KEY) -> str:
    """
    Stable per-browser id with multiple anchors:
    1) st.session_state['browser_id']
    2) ?pid= in URL
    3) localStorage
    4) cookie
    5) generate new → write (2)(3)(4) → one-time rerun
    """
    # 1) session
    if st.session_state.get("browser_id"):
        return st.session_state["browser_id"]

    # 2) URL param
    try:
        qp = getattr(st, "query_params", None) or st.experimental_get_query_params()
        v = qp.get("pid") if isinstance(qp, dict) else None
        url_pid = v[0] if isinstance(v, list) else v
        if url_pid:
            st.session_state["browser_id"] = url_pid
            # mirror everywhere
            _pin_pid_in_url(url_pid)
            _write_cookie(_COOKIE, url_pid)
            if streamlit_js_eval:
                try:
                    streamlit_js_eval(
                        js_expressions=f"localStorage.setItem('{store_key}', '{url_pid}')",
                        key="sync_ls_from_url",
                    )
                except Exception:
                    pass
            return url_pid
    except Exception:
        pass

    # 3) localStorage  (FIX: use `bid`, not `url_pid`)
    try:
        if streamlit_js_eval:
            ls = streamlit_js_eval(
                js_expressions=f"localStorage.getItem('{store_key}')",
                key="read_ls_pid",
            )
        else:
            ls = None
        if isinstance(ls, str) and ls.strip():
            bid = ls.strip()
            st.session_state["browser_id"] = bid
            _pin_pid_in_url(bid)          # <-- put it in the URL
            _write_cookie(_COOKIE, bid)   # <-- FIXED: was url_pid
            if streamlit_js_eval:
                try:
                    streamlit_js_eval(
                        js_expressions=f"localStorage.setItem('{store_key}', '{bid}')",
                        key="echo_ls_pid",
                    )
                except Exception:
                    pass
            return bid
    except Exception:
        pass

    # 4) cookie
    cookie_pid = _read_cookie(_COOKIE)
    if cookie_pid:
        st.session_state["browser_id"] = cookie_pid
        _pin_pid_in_url(cookie_pid)  # mirror URL
        if streamlit_js_eval:
            try:
                streamlit_js_eval(
                    js_expressions=f"localStorage.setItem('{store_key}', '{cookie_pid}')",
                    key="write_ls_from_cookie",
                )
            except Exception:
                pass
        return cookie_pid

    # 5) generate new (one-time wait)
    if not st.session_state.get("_pid_waited_once"):
        st.session_state["_pid_waited_once"] = True
        (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()
        st.stop()

    bid = "brw-" + uuid.uuid4().hex
    st.session_state["browser_id"] = bid
    _pin_pid_in_url(bid)
    _write_cookie(_COOKIE, bid)
    if streamlit_js_eval:
        try:
            streamlit_js_eval(
                js_expressions=f"localStorage.setItem('{store_key}', '{bid}')",
                key="write_ls_pid",
            )
        except Exception:
            pass

    # one-time bootstrap rerun
    if not st.session_state.get("_pid_bootstrapped"):
        st.session_state["_pid_bootstrapped"] = True
        (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()
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
