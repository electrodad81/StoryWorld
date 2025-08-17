import streamlit as st
from streamlit_js_eval import streamlit_js_eval

LS_KEY = "browser_id_v1"

def ensure_browser_id() -> str:
    """
    Get a persistent browser id from localStorage, mint if missing.
    Returns the id and stores it in st.session_state['browser_id'].
    """
    # Ask the browser to read/create the ID and return it to Python.
    bid = streamlit_js_eval(
        js_expressions=f"""
            (function(){{
                const KEY = {LS_KEY!r};
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

    # Fallback if something blocks JS (very rare)
    if not bid:
        import uuid
        bid = st.session_state.get("browser_id") or uuid.uuid4().hex

    st.session_state["browser_id"] = bid
    return bid

def main():
    st.set_page_config(page_title="Stable Browser ID (localStorage)", page_icon="🪪")
    st.title("Stable Browser ID (localStorage)")

    bid = ensure_browser_id()

    st.subheader("Your browser id")
    st.code(bid)

    st.write(
        "This id lives in your browser’s localStorage and will stay the same across "
        "refreshes and full server restarts as long as you open the app from the same "
        "origin (e.g., http://localhost:8501) and don’t clear site data."
    )

    with st.sidebar:
        st.header("Controls")
        if st.button("Clear id"):
            # Clear in the browser, then reload the page
            streamlit_js_eval(
                js_expressions=f"""
                    (function(){{
                        window.localStorage.removeItem({LS_KEY!r});
                        window.location.reload();
                    }})();
                """,
                key="clear_browser_id",
            )
        st.caption("DevTools → Application → Local Storage should show one key "
                   f"`{LS_KEY}` with this same value.")

if __name__ == "__main__":
    main()
