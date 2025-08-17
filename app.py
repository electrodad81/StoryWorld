# app.py
import streamlit as st
from core.identity import ensure_browser_id, clear_browser_id_and_reload
from data.store import init_db, load_snapshot, save_snapshot, delete_snapshot, has_snapshot
from ui.controls import sidebar_controls, scene_and_choices
import json
from story.engine import stream_scene, generate_choices

st.set_page_config(page_title="Gloamreach — Storyworld MVP", layout="wide")

# Load lore.json once
with open("lore.json", "r", encoding="utf-8") as f:
    LORE = json.load(f)

INTRO_SCENE = (
    "In the land of Gloamreach, the sun never fully rises, leaving the world draped "
    "in a curtain of twilight. Mist clings to the cobblestone streets like a whisper..."
)
INTRO_CHOICES = ["Follow the creaking rope.", "Explore a shadowed alley."]

def hydrate_once_for(pid: str):
    """Load snapshot once per pid into session_state."""
    if st.session_state.get("hydrated_for_pid") == pid:
        return
    snap = load_snapshot(pid)
    if snap:
        st.session_state["scene"] = snap["scene"]
        st.session_state["choices"] = snap["choices"]
        st.session_state["history"] = snap["history"]
    else:
        st.session_state["scene"] = None
        st.session_state["choices"] = []
        st.session_state["history"] = []
    st.session_state["hydrated_for_pid"] = pid

def start_new_story(pid: str):
    # Reset and kick off first streamed scene
    st.session_state["history"] = []
    st.session_state["scene"] = None
    st.session_state["choices"] = []
    # First scene streams now
    _stream_and_save_scene(pid)

def apply_choice(pid: str, choice: str):
    # Record the player's choice, then stream a new scene from the LLM
    st.session_state["history"].append({"role":"user","content": choice})
    _stream_and_save_scene(pid)

def _stream_and_save_scene(pid: str):
    """Stream a new scene, then compute choices, and persist both."""
    # Ensure history exists
    hist = st.session_state.setdefault("history", [])

    # Live stream the scene
    area = st.empty()
    buf = ""
    for tok in stream_scene(hist, LORE):
        buf += tok or ""
        area.markdown(buf)
    scene_text = buf.strip()

    # Append to history & persist with placeholder choices
    hist.append({"role": "assistant", "content": scene_text})
    st.session_state["scene"] = scene_text
    st.session_state["choices"] = ["(thinking…)", "(thinking…)"]
    save_snapshot(pid, st.session_state["scene"], st.session_state["choices"], hist)

    # Fast call for two choices → persist again
    choices = generate_choices(hist, scene_text, LORE)
    st.session_state["choices"] = choices
    save_snapshot(pid, st.session_state["scene"], st.session_state["choices"], hist)


def main():
    st.title("Gloamreach — Storyworld MVP")

    import os
    from data.store import store_name, init_db, save_snapshot, load_snapshot

    # Show which backend the app actually picked
    db_url = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
    st.caption(f"Backend: {store_name().capitalize()} • DATABASE_URL present: {bool(db_url)}")

    # Quick self-test panel
    with st.sidebar.expander("DB self-test", expanded=False):
        import uuid, json

        pid_for_test = st.session_state.get("browser_id", "no-id")
        if st.button("Write probe row"):
            init_db()
            save_snapshot(pid_for_test, "PROBE_SCENE", ["A","B"], [{"role":"assistant","content":"probe"}])
            st.success("Wrote probe row")

        if st.button("Read probe row"):
            snap = load_snapshot(pid_for_test)
            st.code(json.dumps(snap, indent=2))


    # 0) Identity (stable across refresh & restart)
    pid = ensure_browser_id()
    from data.store import store_name
    st.caption(f"player_id: {pid[:8]}… • storage: {store_name().capitalize()}")
    # 1) Persistence
    init_db()
    hydrate_once_for(pid)

    # 2) Sidebar controls → actions
    action = sidebar_controls(pid)
    if action == "start":
        start_new_story(pid)
        st.rerun()
    elif action == "reset":
        # clear only in-memory (keep DB row)
        for k in ("scene", "choices", "history"):
            st.session_state.pop(k, None)
        st.experimental_set_query_params()  # no-op; keeps URL tidy for Streamlit <=1.34
        st.rerun()
    elif action == "switch_user":
        # clear storage for this user; simulate "new browser"
        delete_snapshot(pid)
        clear_browser_id_and_reload()
        st.stop()

    # 3) Main panel
    if not st.session_state.get("scene"):
        st.info("Click **Start New Story** in the sidebar to begin.")
    else:
        scene_and_choices(st.session_state["scene"], st.session_state.get("choices", []))

    # 4) Choice handler (after UI triggers)
    picked = st.session_state.pop("_picked", None)
    if picked:
        apply_choice(pid, picked)
        st.rerun()

    # Footnote
    st.caption("Live-streamed scenes later • Choices advance • SQLite persistence ready • Neon next")

if __name__ == "__main__":
    main()
