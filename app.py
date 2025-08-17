# app.py
import streamlit as st
from core.identity import ensure_browser_id, clear_browser_id_and_reload
from data.sqlite_store import init_db, load_snapshot, save_snapshot, delete_snapshot, has_snapshot
from ui.controls import sidebar_controls, scene_and_choices

st.set_page_config(page_title="Gloamreach — Storyworld MVP", layout="wide")

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
    st.session_state["scene"] = INTRO_SCENE
    st.session_state["choices"] = INTRO_CHOICES[:2]
    st.session_state["history"] = [{"role": "assistant", "content": INTRO_SCENE}]
    save_snapshot(pid, st.session_state["scene"], st.session_state["choices"], st.session_state["history"])

def apply_choice(pid: str, choice: str):
    # Placeholder narrative step; swap this later with your LLM scene/choices engine.
    nxt = f"You choose: **{choice}**. The fog parts slightly, revealing a narrow path..."
    st.session_state["history"].append({"role": "user", "content": choice})
    st.session_state["history"].append({"role": "assistant", "content": nxt})
    st.session_state["scene"] = nxt
    # Just alternate the choice labels to prove state changes
    st.session_state["choices"] = ["Keep going", "Turn back"]
    save_snapshot(pid, st.session_state["scene"], st.session_state["choices"], st.session_state["history"])

def main():
    st.title("Gloamreach — Storyworld MVP")

    # 0) Identity (stable across refresh & restart)
    pid = ensure_browser_id()

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
