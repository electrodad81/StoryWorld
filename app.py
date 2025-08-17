# app.py
from __future__ import annotations
from typing import List
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
    # Defer the streaming to the next run so we don't render the old UI first
    st.session_state["_start_new_pending"] = True
    st.rerun()

def apply_choice(pid: str, choice: str):
    # Record the choice to process on the *next* run
    st.session_state["_pending_choice"] = choice
    st.rerun()

def _render_choices(choices: List[str], choices_ph):
    with choices_ph.container():
        st.subheader("Your choices")
        c1, c2 = st.columns(2)
        turn_id = len(st.session_state.get("history", []))  # changes each scene
        if len(choices) >= 1 and c1.button(
            choices[0], key=f"choice1_{turn_id}", use_container_width=True
        ):
            st.session_state["_picked"] = choices[0]
        if len(choices) >= 2 and c2.button(
            choices[1], key=f"choice2_{turn_id}", use_container_width=True
        ):
            st.session_state["_picked"] = choices[1]

def _stream_and_save_scene(pid: str, scene_ph, choices_ph):
    """Stream a new scene into scene_ph, then compute choices into choices_ph."""
    # Ensure history exists
    hist = st.session_state.setdefault("history", [])

    # Clear UI slots
    scene_ph.empty()
    choices_ph.empty()

    # Live stream the scene
    buf = ""
    for tok in stream_scene(hist, LORE):
        buf += tok or ""
        scene_ph.markdown(buf)  # always write into the same placeholder

    scene_text = buf.strip()
    hist.append({"role": "assistant", "content": scene_text})
    st.session_state["scene"] = scene_text

    # Show a tiny “thinking…” stub while we fetch choices
    with choices_ph.container():
        st.caption("drafting choices…")

    # Fast non-stream call for two choices
    choices = generate_choices(hist, scene_text, LORE)
    st.session_state["choices"] = choices
    save_snapshot(pid, st.session_state["scene"], st.session_state["choices"], hist)

    # Render choices in-place
    _render_choices(choices, choices_ph)


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

    # 1.1
    # fixed slots for this run
    scene_ph   = st.empty()
    choices_ph = st.empty()

    #  Starting a brand new story (deferred)
    if st.session_state.pop("_start_new_pending", False):
        st.session_state["history"] = []
        st.session_state["scene"] = None
        st.session_state["choices"] = []
        _stream_and_save_scene(pid, scene_ph, choices_ph)
        return  # prevent the old UI from rendering below the stream

    # 2) Applying a pending choice (deferred)
    pending = st.session_state.pop("_pending_choice", None)
    if pending is not None:
        st.session_state["history"].append({"role": "user", "content": pending})
        _stream_and_save_scene(pid, scene_ph, choices_ph)
        return

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

    # 3) Normal render (no pending work)
    if not st.session_state.get("scene"):
        st.info("Click **Start New Story** in the sidebar to begin.")
    else:
        # Scene goes into the fixed scene placeholder
        scene_ph.markdown(st.session_state["scene"])
        # Choices go into the fixed choices placeholder
        _render_choices(st.session_state.get("choices", []), choices_ph)

    # 4) Choice handler (after UI triggers)
    picked = st.session_state.pop("_picked", None)
    if picked:
        apply_choice(pid, picked)
        st.rerun()

    # Footnote
    st.caption("Live-streamed scenes later • Choices advance • SQLite persistence ready • Neon next")

if __name__ == "__main__":
    main()
