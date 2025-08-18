# app.py
from __future__ import annotations
from typing import List
import streamlit as st
from core.identity import ensure_browser_id, clear_browser_id_and_reload
from data.store import init_db, load_snapshot, save_snapshot, delete_snapshot, has_snapshot
from ui.controls import sidebar_controls, scene_and_choices
import json
from story.engine import stream_scene, generate_choices
from ui.anim import inject_css, render_scene, render_thinking, render_choices

from ui.loader import lantern, show_lantern_loader
from ui.streaming import stream_text
from ui.choices import render_choices_grid

st.set_page_config(page_title="Gloamreach — Storyworld MVP", layout="wide")

# Load lore.json once
with open("lore.json", "r", encoding="utf-8") as f:
    LORE = json.load(f)

INTRO_SCENE = (
    "In the land of Gloamreach, the sun never fully rises, leaving the world draped "
    "in a curtain of twilight. Mist clings to the cobblestone streets like a whisper..."
)
INTRO_CHOICES = ["Follow the creaking rope.", "Explore a shadowed alley."]

def inject_css():
    st.markdown("""
<style>
/* Lantern flicker */
.lantern { display:flex; align-items:center; gap:.5rem; margin:.25rem 0 .5rem 0; }
.lantern .flame{
  width:14px; height:14px; border-radius:50%;
  box-shadow:0 0 10px 2px rgba(255,190,70,.7);
  background:radial-gradient(circle at 40% 40%, #ffd27a 0%, #ff9a00 60%, #c96a00 100%);
  animation:flame 1.3s ease-in-out infinite alternate;
}
@keyframes flame{
  0%{ transform:translateY(0) scale(1); filter:brightness(1) }
  100%{ transform:translateY(-1px) scale(1.08); filter:brightness(1.2) }
}

/* Optional: subtle typing caret used only during streaming */
.typing-caret{ display:inline-block; width:.5ch; }
.typing-caret::after{ content:"▌"; animation:blink .9s steps(1,end) infinite; }
@keyframes blink{ 50% { opacity:0 } }

/* Respect reduced motion */
@media (prefers-reduced-motion: reduce){
  *{ animation:none !important; transition:none !important }
}
</style>
""", unsafe_allow_html=True)

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

# --- Dev UI toggle (off by default) ------------------------------------------
def _truthy(x) -> bool:
    return str(x).strip().lower() in ("1", "true", "yes", "on")

def _dev_default() -> bool:
    import os
    # Secrets/env can turn it on globally
    if _truthy(os.getenv("DEBUG_UI") or st.secrets.get("DEBUG_UI", False)):
        return True
    # URL param ?dev=1 enables it per-session
    try:
        qp = getattr(st, "query_params", {})
        val = qp.get("dev")
        if isinstance(val, list):
            val = val[0] if val else ""
    except Exception:
        q = st.experimental_get_query_params()
        val = (q.get("dev", [""]) or [""])[0]
    return _truthy(val)


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

CHOICE_COUNT = 2

def _advance_turn(pid: str, story_slot, grid_slot, anim_enabled: bool):
    # keep grid visible during work
    render_choices_grid(grid_slot, choices=None, generating=True, count=CHOICE_COUNT)

    # stream the next scene (lantern + caret if enabled)
    if anim_enabled:
        with lantern("Summoning the next scene…"):
            full = stream_text(stream_scene(st.session_state["history"], LORE), story_slot, show_caret=True)
    else:
        full = stream_text(stream_scene(st.session_state["history"], LORE), story_slot, show_caret=False)

    # save to history
    st.session_state["history"].append({"role": "assistant", "content": full})
    st.session_state["scene"] = full

    # compute choices (fast), save, and render them
    choices = generate_choices(st.session_state["history"], full, LORE)
    st.session_state["choices"] = choices
    save_snapshot(pid, full, choices, st.session_state["history"])
    render_choices_grid(grid_slot, choices=choices, generating=False, count=CHOICE_COUNT)


def main():
    st.title("Gloamreach — Storyworld MVP")

    # --- Atmosphere toggle (visible to everyone) ---
    with st.sidebar:
        anim_enabled = st.checkbox("Atmosphere (lantern + typewriter)", value=False)
    if anim_enabled:
        inject_css()

    # --- Developer mode (hidden unless ?dev=1 or DEBUG_UI=true) ---
    # initialize once per session
    if "_dev" not in st.session_state:
        try:
            st.session_state["_dev"] = _dev_default()  # helper you added earlier
        except NameError:
            st.session_state["_dev"] = False  # safe fallback if helper not present
    # allow turning dev UI off during the session
    with st.sidebar:
        if st.session_state["_dev"]:
            st.session_state["_dev"] = st.checkbox("Developer tools", value=True, help="Show debug info & self-tests")
    dev = bool(st.session_state["_dev"])

    # --- Identity & storage ---
    from data.store import store_name, init_db, save_snapshot, load_snapshot, delete_snapshot
    pid = ensure_browser_id()
    init_db()
    hydrate_once_for(pid)

    # --- Debug UI (only in dev mode) ---
    if dev:
        import os, json
        db_url = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
        with st.sidebar.expander("DB self-test", expanded=False):
            pid_for_test = st.session_state.get("browser_id", "no-id")
            if st.button("Write probe row"):
                init_db()
                save_snapshot(pid_for_test, "PROBE_SCENE", ["A","B"], [{"role":"assistant","content":"probe"}])
                st.success("Wrote probe row")
            if st.button("Read probe row"):
                snap = load_snapshot(pid_for_test)
                st.code(json.dumps(snap, indent=2))

    # --- Fixed slots for this run ---
    scene_ph   = st.empty()
    choices_ph = st.empty()

    # --- Deferred work FIRST (no duplicate text) ---
    # 1) Starting a brand new story
    if st.session_state.pop("_start_new_pending", False):
        st.session_state["history"] = []
        st.session_state["scene"] = None
        st.session_state["choices"] = []
        _advance_turn(pid, scene_ph, choices_ph, anim_enabled)
        return  # prevent old UI from rendering under streamed scene

    # 2) Applying a pending choice
    pending = st.session_state.pop("_pending_choice", None)
    if pending is not None:
        st.session_state["history"].append({"role": "user", "content": pending})
        _advance_turn(pid, scene_ph, choices_ph, anim_enabled)
        return

    # --- Sidebar controls → actions ---
    action = sidebar_controls(pid)
    if action == "start":
        start_new_story(pid)
        st.rerun()
    elif action == "reset":
        # clear only in-memory (keep DB row)
        for k in ("scene", "choices", "history"):
            st.session_state.pop(k, None)
        st.experimental_set_query_params()  # no-op on newer Streamlit; tidy on older
        st.rerun()
        
    # --- Normal render (no pending work) ---
    if not st.session_state.get("scene"):
        st.info("Click **Start New Story** in the sidebar to begin.")
    else:
        scene_ph.markdown(st.session_state["scene"])
        turn_id = len(st.session_state.get("history", []))
        render_choices(st.session_state.get("choices", []), choices_ph, turn_id)

    # --- Choice handler (after UI triggers) ---
    picked = st.session_state.pop("_picked", None)
    if picked:
        apply_choice(pid, picked)
        st.rerun()

    # (No public debug/footer text; dev UI is gated above)

if __name__ == "__main__":
    main()
