# app.py
from __future__ import annotations
from typing import List, Optional
import streamlit as st
from core.identity import ensure_browser_id
from data.store import has_snapshot, delete_snapshot, load_snapshot, save_snapshot, save_visit
import json
from story.engine import stream_scene, generate_choices
from ui.anim import inject_css

from ui.loader import lantern
from ui.streaming import stream_text
from ui.choices import render_choices_grid

from data.store import save_visit

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
    """Load snapshot once per pid into session_state (scene, choices, history, username)."""
    if st.session_state.get("hydrated_for_pid") == pid:
        return

    snap = load_snapshot(pid)

    if snap:
        st.session_state["scene"]   = snap.get("scene") or ""
        st.session_state["choices"] = snap.get("choices") or []
        st.session_state["history"] = snap.get("history") or []
        # Only set username if the widget hasn't created the key this run
        if snap.get("username") and "player_username" not in st.session_state:
            st.session_state["player_username"] = snap["username"]
        else:
            st.session_state.setdefault("player_username", "")
    else:
        st.session_state["scene"] = ""
        st.session_state["choices"] = []
        st.session_state["history"] = []
        st.session_state.setdefault("player_username", "")

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

def start_new_story():
    """Kick the first deferred turn without touching browser_id."""
    st.session_state["pending_choice"] = "__start__"
    st.session_state["is_generating"] = True
    st.rerun()

def continue_story():
    """Resume by running the next turn if we already have state."""
    st.session_state["pending_choice"] = "__start__"
    st.session_state["is_generating"] = True
    st.rerun()

def reset_story():
    """Hard reset: wipe persisted snapshot and local state (keeps browser_id)."""
    pid = st.session_state.get("browser_id")
    try:
        if pid and has_snapshot(pid):
            delete_snapshot(pid)
    except Exception:
        pass
    for k in ("scene","choices","history","pending_choice","is_generating","choices_before"):
        if k in st.session_state:
            del st.session_state[k]
    start_new_story()

def sidebar_controls(pid: str) -> Optional[str]:
    st.sidebar.markdown("### Story Controls")
    # Show both buttons explicitly
    new_clicked = st.sidebar.button("Start New Story", use_container_width=True)
    cont_clicked = st.sidebar.button("Continue", type="primary", use_container_width=True)
    reset_clicked = st.sidebar.button("Reset", type="secondary", use_container_width=True)

    if new_clicked:
        reset_story()          # guarantees fresh start, then reruns
        return "start_new"
    if cont_clicked:
        continue_story()       # continues current slot, then reruns
        return "continue"
    if reset_clicked:
        reset_story()          # wipes and starts, then reruns
        return "reset"
    return None

# --- reliable choice grid (inline) -----------------------------------
def _current_turn_id() -> int:
    """0-based count of user decisions; used to keep button keys unique."""
    hist = st.session_state.get("history", [])
    return sum(1 for m in hist if isinstance(m, dict) and m.get("role") == "user")

def apply_choice(choice_label: str) -> None:
    """Set flags that the main loop expects, then rerun."""
    st.session_state["pending_choice"] = choice_label
    st.session_state["is_generating"] = True
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
    from data.store import save_snapshot, save_visit
    # Ensure required state exists
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("choices", [])
    st.session_state.setdefault("scene", "")

    # Capture prior choices (for choice_index) and the picked label
    old_choices = list(st.session_state.get("choices", []))
    picked = st.session_state.get("pending_choice")

    # Record the player's selection as a 'user' turn (once)
    if picked and picked != "__start__":
        hist = st.session_state["history"]
        if not hist or hist[-1].get("role") != "user" or hist[-1].get("content") != picked:
            hist.append({"role": "user", "content": picked})

    # Keep grid visible during work
    render_choices_grid(grid_slot, choices=None, generating=True, count=CHOICE_COUNT)

    # Stream the next scene (lantern + caret if enabled)
    if anim_enabled:
        with lantern("Summoning the next scene…"):
            full = stream_text(stream_scene(st.session_state["history"], LORE),
                               story_slot, show_caret=True)
    else:
        full = stream_text(stream_scene(st.session_state["history"], LORE),
                           story_slot, show_caret=False)

    # Save to history
    st.session_state["history"].append({"role": "assistant", "content": full})
    st.session_state["scene"] = full

    # Compute choices (fast) and stash
    choices = generate_choices(st.session_state["history"], full, LORE)
    st.session_state["choices"] = choices

    # Persist snapshot (decisions_count auto-computed) — pass USERNAME here
    username = st.session_state.get("player_username") or None
    save_snapshot(pid, full, choices, st.session_state["history"], username=username)

    # Log this screen in story_visits with the choice that led here (if any)
    choice_text, choice_index = None, None
    if picked and picked != "__start__":
        choice_text = str(picked)
        try:
            choice_index = int(old_choices.index(choice_text))  # 0-based
        except Exception:
            choice_index = None
    save_visit(pid, full, choice_text, choice_index)

    # Render fresh buttons
    render_choices_grid(grid_slot, choices=choices, generating=False, count=CHOICE_COUNT)



def main():
    st.title("Gloamreach — Storyworld MVP")

    # --- Atmosphere toggle (visible to everyone) ---
    with st.sidebar:
        anim_enabled = st.checkbox("Atmosphere (lantern + typewriter)", value=True)  # default ON if you like
    if anim_enabled:
        inject_css()

    # --- Developer mode (hidden unless ?dev=1 or DEBUG_UI=true) ---
    if "_dev" not in st.session_state:
        try:
            st.session_state["_dev"] = _dev_default()
        except NameError:
            st.session_state["_dev"] = False
    with st.sidebar:
        if st.session_state["_dev"]:
            st.session_state["_dev"] = st.checkbox("Developer tools", value=True, help="Show debug info & self-tests")
    dev = bool(st.session_state["_dev"])

    # --- Identity & storage ---
    from data.store import init_db, save_snapshot, load_snapshot, delete_snapshot
    pid = ensure_browser_id()
    init_db()

    # --- Always render sidebar controls up front (hard-reset semantics) ---
    from ui.controls import sidebar_controls
    action = sidebar_controls(pid)
    if action == "start":
        # Hard reset + immediately start fresh
        try:
            delete_snapshot(pid)      # wipe persisted progress
        except Exception:
            pass
        for k in ("scene", "choices", "history", "pending_choice", "is_generating", "choices_before"):
            st.session_state.pop(k, None)
        st.session_state["pending_choice"] = "__start__"   # queue first turn
        st.session_state["is_generating"] = True
        st.rerun()

    elif action == "reset":
        # Hard reset; do NOT auto-start
        try:
            delete_snapshot(pid)
        except Exception:
            pass
        for k in ("scene", "choices", "history", "pending_choice", "is_generating", "choices_before"):
            st.session_state.pop(k, None)
        st.rerun()

    # --- Hydrate AFTER handling actions so we don't revive old snapshots ---
    hydrate_once_for(pid)

    # now render the sidebar
    with st.sidebar:
        username = st.text_input(
            "Player username (optional)",
            key="player_username",          # don't pass `value=`; let session_state supply it
            placeholder="e.g., Bastion",
            max_chars=24,
            help="Used in NPC dialogue only. You can change this anytime.",
        )

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

    # --- Deferred work (no duplicate text) ---
    if st.session_state.get("pending_choice") is not None:
        _advance_turn(pid, scene_ph, choices_ph, anim_enabled)
        st.session_state["pending_choice"] = None
        return

    # --- Normal render (no pending work) ---
    if not st.session_state.get("scene"):
        st.info("Click **Start New Story** in the sidebar to begin.")
    else:
        scene_ph.markdown(st.session_state["scene"])
        from ui.choices import render_choices_grid
        render_choices_grid(
            choices_ph,
            choices=st.session_state.get("choices", []),
            generating=False,
            count=CHOICE_COUNT if "CHOICE_COUNT" in globals() else 2,
        )



if __name__ == "__main__":
    main()
