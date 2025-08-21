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

import time
from data.store import save_event  # after you export it

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
        st.session_state["player_gender"] = snap["gender"]
        # Only set username if the widget hasn't created the key this run
        if snap and snap.get("username"):
            st.session_state["player_name"] = snap["username"]
            st.session_state.setdefault("player_username", snap["username"])  # compat

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

def render_sidebar_username(story_started: bool):
    with st.sidebar:
        if story_started:
            st.session_state.setdefault("player_username", "")
            st.text_input(
                "Player username (optional)",
                key="player_username",
                placeholder="e.g., Nova",
                max_chars=24,
                help="Used in NPC dialogue only. You can change this anytime."
            )
        else:
            st.caption("Set your username on the start screen.")

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

def _advance_turn(pid: str, story_slot, grid_slot, anim_enabled: bool = True):
    from data.store import save_snapshot, save_visit, save_event
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("choices", [])
    st.session_state.setdefault("scene", "")

    old_choices = list(st.session_state.get("choices", []))
    picked = st.session_state.get("pending_choice")

    # --- CHOICE EVENT + latency (time from choices visible → click)
    if picked and picked != "__start__":
        hist = st.session_state["history"]
        if not hist or hist[-1].get("role") != "user" or hist[-1].get("content") != picked:
            hist.append({"role": "user", "content": picked})

        visible_at = st.session_state.get("t_choices_visible_at")
        latency_ms = int((time.time() - visible_at) * 1000) if visible_at else None
        save_event(pid, "choice", {
            "label": picked,
            "index": (old_choices.index(picked) if picked in old_choices else None),
            "latency_ms": latency_ms,
            "decisions_count": sum(1 for m in hist if m.get("role") == "user"),
        })

    # keep grid visible during work
    render_choices_grid(grid_slot, choices=None, generating=True, count=CHOICE_COUNT)

    # --- mark scene stream start (dwell timer)
    st.session_state["t_scene_start"] = time.time()

    # stream scene
    if anim_enabled:
        with lantern("Summoning the next scene…"):
            full = stream_text(stream_scene(st.session_state["history"], LORE), story_slot, show_caret=True)
    else:
        full = stream_text(stream_scene(st.session_state["history"], LORE), story_slot, show_caret=False)

    # finalize history
    st.session_state["history"].append({"role": "assistant", "content": full})
    st.session_state["scene"] = full

    # choices next
    choices = generate_choices(st.session_state["history"], full, LORE)
    st.session_state["choices"] = choices

    # --- persist snapshot (username + gender/archetype if supported)
    username  = st.session_state.get("player_name") or st.session_state.get("player_username") or None
    gender    = st.session_state.get("player_gender")
    archetype = st.session_state.get("player_archetype")
    try:
        save_snapshot(pid, full, choices, st.session_state["history"],
                    username=username, gender=gender, archetype=archetype)
    except TypeError:
        save_snapshot(pid, full, choices, st.session_state["history"], username=username)


    # visit row
    choice_text, choice_index = None, None
    if picked and picked != "__start__":
        choice_text = str(picked)
        try:
            choice_index = int(old_choices.index(choice_text))
        except Exception:
            choice_index = None
    save_visit(pid, full, choice_text, choice_index)

    # --- SCENE EVENT: word count + dwell (time on scene until next click measured next run)
    word_count = len(full.split())
    save_event(pid, "scene", {
        "word_count": word_count,
        "has_choice": bool(choices),
        "choices_count": len(choices or []),
    })

    # render fresh buttons
    render_choices_grid(grid_slot, choices=choices, generating=False, count=CHOICE_COUNT)


def render_onboarding(pid: str):
    st.header("Begin Your Journey")
    st.markdown("Pick your setup. Name and character are locked once you begin.")

    # Required Name
    name = st.text_input("Name", key="onb_name", placeholder="e.g., Nova", max_chars=24)
    gender = st.selectbox("Gender", ["Unspecified", "Female", "Male", "Nonbinary"], index=0, key="onb_gender")
    archetype = st.selectbox("Character type", ["Default"], index=0, key="onb_archetype")

    c1, c2 = st.columns([1, 1])
    begin = c1.button("Begin Adventure", use_container_width=True, disabled=not name.strip())
    reset  = c2.button("Reset Choices", use_container_width=True)

    if begin:
        # Lock selections for this session
        st.session_state["player_name"] = name.strip()
        st.session_state["player_username"] = st.session_state["player_name"]  # compat with existing code
        st.session_state["player_gender"] = gender
        st.session_state["player_archetype"] = archetype

        # Queue first turn
        st.session_state["pending_choice"] = "__start__"
        st.session_state["is_generating"] = True
        st.rerun()

    if reset:
        for k in ("onb_name", "onb_gender", "onb_archetype"):
            st.session_state.pop(k, None)
        st.rerun()

def main():
    st.title("Gloamreach — Storyworld MVP")

    # --- Atmosphere: always ON, no checkbox ---
    anim_enabled = True
    inject_css()

    # --- Developer mode ---
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
    from data.store import init_db, save_snapshot, load_snapshot, delete_snapshot, has_snapshot, save_event
    pid = ensure_browser_id()
    init_db()

    # --- Fixed body slots ---
    scene_ph   = st.empty()
    choices_ph = st.empty()

    # --- Sidebar controls: render EARLY every run so they never vanish ---
    try:
        from ui.controls import sidebar_controls
    except Exception:
        from controls import sidebar_controls  # fallback if file lives at project root
    action = sidebar_controls(pid)

    if action == "start":
        # Hard reset persisted + in-memory, then queue first turn
        try:
            delete_snapshot(pid)
        except Exception:
            pass
        for k in ("scene", "choices", "history", "pending_choice", "is_generating",
                  "choices_before", "t_choices_visible_at", "t_scene_start"):
            st.session_state.pop(k, None)

        st.session_state["pending_choice"] = "__start__"
        st.session_state["is_generating"] = True
        try:
            save_event(pid, "start", {"username": st.session_state.get("player_username") or None})
        except Exception:
            pass
        st.rerun()

    elif action == "reset":
        try:
            delete_snapshot(pid)
        except Exception:
            pass
        for k in (
            "scene","choices","history","pending_choice","is_generating","choices_before",
            "t_choices_visible_at","t_scene_start",
            # also clear locked selections:
            "player_name","player_username","player_gender","player_archetype"
        ):
            st.session_state.pop(k, None)
        st.rerun()

    # --- Hydrate AFTER handling actions ---
    hydrate_once_for(pid)

    # --- Pending work FIRST (no duplicate text) ---
    # Buttons remain visible because we rendered them already above.
    if st.session_state.get("pending_choice") is not None:
        _advance_turn(pid, scene_ph, choices_ph, anim_enabled)
        st.session_state["pending_choice"] = None
        return

    # --- Onboarding for brand-new players (no scene and no snapshot) ---
    brand_new = (not st.session_state.get("scene")) and (not has_snapshot(pid))
    if brand_new:
        render_onboarding(pid)
        return  # prevents sidebar username from rendering this run

    # --- Sidebar username (render exactly once, post-onboarding) ---
    with st.sidebar:
        # Show locked Name (no editing)
        name_locked = st.session_state.get("player_name") or st.session_state.get("player_username") or ""
        if name_locked:
            st.caption("Name")
            st.text_input("Name", value=name_locked, key="locked_name", disabled=True)

    # --- Debug UI (dev only) ---
    if dev:
        import os, json
        db_url = os.getenv("DATABASE_URL") or st.secrets.get("DATABASE_URL")
        with st.sidebar.expander("DB self-test", expanded=False):
            pid_for_test = st.session_state.get("browser_id", "no-id")
            if st.button("Write probe row"):
                init_db()
                save_snapshot(pid_for_test, "PROBE_SCENE", ["A", "B"], [{"role": "assistant", "content": "probe"}])
                st.success("Wrote probe row")
            if st.button("Read probe row"):
                snap = load_snapshot(pid_for_test)
                st.code(json.dumps(snap, indent=2))

    # --- Normal render ---
    if not st.session_state.get("scene"):
        st.info("Click **Start New Story** in the sidebar to begin.")
    else:
        # Constrain the width of the story text and center it
        scene_html = f"<div style='max-width: 700px; margin-left: auto; margin-right: auto;'>{st.session_state['scene']}</div>"
        scene_ph.markdown(scene_html, unsafe_allow_html=True)
        from ui.choices import render_choices_grid
        render_choices_grid(
            choices_ph,
            choices=st.session_state.get("choices", []),
            generating=False,
            count=CHOICE_COUNT if "CHOICE_COUNT" in globals() else 2,
        )
    
if __name__ == "__main__":
    main()
