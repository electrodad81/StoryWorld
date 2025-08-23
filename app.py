import os
from typing import Optional
import time
import json
import streamlit as st
import html as _html
from html import escape as _esc

# --- UI helpers (assumes these modules exist in your repo) ---
from ui.anim import inject_css, render_thinking
from ui.streaming import stream_text
from ui.choices import render_choices_grid
from ui.controls import sidebar_controls  # adjust import if your project structure differs

# --- Story engine ---
from story.engine import stream_scene, generate_choices

# --- Persistence / telemetry ---
from data.store import (
    init_db, save_snapshot, load_snapshot, delete_snapshot,
    has_snapshot, save_visit, save_event
)

# --- Lore data ---
import pathlib
ROOT = pathlib.Path(__file__).parent
LORE_PATH = ROOT / "lore.json"
LORE = json.loads(LORE_PATH.read_text(encoding="utf-8")) if LORE_PATH.exists() else {}

# --- Constants ---
CHOICE_COUNT = 2
APP_TITLE = "Gloamreach"

# -----------------------------------------------------------------------------
# Risk & Consequence utilities
#
# A set of keywords that indicate a risky choice. Feel free to tweak/extend
# these terms to better fit your storyworld. The words are matched
# case-insensitively as whole words using regular expressions.
import re
_RISKY_WORDS = {
    "charge", "attack", "fight", "steal", "stab", "break", "smash", "dive",
    "jump", "descend", "enter", "drink", "touch", "open the", "confront",
    "cross", "swim", "sprint", "bait", "ambush", "bleed", "sacrifice", "shout",
    "brave", "risk", "gamble", "rush", "kick", "force", "pry", "ignite", "set fire"
}

def is_risky_label(label: str) -> bool:
    """Return True if the given choice label contains any risky keywords."""
    s = (label or "").lower()
    return any(re.search(rf"\b{re.escape(w)}\b", s) for w in _RISKY_WORDS)

def update_consequence_counters(picked_label: Optional[str]) -> None:
    """
    Update the in-session counters that track consecutive risky choices.
    On a risky pick, increment ``danger_streak``. On a safe pick, decrement the streak
    (never below zero). These counters live only in session state and are not
    persisted to the database. A ``picked_label`` of ``None`` or ``"__start__"``
    is ignored.
    """
    if not picked_label or picked_label == "__start__":
        return
    if is_risky_label(picked_label):
        st.session_state["danger_streak"] = st.session_state.get("danger_streak", 0) + 1
    else:
        # Safe picks reduce the streak (but not below zero)
        st.session_state["danger_streak"] = max(0, st.session_state.get("danger_streak", 0) - 1)

def detect_cost_in_scene(text: str) -> bool:
    """
    Heuristic check to see if the scene text contains a visible cost or setback.
    Looks for a variety of words related to injury, loss, time pressure, or being
    trapped. Returns True if any cost indicator is present.
    """
    cost_terms = [
        "bleed", "wound", "cut", "bruis", "sprain", "fracture", "poison",
        "lose", "lost", "dropped", "broke", "shattered", "torn", "spent",
        "captured", "seized", "trapped", "cornered", "exposed", "compromised",
        "too late", "ticking", "deadline", "out of time", "cannot return",
        "ally leaves", "ally falls", "alone now", "weakened", "limp"
    ]
    t = (text or "").lower()
    return any(w in t for w in cost_terms)

def render_death_options(pid: str, slot) -> None:
    """
    Render the death fail-state options panel. This appears only when a scene
    ends with the special ``[DEATH]`` tag. Offers the player two buttons:
    'Restart this story' (continuing with the same Name/Gender/Character) and
    'Start a new story' (fresh run). Both options clear session state and
    persisted snapshot before queuing a new '__start__' turn.
    """
    with slot.container():
        st.subheader("Your journey ends here.")
        st.caption("Fate is not always kind in Gloamreach.")
        c1, c2 = st.columns(2)
        restart = c1.button("Restart this story", use_container_width=True, key="death_restart")
        anew    = c2.button("Start a new story", use_container_width=True, key="death_new")
        if restart or anew:
            # Log the player's choice (restart vs new)
            try:
                save_event(pid, "death_choice", {"action": "restart" if restart else "new"})
            except Exception:
                pass
            # Clear the current run but keep locked selections (Name/Gender/Character)
            for k in (
                "scene", "choices", "history", "pending_choice", "is_generating",
                "t_choices_visible_at", "t_scene_start", "is_dead", "beat_index", "story_complete"
            ):
                st.session_state.pop(k, None)
            try:
                delete_snapshot(pid)
            except Exception:
                pass
            # Reset beat index if story mode is on
            if st.session_state.get("story_mode"):
                st.session_state["beat_index"] = 0
                st.session_state["story_complete"] = False
            st.session_state["pending_choice"] = "__start__"
            st.session_state["is_generating"] = True
            st.rerun()

# --------------------------
# Utility: identity / session
# --------------------------
def ensure_keys():
    st.session_state.setdefault("scene", "")
    st.session_state.setdefault("choices", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending_choice", None)
    st.session_state.setdefault("is_generating", False)
    st.session_state.setdefault("t_scene_start", None)
    st.session_state.setdefault("t_choices_visible_at", None)

    # Player profile
    st.session_state.setdefault("player_name", None)
    st.session_state.setdefault("player_gender", "Unspecified")
    st.session_state.setdefault("player_archetype", "Default")

    # Risk / death flags
    st.session_state.setdefault("danger_streak", 0)
    st.session_state.setdefault("injury_level", 0)
    st.session_state.setdefault("is_dead", False)

    # Story mode toggle + beat tracking
    st.session_state.setdefault("story_mode", False)
    st.session_state.setdefault("beat_index", 0)
    st.session_state.setdefault("story_complete", False)

    # Developer tools toggle (controls whether debug info is shown in sidebar)
    st.session_state.setdefault("_dev", False)

    # Beat scene counter (tracks scenes within the current beat)
    st.session_state.setdefault("_beat_scene_count", 0)


def reset_session(full_reset=False):
    """Clear out session. If full_reset, also clear player profile."""
    keep = {}
    if not full_reset:
        keep["player_name"] = st.session_state.get("player_name")
        keep["player_gender"] = st.session_state.get("player_gender", "Unspecified")
        keep["player_archetype"] = st.session_state.get("player_archetype", "Default")
        keep["story_mode"] = bool(st.session_state.get("story_mode", False))

    for k in list(st.session_state.keys()):
        del st.session_state[k]

    ensure_keys()
    for k, v in keep.items():
        st.session_state[k] = v


# --------------------------
# Story Mode helpers (beats)
# --------------------------
BEATS = ["exposition", "rising_action", "climax", "falling_action", "resolution"]

# Target number of scenes per beat. Once the count meets or exceeds the target,
# the story progresses to the next beat. Feel free to adjust counts to tune
# pacing (classic CYOA arcs usually have ~8–12 scenes total).
BEAT_TARGET_SCENES = {
    "exposition": 2,
    "rising_action": 3,
    "climax": 2,
    "falling_action": 2,
    "resolution": 1,
}

def get_current_beat(state) -> str:
    i = int(state.get("beat_index", 0))
    i = max(0, min(i, len(BEATS) - 1))
    return BEATS[i]

def advance_beat():
    st.session_state["beat_index"] = min(st.session_state["beat_index"] + 1, len(BEATS) - 1)

def mark_story_complete():
    st.session_state["story_complete"] = True

def is_story_complete(state) -> bool:
    return bool(state.get("story_complete", False))


# --------------------------
# Ending UI (non-death)
# --------------------------
# app.py – replace the entire contents of render_end_options
def render_end_options(pid: str, slot) -> None:
    """
    Render the conclusion message. When the story arc completes, inform the player
    that the adventure is over and prompt them to start a new story using the sidebar.
    """
    with slot.container():
        st.subheader("Your adventure concludes.")
        st.caption("The story arc has come to an end.")
        # Display a simple informational message instead of restart buttons
        st.info(
            "To begin a new adventure, use the **Start New Story** button in the left sidebar."
        )

# --------------------------
# Main advance turn
# --------------------------
def _advance_turn(pid: str, story_slot, grid_slot, anim_enabled: bool = True):
    """
    Advance the story by streaming the next scene, updating history and choices,
    and handling death or story completion. In Story Mode, beat progression
    determines when the arc concludes; in classic mode, the story continues
    indefinitely until death.
    """
    old_choices = list(st.session_state.get("choices", []))
    picked = st.session_state.get("pending_choice")

    # Update risk counters (danger streak / injury level) based on the picked choice
    update_consequence_counters(picked)

    # Choice telemetry
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

    # Keep the choice buttons visible and greyed-out while the next scene streams
    # (this draws a disabled “Generating…” button in each slot)
    render_choices_grid(grid_slot, choices=None, generating=True, count=CHOICE_COUNT)

    # Determine beat only if Story Mode is ON
    beat = get_current_beat(st.session_state) if st.session_state.get("story_mode") else None

    # Stream next scene
    st.session_state["t_scene_start"] = time.time()
    if anim_enabled:
        full = stream_text(stream_scene(st.session_state["history"], LORE, beat=beat), story_slot, show_caret=True)
    else:
        full = stream_text(stream_scene(st.session_state["history"], LORE, beat=beat), story_slot, show_caret=False)

    # Check for death sentinel: if scene ends with '[DEATH]', set died flag and strip it
    died = False
    stripped = full.rstrip()
    if stripped.endswith("[DEATH]"):
        died = True
        # Remove the sentinel from the displayed text
        full = stripped[: -len("[DEATH]")].rstrip()

    # Append scene to history
    st.session_state["history"].append({"role": "assistant", "content": full})
    st.session_state["scene"] = full

    # Evaluate expected cost: if pick was risky or must_escalate (danger streak ≥ 2)
    must_escalate = st.session_state.get("danger_streak", 0) >= 2
    expected_cost = (
        bool(picked and picked != "__start__" and is_risky_label(picked)) or must_escalate
    )
    # If we expected a cost but none found and not dead, log a miss
    if expected_cost and not died and not detect_cost_in_scene(full):
        try:
            save_event(pid, "missed_consequence", {
                "picked": picked,
                "danger_streak": st.session_state.get("danger_streak", 0),
            })
        except Exception:
            pass

    # If dead: mark state, log death, persist with no choices, show death panel, and exit
    if died:
        st.session_state["is_dead"] = True
        try:
            save_event(pid, "death", {
                "picked": picked,
                "decisions_count": sum(1 for m in st.session_state["history"] if m.get("role") == "user"),
                "danger_streak": st.session_state.get("danger_streak", 0),
            })
        except Exception:
            pass
        # Persist snapshot with no choices (terminal scene)
        username  = st.session_state.get("player_name") or st.session_state.get("player_username") or None
        gender    = st.session_state.get("player_gender")
        archetype = st.session_state.get("player_archetype")
        try:
            save_snapshot(pid, full, [], st.session_state["history"],
                          username=username, gender=gender, archetype=archetype)
        except TypeError:
            save_snapshot(pid, full, [], st.session_state["history"], username=username)
        # Show the death options panel and halt
        render_death_options(pid, grid_slot)
        return

    # Beat progression (Story Mode only)
    if st.session_state.get("story_mode"):
        # Increment scene counter for the current beat
        st.session_state["_beat_scene_count"] = st.session_state.get("_beat_scene_count", 0) + 1
        current_beat = get_current_beat(st.session_state)
        # Determine target scenes for this beat
        target = BEAT_TARGET_SCENES.get(current_beat, 1)
        # If we've reached or exceeded the target, move to next beat or conclude
        if st.session_state["_beat_scene_count"] >= target:
            if current_beat == "resolution":
                # End of resolution: mark story complete
                mark_story_complete()
            else:
                # Advance to next beat and reset counter
                st.session_state["beat_index"] = min(st.session_state.get("beat_index", 0) + 1, len(BEATS) - 1)
                st.session_state["_beat_scene_count"] = 0

    # If story is complete, don't generate more choices; show ending options
    if is_story_complete(st.session_state):
        # Persist final scene
        username  = st.session_state.get("player_name") or st.session_state.get("player_username") or None
        gender    = st.session_state.get("player_gender")
        archetype = st.session_state.get("player_archetype")
        try:
            save_snapshot(pid, full, [], st.session_state["history"], username=username, gender=gender, archetype=archetype)
        except TypeError:
            save_snapshot(pid, full, [], st.session_state["history"], username=username)
        # Show conclusion panel
        render_end_options(pid, grid_slot)
        return

    # Generate choices (free mode and story mode alike)
    choices = generate_choices(st.session_state["history"], full, LORE)
    st.session_state["choices"] = choices

    # Persist snapshot
    username  = st.session_state.get("player_name") or st.session_state.get("player_username") or None
    gender    = st.session_state.get("player_gender")
    archetype = st.session_state.get("player_archetype")
    try:
        save_snapshot(pid, full, choices, st.session_state["history"],
                      username=username, gender=gender, archetype=archetype)
    except TypeError:
        save_snapshot(pid, full, choices, st.session_state["history"], username=username)

    # Visit record
    choice_text, choice_index = None, None
    if picked and picked != "__start__":
        choice_text = str(picked)
        try:
            choice_index = int(old_choices.index(choice_text))
        except Exception:
            choice_index = None
    save_visit(pid, full, choice_text, choice_index)

    # Scene telemetry
    word_count = len(full.split())
    save_event(pid, "scene", {
        "word_count": word_count,
        "has_choice": bool(choices),
        "choices_count": len(choices or []),
        "beat": beat if beat else "classic",
    })

    # Render real buttons
    render_choices_grid(grid_slot, choices=choices, generating=False, count=CHOICE_COUNT)
    st.session_state["t_choices_visible_at"] = time.time()


# --------------------------
# Onboarding UI
# --------------------------
def onboarding(pid: str):
    st.header("Begin Your Journey")
    st.markdown("Pick your setup. Name and character are locked once you begin.")

    name = st.text_input("Name", value=st.session_state.get("player_name") or "", max_chars=24)
    #gender = st.selectbox("Gender", ["Unspecified", "Female", "Male", "Nonbinary"], index=0)
    archetype = st.selectbox("Character type", ["Default"], index=0)

    # Mode selection: Story Mode vs. Exploration
    mode_default_index = 1 if st.session_state.get("story_mode", True) else 1
    mode = st.radio(
        "Mode",
        options=["Story Mode", "Exploration"],
        index=mode_default_index,
        help=(
            "Story Mode: guided 5‑beat arc (Exposition → Rising Action → Climax → "
            "Falling Action → Resolution). "
            "Exploration: classic freeform with risk/death."
        ),
    )

    col1, col2 = st.columns([1, 1])
    # Provide unique keys for buttons to avoid StreamlitDuplicateElementId errors.
    begin = col1.button("Begin Adventure", use_container_width=True, disabled=not name.strip(), key="onboard_begin")
    reset = col2.button("Reset Session", use_container_width=True, key="onboard_reset")

    if reset:
        delete_snapshot(pid)
        reset_session(full_reset=True)
        st.rerun()

    if begin:
        st.session_state["player_name"] = name.strip()
        st.session_state["player_gender"] = gender
        st.session_state["player_archetype"] = archetype
        st.session_state["story_mode"] = bool(story_mode)
        # Reset beat index on new start
        st.session_state["beat_index"] = 0
        st.session_state["story_complete"] = False
        # Queue first turn
        st.session_state["pending_choice"] = "__start__"
        st.session_state["is_generating"] = True
        st.rerun()


# --------------------------
# Main
# --------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🕯️", layout="wide")
    inject_css()
    ensure_keys()
    init_db()

    pid = "local-user"  # simple identity; adjust as needed

    # Sidebar controls (render early)
    try:
        action = sidebar_controls(pid)
    except Exception:
        action = None

    # Developer tools panel in sidebar. This displays debug information when enabled.
    # Initialize _dev flag on first run if not already set.
    if "_dev" not in st.session_state:
        st.session_state["_dev"] = False
    # Render dev toggle and info in the sidebar after the core controls
    with st.sidebar:
        # Checkbox to toggle developer tools
        st.session_state["_dev"] = st.checkbox(
            "Developer tools", value=bool(st.session_state["_dev"]), help="Show debug info & self-tests"
        )
        if st.session_state["_dev"]:
            # Show basic debug information
            beat = get_current_beat(st.session_state) if st.session_state.get("story_mode") else "classic"
            st.caption(
                f"Story Mode: {'on' if st.session_state.get('story_mode') else 'off'} • "
                f"Beat: {beat} • Danger streak: {st.session_state.get('danger_streak', 0)} • "
                f"Scenes: {len(st.session_state.get('history', []))}"
            )

    # Handle sidebar actions
    if action == "start":
        # Hard reset persisted + in-memory, then queue first turn
        try:
            delete_snapshot(pid)
        except Exception:
            pass
        for k in (
            "scene", "choices", "history", "pending_choice", "is_generating",
            "t_choices_visible_at", "t_scene_start", "is_dead", "beat_index", "story_complete"
        ):
            st.session_state.pop(k, None)
        # Reset beat index for story mode
        if st.session_state.get("story_mode"):
            st.session_state["beat_index"] = 0
            st.session_state["story_complete"] = False
        st.session_state["pending_choice"] = "__start__"
        st.session_state["is_generating"] = True
        st.rerun()
    elif action == "reset":
        # Full reset (clear profile)
        try:
            delete_snapshot(pid)
        except Exception:
            pass
        for k in (
            "scene", "choices", "history", "pending_choice", "is_generating",
            "t_choices_visible_at", "t_scene_start", "is_dead", "beat_index", "story_complete",
            "player_name", "player_gender", "player_archetype", "story_mode"
        ):
            st.session_state.pop(k, None)
        st.rerun()

    # Fixed body slots
    scene_ph   = st.empty()
    choices_ph = st.empty()

    # Keep the storybox visible with the most recent scene
    _last = st.session_state.get("scene")
    if _last:
        scene_ph.markdown(
            f"<div class='storybox'>{_esc(_last).replace('\\n', ' ')}</div>",
            unsafe_allow_html=True
        )

    # Pending work
    if st.session_state.get("pending_choice") is not None:
        _advance_turn(pid, scene_ph, choices_ph, anim_enabled=True)
        st.session_state["pending_choice"] = None
        st.session_state["is_generating"] = False
        st.stop()

    # Brand new or after reset: show onboarding
    if not st.session_state.get("scene"):
        onboarding(pid)
        return

    # Render current scene and choices
    if st.session_state.get("scene"):
        scene_ph.markdown(
            f"<div class='storybox'>{_esc(st.session_state['scene']).replace('\\n',' ')}</div>",
            unsafe_allow_html=True
            )
    render_choices_grid(
        choices_ph,
        choices=st.session_state.get("choices", []),
        generating=False,
        count=CHOICE_COUNT
    )
    st.session_state["t_choices_visible_at"] = time.time()


if __name__ == "__main__":
    main()