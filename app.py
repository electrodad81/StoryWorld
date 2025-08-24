import os
from typing import Optional
import time
import json
import streamlit as st

# --- UI helpers (assumes these modules exist in your repo) ---
from ui.anim import inject_css, render_thinking
from ui.streaming import stream_text
from ui.choices import render_choices_grid
from ui.controls import sidebar_controls  # adjust import if your project structure differs

# --- Story engine ---
from story.engine import stream_scene, generate_choices, generate_illustration

# --- Persistence / telemetry ---
from data.store import (
    init_db, save_snapshot, load_snapshot, delete_snapshot,
    has_snapshot, save_visit, save_event
)

from concurrent.futures import ThreadPoolExecutor
import hashlib, re

@st.cache_resource
def _ill_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)

def _has_first_sentence(txt: str) -> Optional[str]:
    """Return the first complete sentence with a minimum length, else None."""
    m = re.search(r"(.+?[.!?])(\s|$)", txt or "")
    if not m:
        return None
    sent = m.group(1).strip()
    return sent if len(sent.split()) >= 8 else None  # avoid tiny fragments

def _scene_key(text: str, simple: bool) -> str:
    h = hashlib.md5((text.strip() + f"|{simple}").encode("utf-8")).hexdigest()
    return f"ill-{h}"

def _start_illustration_job(seed_sentence: str, simple: bool) -> str:
    """Kick off background illustration generation for a seed sentence."""
    key = _scene_key(seed_sentence, simple)
    jobs = st.session_state.setdefault("ill_jobs", {})
    if key in jobs:
        return key
    fut = _ill_executor().submit(generate_illustration, seed_sentence, simple)  # returns (img_ref, dbg) or single
    jobs[key] = {"future": fut, "result": None}
    return key

def _job_for_scene(scene_text: str, simple: bool):
    """Return (key, job_dict|None) for the current scene."""
    seed = _has_first_sentence(scene_text) or (scene_text or "").strip()
    key  = _scene_key(seed, simple)
    jobs = st.session_state.get("ill_jobs", {})
    return key, jobs.get(key)

# (optional) if you want a caret while streaming and you're not already adding one in CSS:
def _with_caret(txt: str) -> str:
    return f'{txt}<span class="caret">▋</span>'

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
    #st.session_state.setdefault("player_gender", "Unspecified")
    st.session_state.setdefault("player_archetype", "Default")

    # Risk / death flags
    st.session_state.setdefault("danger_streak", 0)
    st.session_state.setdefault("injury_level", 0)
    st.session_state.setdefault("is_dead", False)

    # Story mode toggle + beat tracking
    st.session_state.setdefault("story_mode", True)
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
def render_end_options(pid: str, slot) -> None:
    """
    Render the conclusion options panel. This appears when the story arc completes
    (e.g., after the resolution beat). It offers the player options to restart
    the current story or begin a new one.
    """
    with slot.container():
        st.subheader("Your adventure concludes.")
        st.caption("The story arc has come to an end. What will you do next?")
        c1, c2 = st.columns(2)
        restart = c1.button("Restart this story", use_container_width=True, key="end_restart")
        anew    = c2.button("Start a new story", use_container_width=True, key="end_new")
        if restart or anew:
            # Log the player's choice (restart vs new)
            try:
                save_event(pid, "complete_choice", {"action": "restart" if restart else "new"})
            except Exception:
                pass
            # Clear the current run but keep locked selections (Name/Gender/Character) and story mode
            for k in (
                "scene", "choices", "history", "pending_choice", "is_generating",
                "t_choices_visible_at", "t_scene_start", "is_dead", "beat_index", "story_complete",
            ):
                st.session_state.pop(k, None)
            try:
                delete_snapshot(pid)
            except Exception:
                pass
            # Reset beat index to 0 for new arc if story mode is on
            if st.session_state.get("story_mode"):
                st.session_state["beat_index"] = 0
                st.session_state["story_complete"] = False
            st.session_state["pending_choice"] = "__start__"
            st.session_state["is_generating"] = True
            st.rerun()


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
    st.session_state["last_illustration_url"] = None
    st.session_state["last_illustration_debug"] = {}

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

    # Show lantern in the choices area
    render_thinking(grid_slot)

    # Determine beat only if Story Mode is ON
    beat = get_current_beat(st.session_state) if st.session_state.get("story_mode") else None

    # Stream next scene (manual loop so we can kick off illustration early)
    st.session_state["t_scene_start"] = time.time()

    gen = stream_scene(st.session_state["history"], LORE, beat=beat)  # generator[str]

    simple_flag = bool(st.session_state.get("simple_cyoa", True))
    auto_ill    = bool(st.session_state.get("auto_illustrate", True))  # add a checkbox if you want this user-toggable
    scene_count = int(st.session_state.get("scene_count", 0))
    started_early = False
    buffer = []

    for chunk in gen:
        buffer.append(chunk)
        text_so_far = "".join(buffer)

        # render...
        story_slot.markdown(
            (f'{text_so_far}<span class="caret">▋</span>' if anim_enabled else text_so_far),
            unsafe_allow_html=True
        )

        # EARLY kickoff for 1st scene (and you can add %3 later)
        if auto_ill and not started_early and (scene_count == 0):
            seed = _has_first_sentence(text_so_far)
            if not seed and len(text_so_far.split()) >= 12:
                seed = " ".join(text_so_far.split()[:18]) + "…"
            if seed:
                _start_illustration_job(seed, simple_flag)  # remembers ill_last_key
                started_early = True

    full = "".join(buffer)

    # Check for death sentinel: if scene ends with '[DEATH]', set died flag and strip it
    died = False
    stripped = full.rstrip()
    if stripped.endswith("[DEATH]"):
        died = True
        # Remove the sentinel from the displayed text
        full = stripped[: -len("[DEATH]")].rstrip()

    # Append scene to history
    st.session_state["scene"] = full
    st.session_state["history"].append({"role": "assistant", "content": full})

    # --- Illustration generation: optionally generate and display an image for this scene ---
    scene_count = st.session_state.get("scene_count", 0)
    img_url = None  # <-- make sure it's always defined

    #if scene_count % 3 == 0:
    #    try:
    #        img_url = generate_illustration(full)
    #    except Exception:
    #       img_url = None

    #    if img_url:
    #       story_slot.image(img_url, caption="Illustration", use_column_width=True)

    # Save the last image URL (or None) for the dev panel
    st.session_state["last_illustration_url"] = img_url
    st.session_state["scene_count"] = int(st.session_state.get("scene_count", 0)) + 1

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
    gender = st.selectbox("Gender", ["Unspecified", "Female", "Male", "Nonbinary"], index=0)
    archetype = st.selectbox("Character type", ["Default"], index=0)

    # Mode selection: Story Mode vs. Exploration
    mode_default_index = 0 if st.session_state.get("story_mode", True) else 1
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
        st.session_state["story_mode"] = (mode == "Story Mode")
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

    st.session_state.setdefault("scene_count", 0)
    st.session_state.setdefault("simple_cyoa", True)
    st.session_state.setdefault("auto_illustrate", True)
    st.session_state.setdefault("last_illustration_url", None)

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
            illustration_text = st.session_state.get("last_illustration_url") or "None"
            st.caption(
                f"Story Mode: {'on' if st.session_state.get('story_mode') else 'off'} • "
                f"Beat: {beat} • Danger streak: {st.session_state.get('danger_streak', 0)} • "
                f"Scenes: {len(st.session_state.get('history', []))} • "
                f"Illustration: {illustration_text} • "
                f"Ill key: {st.session_state.get('ill_last_key','–')} • "
                f"Auto: {st.session_state.get('auto_illustrate', True)} • "
                f"Simple: {st.session_state.get('simple_cyoa', True)}"
            )
            last_dbg = st.session_state.get("last_illustration_debug")
            if last_dbg:
                with st.expander("Illustration debug"):
                    try:
                        st.json(last_dbg)
                    except Exception:
                        st.write(last_dbg)

        st.markdown("### Illustration")
        simple_cyoa = st.checkbox("Simple CYOA style", value=True, help="Cleaner line art, minimal detail")
        st.session_state["simple_cyoa"] = simple_cyoa
        started_early = False
        buffer = []

        gen_click = st.button("Generate illustration for this scene", use_container_width=True, key="btn_gen_ill")
        if gen_click:
            scene_text = (st.session_state.get("scene") or "").strip()
            if not scene_text:
                st.warning("No scene text is available yet.")
            else:
                # ---- illustration cache (per scene + style) ----
                ill_cache = st.session_state.setdefault("ill_cache", {})
                import hashlib, json
                cache_key = hashlib.md5(
                    json.dumps({"scene": scene_text, "simple": bool(simple_cyoa)}, ensure_ascii=False).encode("utf-8")
                ).hexdigest()

                if cache_key in ill_cache:
                    img_ref, dbg = ill_cache[cache_key]
                else:
                    with st.spinner("Creating illustration…"):
                        res = generate_illustration(scene_text, simple=simple_cyoa)
                    if isinstance(res, tuple) and len(res) == 2:
                        img_ref, dbg = res
                    else:
                        img_ref, dbg = (res if res is not None else None), {}
                    ill_cache[cache_key] = (img_ref, dbg)
                # -----------------------------------------------

                st.session_state["last_illustration_url"] = img_ref
                st.session_state["last_illustration_debug"] = dbg
                if img_ref:
                    # The main render path should include:
                    # illustration_ph.image(img_ref, caption="Illustration", use_container_width=True)
                    try:
                        st.toast("Illustration added.", icon="✨")
                    except Exception:
                        pass
                else:
                    st.warning("No illustration was returned.")
                    try:
                        st.json(dbg)
                    except Exception:
                        st.write(dbg)
                st.rerun()

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
        st.session_state["scene_count"] = 0
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
    scene_ph = st.empty()
    illustration_ph = st.empty()   # <-- add this
    choices_ph = st.empty()

    def _poll_and_show_illustration(scene_text: str, simple: bool, ph):
        # 1) Locate the job for THIS scene
        key, job = _job_for_scene(scene_text, simple)

        if not job:
            # No job started (maybe early kickoff didn’t fire). Show status and optionally start now.
            ph.caption("Illustration: no job yet.")
            # Optional: if you want to force-start here when auto is ON:
            if st.session_state.get("auto_illustrate", True):
                seed = _has_first_sentence(scene_text) or " ".join(scene_text.split()[:18]) + "…"
                if seed.strip():
                    _start_illustration_job(seed, simple)
                    ph.caption("Illustration: job started…")
            return

        fut = job["future"]
        # 2) Collect result if done
        if fut.done() and job["result"] is None:
            try:
                job["result"] = fut.result(timeout=0)
            except Exception as e:
                job["result"] = (None, {"error": repr(e)})

        res = job.get("result")

        # 3) Render according to state
        if res:
            img_ref, dbg = res if (isinstance(res, tuple) and len(res) == 2) else (res, {})
            if img_ref:
                st.session_state["last_illustration_url"]   = img_ref
                st.session_state["last_illustration_debug"] = dbg
                ph.image(img_ref, caption="Illustration", use_container_width=True)
            else:
                ph.caption("Illustration: unavailable (no image).")
                try:
                    with st.expander("Illustration debug"):
                        st.json(dbg)
                except Exception:
                    pass
        else:
            # Still running
            ph.caption("Illustration brewing…")
            # Gentle refresh using streamlit-js-eval (if installed)
            try:
                from streamlit_js_eval import streamlit_js_eval
                streamlit_js_eval(
                    js_expressions="setTimeout(() => window.parent.location.reload(), 1000)",
                    key=f"ill_poll_{key}"
                )
            except Exception:
                # do nothing; it will show on the next natural rerun
                pass


    # Ensure state key exists
    st.session_state.setdefault("last_illustration_url", None)

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
        # existing: show any previously saved image
        if st.session_state.get("last_illustration_url"):
            illustration_ph.image(st.session_state["last_illustration_url"], caption="", use_container_width=True)

        scene_ph.markdown(st.session_state["scene"], unsafe_allow_html=True)

        # POLL HERE on idle path (NOT in the pending path)
        _poll_and_show_illustration(
            st.session_state["scene"],
            bool(st.session_state.get("simple_cyoa", True)),
            illustration_ph,
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