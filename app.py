# app.py
from __future__ import annotations
from typing import List, Optional
import streamlit as st
from core.identity import ensure_browser_id
from data.store import has_snapshot, delete_snapshot, load_snapshot, save_snapshot, save_visit
import json
from story.engine import stream_scene, generate_choices
from ui.anim import inject_css

import re  # for risk keyword matching

from ui.loader import lantern
from ui.anim import render_thinking
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

# -----------------------------------------------------------------------------
# Risk & Consequence utilities (name-agnostic)
# -----------------------------------------------------------------------------
# A set of keywords that indicate a risky choice. Feel free to tweak/extend
# these terms to better fit your storyworld. The words are matched
# case-insensitively as whole words using regular expressions.
_RISKY_WORDS: set[str] = {
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
                "choices_before", "t_choices_visible_at", "t_scene_start", "is_dead"
            ):
                st.session_state.pop(k, None)
            try:
                delete_snapshot(pid)
            except Exception:
                pass
            st.session_state["pending_choice"] = "__start__"
            st.session_state["is_generating"] = True
            st.rerun()

def inject_css():
    """Inject global and local CSS styles.

    This function first invokes the shared CSS injector from ``ui.anim`` to
    ensure all base styles (including both ``.bulb`` and ``.flame`` classes)
    are available. It then overlays any local customizations specific to
    the top-level app, such as the radial gradient flame and caret styling.
    """
    # Import here to avoid circular import at module load time. Using the
    # module-level ``inject_css`` from ``ui.anim`` ensures our local
    # definition does not override the shared injector.
    try:
        from ui import anim as _anim
        # Call the shared injector; guarded to prevent infinite recursion.
        _anim.inject_css(enabled=True)
    except Exception:
        # Fallback: if import fails, continue silently. This may happen in
        # certain testing contexts but should not prevent the app from
        # rendering.
        pass
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

    # Update risk counters for this pick (increments on risky, decrements on safe)
    # Do this before logging the choice event so that danger_streak reflects the
    # current pick in telemetry.
    try:
        update_consequence_counters(picked)
    except Exception:
        pass

    # --- CHOICE EVENT + latency (time from choices visible → click)
    if picked and picked != "__start__":
        hist = st.session_state["history"]
        if not hist or hist[-1].get("role") != "user" or hist[-1].get("content") != picked:
            hist.append({"role": "user", "content": picked})

        visible_at = st.session_state.get("t_choices_visible_at")
        latency_ms = int((time.time() - visible_at) * 1000) if visible_at else None
        # Include danger_streak in choice telemetry
        save_event(pid, "choice", {
            "label": picked,
            "index": (old_choices.index(picked) if picked in old_choices else None),
            "latency_ms": latency_ms,
            "decisions_count": sum(1 for m in hist if m.get("role") == "user"),
            "danger_streak": st.session_state.get("danger_streak", 0),
        })

    # show the lantern loader in place of the choices while we generate the next scene
    # using ui.anim.render_thinking to ensure the glow appears in the same slot
    try:
        # this writes the lantern into the choices area; it will be overwritten when buttons render
        render_thinking(grid_slot)
    except Exception:
        # if the import fails, fall back to neutral placeholders (no buttons)
        render_choices_grid(grid_slot, choices=None, generating=True, count=CHOICE_COUNT)

    # --- mark scene stream start (dwell timer)
    st.session_state["t_scene_start"] = time.time()

    # stream scene
    if anim_enabled:
        # no need to use the loader context here since the lantern is shown in the grid slot
        full = stream_text(stream_scene(st.session_state["history"], LORE), story_slot, show_caret=True)
    else:
        full = stream_text(stream_scene(st.session_state["history"], LORE), story_slot, show_caret=False)

    # DEATH sentinel detection: model will append [DEATH] on a fatal scene
    died = False
    stripped = full.rstrip()
    if stripped.endswith("[DEATH]"):
        died = True
        full = stripped[: -len("[DEATH]")].rstrip()

    # finalize history
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
            save_snapshot(pid, full, [], st.session_state["history"], username=username, gender=gender, archetype=archetype)
        except TypeError:
            save_snapshot(pid, full, [], st.session_state["history"], username=username)
        # Show the death options panel and halt
        render_death_options(pid, grid_slot)
        return

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
        "danger_streak": st.session_state.get("danger_streak", 0),
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
        scene_ph.markdown(st.session_state["scene"])
        # Choices are about to be visible (for latency calc on click)
        st.session_state["t_choices_visible_at"] = time.time()

        from ui.choices import render_choices_grid
        render_choices_grid(
            choices_ph,
            choices=st.session_state.get("choices", []),
            generating=False,
            count=CHOICE_COUNT if "CHOICE_COUNT" in globals() else 2,
        )

if __name__ == "__main__":
    main()
