from __future__ import annotations

import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import streamlit as st

from ui.choices import render_choices_grid
from story.engine import generate_illustration
from story.explore_engine import apply_outcome, tick_render
from data.explore_store import (
    init_db as _init_db,
    save_snapshot as _save_snapshot,
    load_snapshot as _load_snapshot,
)

# -----------------------------------------------------------------------------
# Lore / constants
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
LORE_PATH = ROOT / "lore.json"
LORE = json.loads(LORE_PATH.read_text(encoding="utf-8")) if LORE_PATH.exists() else {}

_executor = ThreadPoolExecutor(max_workers=2)

# -----------------------------------------------------------------------------
# Small UI helpers copied from story mode
# -----------------------------------------------------------------------------

def _render_separator(ph):
    ph.markdown(
        """
        <div class="illus-sep" aria-hidden="true">
          <span class="line"></span><span class="gem">◆</span><span class="line"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_illustration_inline(ph, url: Optional[str], status_text: str = "") -> None:
    if url:
        ph.markdown(
            f"""
            <div class="illus-inline">
              <img src="{url}" alt="illustration"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        ph.markdown(
            f"""
            <div class="illus-inline illus-skeleton">
              <div class="illus-status">{status_text or "Illustration brewing…"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# Illustration job helpers
# -----------------------------------------------------------------------------

def _start_illustration_job(scene_text: str) -> None:
    fut = _executor.submit(generate_illustration, scene_text, True)
    st.session_state["explore_ill_future"] = fut
    st.session_state["explore_illustration_url"] = None
    st.session_state["explore_poll_count"] = 0


def _poll_illustration(delay: float = 0.5, max_polls: int = 60) -> None:
    fut = st.session_state.get("explore_ill_future")
    url = st.session_state.get("explore_illustration_url")
    if not fut or fut.done() or url:
        return
    count = int(st.session_state.get("explore_poll_count", 0))
    if count >= max_polls:
        return
    st.session_state["explore_poll_count"] = count + 1
    time.sleep(delay)
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()


def _render_illustration(ph) -> None:
    fut = st.session_state.get("explore_ill_future")
    url = st.session_state.get("explore_illustration_url")
    status = "Illustration brewing…"
    if fut and fut.done() and url is None:
        try:
            img_ref, _ = fut.result(timeout=0)
            url = img_ref
            status = ""
        except Exception:
            url = None
            status = "Illustration unavailable"
        st.session_state["explore_illustration_url"] = url
    if url:
        _render_illustration_inline(ph, url)
    else:
        _render_illustration_inline(ph, None, status or "Illustration brewing…")

# -----------------------------------------------------------------------------
# Session state helpers
# -----------------------------------------------------------------------------

def ensure_explore_keys() -> None:
    st.session_state.setdefault("scene", "")
    st.session_state.setdefault("choices", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("pending_choice", "__start__")
    st.session_state.setdefault("is_generating", False)
    st.session_state.setdefault("t_scene_start", None)
    st.session_state.setdefault("t_choices_visible_at", None)
    st.session_state.setdefault("explore_ill_future", None)
    st.session_state.setdefault("explore_illustration_url", None)
    st.session_state.setdefault("explore_poll_count", 0)
    st.session_state.setdefault("scene_count", 0)
    st.session_state.setdefault("onboard_dismissed", True)
    st.session_state.setdefault("explore_illustration_every", 3)

def _maybe_restore_from_snapshot(pid: str) -> bool:
    """Hydrate session_state from a persisted snapshot if empty."""
    if st.session_state.get("scene") or st.session_state.get("choices"):
        return False
    try:
        snap = _load_snapshot("world", pid)
    except Exception:
        snap = None
    if not snap:
        return False
    st.session_state["scene"] = snap.get("scene") or ""
    st.session_state["choices"] = snap.get("choices") or []
    st.session_state["history"] = snap.get("history") or []
    st.session_state["pending_choice"] = None
    st.session_state["scene_count"] = snap.get("decisions_count", 0)
    if not st.session_state.get("explore_illustration_url"):
        _start_illustration_job(st.session_state["scene"])
    return True


def _render_sidebar(state: dict) -> None:
    """Render map, inventory and NPC info in the sidebar."""
    with st.sidebar:
        st.markdown("### Map")
        loc = state.get("location") or ""
        exits = state.get("exits", []) or []
        lines = [f"{e.get('direction')}→{e.get('dst_id')}" for e in exits]
        st.write(loc)
        if lines:
            st.caption("exits: " + ", ".join(lines))

        inv = state.get("inventory") or {}
        if inv:
            st.markdown("### Inventory")
            for item_id, qty in inv.items():
                st.write(f"{item_id} ×{qty}")

        npcs = state.get("npcs") or []
        if npcs:
            st.markdown("### NPCs")
            for npc in npcs:
                name = npc.get("name") or npc.get("id")
                cooldown = npc.get("cooldown")
                if cooldown:
                    st.write(f"{name} (CD {cooldown})")
                else:
                    st.write(name)

# -----------------------------------------------------------------------------
# Core loop
# -----------------------------------------------------------------------------

def _advance_turn(pid: str, story_ph, illus_ph, sep_ph, choices_ph) -> dict:
    choice_label = st.session_state.get("pending_choice", "__start__")
    st.session_state["pending_choice"] = None
    choice_map = st.session_state.get("explore_choice_map", {})
    choice_id = choice_map.get(choice_label, choice_label)

    st.session_state["t_scene_start"] = time.time()
    
    # Depending on whether the user made a choice we either apply the
    # outcome or simply render the current location.
    if choice_label and choice_label != "__start__":
        result = apply_outcome(pid, choice_id, ROOT)
    else:
        result = tick_render(pid, ROOT)

    scene_text = result.get("prose", "")
    story_ph.markdown(
        f'<div class="story-window"><div class="storybox">{scene_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state["scene"] = scene_text
    st.session_state["scene_count"] = st.session_state.get("scene_count", 0) + 1

    ill_every = int(st.session_state.get("explore_illustration_every", 3))
    scene_index = st.session_state["scene_count"]
    if scene_index % ill_every == 1:
        _start_illustration_job(scene_text)
        _render_illustration_inline(illus_ph, None, "Illustration brewing…")
    else:
        _render_illustration(illus_ph)
    _render_separator(sep_ph)

    raw_choices = result.get("choices", []) or []
    labels = []
    choice_map = {}
    for opt in raw_choices:
        if isinstance(opt, dict):
            cid = opt.get("id")
            lab = opt.get("label") or cid
            if lab:
                labels.append(lab)
                if cid:
                    choice_map[lab] = cid
        elif isinstance(opt, str):
            labels.append(opt)
            choice_map[opt] = opt
    st.session_state["choices"] = labels
    st.session_state["explore_choice_map"] = choice_map
    st.session_state["is_generating"] = False
    with choices_ph.container():
        st.markdown('<div class="story-body">', unsafe_allow_html=True)
        slot = st.container()
        render_choices_grid(
            slot, choices=labels, generating=False, count=max(1, len(labels))
        )
    st.session_state["t_choices_visible_at"] = time.time()
    _save_snapshot("world", pid, scene_text, raw_choices, st.session_state.get("history", []))
    return result

# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def render_explore(pid: str) -> None:
    _init_db()
    ensure_explore_keys()
    st.session_state["story_mode"] = False
    _maybe_restore_from_snapshot(pid)

    story_ph = st.empty()
    illus_ph = st.empty()
    sep_ph = st.empty()
    choices_ph = st.empty()
    
    if st.session_state.get("pending_choice") is not None:
        state = _advance_turn(pid, story_ph, illus_ph, sep_ph, choices_ph)
    else:
        story_html = st.session_state.get("scene", "")
        story_ph.markdown(
            f'<div class="story-window"><div class="storybox">{story_html}</div></div>',
            unsafe_allow_html=True,
        )
        _render_illustration(illus_ph)
        _render_separator(sep_ph)
        current_choices = list(st.session_state.get("choices", []))
        with choices_ph.container():
            st.markdown('<div class="story-body">', unsafe_allow_html=True)
            slot = st.container()
            render_choices_grid(
                slot,
                choices=current_choices,
                generating=False,
                count=max(1, len(current_choices)),
            )
        state = tick_render(pid, ROOT)
    _render_sidebar(state)
    _poll_illustration()