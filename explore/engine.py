from __future__ import annotations

import json
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import streamlit as st

from ui.choices import render_choices_grid
from story.engine import stream_scene, generate_choices, generate_illustration

# -----------------------------------------------------------------------------
# Lore / constants
# -----------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent.parent
LORE_PATH = ROOT / "lore.json"
LORE = json.loads(LORE_PATH.read_text(encoding="utf-8")) if LORE_PATH.exists() else {}
CHOICE_COUNT = 2

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

# -----------------------------------------------------------------------------
# Core loop
# -----------------------------------------------------------------------------

def _advance_turn(story_ph, illus_ph, sep_ph, choices_ph) -> None:
    choice = st.session_state.get("pending_choice", "__start__")
    st.session_state["pending_choice"] = None
    history = st.session_state.get("history", [])
    if choice and choice != "__start__":
        history.append({"role": "user", "content": choice})
    st.session_state["t_scene_start"] = time.time()
    gen = stream_scene(history, LORE)
    buf = []
    for chunk in gen:
        buf.append(chunk or "")
        text = "".join(buf)
        story_ph.markdown(
            f'<div class="story-window"><div class="storybox">{text}<span class="typing-caret"></span></div></div>',
            unsafe_allow_html=True,
        )
    scene_text = "".join(buf)
    story_ph.markdown(
        f'<div class="story-window"><div class="storybox">{scene_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state["scene"] = scene_text
    history.append({"role": "assistant", "content": scene_text})
    st.session_state["scene_count"] = st.session_state.get("scene_count", 0) + 1

    _start_illustration_job(scene_text)
    _render_illustration_inline(illus_ph, None, "Illustration brewing…")
    _render_separator(sep_ph)

    choices = generate_choices(history, scene_text, LORE) or []
    st.session_state["choices"] = choices[:CHOICE_COUNT]
    st.session_state["is_generating"] = False
    with choices_ph.container():
        st.markdown('<div class="story-body">', unsafe_allow_html=True)
        slot = st.container()
        render_choices_grid(slot, choices=choices, generating=False, count=CHOICE_COUNT)
    st.session_state["t_choices_visible_at"] = time.time()

# -----------------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------------

def render_explore(pid: str) -> None:
    ensure_explore_keys()
    st.session_state["story_mode"] = False

    story_ph = st.empty()
    illus_ph = st.empty()
    sep_ph = st.empty()
    choices_ph = st.empty()

    if st.session_state.get("pending_choice") is not None:
        _advance_turn(story_ph, illus_ph, sep_ph, choices_ph)
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
            render_choices_grid(slot, choices=current_choices, generating=False, count=CHOICE_COUNT)
    _poll_illustration()