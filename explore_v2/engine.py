from __future__ import annotations

"""Exploration mode renderer (V2).

This module extends the original exploration engine with a live-updating
sidebar that shows lightweight world state such as current location,
available exits, visible NPCs and the player's inventory.  It reuses the
story streaming and illustration generation helpers from the original
engine but keeps illustration generation off the main thread so the UI
remains responsive.
"""

import json
import os
import pathlib
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Generator, Tuple
import re

import streamlit as st

from ui.choices import render_choices_grid
from story.engine import generate_illustration, client
from explore.prompts import SCENE_SYSTEM_PROMPT, CHOICE_SYSTEM_PROMPT
from data.explore_store import (
    init_db as _init_db,
    save_snapshot as _save_snapshot,
    load_snapshot as _load_snapshot,
)

# ---------------------------------------------------------------------------
# Lore / constants
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parent.parent
LORE_PATH = ROOT / "lore.json"
LORE = json.loads(LORE_PATH.read_text(encoding="utf-8")) if LORE_PATH.exists() else {}

CHOICE_COUNT = 2

_executor = ThreadPoolExecutor(max_workers=2)


# ---------------------------------------------------------------------------
# Small UI helpers copied from story mode
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Illustration job helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------


def _history_text(history: List[Dict[str, str]]) -> str:
    return "\n".join(h.get("content", "") for h in history[-10:])


def stream_explore_scene(
    history: List[Dict[str, str]], lore: Dict
) -> Generator[str, None, None]:
    lore_blob = json.dumps(lore)[:10_000]
    user = (
        "Continue free-roam exploration.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        "Player history (latest last):\n"
        f"{_history_text(history)}\n"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        max_tokens=350,
        stream=True,
        messages=[
            {"role": "system", "content": SCENE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    for ev in resp:
        if ev.choices:
            delta = ev.choices[0].delta.content or ""
            if delta:
                yield delta


def generate_explore_choices(
    history: List[Dict[str, str]], last_scene: str, lore: Dict
) -> List[str]:
    lore_blob = json.dumps(lore)[:10_000]
    user = (
        f"{last_scene}\n\n"
        "Player history (latest last):\n"
        f"{_history_text(history)}\n\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        max_tokens=150,
        messages=[
            {"role": "system", "content": CHOICE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
    )
    text = resp.choices[0].message.content.strip()
    try:
        arr = json.loads(text)
    except Exception:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            try:
                arr = json.loads(m.group(0))
            except Exception:
                arr = []
        else:
            arr = []
    if arr:
        return [s.strip().rstrip(".") for s in arr[:CHOICE_COUNT]]
    lines = [ln.strip("-* ") for ln in text.splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0].startswith("[") and lines[0].endswith("]"):
        try:
            arr = json.loads(lines[0])
            return [s.strip().rstrip(".") for s in arr[:CHOICE_COUNT]]
        except Exception:
            pass
    return lines[:CHOICE_COUNT]


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------


def ensure_explore_keys() -> None:
    """Seed session state for exploration mode.

    Unlike ``setdefault`` alone, this also boots the first scene when switching
    into exploration after visiting other modes. If no scene has been
    generated yet (``scene_count`` still 0) and ``pending_choice`` is missing or
    ``None``, we initialize it to ``"__start__"`` so the engine advances
    immediately on first render.
    """

    st.session_state.setdefault("scene", "")
    st.session_state.setdefault("choices", [])
    st.session_state.setdefault("choice_map", {})
    st.session_state.setdefault("choice_objs", [])
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("is_generating", False)
    st.session_state.setdefault("t_scene_start", None)
    st.session_state.setdefault("t_choices_visible_at", None)
    st.session_state.setdefault("explore_ill_future", None)
    st.session_state.setdefault("explore_illustration_url", None)
    st.session_state.setdefault("explore_poll_count", 0)
    st.session_state.setdefault("scene_count", 0)
    st.session_state.setdefault("onboard_dismissed", True)
    st.session_state.setdefault("explore_illustration_every", 3)
    st.session_state.setdefault("current_location", "Unknown")
    st.session_state.setdefault("current_exits", [])
    st.session_state.setdefault("visible_npcs", [])
    st.session_state.setdefault("inventory", [])
    st.session_state.setdefault("pos", (0, 0))
    st.session_state.setdefault("visited", {(0, 0): True})

    # ``pending_choice`` needs extra care: if a previous mode cleared it we
    # still want exploration to auto-start the first scene.  However, story
    # mode may leave ``scene_count`` non-zero when switching modes, which used
    # to prevent the first scene from booting.  Detect the absence of any
    # current scene instead – if no text exists and ``pending_choice`` is
    # missing, schedule the start marker.
    if not st.session_state.get("scene"):
        st.session_state["scene_count"] = 0
        if not st.session_state.get("pending_choice"):
            st.session_state["pending_choice"] = "__start__"


def _dir_to_delta(label: str) -> Tuple[int, int]:
    s = label.lower()
    if "north" in s:
        return (0, 1)
    if "south" in s:
        return (0, -1)
    if "east" in s:
        return (1, 0)
    if "west" in s:
        return (-1, 0)
    return (0, 0)


def _build_ascii_map() -> str:
    visited = dict(st.session_state.get("visited", {(0, 0): True}))
    x, y = st.session_state.get("pos", (0, 0))
    visited[(x, y)] = True
    xs = [p[0] for p in visited]
    ys = [p[1] for p in visited]
    x_min, x_max = min(xs) - 1, max(xs) + 1
    y_min, y_max = min(ys) - 1, max(ys) + 1
    lines = []
    for yy in range(y_max, y_min - 1, -1):
        row = []
        for xx in range(x_min, x_max + 1):
            if (xx, yy) == (x, y):
                row.append("P")
            elif (xx, yy) in visited:
                row.append(".")
            else:
                row.append(" ")
        lines.append("".join(row))
    return "\n".join(lines)


def _update_world_state(scene_text: str, choices: List, picked: Optional[str]) -> None:
    pos = st.session_state.get("pos", (0, 0))
    if picked and picked != "__start__":
        dx, dy = _dir_to_delta(picked)
        pos = (pos[0] + dx, pos[1] + dy)
        st.session_state["pos"] = pos
    visited = dict(st.session_state.get("visited", {}))
    visited[pos] = True
    st.session_state["visited"] = visited
    st.session_state["current_location"] = f"{pos[0]},{pos[1]}"
    exits = []
    for c in choices:
        label = c.get("id") if isinstance(c, dict) else str(c)
        if _dir_to_delta(label) != (0, 0):
            exits.append(label)
    st.session_state["current_exits"] = exits
    st.session_state["visible_npcs"] = [
        {"name": f"NPC {len(visited)}", "romance": min(100, len(visited) * 5)}
    ]
    inv = list(st.session_state.get("inventory", []))
    if len(visited) % 3 == 0 and f"Relic {len(visited)}" not in inv:
        inv.append(f"Relic {len(visited)}")
    st.session_state["inventory"] = inv

def _rehydrate_world_state(history: List[Dict[str, str]], choices: List) -> None:
    """Reconstruct lightweight world state from history and current choices."""
    pos = (0, 0)
    visited: Dict[Tuple[int, int], bool] = {(0, 0): True}
    inv: List[str] = []
    for msg in history:
        if isinstance(msg, dict) and msg.get("role") == "user":
            dx, dy = _dir_to_delta(msg.get("content", ""))
            pos = (pos[0] + dx, pos[1] + dy)
            visited[pos] = True
            if len(visited) % 3 == 0:
                relic = f"Relic {len(visited)}"
                if relic not in inv:
                    inv.append(relic)
    st.session_state["pos"] = pos
    st.session_state["visited"] = visited
    st.session_state["current_location"] = f"{pos[0]},{pos[1]}"
    exits: List[str] = []
    for c in choices:
        label = c.get("id") if isinstance(c, dict) else str(c)
        if _dir_to_delta(label) != (0, 0):
            exits.append(label)
    st.session_state["current_exits"] = exits
    st.session_state["visible_npcs"] = [
        {"name": f"NPC {len(visited)}", "romance": min(100, len(visited) * 5)}
    ]
    st.session_state["inventory"] = inv


def _maybe_restore_from_snapshot(pid: str) -> bool:
    """Hydrate session state from a persisted snapshot if empty."""
    if st.session_state.get("scene") or st.session_state.get("choices"):
        return False
    try:
        snap = _load_snapshot("world", pid)
    except Exception:
        snap = None
    if not snap:
        return False
    st.session_state["scene"] = snap.get("scene") or ""
    raw_choices = snap.get("choices") or []
    st.session_state["history"] = snap.get("history") or []
    st.session_state["pending_choice"] = None
    st.session_state["scene_count"] = snap.get("decisions_count", 0)
    st.session_state["choice_objs"] = raw_choices
    if raw_choices and isinstance(raw_choices[0], dict):
        labels = [c.get("label", "") for c in raw_choices]
        st.session_state["choice_map"] = {
            c.get("label", ""): c.get("id", c.get("label", "")) for c in raw_choices
        }
    else:
        labels = [str(c) for c in raw_choices]
        st.session_state["choice_map"] = {lbl: lbl for lbl in labels}
    st.session_state["choices"] = labels[:CHOICE_COUNT]
    _rehydrate_world_state(st.session_state["history"], raw_choices)
    if not st.session_state.get("explore_illustration_url"):
        _start_illustration_job(st.session_state["scene"])
    return True


def _render_sidebar_state() -> None:
    with st.sidebar.expander("Map", expanded=True):
        st.text(_build_ascii_map())
    with st.sidebar.expander("Adventurer's Notes", expanded=True):
        st.markdown(
            f"**Location:** {st.session_state.get('current_location', 'Unknown')}"
        )
        exits = st.session_state.get("current_exits", [])
        st.markdown("**Exits:** " + (", ".join(exits) if exits else "None"))
        npcs = st.session_state.get("visible_npcs", [])
        st.markdown("**NPCs:**")
        if npcs:
            for npc in npcs:
                st.markdown(
                    f"- {npc.get('name')} ({npc.get('romance', 0)}%)"
                )
        else:
            st.markdown("_None_")
        inv = st.session_state.get("inventory", [])
        st.markdown("**Inventory:**")
        if inv:
            for item in inv:
                st.markdown(f"- {item}")
        else:
            st.markdown("_Empty_")


# ---------------------------------------------------------------------------
# Core loop
# ---------------------------------------------------------------------------


def _advance_turn(pid: str, story_ph, illus_ph, sep_ph, choices_ph) -> None:
    choice_label = st.session_state.get("pending_choice", "__start__")
    st.session_state["pending_choice"] = None
    choice_map = st.session_state.get("choice_map", {})
    choice = choice_map.get(choice_label, choice_label)
    history = st.session_state.get("history", [])
    if choice and choice != "__start__":
        history.append({"role": "user", "content": choice})
    st.session_state["t_scene_start"] = time.time()
    gen = stream_explore_scene(history, LORE)
    raw_chunks: List[str] = []
    scene_chars: List[str] = []
    state = "seek_key"
    key_buf = ""
    escape = False
    scene_done = False
    for chunk in gen:
        raw_chunks.append(chunk or "")
        scene_updated = False
        for ch in chunk:
            if scene_done:
                continue
            if state == "seek_key":
                if ch == '"':
                    state = "read_key"
                    key_buf = ""
            elif state == "read_key":
                if ch == '"':
                    state = "post_key" if key_buf == "scene" else "seek_key"
                else:
                    key_buf += ch
            elif state == "post_key":
                if ch == ':':
                    state = "pre_value"
                elif ch in ' \n\r\t':
                    pass
                else:
                    state = "seek_key"
            elif state == "pre_value":
                if ch == '"':
                    state = "in_value"
                elif ch in ' \n\r\t':
                    pass
                else:
                    state = "seek_key"
            elif state == "in_value":
                if escape:
                    if ch == 'n':
                        scene_chars.append('\n')
                    elif ch == 't':
                        scene_chars.append('\t')
                    elif ch == '"':
                        scene_chars.append('"')
                    elif ch == '\\':
                        scene_chars.append('\\')
                    else:
                        scene_chars.append(ch)
                    escape = False
                    scene_updated = True
                else:
                    if ch == '\\':
                        escape = True
                    elif ch == '"':
                        scene_done = True
                        state = "seek_key"
                    else:
                        scene_chars.append(ch)
                        scene_updated = True
        if scene_updated:
            story_ph.markdown(
                f'<div class="story-window"><div class="storybox">{"".join(scene_chars)}</div></div>',
                unsafe_allow_html=True,
            )
    raw_text = "".join(raw_chunks)
    scene_text = "".join(scene_chars) or raw_text
    choice_objs: List = []
    try:
        m = re.search(r"\{[\s\S]*\}", raw_text)
        data = json.loads(m.group(0) if m else raw_text)
        scene_text = data.get("scene") or data.get("prose") or raw_text
        choice_objs = data.get("choices") or []
        if isinstance(choice_objs, str):
            try:
                choice_objs = json.loads(choice_objs)
            except Exception:
                choice_objs = [choice_objs]
        choice_objs = data.get("choices") or []
        items = data.get("items") or []
        if items:
            inv = list(st.session_state.get("inventory", []))
            for itm in items:
                name = itm.get("name") if isinstance(itm, dict) else str(itm)
                if name not in inv:
                    inv.append(name)
            st.session_state["inventory"] = inv
    except Exception:
        pass
    story_ph.markdown(
        f'<div class="story-window"><div class="storybox">{scene_text}</div></div>',
        unsafe_allow_html=True,
    )
    st.session_state["scene"] = scene_text
    history.append({"role": "assistant", "content": scene_text})
    st.session_state["scene_count"] = st.session_state.get("scene_count", 0) + 1

    ill_every = int(st.session_state.get("explore_illustration_every", 3))
    scene_index = st.session_state["scene_count"]
    if scene_index % ill_every == 1:
        _start_illustration_job(scene_text)
        _render_illustration_inline(illus_ph, None, "Illustration brewing…")
    else:
        _render_illustration(illus_ph)
    _render_separator(sep_ph)

    if not choice_objs:
        choice_labels = generate_explore_choices(history, scene_text, LORE) or []
        choice_objs = choice_labels
    st.session_state["choice_objs"] = choice_objs
    if choice_objs and isinstance(choice_objs[0], dict):
        labels = [c.get("label", "") for c in choice_objs]
        st.session_state["choice_map"] = {
            c.get("label", ""): c.get("id", c.get("label", "")) for c in choice_objs
        }
    else:
        labels = [str(c) for c in choice_objs]
        st.session_state["choice_map"] = {lbl: lbl for lbl in labels}
    st.session_state["choices"] = labels[:CHOICE_COUNT]
    st.session_state["is_generating"] = False
    with choices_ph.container():
        st.markdown('<div class="story-body">', unsafe_allow_html=True)
        slot = st.container()
        render_choices_grid(
            slot, choices=labels, generating=False, count=CHOICE_COUNT
        )
    st.session_state["t_choices_visible_at"] = time.time()
    _update_world_state(scene_text, choice_objs, choice)
    _save_snapshot(
        "world",
        pid,
        scene_text,
        choice_objs,
        st.session_state.get("history", []),
    )

# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


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
        _advance_turn(pid, story_ph, illus_ph, sep_ph, choices_ph)
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
                slot, choices=current_choices, generating=False, count=CHOICE_COUNT
            )
    _render_sidebar_state()
    _poll_illustration()
