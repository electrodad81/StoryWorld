from __future__ import annotations

"""Developer helpers for exploration mode.

These mirror the diagnostics available in story mode and surface common
session-state issues that can lead to a blank UI.  The tools are lightweight
and safe to import even when the underlying data stores are unavailable.
"""

from typing import Any, Dict
import time
import streamlit as st

try:
    from data.store import has_snapshot, load_snapshot
except Exception:  # pragma: no cover - stores may be unavailable in tests
    has_snapshot = lambda pid: False  # type: ignore
    load_snapshot = lambda pid: None  # type: ignore


def render_debug_sidebar(pid: str) -> None:
    """Render a debug panel mirroring story mode's dev tools.

    Shows key session state values, snapshot status and simple self-tests to
    highlight situations where the engine is idle without a pending choice.
    """

    now = time.time()

    scene_text = (st.session_state.get("scene") or "").strip()
    choices = st.session_state.get("choices", [])
    history = st.session_state.get("history", [])

    t_scene = st.session_state.get("t_scene_start")
    t_choices = st.session_state.get("t_choices_visible_at")
    age_scene = f"{now - t_scene:.1f}s" if t_scene else "–"
    age_choices = f"{now - t_choices:.1f}s" if t_choices else "–"

    try:
        snap = load_snapshot(pid) if has_snapshot(pid) else None
        snap_len = len(snap["history"]) if snap else 0
    except Exception:  # pragma: no cover - diagnostics only
        snap_len = 0

    st.caption(
        f"scene_len: {len(scene_text)} • history: {len(history)} • choices: {len(choices)}"
    )
    st.caption(
        f"pending_choice: {st.session_state.get('pending_choice')} • "
        f"is_generating: {int(bool(st.session_state.get('is_generating')))} • "
        f"scene_count: {st.session_state.get('scene_count', 0)} • "
        f"snapshot_history: {snap_len}"
    )
    st.caption(
        f"t_scene_age: {age_scene} • t_choices_age: {age_choices} • "
        f"polls: {st.session_state.get('explore_poll_count', 0)}"
    )

    issues = []
    if not scene_text and st.session_state.get("pending_choice") is None:
        issues.append("no scene & no pending choice")
    if issues:
        st.error("; ".join(issues))
        if st.button("Force start scene", key="dev_force_start"):
            st.session_state["pending_choice"] = "__start__"
            st.session_state["is_generating"] = True
            (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()