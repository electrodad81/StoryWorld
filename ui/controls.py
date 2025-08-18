# controls.py
from __future__ import annotations

import streamlit as st
from typing import Optional, Literal

Action = Optional[Literal["start", "reset"]]

def sidebar_controls(pid: str) -> Action:
    """Sidebar controls: Start New Story and Reset Session.
    Returns an action string your main() handles.
    """
    with st.sidebar:
        st.header("Story Controls")
        col1, col2 = st.columns(2)
        if col1.button("Start New Story", key="btn_start", use_container_width=True):
            return "start"
        if col2.button("Reset Session", key="btn_reset", use_container_width=True):
            return "reset"
    return None

# --- Legacy helper kept for compatibility (no _picked, no direct state writes) ---
def scene_and_choices(scene: str, choices: list[str]) -> None:
    """Compat shim: renders the scene and uses the shared grid renderer.
    Clicks will set session_state['pending_choice'] and rerun via choices.render_choices_grid.
    """
    st.write(scene)
    try:
        # Prefer the shared renderer you’re already using elsewhere
        from choices import render_choices_grid  # local import to avoid circulars
        ph = st.empty()
        render_choices_grid(
            ph,
            choices=choices,
            generating=False,
            count=max(2, len(choices or [])),
        )
    except Exception:
        # Minimal fallback: non-interactive display if import fails
        st.subheader("Your choices")
        for c in (choices or []):
            st.write(f"• {c}")
