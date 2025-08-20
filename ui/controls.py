from __future__ import annotations

import streamlit as st
from typing import Optional, Literal

Action = Optional[Literal["start", "reset", "switch_user"]]

def sidebar_controls(pid: str) -> Action:
    with st.sidebar:
        st.header("Controls")
        c1, c2 = st.columns(2)
        start_clicked = c1.button("Start New Story", use_container_width=True)
        reset_clicked = c2.button("Reset Session", use_container_width=True, type="secondary")

        # Help text (always visible)
        st.caption("**Start New Story:** Restarts the story using your current selections "
                "(Name, gender, character type).")
        st.caption("**Reset Session:** Clears progress and selections and returns to the start screen.")

        if start_clicked:
            return "start"
        if reset_clicked:
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
