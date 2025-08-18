# ui/choices.py
from __future__ import annotations
import streamlit as st
from typing import List, Optional

def _turn_id() -> int:
    """0-based number of player decisions so far (user turns)."""
    hist = st.session_state.get("history", [])
    return sum(1 for m in hist if isinstance(m, dict) and m.get("role") == "user")

def render_choices_grid(
    slot: st.DeltaGenerator,
    choices: Optional[List[str]],
    generating: bool,
    count: int = 2,
) -> Optional[str]:
    """
    Always draws a `count`-column grid inside `slot`.
    - If generating=True or choices empty, shows disabled 'Generating…' buttons.
    - If choices provided, clicking sets flags and reruns immediately.
    """
    with slot.container():
        st.subheader("Your choices")
        cols = st.columns(max(1, min(count, len(choices or [])) or count))

        if generating or not choices:
            placeholder = "Generating..." if generating else "Start New Story → sidebar"
            for i in range(count):
                cols[i % len(cols)].button(
                    placeholder,
                    key=f"waiting_{_turn_id()}_{i}",
                    use_container_width=True,
                    disabled=True,
                )
            return None

        tid = _turn_id()
        for i, label in enumerate(choices[:count]):
            key = f"choice_{tid}_{i}"
            if cols[i % len(cols)].button(label, key=key, use_container_width=True):
                # Defer the turn to the main loop
                st.session_state["pending_choice"] = label
                st.session_state["is_generating"] = True
                st.rerun()
        return None  # We never rely on the return value
