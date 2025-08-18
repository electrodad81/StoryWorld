# ui/choices.py
from __future__ import annotations
import streamlit as st
from typing import List, Optional

def render_choices_grid(
    slot: st.DeltaGenerator,
    choices: Optional[List[str]],
    generating: bool,
    count: int = 2,
) -> Optional[str]:
    """
    Always draws a `count`-column grid inside `slot`.
    - If generating=True or choices empty, shows disabled 'Generating…' buttons.
    - If choices provided, returns the label of the clicked choice (or None).
    """
    clicked: Optional[str] = None
    with slot.container():
        st.subheader("Your choices")
        cols = st.columns(count)

        if not generating and choices:
            for i, label in enumerate(choices[:count]):
                key = f"choice_{i}_{abs(hash(label)) % 10_000}"
                with cols[i]:
                    if st.button(label, key=key, use_container_width=True):
                        clicked = label
        else:
            placeholder = "Generating..." if generating else "Start New Story → sidebar"
            for i in range(count):
                with cols[i]:
                    st.button(placeholder, key=f"waiting_{i}", use_container_width=True, disabled=True)

    return clicked
