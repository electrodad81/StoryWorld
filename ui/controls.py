from __future__ import annotations

import streamlit as st
from typing import Optional, Literal

Action = Optional[Literal["start", "reset", "switch_user"]]

def sidebar_controls(pid: str) -> Action:
    with st.sidebar:
        st.header("Controls")
        st.caption(f"player_id: {pid[:8]}…")
        col1, col2 = st.columns(2)
        if col1.button("Start New Story", use_container_width=True):
            return "start"
        if col2.button("Reset Session", use_container_width=True):
            return "reset"
        if st.button("Switch user (new id)", use_container_width=True):
            return "switch_user"
    return None

def scene_and_choices(scene: str, choices: list[str]) -> None:
    st.write(scene)
    st.subheader("Your choices")
    c1, c2 = st.columns(2)
    if len(choices) >= 1 and c1.button(choices[0], use_container_width=True):
        st.session_state["_picked"] = choices[0]
    if len(choices) >= 2 and c2.button(choices[1], use_container_width=True):
        st.session_state["_picked"] = choices[1]
