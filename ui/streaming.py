# ui/streaming.py
from __future__ import annotations
import html
import streamlit as st
from typing import Iterable

def _render_text(slot: st.DeltaGenerator, text: str, caret: bool):
    safe = html.escape(text).replace("\n", "<br>")
    caret_html = ' <span class="typing-caret"></span>' if caret else ""
    # wrap the streaming text in the storybox container
    slot.markdown(
        f"<div class='storybox'>{safe}{caret_html}</div>",
        unsafe_allow_html=True
        )

def stream_text(chunks: Iterable[str], target_slot: st.DeltaGenerator, show_caret: bool = True) -> str:
    """
    Render text incrementally into target_slot. `chunks` is an iterator of small strings.
    Returns the full text.
    """
    buf = []
    for piece in chunks:
        buf.append(piece or "")
        _render_text(target_slot, "".join(buf), caret=(show_caret))
    final = "".join(buf)
    _render_text(target_slot, final, caret=False)
    return final
