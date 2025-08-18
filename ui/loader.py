# ui/loader.py
from __future__ import annotations
import streamlit as st
from contextlib import contextmanager

def show_lantern_loader(caption: str):
    """Imperative API: returns a callable to stop/hide the loader."""
    slot = st.empty()
    with slot.container():
        st.markdown('<div class="lantern"><span class="flame"></span><b>Working…</b></div>', unsafe_allow_html=True)
        if caption:
            st.caption(caption)
    def stop():
        slot.empty()
    return stop

@contextmanager
def lantern(caption: str):
    """Context manager API: show while in the 'with' block."""
    stop = show_lantern_loader(caption)
    try:
        yield
    finally:
        stop()
