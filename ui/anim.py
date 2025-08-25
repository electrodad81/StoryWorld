# ui/anim.py
from __future__ import annotations
import streamlit as st

def inject_css(enabled: bool = True) -> None:
    if not enabled:
        return
    st.markdown(
        """
        <style>
        :root{
          --ink:#2b2b2b;
          --card:#ffffff;
          --ring:#f59e0b; /* warm lantern */
        }

        /* Readable scene card */
        .scene-block{
          color:var(--ink);
          background:var(--card);
          border:1px solid rgba(0,0,0,.08);
          border-radius:12px;
          padding:18px;
          box-shadow:0 8px 18px rgba(0,0,0,.05);
        }

        /* Story text wrapper used in the CENTER column only */
        width: 80%;                 /* fill the story window */
        max-width: none;             /* remove the old 760px clamp */
        margin: 0;
        padding: 1rem 1.25rem;
        background:#fdf8f2;
        color:#333;
        border:1px solid #e8e0d3;
        border-radius:4px;
        box-shadow:0 1px 1px rgba(0,0,0,.05);
        font-family: "Times New Roman", serif;
        font-size: 1.1rem;
        line-height: 1.6;
        }

        /* Choice buttons */
        .stButton>button {
          background: #f6f4e9;
          color: #333333;
          border: 1px solid #c8c8c8;
          border-radius: 8px;
          padding: 10px 16px;
          font-family: "Times New Roman", serif;
        }
        .stButton>button:hover { background:#eee9d9; border-color:#b8b8b8; }

        /* Lantern pulse (used while choices are being generated) */
        .lantern{ display:inline-flex; align-items:center; gap:8px; color:#7a5a00; font-weight:600; }
        .lantern .bulb{
          width:10px; height:10px; border-radius:50%;
          background:#fcd34d;
          box-shadow:0 0 8px #f59e0b, 0 0 16px rgba(245,158,11,.8);
          animation:lanternPulse 1.1s ease-in-out infinite;
        }
        @keyframes lanternPulse{
          0%,100%{ transform:scale(.95); box-shadow:0 0 6px #f59e0b, 0 0 12px rgba(245,158,11,.5) }
          50%     { transform:scale(1.05); box-shadow:0 0 12px #f59e0b, 0 0 28px rgba(245,158,11,.9) }
        }
        @media (prefers-reduced-motion: reduce){ *{ animation:none !important; transition:none !important } }

        /* IMPORTANT: DO NOT clamp or center the whole app.
           (This was causing the large gap + tiny right column.) */
        /* -- removed your global block-container width rules -- */
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_scene(container, markdown_text: str) -> None:
    container.markdown(f'<div class="scene-block">{markdown_text}</div>', unsafe_allow_html=True)

def render_thinking(choices_container) -> None:
    with choices_container:
        st.markdown(
            '<div class="lantern"><span class="bulb"></span>'
            '<span>The lantern glows while choices take shape…</span></div>',
            unsafe_allow_html=True,
        )
