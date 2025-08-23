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
          --ink:#e9e9f5;
          --card:#12131b;
          --ring:#f59e0b; /* warm lantern */
        }

        /* Readable scene card (no overlay, no opacity tricks) */
        .scene-block{
          color:var(--ink);
          background:var(--card);
          border:1px solid rgba(255,255,255,.08);
          border-radius:14px;
          padding:18px;
          box-shadow:0 8px 18px rgba(0,0,0,.35);
        }

        /* Choices — neutral styling, no popups */
        .choice-zone h3{ margin:0 0 8px 2px; color:#cfd0ff; font-weight:600; }
        .choice-zone .stButton>button{
          width:100%;
          border-radius:12px;
          padding:12px 14px;
          background:#171825;
          color:var(--ink);
          border:1px solid rgba(255,255,255,.08);
        }

        .choice-zone .stButton>button:hover{
          border-color:rgba(245,158,11,.65);
        }

        /* Tiny pulsing lantern used while choices are computed */
        .lantern{
          display:inline-flex; align-items:center; gap:8px;
          color:#ffeab6; font-weight:600;
        }

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

        /* Respect reduced motion */
        @media (prefers-reduced-motion: reduce){
          *{ animation:none !important; transition:none !important }
        }

        /* A fixed story container for streaming text */
        .storybox {
            max-width: 650px;
            margin-left: auto;
            margin-right: auto;
            padding: 1.2rem;
            min-height: 20rem;  /* adjust height as desired */
            background: #fffff
            color: #111111
            border: 1px solid rgba(255,255,255,.08);
            border-radius: 4px;
            box-shadow: 0 8px 18px rgba(0,0,0,.35);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_scene(container, markdown_text: str) -> None:
    """Write the scene into a readable card (no overlays)."""
    container.markdown(f'<div class="scene-block">{markdown_text}</div>', unsafe_allow_html=True)

def render_thinking(choices_container) -> None:
    """Show the lantern pulse while choices are being generated."""
    with choices_container:
        st.markdown(
            '<div class="lantern"><span class="bulb"></span>'
            '<span>The lantern glows while choices take shape…</span></div>',
            unsafe_allow_html=True,
        )

def render_choices(choices: list[str], choices_container, turn_id: int) -> None:
    """Render two buttons in-place (no duplicate scenes, no popups)."""
    with choices_container:
        st.markdown('<div class="choice-zone"><h3>Your choices</h3></div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if len(choices) >= 1 and c1.button(choices[0], key=f"c1_{turn_id}", use_container_width=True):
            st.session_state["_picked"] = choices[0]
        if len(choices) >= 2 and c2.button(choices[1], key=f"c2_{turn_id}", use_container_width=True):
            st.session_state["_picked"] = choices[1]
