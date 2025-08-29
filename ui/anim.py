# ui/anim.py
from __future__ import annotations
import streamlit as st

def inject_css(enabled: bool = True) -> None:
    if not enabled:
        return

    # Font for mobile TikTok-style captions
    st.markdown(
        "<link href='https://fonts.googleapis.com/css2?family=Inter:wght@600;800&display=swap' rel='stylesheet'>",
        unsafe_allow_html=True,
    )

    # One consolidated <style> block
    st.markdown(
                """

<style>
:root { --story-body-width: 680px; }


/* Anchor width for the story content */
.story-window .storybox{
  width: var(--story-body-width);
  max-width: none;
  margin: 0;
  padding: 1rem 1.25rem;
  border:1px solid #e8e0d3;
  border-radius:4px;
  font-family: "Times New Roman", serif;
  font-size: 1.5rem;
  line-height: 1.6;
  letter-spacing: .005em;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}

/* ---- Separator (match story width) ---- */
.illus-sep{
  width: var(--story-body-width) !important;
  max-width: var(--story-body-width) !important;
  margin: .35rem 0 .6rem 0 !important;
  display:flex; align-items:center; gap:.5rem;
}

.illus-sep .line{
  flex:1; height:1px;
  background:linear-gradient(90deg,rgba(0,0,0,0),rgba(0,0,0,.28),rgba(0,0,0,0));
}
.illus-sep .gem{ font-size:.8rem; opacity:.6; line-height:1; }

/* ---- Illustration (match story width) ---- */
.illus-inline{
  width: var(--story-body-width) !important;
  max-width: var(--story-body-width) !important;
  margin: .6rem 0 1rem 0 !important;
  background: transparent !important;
  border: 0 !important;
  padding: 0 !important;
}
.illus-inline img{
  width: 100% !important;
  height: auto !important;
  display: block;
  border-radius: 4px;
}

/* === Clamp the entire choices block to the same width ===
   The marker (.story-body) sits inside an stMarkdown wrapper.
   We clamp the *outer* Streamlit block that CONTAINS that marker. */
div[data-testid="stVerticalBlock"]:has(.story-body){
  width: var(--story-body-width) !important;
  max-width: var(--story-body-width) !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Make every button inside that clamped block fill the width */
div[data-testid="stVerticalBlock"]:has(.story-body) .stButton > button{
  width: 100% !important;
}

/* Tighten the two-column gutter inside that clamped block */
div[data-testid="stVerticalBlock"]:has(.story-body) [data-testid="column"]{
  padding-left: 6px !important;
  padding-right: 6px !important;
}

</style>

                """,
        unsafe_allow_html=True,
    )


def render_scene(container, markdown_text: str) -> None:
    container.markdown(f'<div class="scene-block">{markdown_text}</div>', unsafe_allow_html=True)

def render_mobile_card(text_html: str, choices: list[str], bg_url: Optional[str], brewing: bool):
    """Single 9:16 card for mobile UI."""
    caret = '<span class="tt-caret">▋</span>' if st.session_state.get("_streaming", False) else ""
    safe_bg = bg_url or ""  # empty = gradient only

    st.markdown('<div class="mobile-shell">', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="tt-card">
          <div class="tt-bg" style="background-image:url('{safe_bg}');"></div>
          <div class="tt-grad"></div>
          {'<div class="tt-badge">Illustration brewing…</div>' if brewing and not safe_bg else ''}
          <div class="tt-caption">{text_html}{caret}</div>
          <div class="tt-choices">
            <!-- Buttons are rendered below via Streamlit to wire clicks -->
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Wire the choices into the same .tt-choices zone using Streamlit columns hack
    # (Streamlit renders below; CSS keeps it visually in place.)
    c1 = st.button(choices[0], key=f"m_choice_0", use_container_width=True) if len(choices) > 0 else None
    c2 = st.button(choices[1], key=f"m_choice_1", use_container_width=True) if len(choices) > 1 else None

    if c1:
        st.session_state["pending_choice"] = choices[0]
        st.rerun()
    if c2:
        st.session_state["pending_choice"] = choices[1]
        st.rerun()

    st.markdown('</div>', unsafe_allow_html=True)
