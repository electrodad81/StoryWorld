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

/* ---- Theme tokens ---- */
:root{
  --ink:#2b2b2b;
  --card:#ffffff;
  --ring:#f59e0b;                 /* warm lantern */
  --story-body-width: 80%;      /* single source of truth for content width */
}

/* ---- Scene card (optional) ---- */
.scene-block{
  color:var(--ink);
  background:var(--card);
  border:1px solid rgba(0,0,0,.08);
  border-radius:12px;
  padding:18px;
  box-shadow:0 8px 18px rgba(0,0,0,.05);
}

/* ---- Story text ---- */
.story-window .storybox{
  width: var(--story-body-width);
  max-width: none;
  margin: 0;
  padding: 1rem 1.25rem;
  background:#fdf8f2;
  color:#333;
  border:1px solid #e8e0d3;
  border-radius:4px;
  box-shadow:0 1px 1px rgba(0,0,0,.05);
  font-family: "Times New Roman", serif;
  font-size: 1.5rem;
  line-height: 1.6;
  letter-spacing: .005em;
  text-rendering: optimizeLegibility;
  -webkit-font-smoothing: antialiased;
}
.story-window .storybox p{ margin:0 0 .95rem 0; }
.story-window .storybox p + p{ text-indent:1.25em; }

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

/* ---- Clamp choices to story width ---- */
.story-body,
.story-body > div,
.story-body > div > div{
  width: var(--story-body-width) !important;
  max-width: var(--story-body-width) !important;
  margin: 0 !important;
  padding: 0 !important;
}
.story-body .stButton > button{ width:100% !important; }
.story-body [data-testid="column"]{ padding-left:6px !important; padding-right:6px !important; }

/* ---- Button styling ---- */
.stButton>button{
  background:#f6f4e9;
  color:#333;
  border:1px solid #c8c8c8;
  border-radius:8px;
  padding:10px 16px;
  font-family:"Times New Roman", serif;
}
.stButton>button:hover{ background:#eee9d9; border-color:#b8b8b8; }

/* ---- Lantern pulse ---- */
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

/* ---- (Optional) Mobile TikTok shell kept as-is below ---- */
@media (max-width: 768px){
  .desktop-shell { display:none !important; }
  .mobile-shell  { display:block !important; }
  /* ... your tt-card styles ... */
}
@media (min-width: 850px){
  .desktop-shell { display:block !important; }
  .mobile-shell  { display:none !important; }
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
