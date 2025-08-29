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
  --ring:#f59e0b;                  /* warm lantern */
  --story-body-width: 680px;       /* single source of truth for content width */
}

/* ---- Scene card (optional wrapper you use) ---- */
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
.story-window .storybox p{ margin: 0 0 .95rem 0; }
.story-window .storybox p + p{ text-indent: 1.25em; }

/* ---- Separator (matches story width) ---- */
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

/* --- Clamp choices area to story width (sibling selectors) --- */

/* The marker itself can be invisible; keep it but remove any size */
.choices-wrap{ margin:0 !important; padding:0 !important; }

/* The Streamlit wrappers that follow the marker (immediate + later siblings) */
.choices-wrap + div,
.choices-wrap + div > div,
.choices-wrap + div > div > div,
.choices-wrap ~ div[data-testid="stVerticalBlock"],
.choices-wrap ~ div[data-testid="stVerticalBlock"] > div,
.choices-wrap ~ div[data-testid="stVerticalBlock"] > div > div{
  width: var(--story-body-width) !important;
  max-width: var(--story-body-width) !important;
  margin: 0 !important;
  padding: 0 !important;
}

/* Make the buttons fill the clamped width */
.choices-wrap ~ div .stButton>button{
  width: 100% !important;
}

.choices-wrap{ outline:1px dashed #c99 !important; }
.choices-wrap + div{ outline:1px dashed #9c9 !important; }

/* Tighten gutters on the two-column row */
.choices-wrap ~ div [data-testid="column"]{
  padding-left: 6px !important;
  padding-right: 6px !important;
}

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

/* ---- Lantern pulse (loading hint) ---- */
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

/* ---- Mobile TikTok-style card (scoped; safe to keep idle) ---- */
@media (max-width: 768px){
  .desktop-shell { display: none !important; }
  .mobile-shell  { display: block !important; }

  .tt-card{
    aspect-ratio: 9 / 16;
    width: 100%;
    max-width: 430px;
    max-height: calc(100vh - 2rem);
    margin: 0 auto;
    border-radius: 16px;
    overflow: hidden;
    position: relative;
    background: #111;
    box-shadow: 0 6px 24px rgba(0,0,0,.25);
  }
  .tt-bg{
    position: absolute; inset: 0;
    background-position: center;
    background-size: cover;
    filter: saturate(1.05) contrast(1.05);
  }
  .tt-grad{
    position: absolute; inset: 0;
    background: linear-gradient(180deg,
                rgba(0,0,0,.15) 10%,
                rgba(0,0,0,.25) 35%,
                rgba(0,0,0,.55) 65%,
                rgba(0,0,0,.85) 100%);
  }
  .tt-caption{
    position: absolute;
    left: 14px; right: 14px; bottom: 110px;
    color: #fff;
    font-weight: 600;
    font-size: 1.05rem;
    line-height: 1.35;
    text-shadow: 0 2px 6px rgba(0,0,0,.65), 0 0 1px rgba(0,0,0,.35);
    max-height: 42vh; overflow: hidden;
    display: -webkit-box; -webkit-line-clamp: 8; -webkit-box-orient: vertical;
  }
  .tt-caret{ display:inline-block; animation: ttblink 1s ease-in-out infinite; }
  @keyframes ttblink{ 0%{opacity:.25} 50%{opacity:1} 100%{opacity:.25} }

  .tt-choices{
    position: absolute; left: 12px; right: 12px; bottom: 12px;
    display: grid; gap: 10px;
  }
  .tt-btn{
    width: 100%;
    padding: 12px 14px;
    border-radius: 12px;
    font-weight: 800;
    letter-spacing: .2px;
    border: 1px solid rgba(255,255,255,.18);
    color: #fff;
    background: rgba(0,0,0,.45);
    backdrop-filter: blur(4px);
  }
  .tt-btn:disabled{ opacity: .6; }

  .tt-badge{
    position: absolute; top: 10px; right: 10px;
    background: rgba(0,0,0,.55); color: #fff;
    padding: 6px 10px; border-radius: 999px;
    font-size: .75rem; border: 1px solid rgba(255,255,255,.18);
  }
}

/* Hide mobile shell on larger screens */
@media (min-width: 850px){
  .desktop-shell { display: block !important; }
  .mobile-shell  { display: none !important; }
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
