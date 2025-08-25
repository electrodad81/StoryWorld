# app.py
import os, time, json, hashlib, re, pathlib
from typing import Optional
import streamlit as st
import streamlit.components.v1 as components

# --- UI helpers ---
from ui.anim import inject_css, render_thinking
from ui.choices import render_choices_grid
from ui.controls import sidebar_controls

# --- Story engine ---
from story.engine import stream_scene, generate_choices, generate_illustration

# --- Persistence / telemetry ---
from data.store import (
    init_db, save_snapshot, load_snapshot, delete_snapshot,
    has_snapshot, save_visit, save_event
)

from concurrent.futures import ThreadPoolExecutor

# =============================================================================
# Config / identity helpers
# =============================================================================
def _to_bool(v, default=False):
    if v is None: return default
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return v != 0
    return str(v).strip().lower() in {"1","true","t","yes","y","on"}

def _from_secrets_or_env(*keys, default=None):
    for k in keys:
        try:
            if k in st.secrets:
                return st.secrets[k]
        except Exception:
            pass
        if k in os.environ:
            return os.environ[k]
    return default

DEV_UI_ALLOWED = _to_bool(_from_secrets_or_env("dev_ui","DEV_UI","show_dev","SHOW_DEV"), default=False)

def resolve_pid() -> str:
    if "pid" in st.session_state and st.session_state["pid"]:
        return st.session_state["pid"]
    try:
        qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
    except Exception:
        qp = {}
    qp_pid = None
    if isinstance(qp, dict) and "pid" in qp:
        v = qp["pid"]
        qp_pid = v[0] if isinstance(v, list) else v
    pid = (
        qp_pid
        or _from_secrets_or_env("pid","PID","user_id","USER_ID")
        or os.environ.get("USER") or os.environ.get("USERNAME")
        or "local-user"
    )
    st.session_state["pid"] = str(pid)
    return st.session_state["pid"]

# =============================================================================
# Illustration helpers (non-streaming)
# =============================================================================
@st.cache_resource
def _ill_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)

def _has_first_sentence(txt: str) -> Optional[str]:
    m = re.search(r"(.+?[.!?])(\s|$)", txt or "")
    if not m: return None
    sent = m.group(1).strip()
    return sent if len(sent.split()) >= 8 else None

def _scene_key(text: str, simple: bool) -> str:
    h = hashlib.md5((text.strip() + f"|{simple}").encode("utf-8")).hexdigest()
    return f"ill-{h}"

def _start_illustration_job(seed_sentence: str, simple: bool) -> str:
    key = _scene_key(seed_sentence, simple)
    jobs = st.session_state.setdefault("ill_jobs", {})
    if key in jobs:
        return key
    fut = _ill_executor().submit(generate_illustration, seed_sentence, simple)
    jobs[key] = {"future": fut, "result": None}
    st.session_state["ill_last_key"] = key
    return key

def _job_for_scene(scene_text: str, simple: bool):
    jobs = st.session_state.get("ill_jobs", {})
    last_key = st.session_state.get("ill_last_key")
    if last_key and last_key in jobs:
        return last_key, jobs[last_key]
    seed = _has_first_sentence(scene_text) or (scene_text or "").strip()
    key  = _scene_key(seed, simple)
    return key, jobs.get(key)

def _soft_rerun():
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()

def _gentle_autorefresh_if_pending(job_key: Optional[str], delay: float = 0.6, max_polls: int = 30) -> None:
    if not job_key: return
    job = st.session_state.get("ill_jobs", {}).get(job_key)
    if not job or job["future"].done(): return
    polls = st.session_state.setdefault("ill_polls", {})
    c = int(polls.get(job_key, 0))
    if c >= max_polls: return
    polls[job_key] = c + 1
    st.session_state["ill_polls"] = polls
    time.sleep(delay)
    _soft_rerun()

# =============================================================================
# Lore / constants
# =============================================================================
ROOT = pathlib.Path(__file__).parent
LORE_PATH = ROOT / "lore.json"
LORE = json.loads(LORE_PATH.read_text(encoding="utf-8")) if LORE_PATH.exists() else {}

CHOICE_COUNT = 2
APP_TITLE = "Gloamreach"

# only show an image every N scenes: 1,4,7,…
ILLUSTRATION_EVERY_N = 3
ILLUSTRATION_PHASE   = 1

# =============================================================================
# Risk / end-state helpers
# =============================================================================
_RISKY_WORDS = {
    "charge","attack","fight","steal","stab","break","smash","dive","jump",
    "descend","enter","drink","touch","open the","confront","cross","swim",
    "sprint","bait","ambush","bleed","sacrifice","shout","brave","risk",
    "gamble","rush","kick","force","pry","ignite","set fire"
}
def is_risky_label(label: str) -> bool:
    s = (label or "").lower()
    return any(re.search(rf"\b{re.escape(w)}\b", s) for w in _RISKY_WORDS)

def update_consequence_counters(picked_label: Optional[str]) -> None:
    if not picked_label or picked_label == "__start__": return
    if is_risky_label(picked_label):
        st.session_state["danger_streak"] = st.session_state.get("danger_streak", 0) + 1
    else:
        st.session_state["danger_streak"] = max(0, st.session_state.get("danger_streak", 0) - 1)

def detect_cost_in_scene(text: str) -> bool:
    for w in ["bleed","wound","cut","bruis","sprain","fracture","poison",
              "lose","lost","dropped","broke","shattered","torn","spent",
              "captured","seized","trapped","cornered","exposed","compromised",
              "too late","ticking","deadline","out of time","cannot return",
              "ally leaves","ally falls","alone now","weakened","limp"]:
        if w in (text or "").lower(): return True
    return False

# =============================================================================
# Right-panel card renderer
# =============================================================================
def _render_right_panel(ph, url: Optional[str], status_text: str = "") -> None:
    img = f'<div class="ill-frame"><img src="{url}" alt="illustration" /></div>' if url else '<div class="ill-frame skeleton"></div>'
    status = f'<div class="small">{status_text}</div>' if status_text else ''
    ph.markdown(
        f'''
        <div class="right-sidebar">
          <h4>Illustration</h4>
          {img}
          {status}
        </div>
        ''',
        unsafe_allow_html=True,
    )

# =============================================================================
# Session keys
# =============================================================================
def ensure_keys():
    st.session_state.setdefault("scene","")
    st.session_state.setdefault("choices",[])
    st.session_state.setdefault("history",[])
    st.session_state.setdefault("pending_choice",None)
    st.session_state.setdefault("is_generating",False)
    st.session_state.setdefault("t_scene_start",None)
    st.session_state.setdefault("t_choices_visible_at",None)
    st.session_state.setdefault("player_name",None)
    st.session_state.setdefault("player_archetype","Default")
    st.session_state.setdefault("danger_streak",0)
    st.session_state.setdefault("injury_level",0)
    st.session_state.setdefault("is_dead",False)
    st.session_state.setdefault("story_mode",True)
    st.session_state.setdefault("beat_index",0)
    st.session_state.setdefault("story_complete",False)
    st.session_state.setdefault("_dev",False)
    st.session_state.setdefault("_beat_scene_count",0)
    st.session_state.setdefault("last_illustration_url",None)
    st.session_state.setdefault("app_stage","onboarding")      # "onboarding" | "play"
    st.session_state.setdefault("onboard_dismissed", False)    # sticky hide
    st.session_state.setdefault("scene_count", 0)

def reset_session(full_reset=False):
    keep={}
    if not full_reset:
        keep["player_name"]=st.session_state.get("player_name")
        keep["player_gender"]=st.session_state.get("player_gender","Unspecified")
        keep["player_archetype"]=st.session_state.get("player_archetype","Default")
        keep["story_mode"]=bool(st.session_state.get("story_mode",False))
        keep["app_stage"]="play"
        keep["onboard_dismissed"]=True
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    ensure_keys()
    for k,v in keep.items():
        st.session_state[k]=v

# =============================================================================
# Beats
# =============================================================================
BEATS=["exposition","rising_action","climax","falling_action","resolution"]
BEAT_TARGET_SCENES={"exposition":2,"rising_action":3,"climax":2,"falling_action":2,"resolution":1}
def get_current_beat(state)->str:
    i=int(state.get("beat_index",0)); i=max(0,min(i,len(BEATS)-1)); return BEATS[i]
def mark_story_complete(): st.session_state["story_complete"]=True
def is_story_complete(state)->bool: return bool(state.get("story_complete",False))

# =============================================================================
# Advance turn (stream text; image AFTER; every-N)
# =============================================================================
def _advance_turn(pid: str, story_slot, grid_slot, right_panel_slot, anim_enabled: bool=True):
    old_choices=list(st.session_state.get("choices",[]))
    picked=st.session_state.get("pending_choice")
    st.session_state["last_illustration_debug"]={}

    update_consequence_counters(picked)

    if picked and picked!="__start__":
        hist=st.session_state["history"]
        if not hist or hist[-1].get("role")!="user" or hist[-1].get("content")!=picked:
            hist.append({"role":"user","content":picked})
        visible_at=st.session_state.get("t_choices_visible_at")
        latency_ms=int((time.time()-visible_at)*1000) if visible_at else None
        save_event(pid,"choice",{
            "label":picked,
            "index":(old_choices.index(picked) if picked in old_choices else None),
            "latency_ms":latency_ms,
            "decisions_count":sum(1 for m in hist if m.get("role")=="user"),
        })

    render_thinking(grid_slot)

    beat=get_current_beat(st.session_state) if st.session_state.get("story_mode") else None
    st.session_state["t_scene_start"]=time.time()
    gen=stream_scene(st.session_state["history"], LORE, beat=beat)

    buf=[]
    for chunk in gen:
        buf.append(chunk)
        text_so_far="".join(buf)
        caret_html = '<span class="caret">▋</span>' if anim_enabled else ''
        story_slot.markdown(
            f'<div class="story-window"><div class="storybox">{text_so_far}{caret_html}</div></div>',
            unsafe_allow_html=True,
        )

    full="".join(buf)

    died=False
    stripped=full.rstrip()
    if stripped.endswith("[DEATH]"):
        died=True; full=stripped[:-len("[DEATH]")].rstrip()

    st.session_state["scene"]=full
    st.session_state["history"].append({"role":"assistant","content":full})

    scene_index = st.session_state.get("scene_count", 0) + 1

    must_escalate=st.session_state.get("danger_streak",0)>=2
    expected_cost=bool(picked and picked!="__start__" and is_risky_label(picked)) or must_escalate
    if expected_cost and not died and not detect_cost_in_scene(full):
        try: save_event(pid,"missed_consequence",{"picked":picked,"danger_streak":st.session_state.get("danger_streak",0)})
        except Exception: pass

    if died:
        st.session_state["is_dead"]=True
        try: save_event(pid,"death",{"picked":picked,"decisions_count":sum(1 for m in st.session_state["history"] if m.get("role")=="user"),"danger_streak":st.session_state.get("danger_streak",0)})
        except Exception: pass
        username=st.session_state.get("player_name") or st.session_state.get("player_username") or None
        gender=st.session_state.get("player_gender"); archetype=st.session_state.get("player_archetype")
        try: save_snapshot(pid,full,[],st.session_state["history"],username=username,gender=gender,archetype=archetype)
        except TypeError: save_snapshot(pid,full,[],st.session_state["history"],username=username)
        render_choices_grid(grid_slot, choices=[], generating=False, count=CHOICE_COUNT)
        return

    if st.session_state.get("story_mode"):
        st.session_state["_beat_scene_count"]=st.session_state.get("_beat_scene_count",0)+1
        current=get_current_beat(st.session_state); target=BEAT_TARGET_SCENES.get(current,1)
        if st.session_state["_beat_scene_count"]>=target:
            if current=="resolution": mark_story_complete()
            else:
                st.session_state["beat_index"]=min(st.session_state.get("beat_index",0)+1,len(BEATS)-1)
                st.session_state["_beat_scene_count"]=0

    choices=generate_choices(st.session_state["history"],full,LORE)
    st.session_state["choices"]=choices

    username=st.session_state.get("player_name") or st.session_state.get("player_username") or None
    gender=st.session_state.get("player_gender"); archetype=st.session_state.get("player_archetype")
    try: save_snapshot(pid,full,choices,st.session_state["history"],username=username,gender=gender,archetype=archetype)
    except TypeError: save_snapshot(pid,full,choices,st.session_state["history"],username=username)

    grid_slot.markdown('<div class="choices-band"></div>', unsafe_allow_html=True)
    render_choices_grid(grid_slot, choices=choices, generating=False, count=CHOICE_COUNT)
    st.session_state["t_choices_visible_at"]=time.time()

    # ---- Illustrate AFTER text/choices, only every N scenes ----
    should_illustrate = ((scene_index - ILLUSTRATION_PHASE) % ILLUSTRATION_EVERY_N == 0)
    if st.session_state.get("auto_illustrate",True) and should_illustrate:
        simple_flag=bool(st.session_state.get("simple_cyoa",True))
        seed=_has_first_sentence(full) or " ".join(full.split()[:18]) + "…"
        if seed.strip():
            _start_illustration_job(seed, simple_flag)
            _render_right_panel(
                right_panel_slot,
                st.session_state.get("last_illustration_url"),
                "Generating illustration…"
            )

    st.session_state["scene_count"] = scene_index

# =============================================================================
# Onboarding
# =============================================================================
def onboarding(pid: str):
    st.header("Begin Your Journey")
    st.markdown("Pick your setup. Name and character are locked once you begin.")

    name = st.text_input("Name", value=st.session_state.get("player_name") or "", max_chars=24)
    gender = st.selectbox("Gender", ["Unspecified", "Female", "Male", "Nonbinary"], index=0)
    archetype = st.selectbox("Character type", ["Default"], index=0)

    mode_default_index = 0 if st.session_state.get("story_mode", True) else 1
    mode = st.radio("Mode", options=["Story Mode", "Exploration"], index=mode_default_index)

    c1, c2 = st.columns(2)
    begin = c1.button("Begin Adventure", use_container_width=True, disabled=not name.strip(), key="onboard_begin")
    reset = c2.button("Reset Session", use_container_width=True, key="onboard_reset")

    if reset:
        delete_snapshot(pid); reset_session(full_reset=True); _soft_rerun()
    if begin:
        st.session_state["player_name"]=name.strip()
        st.session_state["player_gender"]=gender
        st.session_state["player_archetype"]=archetype
        st.session_state["story_mode"]=(mode=="Story Mode")
        st.session_state["beat_index"]=0; st.session_state["story_complete"]=False
        st.session_state["pending_choice"]="__start__"
        st.session_state["is_generating"]=True
        st.session_state["app_stage"]="play"
        st.session_state["onboard_dismissed"]=True     # sticky-hide onboarding
        st.session_state["scene_count"]=0
        _soft_rerun()

# =============================================================================
# JS installer for a real right-edge resizer (column width persists)
# =============================================================================
def _install_right_resizer():
    """
    Injects a tiny JS snippet (in a zero-height component) that:
      - finds the RIGHT st.column (via #c3-marker),
      - adds a full-height grab bar on its LEFT edge,
      - drags to set the column's width (flex-basis) live,
      - persists width in localStorage across reruns.
    """
    components.html(
        """
        <script>
        (function(){
          const P = window.parent;  // main Streamlit document
          if (P.__c3ResizerInstalled) return;
          P.__c3ResizerInstalled = true;

          function setup(){
            const marker = P.document.getElementById('c3-marker');
            if(!marker){ setTimeout(setup, 300); return; }

            const col = marker.closest('div[data-testid="column"]');     // RIGHT column element
            if(!col){ setTimeout(setup, 300); return; }
            const row = col.parentElement;  // stHorizontalBlock
            const leftCol = row.querySelector('div[data-testid="column"]:first-child');

            // Prepare columns for manual width control
            col.style.position = 'relative';
            col.style.flex = '0 0 auto';     // use explicit pixel width
            leftCol.style.flex = '1 1 auto'; // center grows/shrinks

            // Apply saved width (persist across reruns)
            try {
              const saved = P.localStorage.getItem('c3WidthPx');
              if(saved){ col.style.width = saved + 'px'; }
            } catch(e) {}

            // Build a full-height left-edge grab bar
            let handle = P.document.getElementById('c3-handle');
            if(!handle){
              handle = P.document.createElement('div');
              handle.id = 'c3-handle';
              Object.assign(handle.style, {
                position: 'absolute',
                left: '-6px', top: '0px', bottom: '0px',
                width: '10px',
                cursor: 'col-resize',
                zIndex: 1000
              });
              col.appendChild(handle);
            }

            let startX = 0, startW = 0;
            function onDown(ev){
              ev.preventDefault();
              startX = ev.clientX;
              startW = col.getBoundingClientRect().width;
              P.document.addEventListener('mousemove', onMove);
              P.document.addEventListener('mouseup', onUp);
            }
            function onMove(ev){
              const dx = ev.clientX - startX;      // drag to right => dx>0
              let newW = startW - dx;               // because handle on LEFT edge
              const min = 240;
              const max = Math.min(P.innerWidth * 0.60, 1200);
              newW = Math.max(min, Math.min(max, newW));
              col.style.width = newW + 'px';
              col.style.flex  = '0 0 auto';
            }
            function onUp(){
              P.document.removeEventListener('mousemove', onMove);
              P.document.removeEventListener('mouseup', onUp);
              try {
                const w = Math.round(col.getBoundingClientRect().width);
                P.localStorage.setItem('c3WidthPx', String(w));
              } catch(e) {}
            }
            handle.addEventListener('mousedown', onDown);

            // Guard against rerenders replacing column internals
            const obs = new P.MutationObserver(function(){
              if(!col.contains(handle)){
                col.appendChild(handle);
              }
              try {
                const w = P.localStorage.getItem('c3WidthPx');
                if(w){
                  col.style.width = w + 'px';
                  col.style.flex  = '0 0 auto';
                  leftCol.style.flex = '1 1 auto';
                }
              } catch(e) {}
            });
            obs.observe(row, {childList: true, subtree: true});
          }
          setup();
        })();
        </script>
        """,
        height=0, width=0, scrolling=False
    )

# =============================================================================
# Main
# =============================================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🕯️", layout="wide")
    inject_css()
    ensure_keys()
    init_db()
    pid = resolve_pid()

    # --- Layout CSS (center auto, novel max width; right card fills its column) ---
    st.markdown("""
    <style>
      [data-testid="stAppViewContainer"] > .main .block-container{
        max-width: 100% !important;
        padding-left: .35rem !important;
        padding-right: 1.0rem !important;
      }

      :root{
        --story-max: 640px;    /* novel-like max width for story+buttons */
        --story-h: 640px;
        --choices-h: 120px;
      }

      /* Center column grows/shrinks, but caps at --story-max for readability */
      [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child [data-testid="stVerticalBlock"]{
        width: 100% !important;
        max-width: var(--story-max) !important;
        margin-left: 0 !important;
        margin-right: auto !important;
      }

      .story-window{ width:100%; height:var(--story-h); overflow-y:auto; padding:.25rem 0; }
      .choices-band{ width:100%; min-height:var(--choices-h); }

      .right-sidebar{
        width: 100%;
        background: var(--secondary-background-color);
        border: 1px solid rgba(49,51,63,.20);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 1px 2px rgba(0,0,0,.06);
        position: sticky; top: .75rem;
      }
      .right-sidebar h4{ margin: 0 0 .5rem; font-weight: 600; }
      .right-sidebar .ill-frame{ width:100%; }
      .right-sidebar .ill-frame img{ width:100%; height:auto; display:block; border-radius:6px; }

      .skeleton{
        min-height: 220px;
        background: linear-gradient(90deg, rgba(0,0,0,.05) 25%, rgba(0,0,0,.08) 37%, rgba(0,0,0,.05) 63%);
        background-size: 400% 100%;
        animation: shimmer 1.2s ease-in-out infinite;
        border-radius:8px;
      }
      @keyframes shimmer{ 0%{background-position:100% 0} 100%{background-position:-100% 0} }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar (C1)
    try:
        action = sidebar_controls(pid)
    except Exception:
        action = None

    with st.sidebar:
        if DEV_UI_ALLOWED:
            st.session_state["_dev"] = st.checkbox("Developer tools", value=bool(st.session_state.get("_dev", False)))
            if st.session_state["_dev"]:
                beat = get_current_beat(st.session_state) if st.session_state.get("story_mode") else "classic"
                st.caption(
                    f"Story Mode: {'on' if st.session_state.get('story_mode') else 'off'} • "
                    f"Beat: {beat} • Danger streak: {st.session_state.get('danger_streak', 0)} • "
                    f"Scenes: {len(st.session_state.get('history', []))} • "
                    f"Ill key: {st.session_state.get('ill_last_key','–')} • "
                    f"Auto: {st.session_state.get('auto_illustrate', True)} • "
                    f"Simple: {st.session_state.get('simple_cyoa', True)}"
                )
        else:
            st.session_state["_dev"] = False

        st.markdown("### Illustration")
        st.session_state["simple_cyoa"] = st.checkbox(
            "Simple CYOA style",
            value=bool(st.session_state.get("simple_cyoa", True)),
            help="Cleaner line art, minimal detail"
        )

        # Optional: manual generate button
        if st.button("Generate illustration for this scene", use_container_width=True, key="btn_gen_ill"):
            scene_text = (st.session_state.get("scene") or "").strip()
            if not scene_text:
                st.warning("No scene text is available yet.")
            else:
                seed = _has_first_sentence(scene_text) or " ".join(scene_text.split()[:18]) + "…"
                if seed.strip():
                    _start_illustration_job(seed, bool(st.session_state["simple_cyoa"]))
                    st.session_state["last_illustration_url"] = None
                    st.session_state["last_illustration_debug"] = {}
                    _soft_rerun()

    # Handle sidebar actions (start/reset)
    if action == "start":
        try: delete_snapshot(pid)
        except Exception: pass
        for k in ("scene","choices","history","pending_choice","is_generating",
                  "t_choices_visible_at","t_scene_start","is_dead","beat_index","story_complete"):
            st.session_state.pop(k, None)
        if st.session_state.get("story_mode"):
            st.session_state["beat_index"]=0; st.session_state["story_complete"]=False
        st.session_state["pending_choice"]="__start__"
        st.session_state["is_generating"]=True
        st.session_state["app_stage"]="play"
        st.session_state["onboard_dismissed"]=True       # sticky-hide onboarding
        st.session_state["scene_count"]=0
        _soft_rerun()

    elif action == "reset":
        try: delete_snapshot(pid)
        except Exception: pass
        for k in ("scene","choices","history","pending_choice","is_generating",
                  "t_choices_visible_at","t_scene_start","is_dead","beat_index","story_complete",
                  "player_name","player_gender","player_archetype","story_mode"):
            st.session_state.pop(k, None)
        _soft_rerun()

    # MAIN: Center (C2) + Right (C3)
    center_col, right_col = st.columns([3.5, 1.1], gap="small")

    with center_col:
        story_ph = st.empty()
        choices_ph = st.empty()

    with right_col:
        # Marker so the JS can find the RIGHT column reliably
        st.markdown('<div id="c3-marker"></div>', unsafe_allow_html=True)
        right_panel_ph = st.empty()

    # Advance queued turn, then return to let the idle path poll the image
    if st.session_state.get("pending_choice") is not None:
        _advance_turn(pid, story_ph, choices_ph, right_panel_ph, anim_enabled=True)
        st.session_state["pending_choice"] = None
        st.session_state["is_generating"] = False
        _soft_rerun()
        return

    # Should we show onboarding?
    show_onboarding = (not st.session_state.get("onboard_dismissed", False))

    # Center content
    if show_onboarding:
        with center_col:
            onboarding(pid)
    else:
        with center_col:
            story_html = st.session_state.get("scene") or ""
            story_ph.markdown(f'<div class="story-window"><div class="storybox">{story_html}</div></div>',
                              unsafe_allow_html=True)
            choices_ph.markdown('<div class="choices-band"></div>', unsafe_allow_html=True)
            render_choices_grid(choices_ph,
                                choices=st.session_state.get("choices", []),
                                generating=False,
                                count=CHOICE_COUNT)
            st.session_state["t_choices_visible_at"] = time.time()

    # Right panel (hidden during onboarding)
    if show_onboarding:
        right_panel_ph.empty()
    else:
        pending_key_for_refresh = None
        url = st.session_state.get("last_illustration_url")
        status_text = ""
        simple_flag = bool(st.session_state.get("simple_cyoa", True))
        key, job = _job_for_scene(st.session_state.get("scene",""), simple_flag)
        if job:
            fut = job["future"]
            if fut.done() and job.get("result") is None:
                try: job["result"] = fut.result(timeout=0)
                except Exception as e: job["result"] = (None, {"error": repr(e)})
            res = job.get("result")
            if res is not None:
                img_ref, dbg = res if (isinstance(res, tuple) and len(res) == 2) else (res, {})
                st.session_state["last_illustration_url"]  = img_ref
                st.session_state["last_illustration_debug"] = dbg
                st.session_state["ill_last_key"] = None
                url = img_ref
            else:
                status_text = "Generating illustration…"
                pending_key_for_refresh = key

        _render_right_panel(right_panel_ph, url, status_text)
        _gentle_autorefresh_if_pending(pending_key_for_refresh)

    # Install the real right-edge resizer (once)
    _install_right_resizer()

if __name__ == "__main__":
    main()
