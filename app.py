# app.py
import os, time, json, hashlib, re, pathlib
from typing import Optional
import streamlit as st

# --- UI helpers you already have ---
from ui.anim import inject_css
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
# Config / identity
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
# Illustration helpers
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
    """Return (key, job_dict|None) for the *current* scene only."""
    seed = _has_first_sentence(scene_text) or (scene_text or "").strip()
    key  = _scene_key(seed, simple)
    jobs = st.session_state.get("ill_jobs", {})
    return key, jobs.get(key)

def _soft_rerun():
    (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()

def _gentle_autorefresh_if_pending(job_key: Optional[str], delay: float = 0.7, max_polls: int = 40) -> None:
    """Call at the end so choices are already on-screen when the rerun happens."""
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

# illustrate scenes 1,4,7,...
ILLUSTRATION_EVERY_N = 3
ILLUSTRATION_PHASE   = 1

# =============================================================================
# Risk helpers
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
# Render helpers: separator + inline illustration
# =============================================================================
def _render_separator(ph):
    ph.markdown(
        """
        <div class="illus-sep" aria-hidden="true">
          <span class="line"></span><span class="gem">◆</span><span class="line"></span>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _render_illustration_inline(ph, url: Optional[str], status_text: str = "") -> None:
    if url:
        ph.markdown(
            f"""
            <div class="illus-inline">
              <img src="{url}" alt="illustration"/>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        ph.markdown(
            f"""
            <div class="illus-inline illus-skeleton">
              <div class="illus-status">{status_text or "Illustration brewing…"}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# =============================================================================
# Session
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
    st.session_state.setdefault("onboard_dismissed", False)
    st.session_state.setdefault("scene_count", 0)
    st.session_state.setdefault("simple_cyoa", True)   # no sidebar control
    st.session_state.setdefault("auto_illustrate", True)

def reset_session(full_reset=False):
    keep={}
    if not full_reset:
        keep["player_name"]=st.session_state.get("player_name")
        keep["player_gender"]=st.session_state.get("player_gender","Unspecified")
        keep["player_archetype"]=st.session_state.get("player_archetype","Default")
        keep["story_mode"]=bool(st.session_state.get("story_mode",False))
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
# Advance turn: stream text → (separator + art slot / persist) → choices
# =============================================================================
def _advance_turn(pid: str, story_slot, sep_slot, illus_slot, grid_slot, anim_enabled: bool=True):
    old_choices=list(st.session_state.get("choices",[]))
    picked=st.session_state.get("pending_choice")

    # record user choice
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

    # stream story text
    beat=get_current_beat(st.session_state) if st.session_state.get("story_mode") else None
    st.session_state["t_scene_start"]=time.time()
    gen=stream_scene(st.session_state["history"], LORE, beat=beat)

    buf=[]
    for chunk in gen:
        buf.append(chunk)
        text_so_far="".join(buf)
        caret = '<span class="caret">▋</span>' if anim_enabled else ''
        story_slot.markdown(
            f'<div class="story-window"><div class="storybox">{text_so_far}{caret}</div></div>',
            unsafe_allow_html=True,
        )

    full="".join(buf)

    # finalize scene
    stripped=full.rstrip()
    if stripped.endswith("[DEATH]"):
        full=stripped[:-len("[DEATH]")].rstrip()
        st.session_state["is_dead"]=True

    st.session_state["scene"]=full
    st.session_state["history"].append({"role":"assistant","content":full})
    scene_index = st.session_state.get("scene_count", 0) + 1

    # separator
    _render_separator(sep_slot)

    last_url = st.session_state.get("last_illustration_url")

    # decide whether this scene requests a NEW illustration
    should_illustrate = ((scene_index - ILLUSTRATION_PHASE) % ILLUSTRATION_EVERY_N == 0)

    # show illustration slot:
    # - if NEW art is requested: show skeleton (and start job)
    # - otherwise: persist last image if available; else keep slot empty
    if should_illustrate and st.session_state.get("auto_illustrate", True):
        _render_illustration_inline(illus_slot, None, "Illustration brewing…")
        seed=_has_first_sentence(full) or " ".join(full.split()[:18]) + "…"
        if seed.strip():
            _start_illustration_job(seed, bool(st.session_state.get("simple_cyoa", True)))

    # If we already have an image, keep showing it; only show a skeleton if we have nothing yet.
    if last_url:
        _render_illustration_inline(illus_slot, last_url)
    else:
        # First ever scene (no image yet) → show a small skeleton card
        _render_illustration_inline(illus_slot, None, "Illustration brewing…")

    # choices (visible immediately)
    choices=generate_choices(st.session_state["history"],full,LORE)
    st.session_state["choices"]=choices
    grid_slot.markdown('<div class="choices-band"></div>', unsafe_allow_html=True)
    render_choices_grid(grid_slot, choices=choices, generating=False, count=CHOICE_COUNT)
    st.session_state["t_choices_visible_at"]=time.time()

    # beat bookkeeping / snapshot
    if st.session_state.get("story_mode"):
        st.session_state["_beat_scene_count"]=st.session_state.get("_beat_scene_count",0)+1
        current=get_current_beat(st.session_state); target=BEAT_TARGET_SCENES.get(current,1)
        if st.session_state["_beat_scene_count"]>=target:
            if current=="resolution": st.session_state["story_complete"]=True
            else:
                st.session_state["beat_index"]=min(st.session_state.get("beat_index",0)+1,len(BEATS)-1)
                st.session_state["_beat_scene_count"]=0

    username=st.session_state.get("player_name") or st.session_state.get("player_username") or None
    gender=st.session_state.get("player_gender"); archetype=st.session_state.get("player_archetype")
    try: save_snapshot(pid, full, choices, st.session_state["history"], username=username, gender=gender, archetype=archetype)
    except TypeError: save_snapshot(pid, full, choices, st.session_state["history"], username=username)

    st.session_state["scene_count"] = scene_index

# =============================================================================
# Onboarding
# =============================================================================
def onboarding(pid: str):
    with st.container():
        st.markdown('<div class="onboard-panel">', unsafe_allow_html=True)

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
            st.session_state["onboard_dismissed"]=True
            st.session_state["scene_count"]=0
            _soft_rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# Main
# =============================================================================
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🕯️", layout="wide")

    inject_css()
    ensure_keys()
    init_db()
    pid = resolve_pid()

    # As soon as a scene exists, permanently hide onboarding
    if (st.session_state.get("scene") or st.session_state.get("history")):
        st.session_state["onboard_dismissed"] = True

    # --- Book-like type; narrower main, added RIGHT margin via container max-width ---
    st.markdown("""
    <style>
        /* Narrow the main container so there's a right-side margin on large screens.
            This also indirectly constrains the illustration width. */
        [data-testid="stAppViewContainer"] > .main .block-container{
        max-width: 1040px !important;      /* was 100% */
        padding-left: .35rem !important;
        padding-right: 10rem !important;    /* a bit more right padding */
        margin-left: auto; margin-right: auto; /* keep it tidy */
        }

        :root{
        --story-max: 680px;      /* novel-like measure */
        --choices-h: 120px;
        }

        /* Keep the center stack narrow like a book */
        [data-testid="stVerticalBlock"] > div:has(> div.story-window){
        width: 100% !important;
        max-width: var(--story-max) !important;
        margin-left: 0 !important;
        margin-right: auto !important;   /* leaves visible space on the right */
        }

        .story-window{ width:100%; margin: .2rem 0 .25rem 0; }
        .storybox{
        font-family: "Georgia","Garamond","Times New Roman",serif;
        font-size: 1.5rem; line-height: 1.75; letter-spacing: .005em;
        text-rendering: optimizeLegibility; -webkit-font-smoothing: antialiased;
        }
        .storybox p{ margin: 0 0 .95rem 0; }
        .storybox p + p{ text-indent: 1.25em; }

        .illus-sep{ display:flex; align-items:center; gap:.5rem; margin:.12rem 0 .32rem 0; }
        .illus-sep .line{ flex:1; height:1px; background:linear-gradient(90deg,rgba(0,0,0,0),rgba(0,0,0,.28),rgba(0,0,0,0)); }
        .illus-sep .gem{ font-size:.8rem; opacity:.6; line-height:1; }

        /* Illustration takes full story width; no fixed min-height, so it can
            expand into the whitespace directly under the text. */
        .illus-inline{
        width: auto;                         /* allow side margins */
        max-width: calc(100% - 10rem);        /* 1rem left + 1rem right */
        margin: .6rem 1rem 1.4rem 1rem;      /* top | right | bottom | left */
        background: var(--secondary-background-color);
        border: 1px solid rgba(49,51,63,.18);
        border-radius: 6px;
        padding: .55rem;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
        display:flex; align-items:center; justify-content:center;
        }
        .illus-inline img{ width:100%; height:auto; border-radius:8px; display:block; object-fit:contain; }
        .illus-inline .illus-status{ font-size:.95rem; opacity:.75; }
        .illus-skeleton{
        min-height: 220px;
        background: linear-gradient(90deg, rgba(0,0,0,.05) 25%, rgba(0,0,0,.08) 37%, rgba(0,0,0,.05) 63%);
        background-size: 400% 100%; animation: shimmer 1.2s ease-in-out infinite;
        }
        @keyframes shimmer{ 0%{background-position:100% 0} 100%{background-position:-100% 0} }

        .choices-band{ width:100%; min-height: var(--choices-h); }
                
        /* Onboarding card: margin + padding + subtle frame */
        .onboard-panel{
        max-width: 640px;
        margin: 0.5rem auto 1.25rem auto;  /* top | right/left (auto center) | bottom */
        padding: 1rem 1.25rem;
        background: var(--secondary-background-color);
        border: 1px solid rgba(49,51,63,.18);
        border-radius: 12px;
        box-shadow: 0 1px 2px rgba(0,0,0,.05);
        }
        .onboard-panel h1, .onboard-panel h2, .onboard-panel h3{
        margin-top: .2rem; margin-bottom: .6rem;
        }
        .onboard-panel [data-testid="column"]{
        gap: .5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar (no illustration controls)
    try:
        action = sidebar_controls(pid)
    except Exception:
        action = None

    with st.sidebar:
        if DEV_UI_ALLOWED:
            st.session_state["_dev"] = st.checkbox("Developer tools", value=bool(st.session_state.get("_dev", False)))
        else:
            st.session_state["_dev"] = False

    # Sidebar actions
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
        st.session_state["onboard_dismissed"]=True
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

    # Main stack: story → separator → illustration slot (persist/new) → choices
    story_ph = st.empty()
    sep_ph   = st.empty()
    illus_ph = st.empty()
    choices_ph = st.empty()

    # queued turn?
    if st.session_state.get("pending_choice") is not None:
        _advance_turn(pid, story_ph, sep_ph, illus_ph, choices_ph, anim_enabled=True)
        st.session_state["pending_choice"] = None
        st.session_state["is_generating"] = False
        _soft_rerun()
        return

    # Show onboarding ONLY if it hasn't been dismissed AND we're not generating/queued AND no scene yet
    show_onboarding = (
        not st.session_state.get("onboard_dismissed", False)
        and not st.session_state.get("pending_choice")
        and not st.session_state.get("is_generating", False)
        and not st.session_state.get("scene")
    )
    if show_onboarding:
        onboarding(pid)
        return

    # regular render
    story_html = st.session_state.get("scene") or ""
    story_ph.markdown(f'<div class="story-window"><div class="storybox">{story_html}</div></div>',
                      unsafe_allow_html=True)
    _render_separator(sep_ph)

    # illustration: if a job is pending, we’ll swap in place when it finishes;
    # otherwise we just keep whatever last image we have.
    pending_key_for_refresh = None
    url = st.session_state.get("last_illustration_url")
    status_text = ""

    simple_flag = bool(st.session_state.get("simple_cyoa", True))
    key, job = _job_for_scene(st.session_state.get("scene",""), simple_flag)

    if job:
        fut = job["future"]
        if fut.done() and job.get("result") is None:
            try:
                job["result"] = fut.result(timeout=0)
            except Exception as e:
                job["result"] = (None, {"error": repr(e)})

        res = job.get("result")
        if res is not None:
            img_ref, dbg = res if (isinstance(res, tuple) and len(res) == 2) else (res, {})
            st.session_state["last_illustration_url"]  = img_ref
            st.session_state["last_illustration_debug"] = dbg
            url = img_ref
        else:
            # Job still running: keep showing the current image if we have one.
            # Only show a skeleton if we have no image yet.
            if not url:
                status_text = "Illustration brewing…"
            pending_key_for_refresh = key

    # choices (always present)
    choices_ph.markdown('<div class="choices-band"></div>', unsafe_allow_html=True)
    render_choices_grid(choices_ph,
                        choices=st.session_state.get("choices", []),
                        generating=False,
                        count=CHOICE_COUNT)
    st.session_state["t_choices_visible_at"] = time.time()

    # Render: persist current image, or skeleton only when we have none.
    _render_illustration_inline(illus_ph, url, status_text)

    # Ask for a gentle rerun *after* choices are on-screen
    _gentle_autorefresh_if_pending(pending_key_for_refresh)

if __name__ == "__main__":
    main()
