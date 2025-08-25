# app.py
import os, time, json, hashlib, re, pathlib
from typing import Optional
import streamlit as st

from core.identity import ensure_browser_id

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

try:
    from streamlit_js_eval import streamlit_js_eval
except Exception:
    streamlit_js_eval = None

def _pin_pid_in_url(pid: str) -> None:
    """
    Ensure ?pid=... is present in the current URL (without reloading the page).
    Uses history.replaceState via JS for maximum compatibility on hosted Streamlit.
    Safe no-op if the JS helper isn't available.
    """
    if not pid:
        return
    # Prefer JS push/replace to avoid Streamlit re-mount quirks in hosted envs
    if streamlit_js_eval:
        try:
            streamlit_js_eval(
                js_expressions=f"""
                    (() => {{
                        const url = new URL(window.location.href);
                        url.searchParams.set('pid', '{pid}');
                        window.history.replaceState({{}}, '', url.toString());
                        return url.toString();
                    }})()
                """,
                key=f"pin_pid_{pid}",
            )
            return
        except Exception:
            pass
    # Fallback to Streamlit's API
    try:
        if hasattr(st, "query_params"):
            st.query_params.update({"pid": pid})
        else:
            cur = st.experimental_get_query_params()
            cur["pid"] = pid
            st.experimental_set_query_params(**cur)
    except Exception:
        pass

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

DEV_UI_ALLOWED = _to_bool(
    _from_secrets_or_env("DEBUG_UI", "dev_ui", "DEV_UI", "show_dev", "SHOW_DEV"),
    default=False,
)

def resolve_pid() -> str:
    """
    Stable per-browser PID resolution:

    Order:
      1) URL ?pid=... (authoritative)
      2) localStorage via ensure_browser_id()
      3) If PID_WAIT_BOOTSTRAP is true: do a one-time soft rerun to wait for localStorage.
         Otherwise: fall back to 'local-user' (keeps local refresh working).

    When we get a browser PID, pin it into the URL (no page reload).
    """
    # 0) Already in session
    if st.session_state.get("pid"):
        return st.session_state["pid"]

    # 1) From URL
    try:
        qp = st.query_params if hasattr(st, "query_params") else st.experimental_get_query_params()
        qp_pid = qp.get("pid")
        if isinstance(qp_pid, list):
            qp_pid = qp_pid[0]
        if qp_pid:
            pid = str(qp_pid)
            st.session_state["pid"] = pid
            _pin_pid_in_url(pid)
            return pid
    except Exception:
        pass

    # 2) From localStorage (component may not be ready on first render in hosted envs)
    pid = None
    try:
        pid = ensure_browser_id()  # returns brw-... when ready; may be None on first paint
    except Exception:
        pid = None

    if pid:
        pid = str(pid)
        st.session_state["pid"] = pid
        _pin_pid_in_url(pid)
        return pid

    # 3) Fallback differs by mode:
    if PID_WAIT_BOOTSTRAP:
        # Hosted: wait exactly one render to let localStorage initialize
        if not st.session_state.get("_pid_bootstrap_waited"):
            st.session_state["_pid_bootstrap_waited"] = True
            time.sleep(0.03)
            (getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None))()
            st.stop()
        # If we already waited and still no pid, degrade gracefully:
        pid = os.environ.get("PID") or "local-user"
    else:
        # Local: do NOT interrupt; preserve your previously-working refresh behavior
        pid = os.environ.get("PID") or "local-user"

    st.session_state["pid"] = str(pid)
    return st.session_state["pid"]

# NEW: enable “hosted PID bootstrap” only when you want it (e.g., on deployed)
PID_WAIT_BOOTSTRAP = _to_bool(_from_secrets_or_env("PID_WAIT_BOOTSTRAP", "DEPLOYED", "FORCE_URL_PID"), default=False)

def _maybe_restore_from_snapshot(pid: str) -> bool:
    """
    If memory is empty and a snapshot exists, hydrate session_state.
    Returns True if a restore occurred.
    """
    # don't clobber an active generation
    if st.session_state.get("is_generating") or st.session_state.get("pending_choice"):
        return False
    # if we already have a scene/history, nothing to do
    if st.session_state.get("scene") or st.session_state.get("history"):
        return False

    try:
        snap = load_snapshot(pid)
    except Exception:
        snap = None

    if not snap:
        return False

    # Core story state
    st.session_state["scene"]   = snap.get("scene") or ""
    st.session_state["choices"] = snap.get("choices") or []
    st.session_state["history"] = snap.get("history") or []

    img = snap.get("last_illustration_url")
    if img:
        st.session_state["display_illustration_url"] = img
        st.session_state["last_illustration_url"]   = img

    # Player profile (so onboarding won’t show)
    st.session_state["player_name"]      = snap.get("username") or st.session_state.get("player_name")
    st.session_state["player_gender"]    = snap.get("gender")    or st.session_state.get("player_gender", "Unspecified")
    st.session_state["player_archetype"] = snap.get("archetype") or st.session_state.get("player_archetype", "Default")

    # If you track counters/flags in the DB, load them too (guarded):
    for k in ("decisions_count", "beat_index", "story_complete", "is_dead"):
        if k in snap and snap[k] is not None:
            st.session_state[k] = snap[k]

    return True

# =============================================================================
# Illustration helpers
# =============================================================================

from typing import Dict

@st.cache_resource
def _illustration_store() -> Dict[str, str]:
    """Process-wide cache mapping scene-key → image URL."""
    return {}

@st.cache_resource
def _ill_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=2)

from typing import List
def _scene_keys_for(scene_text: str, simple: bool) -> List[str]:
    txt = (scene_text or "").strip()
    keys = []
    first = _has_first_sentence(txt)
    if first:
        keys.append(_scene_key(first, simple))
    keys.append(_scene_key(txt, simple))  # always include full-scene as fallback
    return keys

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

def render_persistent_illustration(illus_ph, scene_text: str):
    """
    Keep the currently displayed image on-screen while a new job is pending.
    Only swap to the new image when it is ready. If we’ve never shown any image,
    display a small skeleton during the first-ever generation.
    """
    simple_flag = bool(st.session_state.get("simple_cyoa", True))
    url = st.session_state.get("display_illustration_url")  # what is currently shown
    status_text = ""
    pending_key_for_refresh = None

    key, job = _job_for_scene(scene_text or "", simple_flag)

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
            if img_ref:
                st.session_state["display_illustration_url"] = img_ref
                st.session_state["last_illustration_url"] = img_ref
                st.session_state["last_illustration_debug"] = dbg
                
                # NEW: cache by deterministic key so refresh can restore it
                try:
                    store = _illustration_store()
                    store[_scene_key(seed_sentence := ( _has_first_sentence(scene_text) or scene_text ), 
                                     bool(st.session_state.get("simple_cyoa", True)))] = img_ref
                except Exception:
                    pass
                url = img_ref
        else:
            # Job still running → KEEP current image; only show skeleton if we had none
            if not url:
                status_text = "Illustration brewing…"
            pending_key_for_refresh = key

    # Draw: keep current image if any; skeleton only when we have none
    if url:
        illus_ph.markdown(
            f'<div class="illus-inline"><img src="{url}" alt="illustration"/></div>',
            unsafe_allow_html=True,
        )
    else:
        illus_ph.markdown(
            f'<div class="illus-inline illus-skeleton"><div class="illus-status">{status_text or "Illustration brewing…"}</div></div>',
            unsafe_allow_html=True,
        )

    # Schedule rerun *after* choices are already rendered elsewhere
    _gentle_autorefresh_if_pending(pending_key_for_refresh)

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
    st.session_state.setdefault("onboard_dismissed", False)
    st.session_state.setdefault("scene_count", 0)
    st.session_state.setdefault("simple_cyoa", True)   # no sidebar control
    st.session_state.setdefault("auto_illustrate", True)
    # --- persistent illustration buffer ---
    st.session_state.setdefault("display_illustration_url", None)

def reset_session(full_reset=False):
    keep={}
    if not full_reset:
        keep["player_name"]=st.session_state.get("player_name")
        keep["player_gender"]=st.session_state.get("player_gender","Unspecified")
        keep["player_archetype"]=st.session_state.get("player_archetype","Default")
        keep["story_mode"]=bool(st.session_state.get("story_mode",False))
        keep["onboard_dismissed"]=True
        # keep the last shown illustration if you want it across hard resets:
        # keep["display_illustration_url"]=st.session_state.get("display_illustration_url")
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
# Advance turn: stream text → (separator) → start job (maybe) → choices
# (illustration itself is rendered in the regular render path to avoid flicker)
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

    # separator under the text
    _render_separator(sep_slot)

    # decide whether this scene requests a NEW illustration; do NOT clear current display
    should_illustrate = ((scene_index - ILLUSTRATION_PHASE) % ILLUSTRATION_EVERY_N == 0)
    if should_illustrate and st.session_state.get("auto_illustrate", True):
        seed = _has_first_sentence(full) or " ".join(full.split()[:18]) + "…"
        if seed.strip():
            _start_illustration_job(seed, bool(st.session_state.get("simple_cyoa", True)))

    # choices: compute/store only; render happens once later in main()
    choices = generate_choices(st.session_state["history"], full, LORE)
    st.session_state["choices"] = choices
    # do NOT render here; main() will render once and set t_choices_visible_at

    # beat bookkeeping / snapshot
    if st.session_state.get("story_mode"):
        st.session_state["_beat_scene_count"]=st.session_state.get("_beat_scene_count",0)+1
        current=get_current_beat(st.session_state); target=BEAT_TARGET_SCENES.get(current,1)
        if st.session_state["_beat_scene_count"]>=target:
            if current=="resolution": st.session_state["story_complete"]=True
            else:
                st.session_state["beat_index"]=min(st.session_state.get("beat_index",0)+1,len(BEATS)-1)
                st.session_state["_beat_scene_count"]=0

    username = st.session_state.get("player_name") or st.session_state.get("player_username")
    gender   = st.session_state.get("player_gender")
    arch     = st.session_state.get("player_archetype")

    last_img = st.session_state.get("last_illustration_url")

    # Try kwargs first (newer store signatures), then positional (older).
    try:
        save_snapshot(
            pid,
            scene=full,
            choices=choices,
            history=st.session_state.get("history", []),
            username=st.session_state.get("player_name"),
            gender=st.session_state.get("player_gender"),
            archetype=st.session_state.get("player_archetype"),
            last_illustration_url=last_img,              # <-- NEW
        )
    except TypeError:
        # older store.py without the new param – fall back silently
        save_snapshot(pid, full, choices, st.session_state["history"],
                    username=st.session_state.get("player_name"))


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
    # Try restoring if memory is empty
    _maybe_restore_from_snapshot(pid)

    # As soon as a scene exists, permanently hide onboarding
    if (st.session_state.get("scene") or st.session_state.get("history")):
        st.session_state["onboard_dismissed"] = True

    # --- Book-like type; narrower main, added RIGHT margin via container max-width ---
    st.markdown("""
    <style>
        [data-testid="stAppViewContainer"] > .main .block-container{
          max-width: 1040px !important;
          padding-left: .35rem !important;
          padding-right: 10rem !important;
          margin-left: auto; margin-right: auto;
        }
        :root{
          --story-max: 680px;
          --choices-h: 120px;
        }
        [data-testid="stVerticalBlock"] > div:has(> div.story-window){
          width: 100% !important;
          max-width: var(--story-max) !important;
          margin-left: 0 !important;
          margin-right: auto !important;
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

        .illus-inline{
          width: auto;
          max-width: calc(100% - 10rem);
          margin: .6rem 1rem 1.4rem 1rem;
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

        .onboard-panel{
          max-width: 640px;
          margin: 0.5rem auto 1.25rem auto;
          padding: 1rem 1.25rem;
          background: var(--secondary-background-color);
          border: 1px solid rgba(49,51,63,.18);
          border-radius: 12px;
          box-shadow: 0 1px 2px rgba(0,0,0,.05);
        }
        .onboard-panel h1, .onboard-panel h2, .onboard-panel h3{ margin-top: .2rem; margin-bottom: .6rem; }
        .onboard-panel [data-testid="column"]{ gap: .5rem; }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar (no illustration controls)
    try:
        action = sidebar_controls(pid)
    except Exception:
        action = None

    with st.sidebar:
        if DEV_UI_ALLOWED:
            st.session_state["_dev"] = st.checkbox(
                "Developer tools",
                value=bool(st.session_state.get("_dev", False)),
                help="Show debug info & self-tests",
                key="dev_toggle",
            )
        else:
            st.session_state["_dev"] = False
        if st.session_state.get("_dev"):

            now = time.time()

            try:
                _has = bool(has_snapshot(pid))
                _snap = load_snapshot(pid) if _has else None
            except Exception:
                _has = False
                _snap = None

            scene_text = (st.session_state.get("scene") or "").strip()
            choices = st.session_state.get("choices", [])
            history = st.session_state.get("history", [])

            t_scene   = st.session_state.get("t_scene_start")
            t_choices = st.session_state.get("t_choices_visible_at")
            age_scene   = f"{now - t_scene:.1f}s" if t_scene else "–"
            age_choices = f"{now - t_choices:.1f}s" if t_choices else "–"

            # Illustration job for *current* scene
            simple_flag = bool(st.session_state.get("simple_cyoa", True))
            cur_key, cur_job = _job_for_scene(scene_text, simple_flag)
            cur_done = cur_job["future"].done() if cur_job else None
            cur_has_result = (cur_job and (cur_job.get("result") is not None))

            st.caption(
                f"PID: `{pid}` • has_snapshot: {int(_has)} • "
                f"scene_len: {len(scene_text)} • history: {len(history)} • choices: {len(choices)}"
            )
            st.caption(
                f"pending_choice: {st.session_state.get('pending_choice')} • "
                f"is_generating: {int(bool(st.session_state.get('is_generating')))} • "
                f"onboard_dismissed: {int(bool(st.session_state.get('onboard_dismissed')))} • "
                f"scene_count: {st.session_state.get('scene_count', 0)}"
            )
            st.caption(
                f"beat_index: {st.session_state.get('beat_index', 0)} • story_mode: {int(bool(st.session_state.get('story_mode', True)))} • "
                f"t_scene_age: {age_scene} • t_choices_age: {age_choices}"
            )

            # --- Illustration diagnostics ---
            with st.expander("Illustration state", expanded=False):
                st.write({
                    "display_illustration_url?": bool(st.session_state.get("display_illustration_url")),
                    "ill_last_key": st.session_state.get("ill_last_key"),
                    "cur_job_key": cur_key,
                    "cur_job_exists": bool(cur_job),
                    "cur_job_done": bool(cur_done) if cur_job else None,
                    "cur_job_has_result": bool(cur_has_result) if cur_job else None,
                    "ill_jobs_count": len(st.session_state.get("ill_jobs", {})),
                    "ill_polls": st.session_state.get("ill_polls", {}),
                })

            # --- Snapshot vs memory ---
            with st.expander("Snapshot vs memory", expanded=False):
                snap_scene_len = len((_snap.get("scene") or "")) if _snap else 0
                snap_choices_len = len((_snap.get("choices") or [])) if _snap else 0
                st.write({
                    "snapshot_exists": _has,
                    "snap_scene_len": snap_scene_len,
                    "mem_scene_len": len(scene_text),
                    "snap_choices_len": snap_choices_len,
                    "mem_choices_len": len(choices),
                })

                c1, c2, c3 = st.columns(3)
                if c1.button("Save snapshot now", key="dev_save_snapshot"):
                    try:
                        # Prefer keyword args for newer store APIs
                        save_snapshot(
                            pid,
                            scene=scene_text,
                            choices=choices,
                            history=history,
                            username=st.session_state.get("player_name"),
                            gender=st.session_state.get("player_gender"),
                            archetype=st.session_state.get("player_archetype"),
                        )
                        st.success("Snapshot saved.")
                    except TypeError:
                        # Fallback to older signature
                        save_snapshot(
                            pid,
                            scene_text,
                            choices,
                            history,
                            st.session_state.get("player_name"),
                            st.session_state.get("player_gender"),
                            st.session_state.get("player_archetype"),
                        )
                        st.success("Snapshot saved (legacy signature).")
                    except Exception as e:
                        st.error(f"Save failed: {e}")

                if c2.button("Reload snapshot → memory", key="dev_reload_snapshot"):
                    restored = _maybe_restore_from_snapshot(pid)
                    st.info(f"Restore attempted: {restored}")

                if c3.button("Delete snapshot", key="dev_delete_snapshot"):
                    try:
                        delete_snapshot(pid)
                        st.warning("Snapshot deleted.")
                    except Exception as e:
                        st.error(f"Delete failed: {e}")

            # --- Session dump (compact) ---
            with st.expander("Session dump (compact)", expanded=False):
                dump = {k: v for k, v in st.session_state.items()
                        if k not in ("history", "choices", "scene")}
                dump["scene_preview"]   = scene_text[:200] + ("…" if len(scene_text) > 200 else "")
                dump["history_len"]     = len(history)
                dump["choices_len"]     = len(choices)
                st.json(dump)

            # --- Utilities ---
            c1, c2, c3 = st.columns(3)
            if c1.button("Soft rerun", key="dev_soft_rerun"):
                _soft_rerun()
            if c2.button("Clear ill_jobs", key="dev_clear_ill_jobs"):
                st.session_state["ill_jobs"] = {}
                st.session_state["ill_last_key"] = None
                st.info("Cleared illustration jobs.")
            if c3.button("Clear memory (keep profile)", key="dev_clear_memory"):
                for k in ("scene","choices","history","pending_choice","is_generating",
                        "t_choices_visible_at","t_scene_start","is_dead"):
                    st.session_state.pop(k, None)
                st.info("Cleared in-memory story state.")
                
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

    # Placeholders in visual order: story → sep → illustration → choices
    story_ph  = st.empty()
    sep_ph    = st.empty()
    illus_ph  = st.empty()     # restored
    choices_ph= st.empty()

    # queued turn?
    if st.session_state.get("pending_choice") is not None:
        _advance_turn(pid, story_ph, sep_ph, illus_ph, choices_ph, anim_enabled=True)
        st.session_state["pending_choice"] = None
        st.session_state["is_generating"] = False
        # IMPORTANT: do NOT rerun or return here.
        # Fall through to the normal render below so the illustration
        # stays visible and only swaps when ready.

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

    # Render choices NOW so they remain visible during any gentle reruns
    choices_ph.markdown('<div class="choices-band"></div>', unsafe_allow_html=True)
    render_choices_grid(choices_ph,
                        choices=st.session_state.get("choices", []),
                        generating=False,
                        count=CHOICE_COUNT)
    st.session_state["t_choices_visible_at"] = time.time()

    if not st.session_state.get("display_illustration_url"):
        try:
            store = _illustration_store()
            simple_flag = bool(st.session_state.get("simple_cyoa", True))
            for k in _scene_keys_for(st.session_state.get("scene",""), simple_flag):
                cached_url = store.get(k)
                if cached_url:
                    st.session_state["display_illustration_url"] = cached_url
                    st.session_state["last_illustration_url"]   = cached_url
                    break
        except Exception:
            pass

    # Then render illustration into its placeholder; helper will schedule a rerun if needed.
    render_persistent_illustration(illus_ph, st.session_state.get("scene",""))

if __name__ == "__main__":
    main()
