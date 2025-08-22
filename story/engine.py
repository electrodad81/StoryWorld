# story/engine.py
from __future__ import annotations
import os, json, re
from typing import Dict, List, Generator

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk

from utils.retry import retry_call

# Models: tweak if you prefer
SCENE_MODEL = os.getenv("SCENE_MODEL", "gpt-4o")
CHOICE_MODEL = os.getenv("CHOICE_MODEL", "gpt-4o-mini")

# Put this near your other constants
PERSPECTIVE_RULES = (
    "Narrate strictly in SECOND PERSON (address the player as 'you') and PRESENT TENSE.\n"
    "Do NOT give the player a proper name, title, or gender unless explicitly provided in history "
    "under keys like 'player_name' or 'player_identity'.\n"
    "If earlier text mentions a name for the player, treat it as NPC dialogue; keep narration as 'you'.\n"
    "Avoid third-person references to the player (no 'the hero', 'they', 'she/he' for the protagonist)."
)

CONCISION_RULES = (
    "Keep it tight: 85–120 words, one compact paragraph (5–7 short sentences). "
    "Use concrete verbs and specific nouns; minimize adjectives/adverbs and filler. "
    "No headings, no second paragraph, no meta-talk."
)

@st.cache_resource(show_spinner=False)
def _client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set (env or .streamlit/secrets.toml).")
    return OpenAI(api_key=api_key)

def _lore_text(lore_obj: Dict) -> str:
    # keep prompt small-ish; truncate serialized lore if huge
    try:
        txt = json.dumps(lore_obj, ensure_ascii=False)
        return txt[:4000]  # ~4k chars
    except Exception:
        return str(lore_obj)[:4000]

def _format_history(history: List[Dict[str, str]], max_turns: int = 8) -> str:
    if not history:
        return ""
    # keep the last N messages; compact
    tail = history[-max_turns:]
    lines = []
    for m in tail:
        role = m.get("role","")
        content = m.get("content","")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)

def stream_scene(history: List[Dict[str,str]], lore: Dict) -> Generator[str, None, None]:
    """
    Stream a narrative continuation given conversation history + lore.
    Yields text chunks (markdown). Use in Streamlit to update a container live.
    """
    sys = (
        "You are the narrative engine for a dark-fantasy, middle-school-readable "
        "interactive storyworld called Gloamreach. Write vivid but PG-13 prose, "
        "present tense. Never mention 'lore' explicitly.\n"
        "If a player username is provided (non-empty), you may use that exact string in NPC dialogue only. "
        "If no username is provided, do not invent one and do not use placeholders like <name>, {name}, or titles; "
        "address the protagonist only as 'you'. Do NOT name the protagonist in narration; "
        "the POV remains second-person 'you'.\n\n"
        f"{PERSPECTIVE_RULES}\n"
        f"{CONCISION_RULES}\n"
        "Open with the on-screen consequence of the player’s last choice (one concrete, visible change).\n"
        "Small twist cadence: at most ~1 in 4 scenes, and only if twist_eligible=true.\n"
        "Bonding cadence: include a brief affiliative exchange with an NPC if bonding_nudge is \"strong\" "
        "(and natural to the scene). Keep it subtle.\n"
        "Do not expose or mention control flags; they are guide rails for you only.\n"
        "Consequence contract: If the last player choice was risky, show a visible cost in THIS scene "
        "(wound, gear loss, time pressure, ally setback, exposure to threat). Do NOT undo it immediately. "
        "If the player chose risky paths in two consecutive scenes, "
        "escalate to a serious setback (capture, grave wound, loss), and carry it forward. "
        "Only when fictionally fitting, you may kill the protagonist.\n"
        "If (and only if) the protagonist dies in this scene, append exactly '\n\n[DEATH]' as the final line. "
        "Do not add any text after [DEATH].\n"
    )

    # ---------- CONTROL FLAGS (computed, soft heuristics) ----------
    # Beat index = how many assistant scenes already exist
    beat_index = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "assistant")

    # last_choice = most recent 'user' message content (empty on first beat)
    last_choice = ""
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user":
            last_choice = str(m.get("content") or "").strip()
            break

    # Twist cadence: invite at most ~1 in 4 beats, never two invites back-to-back
    last_twist_invite_at = st.session_state.get("last_twist_invite_at", None)
    twist_invited_last = (last_twist_invite_at == (beat_index - 1))
    twist_eligible = (beat_index % 4 == 0) and not twist_invited_last
    if twist_eligible:
        st.session_state["last_twist_invite_at"] = beat_index

    # Bonding cadence: escalate nudge over time since last "strong" nudge
    last_bonding_strong_at = st.session_state.get("last_bonding_strong_at", -10_000)
    beats_since_bonding = beat_index - last_bonding_strong_at
    if beats_since_bonding >= 8:
        bonding_nudge = "strong"
        st.session_state["last_bonding_strong_at"] = beat_index  # mark that we nudged strongly this beat
    elif beats_since_bonding >= 5:
        bonding_nudge = "soft"
    else:
        bonding_nudge = "none"

    control_flags = (
        "--- CONTROL FLAGS ---\n"
        f"twist_eligible: {str(twist_eligible).lower()}\n"   # 'true' | 'false'
        f"bonding_nudge: {bonding_nudge}\n"                 # none | soft | strong
        f"last_choice: {last_choice}\n"                     # empty on first beat
        "--- END CONTROL FLAGS ---\n"
    )
    # ---------------------------------------------------------------

    player_name = st.session_state.get("player_name") or st.session_state.get("player_username") or ""
    if player_name:
        name_clause = f"Player name (for NPC dialogue only): {player_name}\n"
    else:
        # Safety (shouldn’t happen now, but keeps things robust)
        name_clause = "No name is provided. Do not address the protagonist by any name; use 'you' only.\n"

    # Risk & consequence control flags: compute current danger streak and injury level. If
    # two risky choices in a row have been taken, must_escalate will be true.
    danger_streak = int(st.session_state.get("danger_streak", 0))
    injury_level = int(st.session_state.get("injury_level", 0))
    risk_flags = {
        "allow_fail_state": True,
        "danger_streak": danger_streak,
        "injury_level": injury_level,
        "must_escalate": (danger_streak >= 2),
    }
    try:
        risk_control_str = f"CONTROL FLAGS: {json.dumps(risk_flags)}\n"
    except Exception:
        # Fallback: if json.dumps fails (should not), use repr
        risk_control_str = f"CONTROL FLAGS: {risk_flags}\n"

    # Lore text (serialized JSON)
    lore_blob = _lore_text(lore)
    user = (
        name_clause +
        risk_control_str +
        control_flags +
        "Continue the story by one beat. Consider the world details below.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        "Player history (latest last):\n"
        f"{_format_history(history)}\n\n"
        "Output exactly one paragraph in second person, present tense.\n"
        "Length: ~85–120 words. End cleanly; do not start a new paragraph."
    )

    cli = _client()
    stream = cli.chat.completions.create(
        model=SCENE_MODEL,
        messages=[{"role":"system","content": sys},
                  {"role":"user","content": user}],
        temperature=0.8,
        top_p=0.9,
        max_tokens=180,
        stream=True,
    )
    for chunk in stream:
        if isinstance(chunk, ChatCompletionChunk):
            delta = chunk.choices[0].delta.content
            if delta:
                # If no username, remove any literal placeholder the model might still emit
                if not player_name:
                    delta = delta.replace("<name>", "").replace("  ", " ")
                yield delta

def _coerce_two_choices(text: str) -> List[str]:
    """
    Try hard to extract exactly two short strings from model output.
    Prefers JSON array; falls back to line parsing.
    """
    if not text:
        return []
    # Prefer JSON array (possibly inside a code fence)
    fence_match = re.search(r"\[.*\]", text, re.DOTALL)
    candidate = fence_match.group(0) if fence_match else text
    try:
        arr = json.loads(candidate)
        if isinstance(arr, list):
            items = [str(x).strip() for x in arr if str(x).strip()]
            return items[:2]
    except Exception:
        pass
    # Fallback: split lines / bullets
    lines = [re.sub(r"^[-•\d\.)\s]+", "", ln).strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]
    return lines[:2]

def _to_imperative(label: str) -> str:
    """Convert a 'You ...' style label to imperative: 'Search the archway'."""
    s = (label or "").strip()
    # Strip bullets/quotes
    s = re.sub(r'^[\s\-\•"“”]+', '', s)
    # Common second-person lead-ins → drop them
    s = re.sub(
        r'^(?:you(?:’|\'|)ll|you(?:’|\'|)d|you will|you would|you can|you could|you should|you must|'
        r'you decide to|you choose to|you try to|you begin to|you start to|you attempt to|you)\s+',
        '', s, flags=re.IGNORECASE)
    # Leading 'to ' → drop for imperative
    s = re.sub(r'^(?:to\s+)', '', s, flags=re.IGNORECASE)
    # Remove trailing sentence-ending punctuation
    s = re.sub(r'[\.!?…]+$', '', s).strip()
    # Capitalize first letter (keep rest as-is)
    if s:
        s = s[0].upper() + s[1:]
    return s

def _smart_limit(text: str, limit: int = 48) -> str:
    """Trim to <= limit without mid-word chops; add ellipsis only if needed."""
    if not text:
        return ""
    s = re.sub(r'\s+', ' ', text).strip()
    if len(s) <= limit:
        return s
    # Soft cut to last space within limit
    cut = s[:limit + 1]
    if ' ' in cut:
        cut = cut[:cut.rfind(' ')]
    cut = cut.rstrip(' ,;:-')
    return (cut if cut else s[:limit]).rstrip() + '…'

def generate_choices(history, last_scene, LORE) -> List[str]:
    """
    Fast non-stream call that returns exactly TWO short imperative choice labels (<= 48 chars each).
    """
    # ----- CONTROL FLAGS (read-only; same cadence as stream_scene) -----
    # Beat index = how many assistant scenes already exist
    beat_index = sum(1 for m in history if isinstance(m, dict) and m.get("role") == "assistant")

    # last_choice = most recent 'user' message content (empty on first beat)
    last_choice = ""
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user":
            last_choice = str(m.get("content") or "").strip()
            break

    # Twist cadence: invite at most ~1 in 4 beats, never two invites back-to-back
    last_twist_invite_at = st.session_state.get("last_twist_invite_at", None)
    twist_invited_last = (last_twist_invite_at == (beat_index - 1))
    twist_eligible = (beat_index % 4 == 0) and not twist_invited_last

    # Bonding cadence (do not mutate counters here; stream_scene already does)
    last_bonding_strong_at = st.session_state.get("last_bonding_strong_at", -10_000)
    beats_since_bonding = beat_index - last_bonding_strong_at
    if beats_since_bonding >= 8:
        bonding_nudge = "strong"
    elif beats_since_bonding >= 5:
        bonding_nudge = "soft"
    else:
        bonding_nudge = "none"

    control_flags = (
        "--- CONTROL FLAGS ---\n"
        f"twist_eligible: {str(twist_eligible).lower()}\n"  # 'true' | 'false'
        f"bonding_nudge: {bonding_nudge}\n"                # none | soft | strong
        f"last_choice: {last_choice}\n"                    # empty on first beat
        "--- END CONTROL FLAGS ---\n"
    )
    # ------------------------------------------------------------------

    sys = (
        "You generate concise, imperative next-action options for an interactive story.\n"
        "Return ONLY a JSON array of two strings.\n"
        "Constraints: imperative voice (command form), do NOT start with 'You', ≤ 48 characters each, "
        "no trailing periods, options must be meaningfully different by approach and risk.\n"
        "Provide one safer path and one clearly riskier path. Do not label them; let the risk difference be implied by the action itself.\n"
        "If bonding_nudge is \"strong\", let one option be prosocial (dialogue, help, trust) if it fits.\n"
        "Do not expose or mention control flags to the player."
    )

    lore_blob = _lore_text(LORE)
    user = (
        control_flags +
        "Based on the latest scene, propose two distinct next actions the player can take.\n"
        "Write each as a short IMPERATIVE label (e.g., 'Follow the wisp', 'Search the archway'). "
        "Do NOT begin with 'You'.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        f"Latest scene:\n{last_scene}\n\n"
        'Return ONLY JSON like ["Follow the wisp","Search the archway"]'
    )

    cli = _client()

    def _call():
        resp = cli.chat.completions.create(
            model=CHOICE_MODEL,
            messages=[{"role": "system", "content": sys},
                      {"role": "user", "content": user}],
            temperature=0.7,
            max_tokens=80,  # enough for two short labels in JSON
        )
        txt = (resp.choices[0].message.content or "").strip()
        out = _coerce_two_choices(txt)

        # Post-process: normalize → imperative → smart-limit → dedupe
        cleaned: List[str] = []
        seen = set()
        for c in out:
            c0 = re.sub(r'\s+', ' ', str(c or '')).strip()
            if not c0:
                continue
            c1 = _to_imperative(c0)
            c2 = _smart_limit(c1, 48)
            key = c2.lower()
            if c2 and key not in seen:
                seen.add(key)
                cleaned.append(c2)

        # Ensure exactly two, with safe fallbacks
        if len(cleaned) < 2:
            fallbacks = ["Investigate further", "Retreat to safety"]
            for fb in fallbacks:
                if len(cleaned) >= 2:
                    break
                if fb.lower() not in seen:
                    cleaned.append(fb)
                    seen.add(fb.lower())

        # Risk post-process: ensure that at least one choice conveys risk implicitly.
        # If neither choice appears risky (per our keyword check), we do not append a tag
        # but rely on the narrative model to decide risk. This preserves the user's
        # experience without visibly marking one as risky.
        if len(cleaned) == 2:
            def _local_is_risky(label: str) -> bool:
                try:
                    # Use shared risk detection from the app if available
                    from app import is_risky_label as _ir
                    return _ir(label)
                except Exception:
                    # Fallback to a small set of risky verbs
                    _words = {
                        "charge","attack","fight","steal","stab","break","smash","dive",
                        "jump","descend","enter","drink","touch","open the","confront",
                        "cross","swim","sprint","bait","ambush","bleed","sacrifice","shout",
                        "brave","risk","gamble","rush","kick","force","pry","ignite","set fire"
                    }
                    s = (label or "").lower()
                    import re as _re
                    return any(_re.search(rf"\b{_re.escape(w)}\b", s) for w in _words)
            # We intentionally do nothing if both choices are safe; internal counters
            # will treat them as safe but risk escalation is still tracked.

        return cleaned[:2]

    # Use your existing retry helper if available
    try:
        return retry_call(_call, tries=3, base=0.6)
    except NameError:
        return _call()


    return retry_call(_call, tries=3, base=0.6)