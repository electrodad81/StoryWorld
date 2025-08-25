from typing import List, Dict, Generator, Optional
import json
import os

import streamlit as st
from openai import OpenAI

# Expect OPENAI_API_KEY in environment
client = OpenAI()

# ----------------------------------------------------------------------------- 
# Illustration helper
# -----------------------------------------------------------------------------
from typing import Optional, Tuple, Dict

from typing import Optional, Tuple, Dict

def generate_illustration(scene: str, simple: bool = True) -> Tuple[Optional[str], Dict]:
    """
    Generate an illustration. Returns (image_ref, debug).
    image_ref is a data: URL (base64), a remote URL, or None.
    """
    dbg: Dict = {"summary": None, "attempts": [], "simple": simple}

    summary = (scene or "").split(".")[0].strip()
    dbg["summary"] = summary
    if not summary:
        dbg["error"] = "empty_summary"
        return None, dbg

    # Two prompt styles: "simple" for CYOA-like line art, "detailed" as fallback
    if simple:
        style_prompt = (
            "Simple, clean black-and-white line art with a single blue highlight. Minimal hatching, clear contours, "
            "single focal subject, medium or close-up shot. Lighthearted adventure tone. "
            "the background behind the image subject should be plain white, "
            "lightly intricate textures, careful crosshatching, no gore. " 
        )
    else:
        style_prompt = (
            "Highly detailed black-and-white pen-and-ink illustration with careful hatching. "
            "Dark-fantasy mood, PG-13. Keep composition readable and not overly cluttered."
        )

    prompt = f"{style_prompt} Scene: {summary}."

    size = "1024x1024"  # required valid size

    for model_name in ("gpt-image-1", "dall-e-3"):
        attempt = {"model": model_name, "size": size, "simple": simple}
        try:
            resp = client.images.generate(
                model=model_name,
                prompt=prompt,
                size=size,
                # quality="standard",  # optional
            )
            data = getattr(resp, "data", None)
            attempt["has_data"] = bool(data)
            if data:
                item = data[0]
                b64 = getattr(item, "b64_json", None)
                if b64:
                    dbg["attempts"].append({**attempt, "has_b64": True})
                    return "data:image/png;base64," + b64, dbg
                url = getattr(item, "url", None)
                if url:
                    dbg["attempts"].append({**attempt, "has_url": True})
                    return url, dbg
            dbg["attempts"].append({**attempt, "ok": True, "note": "no data in response"})
        except Exception as e:
            attempt["ok"] = False
            attempt["error"] = repr(e)
            dbg["attempts"].append(attempt)

    dbg["error"] = "all_attempts_failed"
    return None, dbg

def _history_text(history: List[Dict[str, str]]) -> str:
    """Flatten chat history into a string for prompting."""
    lines = []
    for m in history:
        role = m.get("role", "")
        content = m.get("content", "")
        if not content:
            continue
        lines.append(f"{role.upper()}: {content}")
    return "\n".join(lines[-40:])  # Keep last 40 entries


def stream_scene(history: List[Dict[str, str]],
                 lore: Dict,
                 beat: Optional[str] = None) -> Generator[str, None, None]:
    """
    Streams a scene for the interactive story. If `beat` is provided (Story Mode),
    injects a brief line of guidance appropriate to the beat. If `beat` is None,
    the scene is generated in classic freeform mode.
    """
    # System prompt: style and constraints with consequence contract
    sys = (
        "You are the narrative engine for a PG dark-fantasy interactive story called Gloamreach.\n"
        "Write in present tense, second person. Do not name the protagonist; if a Name exists, it may appear only in NPC dialogue.\n"
        "Keep prose tight by default (~85–150 words single paragraph). Maintain continuity with prior scenes.\n"
        # Consequence contract ensures risky choices have visible costs and escalating setbacks
        "Consequence contract: If the last player choice was risky, show a visible cost in THIS scene (wound, gear loss, time pressure, ally setback, exposure to threat). Do NOT undo it immediately. "
        "If the player chose risky paths in two consecutive scenes, escalate to a serious setback (capture, grave wound, loss). Only when fictionally fitting, you may kill the protagonist.\n"
        "If (and only if) the protagonist dies in this scene, append exactly '\n\n[DEATH]' as the final line. Do not add any text after [DEATH].\n"
    )

    # Beat guidance (Story Mode only)
    beat_map = {
        "exposition":     "Focus on world-building, character intro, and foreshadowing.",
        "rising_action":  "Escalate conflict and tension; reveal complications and risky options.",
        "climax":         "High-stakes confrontation; decisive outcomes. Allow real peril.",
        "falling_action": "Show the aftermath and consequences of the climax; start tying threads.",
        "resolution":     "Summarize the journey, provide emotional closure, and hint at replay."
    }
    beat_line = ""
    if beat and beat in beat_map:
        beat_line = f"BEAT GUIDANCE: {beat_map[beat]}\n"

    # Control flags placeholder (risk/death mechanics can be implemented separately)
    # Risk & consequence control flags: compute current danger streak and injury level. If two risky
    # choices in a row have been taken, must_escalate will be true. These are exposed to the model
    # but never shown to the player directly.
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

    control_flags = (
        "Do not expose control flags; they are for you only.\n"
    )

    # Name clause: provide the player's name for NPC dialogue only. If no name is present,
    # instruct the model to avoid using a name (use 'you' only).
    player_name = st.session_state.get("player_name") or st.session_state.get("player_username") or ""
    if player_name:
        name_clause = f"Player name (for NPC dialogue only): {player_name}\n"
    else:
        name_clause = "No name is provided. Do not address the protagonist by any name; use 'you' only.\n"

    lore_blob = json.dumps(lore)[:20_000]
    user = (
        name_clause +
        risk_control_str +
        control_flags +
        beat_line +
        "Continue the story by one beat. Consider the world details below.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        "Player history (latest last):\n"
        f"{_history_text(history)}\n\n"
        # Guidance on output format
        "Output text in second person, present tense.\n"
        "Length: ~85–150 words. End cleanly; do not start a new paragraph."
    )

    # Streaming call
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.9,
        max_tokens=350,
        stream=True,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    for ev in resp:
        if ev.choices and ev.choices[0].delta and ev.choices[0].delta.content:
            yield ev.choices[0].delta.content


def generate_choices(history: List[Dict[str, str]],
                     last_scene: str,
                     lore: Dict) -> List[str]:
    """
    Generates two imperative next actions based on the current scene and history.
    """
    sys = (
        "You generate concise, imperative next-action options for an interactive story.\n"
        "Return ONLY a JSON array of two strings.\n"
        "Constraints: imperative voice (command form), do NOT start with 'You', ≤ 48 characters each,\n"
        "no trailing periods, options must be meaningful and distinct from each other.\n"
    )
    lore_blob = json.dumps(lore)[:20_000]
    user = (
        "Create two next-step options that follow naturally from the latest assistant scene.\n"
        "Avoid near-duplicates.\n\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        "Context (recent turns; latest last):\n"
        f"{_history_text(history)}\n"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        max_tokens=200,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
    )
    import re
    import json as _json
    text = resp.choices[0].message.content
    m = re.search(r"\[[\s\S]*\]", text)
    if not m:
        lines = [ln.strip("-* \t") for ln in text.splitlines() if ln.strip()]
        opts = [ln for ln in lines if ln]
        return opts[:2] if opts else ["Press on", "Hold back"]
    try:
        arr = _json.loads(m.group(0))
        cleaned = []
        for s in arr[:2]:
            s = (s or "").strip()
            if s.endswith("."):
                s = s[:-1]
            cleaned.append(s)
        while len(cleaned) < 2:
            cleaned.append("Continue")
        return cleaned[:2]
    except Exception:
        return ["Press on", "Hold back"]