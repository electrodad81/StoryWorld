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
        "You are the narrative engine for a dark-fantasy, middle-school-readable interactive storyworld "
        "called Gloamreach. Write vivid but PG-13 prose, present tense, 120–180 words per beat. "
        "Avoid headings and meta-talk. Never mention 'lore' explicitly.\n\n"
        f"{PERSPECTIVE_RULES}"
    )
    lore_blob = _lore_text(lore)
    user = (
        "Continue the story by one beat. Consider the world details below.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        "Player history (latest last):\n"
        f"{_format_history(history)}\n\n"
        "Now continue with a single scene paragraph. No choices yet."
    )

    cli = _client()
    # Streaming via Chat Completions for compatibility
    stream = cli.chat.completions.create(
        model=SCENE_MODEL,
        messages=[{"role":"system","content": sys},
                  {"role":"user","content": user}],
        temperature=0.9,
        top_p=0.95,
        stream=True,
    )
    try:
        for chunk in stream:
            if isinstance(chunk, ChatCompletionChunk):
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
    finally:
        # make sure stream closed in case of early exit
        try:
            stream.close()
        except Exception:
            pass

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

def generate_choices(history: List[Dict[str,str]], last_scene: str, lore: Dict) -> List[str]:
    """
    Fast non-stream call that returns exactly TWO short choice labels (<= 48 chars each).
    Uses retry/backoff for robustness, leaving streaming path unchanged.
    """
    sys = (
        "You generate two crisp, enticing next-action choices for an interactive story.\n"
        f"{PERSPECTIVE_RULES}\n"
        "Each choice must read as what the PLAYER does (second person), ideally starting with a verb or 'You ...'. "
        "No third-person names or pronouns for the player. "
        "Return ONLY a JSON array of two strings. Each <= 48 characters."
    )
    lore_blob = _lore_text(lore)
    user = (
        "Given the latest scene, propose two distinct next actions the player can take. "
        "World context below may help.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        f"Latest scene:\n{last_scene}\n\n"
        "Return ONLY JSON like [\"<choice 1>\", \"<choice 2>\"]"
    )
    cli = _client()

    def _call():
        resp = cli.chat.completions.create(
            model=CHOICE_MODEL,
            messages=[{"role":"system","content": sys},
                      {"role":"user","content": user}],
            temperature=0.7,
            max_tokens=100,
        )
        txt = (resp.choices[0].message.content or "").strip()
        out = _coerce_two_choices(txt)
        if len(out) < 2:
            # Minimal safe fallback if model under-delivers
            out = ["Investigate further", "Retreat to safety"]
        # tighten length and ensure exactly two
        out = [c[:48] for c in out][:2]
        if len(out) < 2:
            out = (out + ["Continue deeper into the current thread."])[:2]
        return out

    return retry_call(_call, tries=3, base=0.6)
