# story/engine.py
from __future__ import annotations
import os, json, time, random
from typing import Dict, List, Generator, Optional

import streamlit as st
from openai import OpenAI
from openai.types.chat import ChatCompletionChunk

# Models: tweak if you prefer
SCENE_MODEL = os.getenv("SCENE_MODEL", "gpt-4o")
CHOICE_MODEL = os.getenv("CHOICE_MODEL", "gpt-4o-mini")

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
        "present tense, 120–180 words per beat. Avoid headings and meta-talk. "
        "Never mention 'lore' explicitly."
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

def _retry(n=3, base=0.4, jitter=0.4):
    def decorator(fn):
        def wrapper(*a, **kw):
            last = None
            for i in range(n):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    last = e
                    time.sleep(base*(2**i) + random.random()*jitter)
            raise last
        return wrapper
    return decorator

@_retry()
def generate_choices(history: List[Dict[str,str]], last_scene: str, lore: Dict) -> List[str]:
    """
    Fast non-stream call that returns exactly TWO short choice labels (<= 48 chars each).
    """
    sys = (
        "You generate crisp, enticing choice labels for an interactive story. "
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
    resp = cli.chat.completions.create(
        model=CHOICE_MODEL,
        messages=[{"role":"system","content": sys},
                  {"role":"user","content": user}],
        temperature=0.7,
        max_tokens=100,
    )
    txt = resp.choices[0].message.content or "[]"
    try:
        arr = json.loads(txt)
        out = [str(x).strip() for x in arr][:2]
        if len(out) < 2:
            raise ValueError("need two choices")
        # tighten length
        out = [c[:48] for c in out]
        return out
    except Exception:
        # fallback if model didn't parse
        return ["Investigate further", "Retreat to safety"]
