from typing import List, Dict, Generator, Optional, Tuple, Set
import json
import os
import re
import random
import time

import streamlit as st
from openai import OpenAI

# Expect OPENAI_API_KEY in environment
client = OpenAI()

def render_exploration(pid: str) -> None:
    """Placeholder exploration renderer."""
    st.write("Exploration mode is under construction.")

# ----------------------------------------------------------------------------- 
# Illustration helper
# -----------------------------------------------------------------------------

# ---------------- Image prompt safety helpers ----------------

# Words/phrases we want to down-tone for image gen (not for story text)
_REPLACEMENTS = [
    # gore/explicit -> softened
    (r"\b(blood|bloody|bloodied|gore|gory|guts|entrails|viscera|corpse|cadaver|mutilated|severed|dismembered|decapitat\w*)\b", "damage"),
    (r"\b(wound|wounded|bleed\w*|stab\w*|slash\w*|maim\w*|butcher\w*|impale\w*)\b", "injury"),
    (r"\b(skull|brain|decay|rotting|putrid)\b", "remains"),
    (r"\b(nude|nudity|breast\w*|genital\w*|erotic|sexual|sex|porn\w*)\b", "tasteful attire"),
    (r"\b(suicide|self[-\s]?harm|hang(ed|ing)|cutting)\b", "danger"),
    # firearms/explosives -> medieval analogs
    (r"\b(shotgun|rifle|pistol|handgun|gun|revolver|smg|submachine|uzi|ak-?47|ar-?15|carbine|bullet|ammo|magazine|shell|cartridge)\b", "sword"),
    (r"\b(musket|flintlock|arquebus|blunderbuss)\b", "crossbow"),
    (r"\b(grenade|bomb|dynamite|explosive\w*)\b", "alchemy vial"),
    # modern clothing -> medieval
    (r"\b(trench\s?coat|trenchcoat|peacoat|greatcoat|overcoat|duster)\b", "hooded cloak"),
    (r"\b(suit|tuxedo|necktie|tie|blazer)\b", "tunic"),
    (r"\b(fedora|trilby|bowler|top\s?hat|newsboy cap)\b", "hood"),
    # modern tech/props -> medieval
    (r"\b(camera|photograph|selfie|phone|smartphone|cell\s?phone|radio|microphone|speaker|laptop|computer|tablet)\b", "lantern"),
    (r"\b(flashlight|electric\s?torch)\b", "torch"),
    # transport/infrastructure -> medieval
    (r"\b(car|truck|van|bus|train|subway|tram|airplane|jet|helicopter)\b", "horse cart"),
    (r"\b(skyscraper|elevator|escalator|billboard|neon|streetlight|traffic light|power\s?line|telephone pole|asphalt)\b", "stone tower"),
    # modern textiles
    (r"\b(jeans|t-?shirt|hoodie|sneakers)\b", "woolen garments"),
]

_HARD_FILTER = [
    r"\bsever(ed|ing)\b", r"\bdismember(ed|ment)\b", r"\bdecapitat\w*\b",
    r"\bguts\b", r"\bentrails\b", r"\bviscera\b", r"\bgraphic\b",
    r"\bsex|sexual|erotic\b", r"\bnude|nudity\b",
    # modern elements (drop the entire sentence at level ≥2)
    r"\b(shotgun|rifle|pistol|handgun|gun|revolver|musket|flintlock|arquebus|blunderbuss|bullet|ammo|magazine|cartridge)\b",
    r"\b(trench\s?coat|trenchcoat|peacoat|greatcoat|overcoat|duster|suit|tuxedo|necktie|blazer|fedora|trilby|bowler|top\s?hat)\b",
    r"\b(camera|photograph|phone|smartphone|cell\s?phone|radio|microphone|speaker|laptop|computer|tablet|flashlight)\b",
    r"\b(car|truck|van|bus|train|subway|tram|airplane|jet|helicopter|skyscraper|neon|streetlight|traffic light|power\s?line)\b",
]
# Keep prompt short & visual
_MAX_TOKENS_APPROX = 120  # rough char budget; DALL·E likes concise prompts

# put this just above or inside generate_illustration (module-scope is fine)
def _gender_visual_directive(gender: Optional[str]) -> str:
    g = (gender or "").strip().lower()
    if g == "male":
        return ("If the protagonist appears, depict a masculine-presenting silhouette or attire. "
                "No facial details. Avoid stereotypes; keep clothing practical.")
    if g == "female":
        return ("If the protagonist appears, depict a feminine-presenting silhouette or attire. "
                "No facial details. Avoid sexualization; keep clothing practical.")
    if g == "nonbinary":
        return ("If the protagonist appears, depict an androgynous silhouette with neutral attire. "
                "No facial details. Avoid gendered cues or stereotypes.")
    # unspecified/unknown
    return ("If the protagonist appears, keep the silhouette gender-ambiguous with neutral attire. "
            "No facial details.")

def generate_illustration(scene: str, simple: bool = True) -> Tuple[Optional[str], Dict]:
    """
    Generate an illustration. Returns (image_ref, debug).
    image_ref is a data: URL (base64), a remote URL, or None.

    - Uses ONLY basic style.
    - Progressive prompt sanitization (levels 1→3) to satisfy safety.
    - Final fallback: environment-only prompt.
    - Respects a global ILLUSTRATIONS_ENABLED kill-switch if defined elsewhere.
    """
    import re, time  # local import so you can paste this function as-is
    dbg: Dict = {"summary": None, "attempts": [], "simple": True}

    # ---- Optional global kill switch (define ILLUSTRATIONS_ENABLED elsewhere) ----
    enabled = globals().get("ILLUSTRATIONS_ENABLED", True)
    if not enabled:
        summary = (scene or "").split(".")[0].strip()
        dbg.update({
            "summary": summary,
            "disabled": True,
            "note": "Image generation disabled via ILLUSTRATIONS_ENABLED.",
        })
        return None, dbg

    # ---- Seed summary (keep your current behavior) ----
    summary = (scene or "").split(".")[0].strip()
    dbg["summary"] = summary
    try:
        # read from session; if you prefer, accept `gender` as a function arg instead
        gender_choice = st.session_state.get("player_gender", "Unspecified")
    except Exception:
        gender_choice = "Unspecified"

    gdir = _gender_visual_directive(gender_choice)
    if not summary:
        dbg["error"] = "empty_summary"
        return None, dbg

    # ---- Basic style only ----
    def _basic_style() -> str:
        return (
            "Simple, clean black-and-white line-and-wash illustration with a single spot color accent "
            "(gold leaf or lapis). Light crosshatching, clear contours, single focal subject, "
            "medium or close-up shot. Plain white background behind the subject. "
            "Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). "
            "Architecture: stone keeps, timber framing, castles, market stalls; natural materials "
            "like wood, leather, linen, iron. Props/weapons allowed: swords, daggers, axes, spears, "
            "shields, bows, torches, lanterns, books, scrolls, potions. "
            "Strictly no firearms or explosives, no modern clothing (no trenchcoats, suits, ties), "
            "no modern tech/vehicles (no cameras, phones, radios, cars, trains, planes, neon, streetlights, power lines). "
            "Do not depict the protagonist’s face directly (use silhouette, hood, or a cropped angle)."
        )

    _REPLACEMENTS = [
        # gore/explicit -> softened
        (r"\b(blood|bloody|bloodied|gore|gory|guts|entrails|viscera|corpse|cadaver|mutilated|severed|dismembered|decapitat\w*)\b", "damage"),
        (r"\b(wound|wounded|bleed\w*|stab\w*|slash\w*|maim\w*|butcher\w*|impale\w*)\b", "injury"),
        (r"\b(skull|brain|decay|rotting|putrid)\b", "remains"),
        (r"\b(nude|nudity|breast\w*|genital\w*|erotic|sexual|sex|porn\w*)\b", "tasteful attire"),
        (r"\b(suicide|self[-\s]?harm|hang(ed|ing)|cutting)\b", "danger"),
        # firearms/explosives -> medieval analogs
        (r"\b(shotgun|rifle|pistol|handgun|gun|revolver|smg|submachine|uzi|ak-?47|ar-?15|carbine|bullet|ammo|magazine|shell|cartridge)\b", "sword"),
        (r"\b(musket|flintlock|arquebus|blunderbuss)\b", "crossbow"),
        (r"\b(grenade|bomb|dynamite|explosive\w*)\b", "alchemy vial"),
        # modern clothing -> medieval
        (r"\b(trench\s?coat|trenchcoat|peacoat|greatcoat|overcoat|duster)\b", "hooded cloak"),
        (r"\b(suit|tuxedo|necktie|tie|blazer)\b", "tunic"),
        (r"\b(fedora|trilby|bowler|top\s?hat|newsboy cap)\b", "hood"),
        # modern tech/props -> medieval
        (r"\b(camera|photograph|selfie|phone|smartphone|cell\s?phone|radio|microphone|speaker|laptop|computer|tablet)\b", "lantern"),
        (r"\b(flashlight|electric\s?torch)\b", "torch"),
        # transport/infrastructure -> medieval
        (r"\b(car|truck|van|bus|train|subway|tram|airplane|jet|helicopter)\b", "horse cart"),
        (r"\b(skyscraper|elevator|escalator|billboard|neon|streetlight|traffic light|power\s?line|telephone pole|asphalt)\b", "stone tower"),
        # modern textiles
        (r"\b(jeans|t-?shirt|hoodie|sneakers)\b", "woolen garments"),
    ]


    def _strip_quotes(s: str) -> str:
        return s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")

    def _truncate(s: str, n: int = 400) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

    def _sanitize(text: str, level: int) -> str:
        """
        level 1: replace problematic terms
        level 2: replace + drop sentences with hard filters
        level 3: abstract → environment/setting only
        """
        src = _strip_quotes(text or "")
        if level >= 1:
            for pat, repl in _REPLACEMENTS:
                src = re.sub(pat, repl, src, flags=re.IGNORECASE)

        if level >= 2:
            sents = re.split(r'(?<=[.!?])\s+', src)
            keep = []
            for s in sents:
                low = s.lower()
                if any(re.search(p, low) for p in _HARD_FILTER):
                    continue
                keep.append(s)
            src = " ".join(keep) if keep else "mysterious scene, danger implied but not shown"

        if level >= 3:
            # Environment-only, no distress; keep it short and visual.
            src = ("atmospheric medieval-fantasy environment; focus on landscape, props, and architecture; "
       "no modern elements, no depiction of injuries")

        safety = "family-friendly, PG-13, no gore, no explicit injuries, no nudity"
        return _truncate(f"{src}. {safety}.")

    def _build_prompt(seed: str, level: int) -> str:
        style = _basic_style()
        safe = _sanitize(seed, level)
        # Keep it visual; avoid literal reenactment of violence
        return f"{style} Depict the scene mood and setting with tasteful restraint. {safe} {gdir}"

    size = "1024x1024"
    models = ("dall-e-3",)  # avoid gpt-image-1 (403 in your org)

    # Try levels 1→3 (increasingly safe), only with dall-e-3.
    for level in (1, 2, 3):
        prompt = _build_prompt(summary, level)
        for model_name in models:
            attempt = {"model": model_name, "size": size, "level": level, "prompt": prompt}
            try:
                resp = client.images.generate(
                    model=model_name,
                    prompt=prompt,
                    size=size,
                    # quality="standard",
                )
                data = getattr(resp, "data", None)
                attempt["has_data"] = bool(data)
                if data:
                    item = data[0]
                    b64 = getattr(item, "b64_json", None)
                    if b64:
                        dbg["attempts"].append({**attempt, "has_b64": True, "ok": True})
                        return "data:image/png;base64," + b64, dbg
                    url = getattr(item, "url", None)
                    if url:
                        dbg["attempts"].append({**attempt, "has_url": True, "ok": True})
                        return url, dbg
                # no data, but call succeeded
                dbg["attempts"].append({**attempt, "ok": True, "note": "no data in response"})
            except Exception as e:
                msg = repr(e)
                attempt["ok"] = False
                attempt["error"] = msg
                dbg["attempts"].append(attempt)
                # If it's a policy rejection, escalate to the next level
                if ("content_policy_violation" in msg
                        or "Your request was rejected" in msg
                        or "safety system" in msg):
                    continue
                # If it's a permission / org verification issue, don't spin further
                if ("PermissionDeniedError" in msg
                        or "must be verified" in msg
                        or "Error code: 403" in msg):
                    dbg["permission_error"] = True
                    return None, dbg
                # Other errors → try next level/model anyway
        time.sleep(0.25)  # tiny backoff between levels

    # Final fallback: very tame environment prompt
    fallback_prompt = (
        f"{_basic_style()} "
        "Atmospheric medieval-fantasy environment; focus on scenery and architecture; "
        "no modern elements, no characters in distress, no firearms or explosives; family-friendly, PG-13. "
        f"{gdir}"
    )
    for model_name in models:
        attempt = {"model": model_name, "size": size, "level": "fallback_env", "prompt": fallback_prompt}
        try:
            resp = client.images.generate(
                model=model_name,
                prompt=fallback_prompt,
                size=size,
            )
            data = getattr(resp, "data", None)
            attempt["has_data"] = bool(data)
            if data:
                item = data[0]
                b64 = getattr(item, "b64_json", None)
                if b64:
                    dbg["attempts"].append({**attempt, "has_b64": True, "ok": True})
                    return "data:image/png;base64," + b64, dbg
                url = getattr(item, "url", None)
                if url:
                    dbg["attempts"].append({**attempt, "has_url": True, "ok": True})
                    return url, dbg
            dbg["attempts"].append({**attempt, "ok": True, "note": "no data in response"})
        except Exception as e:
            attempt["ok"] = False
            attempt["error"] = repr(e)
            dbg["attempts"].append(attempt)

    dbg["error"] = "all_attempts_failed"
    return None, dbg
# -----------------------------------------------------------------------------
# Story generation helpers


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

# Beat-specific generation guidelines (closure rules baked in)
BEAT_GUIDELINES = {
    "exposition": (
        "Open in motion; establish POV, desire, and a concrete obstacle. "
        "Plant an opening image to echo later."
    ),
    "rising_action": (
        "Escalate with 1–2 concrete complications that corner the protagonist. "
        "Narrow options; raise stakes. Introduce at most one minor element."
    ),
    "climax": (
        "Force a single, irreversible decision. Make the price/cost explicit. "
        "No new locations or characters; cash the promise of the hook."
    ),
    "falling_action": (
        "Show immediate, visible consequences in the world. Resolve dangling promises. "
        "No new mysteries or hooks; compress time if needed."
    ),
    "resolution": (
        "Answer the central question (yes / no / yes-but / no-and). "
        "Echo the opening image in a changed way. State the internal change in one clear line. "
        "End on a specific sensory image. No next-scene hooks; no choices."
    ),
}

def _opening_image_hint(history) -> str:
    # very light heuristic: first assistant scene's first sentence
    try:
        first = next(m["content"] for m in history if m.get("role") == "assistant")
        import re
        sent = re.split(r'(?<=[.!?])\s+', first.strip())[0]
        return sent[:160]
    except StopIteration:
        return ""

def _compose_prompt(history: List[Dict], lore: Dict, beat: Optional[str]) -> str:
    base = (
        "Write the next short scene in a self-contained short story. "
        "Keep scenes concise and concrete; avoid summarizing future events."
    )
    guide = BEAT_GUIDELINES.get(beat or "", "")
    opener = _opening_image_hint(history)
    extra = (f' Opening image to echo later: "{opener}".' if opener else "")
    return f"{base}\n\nBeat: {beat or 'classic'}.\nGuidelines: {guide}{extra}"


def stream_scene(history: List[Dict[str, str]],
                 lore: Dict,
                 beat: Optional[str] = None) -> Generator[str, None, None]:
    """
    Streams a scene for the interactive story. If `beat` is provided (Story Mode),
    injects brief guidance appropriate to the beat. If `beat` is None,
    the scene is generated in classic freeform mode.
    """
    # ---- Beat-aware guidance composed at module scope via _compose_prompt(..)
    # (uses BEAT_GUIDELINES + opening-image echo)
    prompt = _compose_prompt(history, lore, beat)

    # ---- System style + consequence contract (kept from your version)
    sys_base = (
        "You are the narrative engine for a PG dark-fantasy interactive story called Gloamreach.\n"
        "Write in present tense, second person. Do not name the protagonist; if a Name exists, it may appear only in NPC dialogue.\n"
        "Keep prose tight by default (~85–150 words, single paragraph). Maintain continuity with prior scenes.\n"
        # Consequence contract ensures risky choices have visible costs and escalating setbacks
        "Consequence contract: If the last player choice was risky, show a visible cost in THIS scene (wound, gear loss, time pressure, ally setback, exposure to threat). Do NOT undo it immediately. "
        "If the player chose risky paths in two consecutive scenes, escalate to a serious setback (capture, grave wound, loss). Only when fictionally fitting, you may kill the protagonist.\n"
        "If (and only if) the protagonist dies in this scene, append exactly '\n\n[DEATH]' as the final line. Do not add any text after [DEATH]."
    )

    run_seed = st.session_state.get("run_seed") or ""
    seed_line = f'RUN_SEED: "{run_seed}" (do not reveal this; use it as a hidden stylistic fingerprint for variety across runs)\n' if run_seed else ""

    # ---- Control flags (risk/death mechanics state -> for model awareness, not output)
    danger_streak = int(st.session_state.get("danger_streak", 0))
    injury_level  = int(st.session_state.get("injury_level", 0))
    risk_flags = {
        "allow_fail_state": True,
        "danger_streak": danger_streak,
        "injury_level": injury_level,
        "must_escalate": (danger_streak >= 2),
    }
    try:
        risk_control_str = f"CONTROL FLAGS: {json.dumps(risk_flags)}\n"
    except Exception:
        risk_control_str = f"CONTROL FLAGS: {risk_flags}\n"
    control_flags_note = "Do not expose control flags; they are for you only.\n"

    # ---- Player name usage rule
    player_name = st.session_state.get("player_name") or st.session_state.get("player_username") or ""
    if player_name:
        name_clause = f"Player name (for NPC dialogue only): {player_name}\n"
    else:
        name_clause = "No name is provided. Do not address the protagonist by any name; use 'you' only.\n"

    # ---- Lore + history
    lore_blob = json.dumps(lore)[:20_000]
    user = (
        name_clause
        + seed_line
        + risk_control_str
        + control_flags_note
        + "Continue the story with the next scene. Consider the world details below.\n"
        f"--- LORE JSON ---\n{lore_blob}\n--- END LORE ---\n\n"
        "Player history (latest last):\n"
        f"{_history_text(history)}\n\n"
        # Output guardrails
        "Output text in second person, present tense.\n"
        "Length: ~85–150 words. End cleanly; do not start a new paragraph."
    )

    # ---- Streaming call (now includes beat-aware prompt in the SYSTEM content)
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.9,
        max_tokens=350,
        stream=True,
        messages=[
            {"role": "system", "content": sys_base + "\n\n" + prompt},  # <-- incorporate _compose_prompt(..)
            {"role": "user", "content": user},
        ],
    )

    for ev in resp:
        if getattr(ev, "choices", None):
            delta = getattr(ev.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                yield delta.content

# -----------------------------------------------------------------------------
# SQLite storage helpers

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
    
def _basic_style_clause(simple: bool) -> str:
    if simple:
        return ("black-and-white line art, clean inked outlines, minimal shading, "
                "storybook illustration, high contrast, centered composition")
    return ("painterly fantasy illustration, soft light, cinematic composition, "
            "rich texture, professional artbook style")

def _strip_quotes(s: str) -> str:
    return s.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")

def _truncate_chars(s: str, n: int) -> str:
    s = re.sub(r"\s+", " ", s).strip()
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "…"

def _soft_sanitize(text: str, level: int) -> str:
    """
    level 1: replace problematic terms
    level 2: replace + drop sentences with hard filters
    level 3: heavily abstract → setting/items only
    """
    src = _strip_quotes(text or "")
    if level >= 1:
        for pat, repl in _REPLACEMENTS.items():
            src = re.sub(pat, repl, src, flags=re.IGNORECASE)

    if level >= 2:
        # remove sentences that still contain hard patterns
        sents = re.split(r'(?<=[.!?])\s+', src)
        safe_sents = []
        for s in sents:
            low = s.lower()
            if any(re.search(p, low) for p in _HARD_FILTER):
                continue
            safe_sents.append(s)
        src = " ".join(safe_sents) if safe_sents else "mysterious scene"

    if level >= 3:
        # extract a gentle environment + subject sketch
        # keep only nouns-ish words; drop people-injury specifics
        words = re.findall(r"[a-zA-Z][a-zA-Z\-']+", src)
        keep = []
        banned = {"injury","damage","remains","weapon","danger","kill","dead","death","corpse"}
        for w in words:
            lw = w.lower()
            if lw in banned: 
                continue
            if len(lw) <= 2:
                continue
            keep.append(w)
        # fallback scene
        if not keep:
            src = "misty forest clearing with ancient stones and soft moonlight"
        else:
            src = " ".join(keep[:40])

    # Always add a friendly safety clause and style guidance
    safety = ("family-friendly, PG-13, no gore, no explicit injuries, no nudity, "
              "suggestive elements are omitted")
    src = _truncate_chars(src, 400)
    return f"{src}. {safety}."

def _build_image_prompt(scene_text: str, simple: bool, level: int) -> str:
    style = _basic_style_clause(simple)
    safe = _soft_sanitize(scene_text, level)
    return (f"{style}. Depict the scene mood and setting suggested here, with tasteful restraint. "
            f"{safe}")