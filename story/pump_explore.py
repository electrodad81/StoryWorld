from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any, Dict, List

from openai import OpenAI

client = OpenAI()


def location_render(map_state: Dict[str, Any], inventory: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Render the current location with available actions and items.

    The model returns a JSON object with keys:
    - scene: string describing the location (60-110 words)
    - choices: list of {"id", "label"}
    - items: list of {"id", "name", "description"}
    - map_hint: short directional hint for the player

    Choice IDs must use the following prefixes:
    - move:<DIR> (north, south, east, west, up, down)
    - pickup:item:<id>
    - inspect:item:<id>
    - talk:npc:<id>
    - use:item:<id>
    """
    sys = dedent(
        """
        You are the narrative engine for Gloamreach's exploration mode.
        Return ONLY a JSON object with keys: scene, choices, items, map_hint.
        Choices must use explicit IDs (move:<DIR>, pickup:item:<id>, inspect:item:<id>, talk:npc:<id>, use:item:<id>).\n"""
    ).strip()
    user = (
        "Current map state: "
        + json.dumps(map_state)
        + "\nPlayer inventory: "
        + json.dumps(inventory)
    )
    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.8,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    text = resp.output_text  # type: ignore[attr-defined]
    return json.loads(text)


def resolve_outcome(choice_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the outcome of a chosen action.

    Returns a JSON object with:
    - result: narrative outcome text (40-90 words)
    - items: {"gained": [item objects], "lost": ["id"]}
    - map_hint: optional directional hint
    """
    sys = dedent(
        """
        You resolve action outcomes for Gloamreach's exploration mode.
        Accept a choice ID and context, and return ONLY a JSON object with keys:
        result, items (gained/lost), and map_hint.
        Choice IDs follow prefixes: move:<DIR>, pickup:item:<id>, inspect:item:<id>, talk:npc:<id>, use:item:<id>.
        """
    ).strip()
    user = (
        f"Choice ID: {choice_id}\n" + f"Context: {json.dumps(context)}"
    )
    resp = client.responses.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0.8,
        messages=[
            {"role": "system", "content": sys},
            {"role": "user", "content": user},
        ],
        response_format={"type": "json_object"},
    )
    text = resp.output_text  # type: ignore[attr-defined]
    return json.loads(text)