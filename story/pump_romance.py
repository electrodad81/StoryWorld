from __future__ import annotations

import json
import os
from textwrap import dedent
from typing import Any, Dict

from openai import OpenAI

client = OpenAI()


def romance_scene(context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a romantic interlude scene.

    Returns JSON with:
    - scene: narrative text (50-100 words)
    - choices: list of {"id", "label"}
    Valid choice IDs include: flirt, confess, gift:item:<id>, hold_back.
    """
    sys = dedent(
        """
        You write PG-13 romantic scenes in the world of Gloamreach.
        Return ONLY a JSON object with keys: scene and choices.
        Each choice must include an explicit id (flirt, confess, gift:item:<id>, hold_back).
        """
    ).strip()
    user = "Context: " + json.dumps(context)
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


def romance_outcome(choice_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve the outcome of a romance choice.

    Returns JSON with:
    - result: short narrative outcome
    - affection_change: integer (-2..2)
    - next_hint: optional suggestion for next steps
    """
    sys = dedent(
        """
        You resolve romance choices in Gloamreach.
        Accept a choice id and context and return ONLY a JSON object with keys:
        result, affection_change, and next_hint.
        Choice ids may be: flirt, confess, gift:item:<id>, hold_back.
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