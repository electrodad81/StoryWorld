"""Lightweight exploration engine utilities.

This module provides thin wrappers around the data store so the
`explore` UI can render deterministic scenes without relying on the
LLM driven story engine.  The functions here deliberately have a very
small surface area so the front‑end can call into them and receive a
JSON compatible structure describing the current state of the world.

The actual persistence layer lives in :mod:`data.store`.  The store in
this repository is intentionally minimal – some deployments may provide
additional helpers such as ``pickup_item`` or ``set_romance_cooldown``.
To keep the code resilient we look up these helpers dynamically via
``getattr`` and simply no‑op if a particular function is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

from data import store


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _call_store(name: str, *args, **kwargs):
    """Call a function on :mod:`data.store` if it exists.

    The production application exposes quite a few helpers on the store
    object (``load_location``, ``pickup_item`` …).  The open source
    version used in tests is far more bare‑bones.  Using ``getattr``
    keeps these calls safe – if a method is missing we simply return
    ``None`` and allow the caller to continue with defaults.
    """

    fn = getattr(store, name, None)
    if callable(fn):  # pragma: no cover - defensive programming
        return fn(*args, **kwargs)
    return None


def _ensure_iter(val: Any) -> Iterable[Any]:
    if val is None:
        return []
    if isinstance(val, (list, tuple, set)):
        return val
    return [val]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tick_render(player_id: str, repo: Path | str) -> Dict[str, Any]:
    """Return a snapshot of the player's surroundings.

    The returned dictionary follows a very small JSON contract used by
    ``explore/engine.py``:

    ``{"prose": str, "choices": list, "map_hint": str | None, ...}``

    ``repo`` is currently unused but kept for API parity with the live
    codebase where location data may be read from disk.
    """

    # Pull state from the persistence layer.  Each ``load_*`` helper is
    # optional; the store decides how much information it can provide.
    location = _call_store("load_location", player_id) or {}
    exits = _call_store("load_exits", player_id) or []
    npcs = _call_store("load_npcs", player_id) or []
    objects = _call_store("load_objects", player_id) or []
    items = _call_store("load_items", player_id) or []

    # Basic prose/choice extraction.  Location dictionaries in the real
    # game contain rich metadata; here we only use a couple of common
    # keys and otherwise fall back to empty strings/lists.
    prose = location.get("description", "")
    map_hint = location.get("map_hint")

    # Choices may come precomputed from the store.  If not, we fall back
    # to generating them from exits.  Each exit is expected to have a
    # ``choice`` field describing the action.
    raw_choices = location.get("choices") or exits
    choices = []
    for opt in raw_choices:
        if isinstance(opt, str):
            choices.append(opt)
        elif isinstance(opt, dict):
            # Common keys: ``choice`` or ``label``
            choices.append(opt.get("choice") or opt.get("label") or "")

    return {
        "location": location.get("id"),
        "prose": prose,
        "choices": [c for c in choices if c],
        "map_hint": map_hint,
        "exits": exits,
        "npcs": npcs,
        "objects": objects,
        "items": items,
    }


def apply_outcome(player_id: str, choice: Any, repo: Path | str) -> Dict[str, Any]:
    """Apply the result of a player's choice.

    ``choice`` is expected to be a structure (often a dict) containing
    any of the following keys:

    - ``pickup`` / ``drop`` / ``use`` : items to modify in the player's
      inventory.
    - ``flags`` : mapping of flag names to boolean values.
    - ``move`` : new location identifier.
    - ``romance_cooldown`` : mapping of NPC identifiers to cooldown
      timestamps.

    After mutating the store the function delegates to :func:`tick_render`
    to produce the updated view of the world.
    """

    data = choice if isinstance(choice, dict) else {}

    # Inventory changes --------------------------------------------------
    for item in _ensure_iter(data.get("pickup")):
        _call_store("pickup_item", player_id, item)
    for item in _ensure_iter(data.get("drop")):
        _call_store("drop_item", player_id, item)
    for item in _ensure_iter(data.get("use")):
        _call_store("use_item", player_id, item)

    # Flag updates -------------------------------------------------------
    flags = data.get("flags", {}) or {}
    for flag, value in flags.items():
        _call_store("set_flag", player_id, flag, value)

    # Movement -----------------------------------------------------------
    if data.get("move") is not None:
        _call_store("move", player_id, data.get("move"))

    # Romance cooldowns --------------------------------------------------
    rc = data.get("romance_cooldown") or {}
    for npc_id, cooldown in rc.items():
        _call_store("set_romance_cooldown", player_id, npc_id, cooldown)

    # Delegates to ``tick_render`` to obtain the resulting scene/state.
    return tick_render(player_id, repo)
