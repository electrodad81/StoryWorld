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

    # Pull state from the persistence layer.  Each helper is optional; the
    # store decides how much information it can provide.
    player = _call_store("get_player_world_state", player_id) or {}
    loc_id = player.get("loc_id")
    location = _call_store("get_location", loc_id) or {"id": loc_id}
    npcs, objects, exits = _call_store("visible_interactables", loc_id) or ([], [], [])
    items = _call_store("list_location_items", loc_id) or []
    # Use the inventory already contained in the player state rather than
    # hitting the store again. Some store implementations may have side
    # effects when listing the player's inventory which could cause items to
    # appear without an explicit pickup action. By relying on the copy from
    # ``get_player_world_state`` we guarantee that merely rendering a scene is
    # read-only.
    inventory = dict(player.get("inventory") or {})

    # Basic prose/choice extraction.  Location dictionaries in the real
    # game contain rich metadata; here we only use a couple of common
    # keys and otherwise fall back to empty strings/lists.
    prose = location.get("description", "")
    map_hint = location.get("map_hint")

    # Assemble dynamic choices from exits, NPCs, objects, and items.
    choices = []
    dir_names = {"N": "north", "S": "south", "E": "east", "W": "west"}
    for ex in exits:
        if ex.get("locked"):
            continue
        direction = ex.get("direction")
        if direction:
            name = dir_names.get(direction, direction)
            choices.append({
                "id": f"move:{direction}",
                "label": f"Go {name}",
            })
    for npc in npcs:
        npc_id = npc.get("id")
        name = npc.get("name") or npc_id
        if npc_id:
            choices.append({
                "id": f"talk:npc:{npc_id}",
                "label": f"Talk to {name}",
            })
    for obj in objects:
        obj_id = obj.get("id")
        name = obj.get("name") or obj_id
        if obj_id:
            choices.append({
                "id": f"investigate:obj:{obj_id}",
                "label": f"Investigate {name}",
            })
    for item in items:
        item_id = item.get("id")
        name = item.get("name") or item_id
        if item_id:
            choices.append({
                "id": f"pickup:item:{item_id}",
                "label": f"Pick up {name}",
            })
    for inv_id in inventory.keys():
        choices.append({"id": f"use:item:{inv_id}", "label": f"Use {inv_id}"})
        choices.append({"id": f"drop:item:{inv_id}", "label": f"Drop {inv_id}"})

    return {
        "location": location.get("id"),
        "prose": prose,
        "choices": [c for c in choices if c],
        "map_hint": map_hint,
        "exits": exits,
        "npcs": npcs,
        "objects": objects,
        "items": items,
        "inventory": inventory,
    }


def apply_outcome(player_id: str, choice: Any, repo: Path | str) -> Dict[str, Any]:
    """Apply the result of a player's choice."""

    if isinstance(choice, str):
        action = choice
        player = _call_store("get_player_world_state", player_id) or {}
        loc_id = player.get("loc_id")
        if action.startswith("move:"):
            direction = action.split(":", 1)[1]
            _, _, exits = _call_store("visible_interactables", loc_id) or ([], [], [])
            for ex in exits:
                if ex.get("direction") == direction and not ex.get("locked"):
                    _call_store("move", player_id, ex.get("dst_id"))
                    break
        elif action.startswith("talk:npc:"):
            pass  # talking has no mechanical effect yet
        elif action.startswith("pickup:item:"):
            item_id = action.split(":")[-1]
            _call_store("pickup_item", player_id, loc_id, item_id, 1)
        elif action.startswith("drop:item:"):
            item_id = action.split(":")[-1]
            _call_store("drop_item", player_id, loc_id, item_id, 1)
        elif action.startswith("use:item:"):
            item_id = action.split(":")[-1]
            _call_store("use_item", player_id, item_id)
        elif action.startswith("investigate:obj:"):
            obj_id = action.split(":")[-1]
            _call_store("set_flag", player_id, f"seen:{obj_id}", True)
        return tick_render(player_id, repo)

    data = choice if isinstance(choice, dict) else {}

    # Inventory changes --------------------------------------------------
    player = _call_store("get_player_world_state", player_id) or {}
    loc_id = player.get("loc_id")
    inv_delta = data.get("inventory_delta") or {}
    for item_id, delta in inv_delta.items():
        if delta > 0:
            _call_store("pickup_item", player_id, loc_id, item_id, delta)
        elif delta < 0:
            _call_store("drop_item", player_id, loc_id, item_id, -delta)
    for item in _ensure_iter(data.get("use")):
        _call_store("use_item", player_id, item)

    # Flag updates -------------------------------------------------------
    flags = data.get("flags", {}) or {}
    for flag, value in flags.items():
        _call_store("set_flag", player_id, flag, value)

    # Movement -----------------------------------------------------------
    new_loc = data.get("player_loc") or data.get("move")
    if new_loc is not None:
        _call_store("move", player_id, new_loc)

    # Romance cooldowns --------------------------------------------------
    rc = data.get("romance_cooldown") or {}
    for npc_id, cooldown in rc.items():
        _call_store("set_romance_cooldown", player_id, npc_id, cooldown)

    # Delegates to ``tick_render`` to obtain the resulting scene/state.
    return tick_render(player_id, repo)
