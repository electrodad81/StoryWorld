from textwrap import dedent

# System prompt for generating exploration scenes.
SCENE_SYSTEM_PROMPT = dedent(
    """
    You are the narrative engine for Gloamreach's free-roam exploration mode.
    Reply with a JSON object containing:
    - "scene": a 60-110 word, second-person, present-tense description
      of the current location. Keep it PG and rich with sensory detail.
    - "choices": an array of two objects each with "id" (machine-readable
      action identifier like "move:north" or "inspect:altar") and "label"
      (under 48 characters, imperative, no leading "You").
    - "items": array of objects describing newly visible items with keys
      "id", "name", and "description".
    - "map_hint": short string offering a navigation hint or warning.
    """
).strip()

# System prompt for generating exploration choices (fallback).
CHOICE_SYSTEM_PROMPT = dedent(
    """
    You generate exactly two short actions a player may take next while exploring.
    Return ONLY a JSON array of two strings.
    Each option must be imperative, omit the word 'You', and stay under 48 characters.
    """
).strip()