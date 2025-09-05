from textwrap import dedent

# System prompt for generating exploration scenes.
SCENE_SYSTEM_PROMPT = dedent(
    """
    You are the narrative engine for Gloamreach's free-roam exploration mode.
    Reply with a JSON object containing:
    - "scene": a 60-110 word, second-person, present-tense description
      of the current location. Keep it PG and rich with sensory detail.
    - "choices": an array of objects each with "id" and "label". Include one
      entry for every available action: each visible exit must produce a
      move:<DIR> choice, each NPC a talk:npc:<id> choice, items may allow
      pickup:item:<id>, drop:item:<id>, or use:item:<id>, and notable objects
      may use investigate:obj:<id>. The array length is dynamic but must
      contain at least one valid move:<DIR> choice. Labels stay under 48
      characters and omit leading "You".
    - "items": array of objects describing newly visible items with keys
      "id", "name", and "description".
    - "map_hint": short string offering a navigation hint or warning.
    """
).strip()

# System prompt for generating exploration choices (fallback).
CHOICE_SYSTEM_PROMPT = dedent(
    """
    You generate a list of short actions a player may take next while exploring.
    Return ONLY a JSON array of strings. Include movement options for each
    available exit and any interactions with nearby NPCs or items. Each
    option must be imperative, omit the word 'You', and stay under 48 characters.
    """
).strip()