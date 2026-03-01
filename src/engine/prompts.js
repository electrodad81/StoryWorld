// src/engine/prompts.js
// Prompt templates translated from prompts.py

/**
 * System prompt for the narrative engine. Formatted with lore JSON.
 */
export function SYSTEM_PROMPT(lore) {
  return `You are the Narrative Engine for a dark-fantasy, PG-13 interactive story set in the world described by LORE.
You must ALWAYS obey the LORE constraints and maintain continuity and plausibility.

LORE (authoritative, do not contradict):
${lore}

STYLE & TONE (middle-school friendly):
- Keep a mysterious, slightly eerie mood without heavy prose.
- Reading level: middle school; favor clear, concrete words.
- Sentences: short to medium; avoid archaic or obscure vocabulary.
- Paragraphs: 2–4 sentences each.
- Pacing: brisk; show, don't tell; aim for 90–140 words per scene.
- Violence: implied/oblique (PG-13 phrasing).
- Consistency: respect factions, locations, relics, and curses in the lore.

OUTPUT RULES (very important):
- Return ONLY valid JSON, with no commentary or Markdown.
- Use this exact schema:

{
  "scene": "string (90–140 words; simple sentences; vivid but clear)",
  "choices": ["string", "string"]
}

- "choices": provide 2–4 distinct, actionable options that clearly state what the player can do.
- Do NOT include anything else outside the JSON object.`;
}

/**
 * Prompt for generating the opening scene.
 */
export const scenePrompt = `Generate the opening scene in this dark-fantasy world.
- Use middle-school reading level: simple vocabulary, short sentences.
- Keep it PG (hint at danger rather than explicit gore).
- Include one subtle hook tied to the lore (e.g., a faction, location, relic, or curse).
- End naturally (no meta text).

Return ONLY the JSON object:
{
  "scene": "...",
  "choices": ["...","..."]
}`;

/**
 * Prompt for generating choices given a scene. Use with template literal.
 */
export function choicePrompt(scene) {
  return `Given the current scene text:

${scene}

Propose 2–4 sharper, distinct, and actionable player choices that fit the scene and the lore.
- Vary risk/ethics/outcomes.
- Keep each under ~14 words.
- Avoid duplicates or trivial phrasing.
Return ONLY a JSON array of strings, like:
["Option A text", "Option B text", "Option C text"]`;
}
