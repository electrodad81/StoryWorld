// src/engine/StoryEngine.js
// Translated from story/engine.py — core story generation engine.

import { chatCompletion, chatCompletionFull, imageGeneration } from '../services/openai.js';

// Beat system constants
export const BEATS = ['exposition', 'rising_action', 'climax', 'falling_action', 'resolution'];

export const BEAT_TARGET_SCENES = {
  exposition: 3,
  rising_action: 4,
  climax: 2,
  falling_action: 3,
  resolution: 1,
};

export const BEAT_GUIDELINES = {
  exposition:
    'Open in motion; establish POV, desire, and a concrete obstacle. Plant an opening image to echo later.',
  rising_action:
    'Escalate with 1–2 concrete complications that corner the protagonist. Narrow options; raise stakes. Introduce at most one minor element.',
  climax:
    'Force a single, irreversible decision. Make the price/cost explicit. No new locations or characters; cash the promise of the hook.',
  falling_action:
    'Show immediate, visible consequences in the world. Resolve dangling promises. No new mysteries or hooks; compress time if needed.',
  resolution:
    'Answer the central question (yes / no / yes-but / no-and). Echo the opening image in a changed way. State the internal change in one clear line. End on a specific sensory image. No next-scene hooks; no choices.',
};

// ---------------------------------------------------------------------------
// Illustration safety helpers
// ---------------------------------------------------------------------------

const REPLACEMENTS = [
  [/\b(blood|bloody|bloodied|gore|gory|guts|entrails|viscera|corpse|cadaver|mutilated|severed|dismembered|decapitat\w*)\b/gi, 'damage'],
  [/\b(wound|wounded|bleed\w*|stab\w*|slash\w*|maim\w*|butcher\w*|impale\w*)\b/gi, 'injury'],
  [/\b(skull|brain|decay|rotting|putrid)\b/gi, 'remains'],
  [/\b(nude|nudity|breast\w*|genital\w*|erotic|sexual|sex|porn\w*)\b/gi, 'tasteful attire'],
  [/\b(suicide|self[-\s]?harm|hang(?:ed|ing)|cutting)\b/gi, 'danger'],
  [/\b(shotgun|rifle|pistol|handgun|gun|revolver|smg|submachine|uzi|ak-?47|ar-?15|carbine|bullet|ammo|magazine|shell|cartridge)\b/gi, 'sword'],
  [/\b(musket|flintlock|arquebus|blunderbuss)\b/gi, 'crossbow'],
  [/\b(grenade|bomb|dynamite|explosive\w*)\b/gi, 'alchemy vial'],
  [/\b(trench\s?coat|trenchcoat|peacoat|greatcoat|overcoat|duster)\b/gi, 'hooded cloak'],
  [/\b(suit|tuxedo|necktie|tie|blazer)\b/gi, 'tunic'],
  [/\b(fedora|trilby|bowler|top\s?hat|newsboy cap)\b/gi, 'hood'],
  [/\b(camera|photograph|selfie|phone|smartphone|cell\s?phone|radio|microphone|speaker|laptop|computer|tablet)\b/gi, 'lantern'],
  [/\b(flashlight|electric\s?torch)\b/gi, 'torch'],
  [/\b(car|truck|van|bus|train|subway|tram|airplane|jet|helicopter)\b/gi, 'horse cart'],
  [/\b(skyscraper|elevator|escalator|billboard|neon|streetlight|traffic light|power\s?line|telephone pole|asphalt)\b/gi, 'stone tower'],
  [/\b(jeans|t-?shirt|hoodie|sneakers)\b/gi, 'woolen garments'],
];

const HARD_FILTER = [
  /\bsever(?:ed|ing)\b/i, /\bdismember(?:ed|ment)\b/i, /\bdecapitat\w*\b/i,
  /\bguts\b/i, /\bentrails\b/i, /\bviscera\b/i, /\bgraphic\b/i,
  /\bsex|sexual|erotic\b/i, /\bnude|nudity\b/i,
  /\b(shotgun|rifle|pistol|handgun|gun|revolver|musket|flintlock|arquebus|blunderbuss|bullet|ammo|magazine|cartridge)\b/i,
  /\b(trench\s?coat|trenchcoat|peacoat|greatcoat|overcoat|duster|suit|tuxedo|necktie|blazer|fedora|trilby|bowler|top\s?hat)\b/i,
  /\b(camera|photograph|phone|smartphone|cell\s?phone|radio|microphone|speaker|laptop|computer|tablet|flashlight)\b/i,
  /\b(car|truck|van|bus|train|subway|tram|airplane|jet|helicopter|skyscraper|neon|streetlight|traffic light|power\s?line)\b/i,
];

function stripQuotes(s) {
  return s.replace(/\u201c/g, '"').replace(/\u201d/g, '"').replace(/\u2018/g, "'").replace(/\u2019/g, "'");
}

function truncate(s, n = 400) {
  s = s.replace(/\s+/g, ' ').trim();
  if (s.length <= n) return s;
  const cut = s.slice(0, n);
  const last = cut.lastIndexOf(' ');
  return (last > 0 ? cut.slice(0, last) : cut) + '\u2026';
}

function sanitize(text, level) {
  let src = stripQuotes(text || '');
  if (level >= 1) {
    for (const [pat, repl] of REPLACEMENTS) {
      src = src.replace(pat, repl);
    }
  }
  if (level >= 2) {
    const sents = src.split(/(?<=[.!?])\s+/);
    const keep = sents.filter((s) => !HARD_FILTER.some((p) => p.test(s)));
    src = keep.length ? keep.join(' ') : 'mysterious scene, danger implied but not shown';
  }
  if (level >= 3) {
    src = 'atmospheric medieval-fantasy environment; focus on landscape, props, and architecture; no modern elements, no depiction of injuries';
  }
  const safety = 'family-friendly, PG-13, no gore, no explicit injuries, no nudity';
  return truncate(`${src}. ${safety}.`);
}

function genderVisualDirective(gender) {
  const g = (gender || '').trim().toLowerCase();
  if (g === 'male')
    return 'If the protagonist appears, depict a masculine-presenting silhouette or attire. No facial details. Avoid stereotypes; keep clothing practical.';
  if (g === 'female')
    return 'If the protagonist appears, depict a feminine-presenting silhouette or attire. No facial details. Avoid sexualization; keep clothing practical.';
  if (g === 'nonbinary')
    return 'If the protagonist appears, depict an androgynous silhouette with neutral attire. No facial details. Avoid gendered cues or stereotypes.';
  return 'If the protagonist appears, keep the silhouette gender-ambiguous with neutral attire. No facial details.';
}

function basicStyle() {
  return (
    'Simple, clean black-and-white line-and-wash illustration with a single spot color accent ' +
    '(gold leaf or lapis). Light crosshatching, clear contours, single focal subject, ' +
    'medium or close-up shot. Plain white background behind the subject. ' +
    'Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). ' +
    'Architecture: stone keeps, timber framing, castles, market stalls; natural materials ' +
    'like wood, leather, linen, iron. Props/weapons allowed: swords, daggers, axes, spears, ' +
    'shields, bows, torches, lanterns, books, scrolls, potions. ' +
    'Strictly no firearms or explosives, no modern clothing (no trenchcoats, suits, ties), ' +
    'no modern tech/vehicles (no cameras, phones, radios, cars, trains, planes, neon, streetlights, power lines). ' +
    'Do not depict the protagonist\'s face directly (use silhouette, hood, or a cropped angle). ' +
    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.'
  );
}

function buildPrompt(seed, level, gdir) {
  const style = basicStyle();
  const safe = sanitize(seed, level);
  return `${style} Depict the scene mood and setting with tasteful restraint. ${safe} ${gdir}`;
}

// ---------------------------------------------------------------------------
// History helper
// ---------------------------------------------------------------------------

function historyText(history) {
  const lines = [];
  for (const m of history) {
    const role = m.role || '';
    const content = m.content || '';
    if (!content) continue;
    lines.push(`${role.toUpperCase()}: ${content}`);
  }
  return lines.slice(-40).join('\n');
}

function openingImageHint(history) {
  const first = history.find((m) => m.role === 'assistant');
  if (!first) return '';
  const sent = (first.content || '').trim().split(/(?<=[.!?])\s+/)[0] || '';
  return sent.slice(0, 160);
}

function composePrompt(history, lore, beat) {
  const base =
    'Write the next short scene in a self-contained short story. Keep scenes concise and concrete; avoid summarizing future events.';
  const guide = BEAT_GUIDELINES[beat] || '';
  const opener = openingImageHint(history);
  const extra = opener ? ` Opening image to echo later: "${opener}".` : '';
  return `${base}\n\nBeat: ${beat || 'classic'}.\nGuidelines: ${guide}${extra}`;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Stream a scene. Returns an async generator of text chunks.
 */
export async function* streamScene(history, lore, {
  beat = null,
  playerName = '',
  gender = 'Unspecified',
  dangerStreak = 0,
  injuryLevel = 0,
  runSeed = '',
} = {}) {
  const prompt = composePrompt(history, lore, beat);
  const loreBlob = JSON.stringify(lore).slice(0, 20000);

  const sysBase =
    'You are the narrative engine for a PG dark-fantasy interactive story called Gloamreach.\n' +
    'Write in present tense, second person. Do not name the protagonist; if a Name exists, it may appear only in NPC dialogue.\n' +
    'Keep prose very tight (~60–90 words, single short paragraph). Every word must earn its place. Maintain continuity with prior scenes.\n' +
    'Consequence contract: If the last player choice was risky, show a visible cost in THIS scene (wound, gear loss, time pressure, ally setback, exposure to threat). Do NOT undo it immediately. ' +
    'If the player chose risky paths in two consecutive scenes, escalate to a serious setback (capture, grave wound, loss). Only when fictionally fitting, you may kill the protagonist.\n' +
    "If (and only if) the protagonist dies in this scene, append exactly '\\n\\n[DEATH]' as the final line. Do not add any text after [DEATH].";

  const seedLine = runSeed
    ? `RUN_SEED: "${runSeed}" (do not reveal this; use it as a hidden stylistic fingerprint for variety across runs)\n`
    : '';

  const riskFlags = {
    allow_fail_state: true,
    danger_streak: dangerStreak,
    injury_level: injuryLevel,
    must_escalate: dangerStreak >= 2,
  };
  const riskControlStr = `CONTROL FLAGS: ${JSON.stringify(riskFlags)}\n`;
  const controlFlagsNote = 'Do not expose control flags; they are for you only.\n';

  const nameClause = playerName
    ? `Player name (for NPC dialogue only): ${playerName}\n`
    : "No name is provided. Do not address the protagonist by any name; use 'you' only.\n";

  const user =
    nameClause +
    seedLine +
    riskControlStr +
    controlFlagsNote +
    'Continue the story with the next scene. Consider the world details below.\n' +
    `--- LORE JSON ---\n${loreBlob}\n--- END LORE ---\n\n` +
    'Player history (latest last):\n' +
    `${historyText(history)}\n\n` +
    'Output text in second person, present tense.\n' +
    'Length: ~60–90 words. One compact paragraph. End cleanly; no new paragraphs.';

  const messages = [
    { role: 'system', content: sysBase + '\n\n' + prompt },
    { role: 'user', content: user },
  ];

  yield* chatCompletion(messages, { temperature: 0.9, max_tokens: 200 });
}

/**
 * Generate choices for the current scene. Returns array of choice strings.
 */
export async function generateChoices(history, lastScene, lore) {
  const sys =
    'You generate concise, imperative next-action options for an interactive story.\n' +
    'Return ONLY a JSON array of two strings.\n' +
    "Constraints: imperative voice (command form), do NOT start with 'You', \u2264 48 characters each,\n" +
    'no trailing periods, options must be meaningful and distinct from each other.\n';

  const loreBlob = JSON.stringify(lore).slice(0, 20000);
  const user =
    'Create two next-step options that follow naturally from the latest assistant scene.\n' +
    'Avoid near-duplicates.\n\n' +
    `--- LORE JSON ---\n${loreBlob}\n--- END LORE ---\n\n` +
    'Context (recent turns; latest last):\n' +
    `${historyText(history)}\n`;

  const messages = [
    { role: 'system', content: sys },
    { role: 'user', content: user },
  ];

  const text = await chatCompletionFull(messages, { temperature: 0.7, max_tokens: 200 });

  const match = text.match(/\[[\s\S]*\]/);
  if (!match) {
    const lines = text.split('\n').map((l) => l.replace(/^[-* \t]+/, '').trim()).filter(Boolean);
    return lines.length ? lines.slice(0, 2) : ['Press on', 'Hold back'];
  }
  try {
    const arr = JSON.parse(match[0]);
    const cleaned = arr.slice(0, 2).map((s) => {
      s = (s || '').trim();
      if (s.endsWith('.')) s = s.slice(0, -1);
      return s;
    });
    while (cleaned.length < 2) cleaned.push('Continue');
    return cleaned.slice(0, 2);
  } catch {
    return ['Press on', 'Hold back'];
  }
}

/**
 * Generate an illustration for a scene.
 * Progressive sanitization levels 1→3, then environment-only fallback.
 * Returns image URL/data URI or null.
 */
export async function generateIllustration(scene, gender = 'Unspecified') {
  const summary = (scene || '').split('.')[0].trim();
  if (!summary) return null;

  const gdir = genderVisualDirective(gender);

  // Try levels 1→3
  for (const level of [1, 2, 3]) {
    const prompt = buildPrompt(summary, level, gdir);
    try {
      const url = await imageGeneration(prompt);
      if (url) return url;
    } catch (e) {
      const msg = e.message || '';
      // If permission/org issue, stop trying
      if (msg.includes('403') || msg.includes('PermissionDenied') || msg.includes('must be verified')) {
        return null;
      }
      // Policy rejection → try next level
      if (msg.includes('content_policy') || msg.includes('rejected') || msg.includes('safety')) {
        continue;
      }
      // Other errors → try next level
    }
  }

  // Final fallback: very tame environment prompt
  const fallbackPrompt =
    `${basicStyle()} ` +
    'Atmospheric medieval-fantasy environment; focus on scenery and architecture; ' +
    'no modern elements, no characters in distress, no firearms or explosives; family-friendly, PG-13. ' +
    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. ' +
    gdir;

  try {
    return await imageGeneration(fallbackPrompt);
  } catch {
    return null;
  }
}
