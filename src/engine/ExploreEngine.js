// src/engine/ExploreEngine.js
// AI-driven NPC dialogue for exploration mode.

import { chatCompletion, chatCompletionFull } from '../services/openai.js';

/**
 * Stream an NPC dialogue response.
 * Returns an async generator of text chunks.
 */
export async function* streamNPCDialogue(npc, playerMessage, context) {
  const {
    disposition = 50,
    playerName = '',
    reputation = {},
    locationDescription = '',
    conversationHistory = [],
    currentTopic = null,
    dialogueFlags = [],
    exchangeCount = 0,
    isDismissal = false,
    crossQuestRebuff = null,
  } = context;

  const dispositionLabel =
    disposition >= 70 ? 'friendly and open' :
    disposition >= 50 ? 'neutral but guarded' :
    disposition >= 30 ? 'suspicious and terse' :
    'hostile and dismissive';

  const factionNote = npc.faction
    ? `${npc.name} is affiliated with the ${npc.faction}. This colors their worldview but they don't announce it openly.`
    : `${npc.name} has no faction loyalty.`;

  const topicGuidance = currentTopic
    ? `\nThe player is asking about: ${currentTopic.subject}.\n` +
      `What you know (convey naturally in character, never read verbatim): ${currentTopic.summary}\n` +
      'Share this information across 1–2 responses. Be in character — reveal reluctantly if disposition is low, more openly if high.\n'
    : '';

  const flagsNote = dialogueFlags?.length
    ? `\nContext from recent events (reference naturally if relevant): ${dialogueFlags.join(', ').replace(/_/g, ' ')}.\n`
    : '';

  const exchangeNote = exchangeCount >= 3
    ? '\nThis conversation is nearing its end. Begin wrapping up naturally.\n'
    : '';

  const dismissalNote = isDismissal
    ? `\nEnd the conversation now. Your dismissal style: "${npc.dismissal}". Say something brief in this tone.\n`
    : '';

  const crossQuestNote = crossQuestRebuff
    ? `\nThe player is asking about something on behalf of ${crossQuestRebuff.sourceNpcName} (related to: ${crossQuestRebuff.topicSubject}).` +
      ` However, you don't trust this person enough yet to discuss it. Rebuff them — be evasive or dismissive about the subject,` +
      ` but hint that you MIGHT discuss it if they earned more of your trust or proved themselves to you first.` +
      ` Do NOT reveal the actual information. Keep it brief (2-3 sentences).\n`
    : '';

  const sys =
    `You are ${npc.name}, a character in the dark-fantasy world of Gloamreach.\n` +
    `Archetype: ${npc.archetype}.\n` +
    `Personality: ${npc.personalityTraits.join(', ')}.\n` +
    `Backstory: ${npc.backstory}\n` +
    `${factionNote}\n` +
    `Current disposition toward the player: ${disposition}/100 (${dispositionLabel}).\n` +
    `Location: ${locationDescription}\n` +
    topicGuidance +
    flagsNote +
    exchangeNote +
    dismissalNote +
    crossQuestNote +
    '\nRules:\n' +
    '- Stay in character at all times. Never break the fourth wall.\n' +
    '- Speak in first person as this character. Keep responses to 2–4 sentences.\n' +
    '- Your tone and willingness to share information should reflect your disposition.\n' +
    '- Do not reveal your backstory directly; let it color your words.\n' +
    '- If disposition is low, be evasive or curt. If high, be warmer and more forthcoming.\n' +
    '- Reference the mood and sounds of the location when natural.\n' +
    '- Do not use modern language or references.\n';

  const messages = [
    { role: 'system', content: sys },
    ...conversationHistory.slice(-6),  // Last 3 exchanges max
    { role: 'user', content: playerMessage },
  ];

  yield* chatCompletion(messages, {
    temperature: 0.85,
    max_tokens: 150,
  });
}

/**
 * Generate a single AI flavor option for the interaction menu.
 * Returns a single string (general conversation flavor).
 */
export async function generateFlavorOption(npc, context) {
  const { disposition = 50, conversationHistory = [] } = context;

  const lastExchange = conversationHistory.slice(-2)
    .map(m => `${m.role}: ${m.content}`)
    .join('\n');

  const sys =
    'Generate 1 short, atmospheric interaction option for a player talking to an NPC in a dark-fantasy world.\n' +
    'Return ONLY a single JSON string (not an array).\n' +
    'The option should be conversational small-talk or atmosphere-related.\n' +
    'Keep it ≤ 40 characters. Imperative voice. No trailing periods.\n';

  const user =
    `NPC: ${npc.name} (${npc.archetype}), disposition: ${disposition}/100\n` +
    `Recent exchange:\n${lastExchange || 'No conversation yet.'}\n` +
    `Generate 1 option.`;

  const messages = [
    { role: 'system', content: sys },
    { role: 'user', content: user },
  ];

  try {
    let text = await chatCompletionFull(messages, { temperature: 0.7, max_tokens: 60 });
    text = text.trim();
    // Strip markdown code fences
    text = text.replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
    // If it looks like a JSON object, extract first string value
    if (text.startsWith('{')) {
      try {
        const obj = JSON.parse(text);
        const val = Object.values(obj).find(v => typeof v === 'string');
        if (val) return val.replace(/\.$/, '') || 'Make small talk';
      } catch { /* fall through */ }
      // Brute-force: grab first quoted string
      const m = text.match(/"([^"]{3,})"/);
      if (m) return m[1].replace(/\.$/, '');
      return 'Make small talk';
    }
    // If it's a JSON array, take first element
    if (text.startsWith('[')) {
      try {
        const arr = JSON.parse(text);
        if (typeof arr[0] === 'string') return arr[0].replace(/\.$/, '');
      } catch { /* fall through */ }
    }
    // Plain string — strip wrapping quotes and trailing period
    const cleaned = text.replace(/^["']|["']$/g, '').replace(/\.$/, '').trim();
    return cleaned || 'Make small talk';
  } catch {
    return 'Make small talk';
  }
}

/**
 * Calculate disposition change from a player message.
 * Simple keyword heuristic (can be upgraded to LLM-based later).
 * Returns a delta: positive = friendlier, negative = more hostile.
 */
export function calculateDispositionDelta(playerMessage, npc) {
  const msg = (playerMessage || '').toLowerCase();

  const friendlyPatterns = /thank|please|help|appreciate|understand|agree|respect|trade|buy|offer/i;
  const hostilePatterns = /threaten|demand|force|steal|lie|attack|insult|mock|coerce/i;
  const investigatePatterns = /ask about|tell me|what do you know|heard any|rumor|secret/i;

  if (hostilePatterns.test(msg)) return -8;
  if (friendlyPatterns.test(msg)) return +5;
  if (investigatePatterns.test(msg)) return -2;  // Prying is slightly negative
  return 0;
}
