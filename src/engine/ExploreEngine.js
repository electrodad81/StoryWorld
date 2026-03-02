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
  } = context;

  const dispositionLabel =
    disposition >= 70 ? 'friendly and open' :
    disposition >= 50 ? 'neutral but guarded' :
    disposition >= 30 ? 'suspicious and terse' :
    'hostile and dismissive';

  const factionNote = npc.faction
    ? `${npc.name} is affiliated with the ${npc.faction}. This colors their worldview but they don't announce it openly.`
    : `${npc.name} has no faction loyalty.`;

  const sys =
    `You are ${npc.name}, a character in the dark-fantasy world of Gloamreach.\n` +
    `Archetype: ${npc.archetype}.\n` +
    `Personality: ${npc.personalityTraits.join(', ')}.\n` +
    `Backstory: ${npc.backstory}\n` +
    `${factionNote}\n` +
    `Current disposition toward the player: ${disposition}/100 (${dispositionLabel}).\n` +
    `Location: ${locationDescription}\n\n` +
    'Rules:\n' +
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
 * Generate contextual interaction options for an NPC.
 * Returns array of 3 action strings.
 */
export async function generateInteractionOptions(npc, context) {
  const { disposition = 50, conversationHistory = [] } = context;

  const lastExchange = conversationHistory.slice(-2)
    .map(m => `${m.role}: ${m.content}`)
    .join('\n');

  const sys =
    'Generate 3 interaction options for a player talking to an NPC in a dark-fantasy world.\n' +
    'Return ONLY a JSON array of 3 strings.\n' +
    'Options should be: one conversational, one investigative, one that could shift disposition.\n' +
    'Keep each option ≤ 40 characters. Imperative voice. No trailing periods.\n';

  const user =
    `NPC: ${npc.name} (${npc.archetype}), disposition: ${disposition}/100\n` +
    `Recent exchange:\n${lastExchange || 'No conversation yet.'}\n` +
    `Generate 3 options.`;

  const messages = [
    { role: 'system', content: sys },
    { role: 'user', content: user },
  ];

  const text = await chatCompletionFull(messages, { temperature: 0.7, max_tokens: 120 });

  try {
    const match = text.match(/\[[\s\S]*\]/);
    if (match) {
      const arr = JSON.parse(match[0]);
      return arr.slice(0, 3).map(s => s.trim().replace(/\.$/, ''));
    }
  } catch {}

  // Fallback
  return ['Ask about the area', 'Inquire about rumors', 'Bid farewell'];
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
