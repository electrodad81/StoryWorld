// src/engine/ExploreEngine.js
// Exploration mode — not yet wired into gameplay.
// Translated from explore/engine.py

import { chatCompletion, chatCompletionFull } from '../services/openai.js';
import { SCENE_SYSTEM_PROMPT, CHOICE_SYSTEM_PROMPT } from './explorePrompts.js';

const CHOICE_COUNT = 2;

function historyText(history) {
  return history
    .slice(-10)
    .map((h) => h.content || '')
    .join('\n');
}

/**
 * Stream an exploration scene. Returns async generator of text chunks.
 */
export async function* streamExploreScene(history, lore) {
  const loreBlob = JSON.stringify(lore).slice(0, 10000);
  const user =
    'Continue free-roam exploration.\n' +
    `--- LORE JSON ---\n${loreBlob}\n--- END LORE ---\n\n` +
    'Player history (latest last):\n' +
    `${historyText(history)}\n`;

  const messages = [
    { role: 'system', content: SCENE_SYSTEM_PROMPT },
    { role: 'user', content: user },
  ];

  yield* chatCompletion(messages, { temperature: 0.7, max_tokens: 350 });
}

/**
 * Generate exploration choices. Returns array of choice strings.
 */
export async function generateExploreChoices(history, lastScene, lore) {
  const loreBlob = JSON.stringify(lore).slice(0, 10000);
  const user =
    `${lastScene}\n\n` +
    'Player history (latest last):\n' +
    `${historyText(history)}\n\n` +
    `--- LORE JSON ---\n${loreBlob}\n--- END LORE ---\n`;

  const messages = [
    { role: 'system', content: CHOICE_SYSTEM_PROMPT },
    { role: 'user', content: user },
  ];

  const text = await chatCompletionFull(messages, { temperature: 0.7, max_tokens: 150 });

  try {
    const arr = JSON.parse(text);
    if (Array.isArray(arr)) {
      return arr.slice(0, CHOICE_COUNT).map((s) => String(s).trim().replace(/\.$/, ''));
    }
  } catch {
    // try extracting JSON array
    const m = text.match(/\[[\s\S]*\]/);
    if (m) {
      try {
        const arr = JSON.parse(m[0]);
        return arr.slice(0, CHOICE_COUNT).map((s) => String(s).trim().replace(/\.$/, ''));
      } catch {
        // fall through
      }
    }
  }

  const lines = text
    .split('\n')
    .map((l) => l.replace(/^[-* ]+/, '').trim())
    .filter(Boolean);
  return lines.slice(0, CHOICE_COUNT);
}

/**
 * Direction label to grid delta.
 */
export function dirToDelta(label) {
  const s = label.toLowerCase();
  if (s.includes('north')) return [0, 1];
  if (s.includes('south')) return [0, -1];
  if (s.includes('east')) return [1, 0];
  if (s.includes('west')) return [-1, 0];
  return [0, 0];
}

/**
 * Build a simple ASCII map from visited positions.
 */
export function buildAsciiMap(visited, pos) {
  const keys = Object.keys(visited).map((k) => k.split(',').map(Number));
  keys.push(pos);
  const xs = keys.map((p) => p[0]);
  const ys = keys.map((p) => p[1]);
  const xMin = Math.min(...xs) - 1;
  const xMax = Math.max(...xs) + 1;
  const yMin = Math.min(...ys) - 1;
  const yMax = Math.max(...ys) + 1;
  const lines = [];
  for (let y = yMax; y >= yMin; y--) {
    let row = '';
    for (let x = xMin; x <= xMax; x++) {
      if (x === pos[0] && y === pos[1]) row += 'P';
      else if (visited[`${x},${y}`]) row += '.';
      else row += ' ';
    }
    lines.push(row);
  }
  return lines.join('\n');
}
