// src/services/persistence.js
// LocalStorage-based persistence stub.
// Interface designed so a Neon DB backend can be swapped in later.
// Future: replace localStorage calls with fetch() to a Neon-backed API.

const SNAPSHOT_PREFIX = 'storyworld_snapshot_';

/**
 * Save a game state snapshot.
 * @param {string} pid - Player/browser ID
 * @param {object} state - Game state to persist
 */
export function saveSnapshot(pid, state) {
  const key = SNAPSHOT_PREFIX + pid;
  localStorage.setItem(key, JSON.stringify(state));
}

/**
 * Load a game state snapshot.
 * @param {string} pid - Player/browser ID
 * @returns {object|null} The saved state, or null if none exists
 */
export function loadSnapshot(pid) {
  const key = SNAPSHOT_PREFIX + pid;
  const raw = localStorage.getItem(key);
  if (!raw) return null;
  try {
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

/**
 * Delete a game state snapshot.
 * @param {string} pid - Player/browser ID
 */
export function deleteSnapshot(pid) {
  const key = SNAPSHOT_PREFIX + pid;
  localStorage.removeItem(key);
}
