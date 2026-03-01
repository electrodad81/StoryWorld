// src/services/identity.js
// Generate/retrieve a stable browser ID stored in localStorage.

const LS_KEY = 'storyworld_browser_id';

function uuid() {
  return crypto.randomUUID
    ? crypto.randomUUID()
    : 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
        const r = (Math.random() * 16) | 0;
        return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
      });
}

/**
 * Returns a stable per-browser ID (brw-<uuid>).
 * Creates one on first call and persists it in localStorage.
 */
export function ensureBrowserId() {
  let id = localStorage.getItem(LS_KEY);
  if (id) return id;
  id = `brw-${uuid()}`;
  localStorage.setItem(LS_KEY, id);
  return id;
}

export function clearBrowserId() {
  localStorage.removeItem(LS_KEY);
}
