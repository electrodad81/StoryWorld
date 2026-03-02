// src/services/journal.js
// Persistent player journal — localStorage-based, designed for backend migration.

const JOURNAL_KEY = 'gloamreach_journal';

// ---- Achievement definitions ----

const ACHIEVEMENT_DEFS = [
  { id: 'first_death',    title: 'Mortal After All',   condition: (j) => j.stats.totalDeaths >= 1 },
  { id: 'first_survival', title: 'Against the Dark',   condition: (j) => j.stats.totalStories - j.stats.totalDeaths >= 1 },
  { id: 'three_stories',  title: 'Seasoned Traveler',  condition: (j) => j.stats.totalStories >= 3 },
  { id: 'all_vale_npcs',  title: 'Voices of the Vale', condition: (j) => ['Maren', 'Edric', 'Sibyl'].every(n => j.loreDiscovered.npcs.includes(n)) },
  { id: 'all_factions',   title: 'Between Powers',     condition: (j) => ['lantern_guild', 'pale_court', 'inquisition'].every(f => j.loreDiscovered.factions.includes(f)) },
  { id: 'five_deaths',    title: 'Dust to Dust',       condition: (j) => j.stats.totalDeaths >= 5 },
];

// ---- Default journal shape ----

function createDefault() {
  return {
    playerId: localStorage.getItem('browser_id') || 'anonymous',
    storiesPlayed: [],
    loreDiscovered: {
      locations: [],
      factions: [],
      relics: [],
      curses: [],
      npcs: [],
    },
    stats: {
      totalStories: 0,
      totalDeaths: 0,
      totalScenes: 0,
      survivalRate: 0,
    },
    achievements: [],
    unlockedHooks: ['HOOK-01', 'HOOK-02'],
    explorationUnlocked: false,
  };
}

// ---- Public API ----

export function loadJournal() {
  try {
    const raw = localStorage.getItem(JOURNAL_KEY);
    if (!raw) return createDefault();
    const parsed = JSON.parse(raw);
    // Merge with defaults for forward-compatibility
    const def = createDefault();
    return {
      ...def,
      ...parsed,
      loreDiscovered: { ...def.loreDiscovered, ...parsed.loreDiscovered },
      stats: { ...def.stats, ...parsed.stats },
    };
  } catch {
    return createDefault();
  }
}

export function saveJournal(journal) {
  localStorage.setItem(JOURNAL_KEY, JSON.stringify(journal));
}

/**
 * Add a completed story entry and update stats/lore/unlocks.
 * Returns { journal, newAchievements, newUnlocks }.
 */
export function addStoryEntry(journal, {
  hookId,
  hookTitle,
  outcome,       // 'survived' | 'died'
  beatReached,
  sceneCount,
  keyChoices,
  factionsEncountered,
  locationsReferenced,
  relicsFound,
  cursesFound,
}) {
  const entry = {
    hookId,
    hookTitle,
    outcome,
    beatReached,
    sceneCount,
    keyChoices: keyChoices || [],
    factionsEncountered: factionsEncountered || [],
    locationsReferenced: locationsReferenced || [],
    timestamp: new Date().toISOString(),
  };

  const updated = { ...journal };
  updated.storiesPlayed = [...updated.storiesPlayed, entry];

  // Update stats
  const totalStories = updated.storiesPlayed.length;
  const totalDeaths = updated.storiesPlayed.filter(s => s.outcome === 'died').length;
  const totalScenes = updated.stats.totalScenes + sceneCount;
  updated.stats = {
    totalStories,
    totalDeaths,
    totalScenes,
    survivalRate: totalStories > 0 ? (totalStories - totalDeaths) / totalStories : 0,
  };

  // Merge lore discoveries
  updated.loreDiscovered = mergeLore(updated.loreDiscovered, {
    locations: locationsReferenced || [],
    factions: factionsEncountered || [],
    relics: relicsFound || [],
    curses: cursesFound || [],
  });

  // Check unlocks
  const newUnlocks = checkUnlocks(updated);
  updated.unlockedHooks = newUnlocks.unlockedHooks;
  updated.explorationUnlocked = newUnlocks.explorationUnlocked;

  // Check achievements
  const newAchievements = getNewAchievements(updated);
  updated.achievements = [...updated.achievements, ...newAchievements];

  saveJournal(updated);

  return {
    journal: updated,
    newAchievements,
    newUnlocks: {
      hooks: newUnlocks.newlyUnlockedHooks,
      explorationJustUnlocked: newUnlocks.explorationJustUnlocked,
    },
  };
}

/**
 * Merge explore-mode lore discoveries into the journal.
 */
export function addExploreLore(journal, { locations, npcs, factions }) {
  const updated = { ...journal };
  updated.loreDiscovered = mergeLore(updated.loreDiscovered, {
    locations: locations || [],
    factions: factions || [],
    npcs: npcs || [],
  });

  // Check achievements after lore update
  const newAchievements = getNewAchievements(updated);
  updated.achievements = [...updated.achievements, ...newAchievements];

  saveJournal(updated);
  return updated;
}

/**
 * Evaluate unlock conditions based on current journal state.
 */
export function checkUnlocks(journal) {
  const { stats } = journal;
  const currentHooks = new Set(journal.unlockedHooks);
  const prevHooks = new Set(journal.unlockedHooks);

  // Always unlocked
  currentHooks.add('HOOK-01');
  currentHooks.add('HOOK-02');

  // HOOK-03: Complete any story
  if (stats.totalStories >= 1) currentHooks.add('HOOK-03');

  // HOOK-04: Survive a story
  if (stats.totalStories - stats.totalDeaths >= 1) currentHooks.add('HOOK-04');

  // HOOK-05: Die in a story
  if (stats.totalDeaths >= 1) currentHooks.add('HOOK-05');

  // HOOK-06: Complete 3 stories
  if (stats.totalStories >= 3) currentHooks.add('HOOK-06');

  const unlockedHooks = [...currentHooks];
  const newlyUnlockedHooks = unlockedHooks.filter(h => !prevHooks.has(h));

  // Exploration unlocked after completing 1 story
  const explorationUnlocked = stats.totalStories >= 1;
  const explorationJustUnlocked = explorationUnlocked && !journal.explorationUnlocked;

  return {
    unlockedHooks,
    newlyUnlockedHooks,
    explorationUnlocked,
    explorationJustUnlocked,
  };
}

// ---- Internal helpers ----

function mergeLore(existing, additions) {
  const merged = { ...existing };
  for (const key of Object.keys(additions)) {
    if (!merged[key]) merged[key] = [];
    const newItems = (additions[key] || []).filter(item => !merged[key].includes(item));
    if (newItems.length > 0) {
      merged[key] = [...merged[key], ...newItems];
    }
  }
  return merged;
}

function getNewAchievements(journal) {
  const earned = new Set(journal.achievements.map(a => a.id));
  const newOnes = [];

  for (const def of ACHIEVEMENT_DEFS) {
    if (!earned.has(def.id) && def.condition(journal)) {
      newOnes.push({
        id: def.id,
        title: def.title,
        unlockedAt: new Date().toISOString(),
      });
    }
  }

  return newOnes;
}

/**
 * Extract lore keywords from story history text.
 */
export function extractLoreFromHistory(history, lore) {
  const fullText = history.map(m => m.content || '').join(' ').toLowerCase();

  const factions = Object.keys(lore.factions || {}).filter(key => {
    const name = (lore.factions[key].name || '').toLowerCase();
    return fullText.includes(key.replace(/_/g, ' ')) || fullText.includes(name);
  });

  const locations = Object.keys(lore.locations || {}).filter(key => {
    return fullText.includes(key.replace(/_/g, ' '));
  });

  const relics = Object.keys(lore.relics || {}).filter(key => {
    return fullText.includes(key.replace(/_/g, ' '));
  });

  const curses = Object.keys(lore.curses || {}).filter(key => {
    return fullText.includes(key.replace(/_/g, ' '));
  });

  return { factions, locations, relics, curses };
}

/** All achievement definitions (for UI display of locked achievements) */
export const ACHIEVEMENTS = ACHIEVEMENT_DEFS.map(({ id, title }) => ({ id, title }));
