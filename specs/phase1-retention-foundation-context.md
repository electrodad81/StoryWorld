# StoryWorld — Phase 1 Retention Foundation Context

**Date:** 2026-03-02
**Commit:** `82c888f` (on `main`)
**Previous commit:** `3fa2d49` (Add Exploration Mode)

This document captures the Phase 1 Retention Foundation implementation for Claude Chat context transfer.

---

## Written Summary

### What was built

Phase 1 of the product roadmap ("Retention Foundation") is now implemented. This gives players a reason to come back by connecting story playthroughs and exploration sessions into persistent progression.

### New systems

**1. Player Journal (`src/services/journal.js`, 253 lines)**

A localStorage-based persistence layer tracking everything across sessions:
- **Story completions** — hook played, outcome (survived/died), beat reached, scene count, key choices, factions/locations/relics/curses referenced
- **Lore discoveries** — merged from both story text (keyword extraction) and exploration (location visits, NPC conversations)
- **Statistics** — total stories, deaths, survival rate, total scenes
- **6 achievements** — Mortal After All (first death), Against the Dark (first survival), Seasoned Traveler (3 stories), Voices of the Vale (all 3 NPCs), Between Powers (all factions), Dust to Dust (5 deaths)
- **Unlock tracking** — which hooks and exploration mode are available

Key functions: `loadJournal()`, `saveJournal()`, `addStoryEntry()`, `addExploreLore()`, `checkUnlocks()`, `extractLoreFromHistory()`

**2. Hook Unlock System**

4 of 6 hooks are now locked behind progression:

| Hook | Condition |
|------|-----------|
| HOOK-01 "Bells in the Fog" | Always unlocked |
| HOOK-02 "The Last Oathkeeper" | Always unlocked |
| HOOK-03 "Lanterns for the Dead" | Complete any story |
| HOOK-04 "White Sigils, Black Debts" | Survive a story |
| HOOK-05 "Drowned Star's Whisper" | Die in a story |
| HOOK-06 "Veilbound Audit" | Complete 3 stories |

Locked hooks show as dimmed cards with lock icon and unlock hint text in `CharacterSetup.jsx`.

**3. Exploration Mode Gating**

Explore mode is locked on the mode select screen until the player completes at least 1 story. Shows dimmed card with "Complete a story to unlock the Withered Vale".

**4. Story Complete Screen (`src/components/StoryComplete.jsx`, 146 lines)**

End-of-story summary that appears after death or resolution:
- Outcome banner (gold "You Survived" / red "You Died")
- Hook title, key stats (scenes, beat reached, total stories)
- Key choices (last 4 player decisions)
- New unlock notifications (hooks + exploration mode)
- Achievement badges earned this run
- Actions: Play Again, View Journal, Return to Menu
- Writes journal entry on mount (ref-guarded against React StrictMode double-mount)

**5. Story Engine `complete` Phase**

`useStoryEngine.js` now has:
- `isResolution` state — set when beat reaches `resolution`, suppresses choice generation
- `completeStory()` — transitions to `phase: 'complete'`
- Death screen shows "Continue" button (to complete screen)
- Resolution scene shows "Story Complete" button (to complete screen)

**6. Journal UI (`src/components/JournalView.jsx`, 212 lines)**

Three-tab journal accessible from mode select:
- **Stories tab** — Chronological entries, expandable to show choices/factions/beat
- **Codex tab** — Lore organized by category (Locations, Factions, NPCs, Relics, Curses) with descriptions from lore.json
- **Stats tab** — 2x2 stats grid + achievement badges (earned vs locked)

**7. Exploration → Journal Integration**

`useExploreEngine.js` now writes to journal:
- `travelTo()` → adds location ID to `loreDiscovered.locations`
- `engageNpc()` → adds NPC name to `loreDiscovered.npcs` + faction to `loreDiscovered.factions`

**8. Journal Card on Mode Select**

Mode select now has 3 cards: Story, Explore (gated), Journal. Journal card shows empty state or active state based on entries.

**9. Dev Reset Buttons**

Two buttons at bottom of mode select for testing:
- "Reset Journal" — clears journal only
- "Reset Everything" — clears journal + explore state + story state (back to fresh start)

### Data flow

```
Story ends (death or resolution)
  → StoryView shows "Continue" / "Story Complete" button
  → engine.completeStory() sets phase to 'complete'
  → App.jsx renders StoryComplete
  → StoryComplete.useEffect writes journal entry via addStoryEntry()
    → addStoryEntry updates stats, merges lore, checks unlocks, evaluates achievements
    → Saves to localStorage under 'gloamreach_journal'
  → User sees summary with unlocks/achievements
  → "Return to Menu" refreshes journal from localStorage, shows updated mode select
```

```
Explore mode actions
  → travelTo() / engageNpc() call addExploreLore()
  → Journal updated in localStorage
  → On return to menu, refreshJournal() reads updated state
```

---

## Current Architecture

```
src/
├── App.jsx                    — Routes: API key → Mode select → Story|Explore|Journal|Complete
├── engine/
│   ├── StoryEngine.js         — Narrative generation + illustration (unchanged)
│   ├── ExploreEngine.js       — NPC dialogue + interaction generation (unchanged)
│   ├── prompts.js             — Story mode prompt templates
│   └── explorePrompts.js      — (dormant)
├── hooks/
│   ├── useStoryEngine.js      — Story state + 'complete' phase + isResolution
│   └── useExploreEngine.js    — Explore state + journal writes on travel/NPC
├── components/
│   ├── ApiKeyInput.jsx
│   ├── CharacterSetup.jsx     — MODIFIED: locked hook cards + unlock hints
│   ├── ModeSelect.jsx         — MODIFIED: explore gating, journal card, dev resets
│   ├── StoryView.jsx          — MODIFIED: Continue/Story Complete buttons
│   ├── StoryComplete.jsx      — NEW: end-of-story summary + journal write
│   ├── JournalView.jsx        — NEW: 3-tab journal UI
│   ├── ExploreView.jsx
│   ├── ExploreMap.jsx
│   ├── LocationView.jsx
│   ├── NPCConversation.jsx
│   ├── ChoiceGrid.jsx
│   ├── Sidebar.jsx
│   ├── StreamingText.jsx
│   └── LanternLoader.jsx
├── services/
│   ├── journal.js             — NEW: journal data layer (localStorage)
│   ├── openai.js              — Raw fetch + SSE streaming (unchanged)
│   ├── persistence.js         — Story snapshots (unchanged)
│   └── identity.js            — Browser ID (unchanged)
└── index.css                  — +547 lines for journal, complete, locked states, dev resets
```

---

## Journal Data Shape

```json
{
  "playerId": "brw-uuid",
  "storiesPlayed": [
    {
      "hookId": "HOOK-01",
      "hookTitle": "Bells in the Fog",
      "outcome": "died",
      "beatReached": "rising_action",
      "sceneCount": 9,
      "keyChoices": ["Confront the bell-keeper", "Rush into the mist"],
      "factionsEncountered": ["lantern_guild"],
      "locationsReferenced": ["gloamreach"],
      "timestamp": "2026-03-02T..."
    }
  ],
  "loreDiscovered": {
    "locations": ["gloamreach", "vale-crossroads"],
    "factions": ["lantern_guild"],
    "relics": [],
    "curses": [],
    "npcs": ["Maren"]
  },
  "stats": {
    "totalStories": 1,
    "totalDeaths": 1,
    "totalScenes": 9,
    "survivalRate": 0
  },
  "achievements": [
    { "id": "first_death", "title": "Mortal After All", "unlockedAt": "2026-03-02T..." }
  ],
  "unlockedHooks": ["HOOK-01", "HOOK-02", "HOOK-03", "HOOK-05"],
  "explorationUnlocked": true
}
```

---

## Pending / Not Yet Built (from Roadmap Phase 2+)

- Archetype system (only "Default")
- Hidden locations (data fields exist, logic not implemented)
- Inventory system
- Player states (Wounded, Blessed, Cursed, etc.)
- Story ↔ Explore shared state (independent modes, no crossover yet)
- Backend / Neon DB (all localStorage)
- Premium regions (Blackfen Marsh, Emberfall City, Ironwood Frontier)
- Premium hook packs
- Content gating / payment system
- Story cards (shareable)
- World statistics (aggregate)

---

## Stats

```
14 files changed, 2207 insertions(+), 35 deletions(-)
```

New files: `journal.js` (253), `StoryComplete.jsx` (146), `JournalView.jsx` (212)
Modified: `App.jsx`, `CharacterSetup.jsx`, `ModeSelect.jsx`, `StoryView.jsx`, `useStoryEngine.js`, `useExploreEngine.js`, `lore.json`, `index.css`

---

## Full Diff

The complete code diff follows below. CSS diff is truncated for readability (547 new lines of styles for locked cards, story complete, journal view, dev resets).

### `public/data/lore.json` — Added unlockCondition to hooks

```diff
-      "coverImage": "/images/hook-01.png"
+      "coverImage": "/images/hook-01.png",
+      "unlockCondition": null

-      "coverImage": "/images/hook-02.png"
+      "coverImage": "/images/hook-02.png",
+      "unlockCondition": null

-      "coverImage": "/images/hook-03.png"
+      "coverImage": "/images/hook-03.png",
+      "unlockCondition": { "type": "stories_completed", "min": 1 }

-      "coverImage": "/images/hook-04.png"
+      "coverImage": "/images/hook-04.png",
+      "unlockCondition": { "type": "stories_survived", "min": 1 }

-      "coverImage": "/images/hook-05.png"
+      "coverImage": "/images/hook-05.png",
+      "unlockCondition": { "type": "stories_died", "min": 1 }

-      "coverImage": "/images/hook-06.png"
+      "coverImage": "/images/hook-06.png",
+      "unlockCondition": { "type": "stories_completed", "min": 3 }
```

### `src/App.jsx` — Journal state, routing, gating

```diff
+import { loadJournal, saveJournal } from './services/journal.js';
+import StoryComplete from './components/StoryComplete.jsx';
+import JournalView from './components/JournalView.jsx';

-  const [gameMode, setGameMode] = useState(null); // 'story' | 'explore' | null
+  const [gameMode, setGameMode] = useState(null); // 'story' | 'explore' | 'journal' | null
+  const [journal, setJournal] = useState(() => loadJournal());
+
+  const refreshJournal = useCallback(() => {
+    setJournal(loadJournal());
+  }, []);

+  const handleReturnToMenu = useCallback(() => {
+    setSidebarOpen(false);
+    setGameMode(null);
+    refreshJournal();
+  }, [refreshJournal]);

   // Mode select now receives journal unlock state + reset callbacks
-  <ModeSelect onSelectMode={setGameMode} />
+  <ModeSelect
+    onSelectMode={setGameMode}
+    explorationUnlocked={journal.explorationUnlocked}
+    hasJournalEntries={journal.storiesPlayed.length > 0}
+    onResetJournal={() => { localStorage.removeItem('gloamreach_journal'); refreshJournal(); }}
+    onResetAll={() => { /* clears journal + explore + story state */ }}
+  />

   // New routes for journal and story complete
+  if (gameMode === 'journal') return <JournalView ... />;
+  if (gameMode === 'story' && engine.phase === 'complete') return <StoryComplete ... />;

   // CharacterSetup now receives unlockedHooks
-  <CharacterSetup lore={engine.lore} onBegin={engine.startGame} />
+  <CharacterSetup lore={engine.lore} unlockedHooks={journal.unlockedHooks} onBegin={engine.startGame} />

   // StoryView receives new props
+  isResolution={engine.isResolution}
+  onComplete={engine.completeStory}
```

### `src/hooks/useStoryEngine.js` — Complete phase + resolution detection

```diff
+  const [isResolution, setIsResolution] = useState(false);

   // In runTurn, after beat advance:
+  if (newBeat === 'resolution') {
+    setChoices([]);
+    setIsResolution(true);
+    setIsGenerating(false);
+    return;
+  }

+  const completeStory = useCallback(() => {
+    setPhase('complete');
+  }, []);

   // resetGame and newStory now also clear isResolution
+  setIsResolution(false);

   // New exports:
+  isResolution, completeStory, beatIndex
```

### `src/components/StoryView.jsx` — Continue / Story Complete buttons

```diff
+  isResolution,
+  onComplete,

   // Death screen button changed:
-  <button onClick={onNewStory}>Start New Story</button>
+  <button onClick={onComplete}>Continue</button>

   // Resolution screen added:
+  {isResolution && !isDead && (
+    <div className="resolution-screen">
+      <button className="sidebar-btn accent" onClick={onComplete}>Story Complete</button>
+    </div>
+  )}
```

### `src/components/CharacterSetup.jsx` — Locked hooks

```diff
+const UNLOCK_HINTS = {
+  stories_completed: (min) => min === 1 ? 'Complete any story to unlock' : `Complete ${min} stories to unlock`,
+  stories_survived: (min) => min === 1 ? 'Survive a story to unlock' : `Survive ${min} stories to unlock`,
+  stories_died: (min) => min === 1 ? 'Die in a story to unlock' : `Die ${min} times to unlock`,
+};

-export default function CharacterSetup({ lore, onBegin }) {
+export default function CharacterSetup({ lore, unlockedHooks, onBegin }) {

   // Auto-select first unlocked hook instead of first hook
   // Locked hooks render with .locked class, lock icon, hint text instead of blurb
```

### `src/components/ModeSelect.jsx` — Gating + Journal + Dev resets

```diff
-export default function ModeSelect({ onSelectMode }) {
+export default function ModeSelect({ onSelectMode, explorationUnlocked, hasJournalEntries, onResetJournal, onResetAll }) {

   // Explore card: locked class + lock icon + hint when !explorationUnlocked
   // Journal card: new third option with empty/active state
   // Dev reset buttons at bottom: Reset Journal / Reset Everything
```

### `src/hooks/useExploreEngine.js` — Journal writes

```diff
+import { loadJournal, addExploreLore } from '../services/journal.js';

   // After travelTo discovers a location:
+  try {
+    const journal = loadJournal();
+    addExploreLore(journal, { locations: [locationId] });
+  } catch {}

   // After engageNpc:
+  try {
+    const journal = loadJournal();
+    const loreUpdate = { npcs: [npc.name] };
+    if (npc.faction) loreUpdate.factions = [npc.faction];
+    addExploreLore(journal, loreUpdate);
+  } catch {}
```

### New files (full source in repo)

- **`src/services/journal.js`** (253 lines) — Complete journal data layer
- **`src/components/StoryComplete.jsx`** (146 lines) — End-of-story summary
- **`src/components/JournalView.jsx`** (212 lines) — Three-tab journal UI

### `src/index.css` — +547 lines

New CSS sections: `.mode-card-header`, `.mode-hint`, `.dev-resets`, `.dev-reset-btn`, locked state (`.mode-card.locked`, `.hook-card.locked`, `.lock-icon`, `.hook-hint`), `.resolution-screen`, `.sidebar-btn.accent`, story complete (`.story-complete-panel`, `.outcome-banner`, `.complete-stats`, `.complete-section`, `.unlock-item`, `.achievement-item`), journal (`.journal-panel`, `.journal-tabs`, `.journal-tab`, `.journal-stories`, `.journal-story-card`, `.outcome-badge`, `.codex-*`, `.stats-grid`, `.stat-card`, `.achievements-grid`, `.achievement-card`).
