# Gloamreach — Product Roadmap (Updated 2026-03-02)

## Vision

Gloamreach is an authored dark-fantasy world that uses AI as its rendering engine. Players explore, make consequential choices, build relationships, and accumulate a persistent history that reshapes their version of the world over time. It's closer to Fallen London or 80 Days than to AI Dungeon — the world is designed, not generated.

**Core positioning:** A curated interactive fiction experience with authored world depth, persistent progression, and AI-driven prose. Free to start, with premium world content for players who want more.

---

## Product Principles

1. **The world is the product.** Lore, factions, NPCs, locations, tone rules, and consequence systems are the authored value. AI renders the prose but the design is what players pay for.
2. **Every session changes the world.** No playthrough is throwaway. Reputation shifts, lore discoveries, NPC memories, and unlocked paths carry forward.
3. **Depth over breadth.** Three deeply developed locations with rich NPC interactions beat ten shallow ones. Quality of world-feel is the differentiator.
4. **Free is complete, paid is more.** The free experience must be satisfying on its own — a full region, multiple story arcs, meaningful progression. Paid content expands the world laterally, not vertically.

---

## Current State (as of commit `3fa2d49`)

### What's built and working

**Story Mode (complete):**
- 5-beat narrative arc (~13 scenes), binary choices, consequence contract, death mechanic, danger streak
- Immersive full-bleed UI: DALL-E backgrounds with crossfade, frosted-glass storybox, glassmorphism choices
- Pre-generated cover images for all 6 starter hooks
- Dark moody art style: detailed ink, borderless/frameless, no text/runes, portrait orientation (1024×1792)
- Scenes at ~60–90 words, max_tokens 200
- Drawer sidebar with hamburger toggle

**Exploration Mode (complete MVP):**
- Mode selection screen (ModeSelect.jsx) after API key entry
- The Withered Vale: 3 locations, 3 NPCs (with portraits), 3 factions
- SVG node map with pulse animation and fog-of-war discovery
- Location detail views with cover images, NPC cards, disposition badges, travel buttons
- AI-driven NPC dialogue: streaming via `generateNpcResponse()`, disposition-aware
- Dynamic interaction options via `generateInteractionOptions()`
- Disposition tracking, faction reputation tracking
- Full localStorage persistence for exploration state

**Infrastructure:**
- React + Vite SPA, client-side only
- OpenAI API (BYOK) via raw fetch + SSE streaming (`openai.js`)
- localStorage persistence (snapshots, browser ID, explore state)
- Lore system: `lore.json` (story mode) + `withered-vale.json` (explore mode)
- DALL-E image generation scripts for covers and portraits

### What's NOT built yet

- Player Journal (cross-session progression tracking)
- Hook unlock system (all 6 hooks available immediately)
- Exploration mode gating (both modes available immediately, should require story completion)
- Story Complete screen (no end-of-story summary or journal entry creation)
- Story ↔ Explore integration (the two modes share no state)
- Archetype system (only "Default")
- Hidden locations (data fields exist, logic not implemented)
- Inventory system
- Player states (Wounded, Blessed, Cursed, etc.)
- Backend / database (all localStorage)
- Payment system

---

## Phase 1: Retention Foundation ← BUILD THIS NEXT

**Goal:** Give players a reason to come back. Connect individual story playthroughs and exploration sessions into an ongoing relationship with the world through persistent progression.

**Dependency order:** Journal data layer → Hook unlock logic → Story Complete screen → Exploration gating → Journal UI

### 1A. Player Journal

A persistent record that survives across stories and exploration sessions. Stored in localStorage under a dedicated key (e.g., `gloamreach_journal`). The data shape is designed to migrate cleanly to a backend later.

**What the journal tracks:**

- **Story completions:** Hook played, outcome (survived/died), beat reached, scene count, key choices made
- **Lore discoveries:** Locations visited, faction encounters, relics mentioned, curses experienced, NPC names encountered
- **Exploration log:** Locations discovered, NPCs spoken to, faction rep changes
- **Statistics:** Total stories played, deaths, survival rate, total scenes experienced
- **Achievements:** Milestone markers (first death, first survival, met all NPCs, etc.)
- **Unlocks:** Which hooks are available, whether exploration mode is unlocked

**Journal data structure:**

```json
{
  "playerId": "brw-uuid",
  "storiesPlayed": [
    {
      "hookId": "HOOK-01",
      "hookTitle": "Bells in the Fog",
      "outcome": "survived",
      "beatReached": "resolution",
      "sceneCount": 13,
      "keyChoices": ["Investigated the bell tower", "Confronted the mist-walker"],
      "factionsEncountered": ["lantern_guild"],
      "locationsReferenced": ["gloamreach", "sunless_abbey"],
      "timestamp": "2026-03-02T..."
    }
  ],
  "loreDiscovered": {
    "locations": ["gloamreach", "mire_of_thorns", "vale-crossroads"],
    "factions": ["lantern_guild", "pale_court"],
    "relics": ["ink_of_concord"],
    "curses": ["salt_tongue"],
    "npcs": ["Maren", "Edric", "Sibyl"]
  },
  "stats": {
    "totalStories": 3,
    "totalDeaths": 1,
    "totalScenes": 34,
    "survivalRate": 0.67
  },
  "achievements": [
    { "id": "first_death", "title": "Mortal After All", "unlockedAt": "2026-03-02T..." },
    { "id": "first_survival", "title": "Against the Dark", "unlockedAt": "..." }
  ],
  "unlockedHooks": ["HOOK-01", "HOOK-02"],
  "explorationUnlocked": false
}
```

**Implementation — new files:**

`src/services/journal.js` — Pure data layer for reading/writing journal state:
- `loadJournal()` → returns journal object or creates default
- `saveJournal(journal)` → writes to localStorage
- `addStoryEntry(journal, storyData)` → appends a story completion, updates stats and lore
- `addExploreLore(journal, { locations, npcs, factions })` → merges explore discoveries
- `checkUnlocks(journal)` → returns `{ newHooks: [...], explorationUnlocked: bool }` based on current journal state
- `getAchievements(journal)` → evaluates achievement conditions, returns any newly earned

**Achievement definitions** (hardcoded in journal.js for now):

| ID | Title | Condition |
|----|-------|-----------|
| `first_death` | Mortal After All | Die in any story |
| `first_survival` | Against the Dark | Survive any story to resolution |
| `three_stories` | Seasoned Traveler | Complete 3 stories (any outcome) |
| `all_vale_npcs` | Voices of the Vale | Speak to Maren, Edric, and Sibyl in exploration |
| `all_factions` | Between Powers | Encounter all 3 factions across any mode |
| `five_deaths` | Dust to Dust | Die 5 times total |

### 1B. Hook Unlock System

Not all 6 hooks are available from the start.

**Unlock rules:**

| Hook | Default State | Unlock Condition |
|------|--------------|-----------------|
| HOOK-01 "Bells in the Fog" | Unlocked | Always available |
| HOOK-02 "The Last Oathkeeper" | Unlocked | Always available |
| HOOK-03 "Lanterns for the Dead" | Locked | Complete any story (survive or die) |
| HOOK-04 "White Sigils, Black Debts" | Locked | Survive a story (reach resolution without dying) |
| HOOK-05 "Drowned Star's Whisper" | Locked | Die in any story |
| HOOK-06 "Veilbound Audit" | Locked | Complete 3 stories |

**Implementation:**

Add an `unlockCondition` field to each hook in `lore.json`:

```json
{
  "id": "HOOK-03",
  "title": "Lanterns for the Dead",
  "blurb": "...",
  "coverImage": "/images/hook-03.png",
  "unlockCondition": { "type": "stories_completed", "min": 1 }
}
```

Condition types:
- `null` or absent → always unlocked
- `{ "type": "stories_completed", "min": N }` → player has completed N stories
- `{ "type": "stories_survived", "min": N }` → player has survived N stories
- `{ "type": "stories_died", "min": N }` → player has died N times

**UI in CharacterSetup.jsx:**

Locked hooks render as dimmed cards with a lock icon and a hint:
- The hook title is visible (not hidden — players should see what they're working toward)
- The blurb is replaced with the unlock hint: "Complete any story to unlock" / "Survive a story to unlock" / "Die in a story to unlock" / "Complete 3 stories to unlock"
- The card is not clickable
- CSS: `opacity: 0.5; pointer-events: none;` with a small lock icon (🔒 or SVG)

`CharacterSetup.jsx` needs to receive the journal (or just `unlockedHooks` array) as a prop so it can determine which hooks to enable.

### 1C. Exploration Mode Gating

Exploration mode is locked on the mode select screen until the player has completed at least one story.

**Implementation in ModeSelect.jsx:**

The component receives `explorationUnlocked` (boolean) as a prop. When `false`:
- The Explore Mode card renders dimmed with a lock icon
- The description is replaced with: "Complete a story to unlock the Withered Vale"
- The card is not clickable

When `true`, it works as it does today.

**Data flow:** `App.jsx` loads the journal on mount, reads `journal.explorationUnlocked`, and passes it to `ModeSelect`.

### 1D. Story Complete Screen

A new screen that appears when a story ends (either via the resolution beat or via death). This is the critical integration point — it's where the journal gets written and unlocks are evaluated.

**New component: `StoryComplete.jsx`**

Triggered when:
- The story reaches the `resolution` beat and the final scene is generated (survived)
- The player dies (isDead becomes true)

**What it shows:**
- **Outcome banner:** "You Survived" (gold) or "You Died" (red)
- **Hook title:** Which story was played
- **Key stats:** Scenes played, beat reached, danger streak peak
- **Key choices:** The player's last 3–4 choices (extracted from history where `role === 'user'`)
- **Lore discovered:** Any faction/location/relic names detected in the story text
- **New unlocks:** If the journal check reveals newly unlocked hooks or exploration mode, show them with emphasis: "NEW: Unlocked 'Lanterns for the Dead'" / "NEW: The Withered Vale is now open to explore"
- **Achievements earned:** Any new achievements from this run
- **Buttons:** "Play Again" (→ character setup) / "Return to Menu" (→ mode select) / "View Journal" (→ journal UI)

**Lore extraction from story history:**

To populate `factionsEncountered` and `locationsReferenced` in the journal entry, scan the full story history for known lore keywords:

```js
function extractLoreFromHistory(history, lore) {
  const fullText = history.map(m => m.content || '').join(' ').toLowerCase();
  
  const factions = Object.keys(lore.factions || {}).filter(key => {
    const name = (lore.factions[key].name || '').toLowerCase();
    return fullText.includes(key) || fullText.includes(name);
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
```

This is simple keyword matching. It won't catch everything, but it catches enough for a satisfying journal entry.

**Integration with useStoryEngine.js:**

The hook needs a new phase: `'complete'`. When the story ends (resolution or death), instead of staying in `'playing'` with `isDead: true`, transition to `phase: 'complete'` after a brief delay (let the player read the final scene first).

Option A: Auto-transition to complete screen after the death screen is shown for a few seconds.
Option B: Add a "Continue" button on the death screen and on the final resolution scene that transitions to `'complete'`.

**Recommend Option B** — let the player control the transition. On death, the existing death overlay stays but the "Start New Story" button is replaced with "Continue →". On resolution (the final scene has no choices), show a "Story Complete →" button in the choice area.

`App.jsx` then renders `StoryComplete` when `engine.phase === 'complete'`, passing it the story data needed to build the journal entry.

### 1E. Journal UI Component

**New component: `JournalView.jsx`**

Accessible from the mode select screen as a third option (alongside Story and Explore). Shows the player's persistent record across all sessions.

**Layout — three tabs:**

**Tab 1: Stories** — Chronological list of completed stories. Each entry shows: hook title, outcome badge (survived/died), date, scene count. Tapping/clicking an entry expands it to show key choices and lore discovered.

**Tab 2: Codex** — Lore discoveries organized by category: Locations, Factions, Relics, Curses, NPCs. Each item is a simple card with the name and a brief description (pulled from lore.json or withered-vale.json). Undiscovered items can optionally show as "???" to hint at content the player hasn't found yet.

**Tab 3: Stats & Achievements** — Statistics panel (total stories, deaths, survival rate, total scenes) and achievement badges. Locked achievements show as dimmed with their unlock condition visible.

**Styling:** Dark card-based layout matching the existing aesthetic. Use the `var(--bg-card)`, `var(--border)`, `var(--accent)` design tokens. Tab navigation at the top. Scrollable content area.

### 1F. Exploration → Journal Integration

When the player does things in exploration mode, the journal should update:

- **Visiting a location:** Add to `loreDiscovered.locations` if not already present
- **Speaking to an NPC:** Add NPC name to `loreDiscovered.npcs`
- **Faction reputation change:** Add faction to `loreDiscovered.factions`

**Implementation:** In `useExploreEngine.js`, after `travelTo()` and `talkToNpc()` complete, call journal update functions. This requires the explore hook to have access to the journal. Simplest approach: `App.jsx` loads the journal into state, passes journal update callbacks down to the explore components, or the explore hook imports `journal.js` directly and reads/writes localStorage.

Recommend: The explore hook imports `journal.js` directly. This avoids prop-drilling and keeps the two hooks independent (they share localStorage but don't share React state). When the player returns to the mode select screen, the journal is re-read from localStorage and reflects all explore discoveries.

---

## Phase 2: Content Depth and Monetization

**Goal:** Expand the world and introduce the freemium model.

### 2A. Free Tier Content

Everything the player has access to without paying:
- Story Mode with 2 starter hooks always available + 4 unlockable
- Exploration Mode: The Withered Vale (3 locations, 3 NPCs)
- Player Journal with full tracking
- All core mechanics

This must feel like a complete game, not a demo.

### 2B. Premium Regions (Paid Content)

Each premium region is a self-contained expansion:

**Region: Blackfen Marsh** ($2.99–$4.99)
- 4 new locations with pre-generated cover art
- 3 new NPCs with portraits and unique archetypes
- 2 premium story hooks set in the marsh
- New faction interactions (Hollow Choir introduction)
- New relics and curses
- ~2–3 hours of new content

**Region: Emberfall City** ($2.99–$4.99)
- Urban setting, different tone from the Vale
- 5 locations, 4 NPCs
- 3 premium story hooks
- Pale Court headquarters

**Region: Ironwood Frontier** ($2.99–$4.99)
- Wilderness/frontier tone
- 3 locations, 3 NPCs
- 2 premium story hooks
- Inquisition outpost

### 2C. Premium Hook Packs

Standalone story hooks:
- "The Ashen Tribunal" hook pack (3 interconnected hooks, $1.99)
- "Oathbreaker's Path" hook pack (2 hooks following Edric's storyline, $1.49)
- Seasonal/event hooks (limited-time free hooks to drive re-engagement)

### 2D. Content Gating Implementation

All content ships with the app. Premium content is locked behind flags:

```json
{
  "purchases": {
    "region_blackfen": false,
    "region_emberfall": false,
    "hookpack_tribunal": false
  }
}
```

For MVP, a localStorage flag set by unlock code. Real payment integration comes in Phase 3.

---

## Phase 3: Backend Migration

**Build when:** Product is validated with real users, need cross-device sync, ready for real payments.

### Stack
- **Database:** Neon PostgreSQL (serverless, free tier)
- **API:** Vercel Edge Functions or Cloudflare Workers
- **Auth:** Clerk or Auth.js
- **Payments:** Stripe (web) or RevenueCat (native mobile)

### Migration Path
The localStorage persistence interfaces (`persistence.js`, `identity.js`, `journal.js`) are designed to swap backends without changing calling code. The data shapes stay identical.

### API-Key-Free Experience
Backend enables server-side OpenAI calls so players don't need BYOK. Critical for mainstream adoption. Model: Free tier gets N stories/month, paid tier unlimited.

---

## Phase 4: Social and Growth

### Story Cards (Shareable)
After story completion, generate a shareable image card with hook title, outcome, key stat, and Gloamreach branding. Canvas API or server-side generation.

### World Statistics
Aggregate anonymized data: "73% survived Bells in the Fog", "Only 12% found the Ink of Concord". Social proof and "did I miss something?" motivation.

### Seasonal Events
Limited-time hooks or NPC encounters. Drives re-engagement.

---

## Development Sequence

### NOW — Phase 1 Retention Foundation

Build order (by dependency):

1. `src/services/journal.js` — Data layer: load, save, add entries, check unlocks, achievements
2. `lore.json` update — Add `unlockCondition` field to each hook
3. `CharacterSetup.jsx` update — Locked hook UI (dimmed, lock icon, hint text)
4. `ModeSelect.jsx` update — Exploration gating (locked card when not unlocked)
5. `useStoryEngine.js` update — Add `'complete'` phase, transition logic on story end
6. `StoryComplete.jsx` — New component: end-of-story summary, journal write, unlock check
7. `JournalView.jsx` — New component: stories tab, codex tab, stats/achievements tab
8. `App.jsx` update — Load journal on mount, pass unlock state to ModeSelect and CharacterSetup, render JournalView and StoryComplete
9. `useExploreEngine.js` update — Write lore discoveries to journal after travel/NPC interactions

### NEXT — Phase 2 Content

10. Playtesting and world-feel tuning
11. Second region design (Blackfen Marsh)
12. Content gating system
13. Story cards

### LATER — Phase 3–4

14. Backend migration
15. Payment integration
16. API-key-free experience
17. World statistics
18. Additional regions and hook packs
19. Seasonal events

---

## New Files for Phase 1

| File | Purpose |
|------|---------|
| `src/services/journal.js` | Journal data layer (localStorage) |
| `src/components/StoryComplete.jsx` | End-of-story summary + journal integration |
| `src/components/JournalView.jsx` | Journal UI (stories, codex, stats/achievements) |

## Modified Files for Phase 1

| File | Change |
|------|--------|
| `public/data/lore.json` | Add `unlockCondition` to each hook |
| `src/components/CharacterSetup.jsx` | Locked hook cards with hints |
| `src/components/ModeSelect.jsx` | Exploration gating |
| `src/hooks/useStoryEngine.js` | Add `'complete'` phase, transition on story end |
| `src/hooks/useExploreEngine.js` | Write discoveries to journal |
| `src/App.jsx` | Load journal, route to JournalView and StoryComplete, pass unlock state |
| `src/index.css` | Styles for locked cards, journal view, story complete screen |

## Unchanged Files

| File | Why |
|------|-----|
| `src/engine/StoryEngine.js` | No changes needed |
| `src/engine/ExploreEngine.js` | No changes needed |
| `src/services/openai.js` | No changes needed |
| `src/components/StoryView.jsx` | No changes (death screen button behavior changes via hook phase) |
| `src/components/ExploreView.jsx` | No changes needed |
| `src/components/ExploreMap.jsx` | No changes needed |
| `src/components/LocationView.jsx` | No changes needed |
| `src/components/NPCConversation.jsx` | No changes needed |

---

## Success Metrics

### Engagement (Pre-Monetization)
- **Retention:** % of players who play a second story
- **Completion rate:** % of stories reaching resolution
- **Exploration adoption:** % of story-completers who try exploration
- **Session length:** Target 8–15 minutes per story

### Monetization (Post-Launch)
- **Conversion rate:** % of free players who purchase premium content
- **ARPU:** Target $3–5 for engaged users
- **LTV:** Target $8–15 across multiple purchases

---

## Key Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| AI costs unsustainable for free tier | BYOK now; server-side with metering later; short scenes reduce tokens |
| DALL-E slow and inconsistent | Pre-generated covers; persistent backgrounds |
| Players don't return | Journal + unlocks create visible progression; exploration adds open-ended play |
| World feels AI-generated | Strong authored lore and constraints; world design is the moat |
| BYOK limits audience | Phase 3 backend removes this; validate with dev audience first |
| Premium feels like paywall | Free tier is complete; premium is lateral expansion |
