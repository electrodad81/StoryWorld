# StoryWorld — Codebase Catchup Context

**Date:** 2026-03-02
**Branch:** `main`
**Base commit:** `cb6fe19` (Migrate StoryWorld from Streamlit to React + Vite SPA)

This document captures all changes made across 7 commits since the initial React migration. Use it to get caught up on the current state of the codebase.

---

## Written Summary

### What happened since the migration

After the core Streamlit → React + Vite migration (`cb6fe19`), the project went through two major phases:

**Phase 1: Immersive UI Overhaul (4 commits)**

The original React UI used a static three-panel layout (illustration / text / choices). This was completely replaced with a full-bleed immersive viewport:

- **Full-viewport background illustrations** — DALL-E images now fill the entire screen with a crossfade transition between scenes. A persistent background state ensures the screen never goes blank.
- **Gradient scrim + frosted-glass storybox** — Text floats over the illustration with a `backdrop-filter: blur(8px)` frosted card and a gradient scrim for legibility.
- **Glassmorphism choice buttons** — Pinned at the bottom of the viewport with semi-transparent backgrounds.
- **Drawer sidebar** — The static 260px sidebar was replaced with a slide-over drawer toggled by a hamburger button in the top bar.
- **Pre-generated cover art** — 6 static hook cover images generated via `scripts/generate-covers.js` using DALL-E 3, stored in `public/images/`.
- **Dark moody art style** — All DALL-E prompts (runtime + static) were updated to: detailed ink, rich crosshatching, deep shadows, edge-to-edge detail, no white backgrounds, no text/runes, borderless/frameless, 104% scale with -2% inset to crop residual borders.
- **Scene brevity** — Narrative word count target reduced to 60–90 words, `max_tokens` to 200.
- **Death screen** restyled with a subtle `border-left` accent instead of a double-glass card.

**Phase 2: Exploration Mode (1 commit)**

A fully functional MVP exploration mode was added alongside Story Mode:

- **Mode selection screen** (`ModeSelect.jsx`) — After API key entry, players choose Story or Explore mode.
- **World data** — `public/data/withered-vale.json` defines the Withered Vale region: 3 locations (The Crossroads, Thornwatch Inn, The Old Mill), 3 NPCs (Maren, Edric, Sibyl), 3 factions (Lantern Guild, Pale Court, Veilbound Inquisition).
- **SVG node map** (`ExploreMap.jsx`) — Hand-positioned nodes with dashed path edges, pulse animation on current location, fog-of-war discovery.
- **Location detail view** (`LocationView.jsx`) — Cover image header, location description, NPC cards with portraits and disposition badges, travel buttons.
- **NPC conversation** (`NPCConversation.jsx`) — AI-driven dialogue via streaming SSE, disposition tracking, dynamically generated interaction options, "Walk away" exit.
- **Exploration engine** (`useExploreEngine.js`, 317 lines) — Full state management hook: travel, NPC engagement, conversation history, disposition changes, faction rep, localStorage persistence.
- **ExploreEngine.js** rewritten — Now a clean service layer with `generateNpcResponse()` and `generateInteractionOptions()` using the existing `openai.js` streaming infrastructure.
- **3 location cover images + 3 NPC portraits** generated and stored in `public/images/explore/`.

### Current architecture

```
src/
├── App.jsx                    — Routes: API key → Mode select → Story|Explore
├── engine/
│   ├── StoryEngine.js         — Narrative generation + illustration
│   ├── ExploreEngine.js       — NPC dialogue + interaction generation
│   ├── prompts.js             — Story mode prompt templates
│   └── explorePrompts.js      — (dormant, not used by new engine)
├── hooks/
│   ├── useStoryEngine.js      — Story mode state hook
│   └── useExploreEngine.js    — Explore mode state hook (NEW)
├── components/
│   ├── ApiKeyInput.jsx
│   ├── CharacterSetup.jsx
│   ├── ModeSelect.jsx         — NEW: Story vs Explore picker
│   ├── StoryView.jsx          — Immersive full-bleed story viewport
│   ├── ExploreView.jsx        — NEW: Explore mode root
│   ├── ExploreMap.jsx         — NEW: SVG node map
│   ├── LocationView.jsx       — NEW: Location detail + NPC cards
│   ├── NPCConversation.jsx    — NEW: Streaming NPC dialogue
│   ├── ChoiceGrid.jsx
│   ├── Sidebar.jsx
│   ├── StreamingText.jsx
│   └── LanternLoader.jsx
├── services/
│   ├── openai.js              — Raw fetch + SSE streaming
│   ├── persistence.js         — localStorage (Neon DB later)
│   └── identity.js
└── index.css                  — 1010 lines (immersive + explore styles)

public/
├── data/
│   ├── lore.json              — World lore + hooks (now with coverImage paths)
│   └── withered-vale.json     — NEW: Region/location/NPC/faction data
└── images/
    ├── hook-01..06.png        — Static story hook cover art
    └── explore/
        ├── vale-crossroads.png
        ├── thornwatch-inn.png
        ├── old-mill.png
        ├── npc-maren.png
        ├── npc-edric.png
        └── npc-sibyl.png

scripts/
├── generate-covers.js         — DALL-E 3 cover generator for story hooks
├── generate-explore-covers.js — DALL-E 3 location cover generator
└── generate-npc-portraits.js  — DALL-E 3 NPC portrait generator
```

### Known pending work

- **Neon DB persistence** — `persistence.js` is still a localStorage stub
- **Archetype system** — Only "Default" archetype exists for story mode
- **Faction system depth** — Faction rep is tracked but doesn't gate content yet
- **Hidden locations** — `isHidden`/`unlockCondition` fields exist in data but aren't implemented
- **Inventory** — Not implemented
- **Player states** — Not implemented (Wounded, Blessed, Cursed, etc.)
- **Story ↔ Explore integration** — The two modes are independent; no shared state yet

---

## Commit Log

### `3fa2d49` — Add Exploration Mode: map, locations, NPC dialogue, portraits
- Mode selection screen (Story vs Explore) after API key entry
- Withered Vale region with 3 locations and 3 NPCs
- SVG node map with pulse animation on current location
- Location detail view with cover images, NPC cards, travel options
- AI-driven NPC dialogue with streaming, disposition tracking, faction rep
- NPC portraits (circular, dark ink style) in location and conversation views
- Exploration state persisted to localStorage
- Generated 3 location covers and 3 NPC portrait images
- New engine, hook, and 5 components for exploration mode

### `d36ab03` — Add final UI polish changelog covering image borders and dark style

### `f79af32` — Dark moody image style, remove white backgrounds, regenerate covers
- Replaced simple/clean style with detailed ink, rich crosshatching, deep tonal range
- Removed "Plain white background" and "single focal subject" directives
- Added dark atmosphere, deep shadows, edge-to-edge environmental detail

### `3686e8a` — Borderless image prompts, 104% bg scale, regenerated covers
- Added borderless/frameless directive to all DALL-E prompts
- Scaled `.bg-layer img` to 104% with inset -2% to crop residual borders

### `50e52ee` — Add UI polish changelog with full diffs

### `6fad244` — UI polish: frosted storybox, text overlap fixes, no-text image regeneration
- Frosted-glass card behind story text (opacity 0.35, side margins)
- Removed "Your choices" heading from choice grid
- Increased text-overlay clearance (120px desktop, 140px mobile)
- Death screen restyled with border-left accent
- Added no-text directive to all DALL-E prompts
- Reduced scene word count to 60–90, max_tokens to 200

### `0d84652` — Immersive full-bleed UI overhaul with pre-generated cover art
- Full-viewport background illustrations with crossfade transitions
- Glassmorphism choice buttons pinned at bottom with gradient scrim
- Drawer sidebar with hamburger toggle
- Pre-generated static cover images for all 6 starter hooks
- Portrait DALL-E images (1024×1792) for mobile-first layout
- Removed stale exploration mode Python files

---

## Key Diffs

### `src/App.jsx` — Mode routing + drawer sidebar

```diff
-import { useEffect } from 'react';
+import { useState, useEffect } from 'react';
 import useStoryEngine from './hooks/useStoryEngine.js';
+import useExploreEngine from './hooks/useExploreEngine.js';
 import ApiKeyInput from './components/ApiKeyInput.jsx';
+import ModeSelect from './components/ModeSelect.jsx';
 import CharacterSetup from './components/CharacterSetup.jsx';
 import StoryView from './components/StoryView.jsx';
+import ExploreView from './components/ExploreView.jsx';
 import Sidebar from './components/Sidebar.jsx';

 export default function App() {
   const engine = useStoryEngine();
+  const explore = useExploreEngine();
+  const [sidebarOpen, setSidebarOpen] = useState(false);
+  const [gameMode, setGameMode] = useState(null); // 'story' | 'explore' | null

   useEffect(() => {
     engine.loadLore();
+    explore.loadWorldData();
   }, []);

   // After API key: mode selection
+  if (!gameMode) {
+    return <ModeSelect onSelectMode={setGameMode} />;
+  }

   // Explore mode renders ExploreView with drawer sidebar
   // Story mode renders StoryView with Sidebar in drawer
```

### `src/components/StoryView.jsx` — Immersive full-bleed viewport

The component was rewritten from a three-panel layout to layered viewport:
- **Layer 0**: Full-viewport background image with crossfade (`persistedBg` → `incomingSrc`)
- **Layer 1**: Gradient scrim (`rgba(10,10,20)` from 92% at bottom to transparent at 75%)
- **Layer 2**: Text overlay (`.storybox` with frosted glass) + choice overlay
- **Layer 3**: Top bar with hamburger menu + beat indicator

Key state additions:
```jsx
const [persistedBg, setPersistedBg] = useState(null);
const [incomingSrc, setIncomingSrc] = useState(null);
const [showIncoming, setShowIncoming] = useState(false);
```

### `public/data/withered-vale.json` (NEW)

Full region data file:
```json
{
  "region": { "id": "withered-vale", "name": "The Withered Vale", "description": "..." },
  "locations": [
    { "id": "vale-crossroads", "name": "The Crossroads", "connections": ["thornwatch-inn", "old-mill"], "coverImage": "/images/explore/vale-crossroads.png" },
    { "id": "thornwatch-inn", "name": "Thornwatch Inn", "connections": ["vale-crossroads", "old-mill"], "coverImage": "/images/explore/thornwatch-inn.png" },
    { "id": "old-mill", "name": "The Old Mill", "connections": ["vale-crossroads", "thornwatch-inn"], "coverImage": "/images/explore/old-mill.png" }
  ],
  "npcs": [
    { "id": "maren", "name": "Maren", "locationId": "thornwatch-inn", "archetype": "guarded innkeeper", "faction": "lantern_guild", "baseDisposition": 40, "portrait": "/images/explore/npc-maren.png" },
    { "id": "edric", "name": "Edric", "locationId": "vale-crossroads", "archetype": "wandering oathbreaker", "faction": null, "baseDisposition": 55, "portrait": "/images/explore/npc-edric.png" },
    { "id": "sibyl", "name": "Sibyl", "locationId": "old-mill", "archetype": "reclusive herbalist", "faction": "pale_court", "baseDisposition": 30, "portrait": "/images/explore/npc-sibyl.png" }
  ],
  "factions": {
    "lantern_guild": { "name": "Lantern Guild" },
    "pale_court": { "name": "The Pale Court" },
    "inquisition": { "name": "Veilbound Inquisition" }
  }
}
```

### `src/hooks/useExploreEngine.js` (NEW, 317 lines)

State hook managing all explore mode state:
- `worldData`, `currentLocationId`, `discoveredLocations` — world/map state
- `npcDispositions`, `factionRep` — reputation tracking
- `explorePhase` — `'map' | 'location' | 'conversation'`
- `activeNpc`, `conversationHistory`, `streamedText` — NPC dialogue state
- `interactionOptions` — AI-generated dialogue choices
- Methods: `loadWorldData()`, `travelTo(id)`, `engageNpc(id)`, `talkToNpc(option)`, `leaveConversation()`, `openMap()`, `resetExplore()`
- All state persisted to localStorage under `storyworld_explore_*` keys

### `src/engine/ExploreEngine.js` (rewritten, 115 lines)

Service layer with two main functions:
- `generateNpcResponse({ npc, location, playerMessage, conversationHistory, disposition, apiKey })` — Streams NPC dialogue with personality/faction/disposition constraints
- `generateInteractionOptions({ npc, conversationHistory, disposition, apiKey })` — Returns 3 contextual dialogue options as JSON array

### New Components

| Component | Lines | Purpose |
|-----------|-------|---------|
| `ModeSelect.jsx` | 24 | Two-card mode picker (Story vs Explore) |
| `ExploreView.jsx` | 59 | Explore mode root — routes between map/location/conversation |
| `ExploreMap.jsx` | 73 | SVG node map with hand-tuned positions, edges, pulse animation |
| `LocationView.jsx` | 84 | Location header image, NPC cards with portraits, travel buttons |
| `NPCConversation.jsx` | 85 | NPC header, streaming dialogue display, interaction options |

### `src/index.css` — Major additions (622→1010 lines)

Key CSS changes:
- **Removed**: `.panel-illustration`, `.panel-text`, `.panel-choices`, `.illus-sep`, static `.app-sidebar`
- **Added**: `.story-viewport` (layered absolute positioning), `.bg-layer` (104% scale, crossfade), `.scrim`, `.text-overlay`, `.choice-overlay`, `.top-bar`, `.menu-btn`, `.drawer`/`.drawer-backdrop`
- **Added**: Full exploration mode styles — `.explore-viewport`, `.explore-map-*`, `.map-node`/`.map-edge`, `.location-*`, `.npc-card`, `.npc-conversation`, `.npc-header-portrait`, `.mode-select-*`
- **Modified**: `.storybox` got frosted glass (`backdrop-filter: blur(8px)`, `rgba(10,10,20,0.35)` bg)
- **Modified**: `.choice-btn` got glassmorphism styling

### `src/engine/StoryEngine.js` — DALL-E prompt updates

```diff
 // basicStyle() now includes:
+"Dark, moody atmosphere — no white or bright backgrounds. Deep shadows and low ambient light."
+"The image must be borderless and frameless"
+"Absolutely no text, letters, words..."

 // Scene generation:
-"Write about 100–150 words of atmospheric prose"
+"Write about 60–90 words of atmospheric prose"
-max_tokens: 300
+max_tokens: 200
```

### Deleted files

- `story/explore_engine.py` (203 lines) — Legacy Python explore engine
- `story/pump_explore.py` (83 lines) — Legacy Python explore driver
- `story/pump_romance.py` (71 lines) — Legacy Python romance driver
- `explore_v2/devtools.py` (67 lines) — Legacy devtools

---

## Stats

```
46 files changed, 4990 insertions(+), 695 deletions(-)
```

Binary assets added: 6 hook covers (~3–4 MB each), 3 location covers, 3 NPC portraits.
