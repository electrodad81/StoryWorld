# Exploration Mode — Phase 1 Spec (Client-Side MVP)

## Goal

Add a separate Exploration Mode to Gloamreach — a persistent, map-based RPG layer where the player moves between locations, interacts with NPCs via AI-generated dialogue, and tracks reputation. Runs entirely client-side using localStorage for persistence and OpenAI for NPC conversations. Accessed via a mode toggle on the character setup screen.

**Scope:** 1 region (The Withered Vale), 3 locations, 3 NPCs, basic movement, AI-driven NPC dialogue, reputation tracking.

---

## 1. Mode Selection

### Changes to App.jsx

After the API key is set, the player sees a **mode selection screen** before character setup. This is a new phase: `modeSelect`.

Phase flow becomes:
```
needKey → modeSelect → setup (story) → playing
                     → explore (exploration mode)
```

The mode select screen is simple: two large cards — "Story Mode" and "Explore Mode" — with a brief description of each. Selecting one sets a `gameMode` state (`'story'` or `'explore'`) and transitions to the appropriate phase.

**Story Mode** flows exactly as it does today (setup → playing).
**Explore Mode** flows to the exploration UI (map → location → interaction).

### New Component: `ModeSelect.jsx`

```jsx
// src/components/ModeSelect.jsx
// Two-card mode selection screen.

export default function ModeSelect({ onSelectMode }) {
  return (
    <div className="app-main">
      <div className="mode-select-panel">
        <h2>Choose Your Path</h2>

        <div className="mode-cards">
          <div className="mode-card" onClick={() => onSelectMode('story')}>
            <h3>Story Mode</h3>
            <p>A guided narrative arc. Choose your hook, shape the story through decisions, and face consequences.</p>
          </div>

          <div className="mode-card" onClick={() => onSelectMode('explore')}>
            <h3>Explore Mode</h3>
            <p>Wander the Withered Vale freely. Speak with its inhabitants, build your reputation, and uncover secrets.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
```

### App.jsx Changes

Add `gameMode` state. Render `ModeSelect` when phase is `modeSelect`. When `gameMode === 'explore'`, render the exploration UI instead of CharacterSetup/StoryView.

---

## 2. World Data

### New File: `public/data/withered-vale.json`

This defines the region, its locations, connections, and NPCs. All world rules live in data, not code.

```json
{
  "region": {
    "id": "withered-vale",
    "name": "The Withered Vale",
    "description": "A sunken valley where the mist never lifts. Crooked trees line paths that shift between visits, and the locals speak in half-truths."
  },
  "locations": [
    {
      "id": "vale-crossroads",
      "name": "The Crossroads",
      "description": "A junction of three muddy paths marked by a leaning signpost. The wood is too rotted to read, but someone has scratched fresh arrows into it with a knife.",
      "mood": "uneasy stillness, low wind",
      "connections": ["thornwatch-inn", "old-mill"],
      "isHidden": false,
      "unlockCondition": null,
      "coverImage": "/images/explore/vale-crossroads.png"
    },
    {
      "id": "thornwatch-inn",
      "name": "Thornwatch Inn",
      "description": "A squat building with boarded windows and a fire that never seems to warm the room. The innkeeper watches everyone but says little.",
      "mood": "smoky warmth masking tension",
      "connections": ["vale-crossroads", "old-mill"],
      "isHidden": false,
      "unlockCondition": null,
      "coverImage": "/images/explore/thornwatch-inn.png"
    },
    {
      "id": "old-mill",
      "name": "The Old Mill",
      "description": "The waterwheel still turns, though no one tends it. Inside, grain sacks rot and something scratches behind the walls at night.",
      "mood": "grinding rhythm, damp decay",
      "connections": ["vale-crossroads", "thornwatch-inn"],
      "isHidden": false,
      "unlockCondition": null,
      "coverImage": "/images/explore/old-mill.png"
    }
  ],
  "npcs": [
    {
      "id": "maren",
      "name": "Maren",
      "locationId": "thornwatch-inn",
      "archetype": "guarded innkeeper",
      "faction": "lantern_guild",
      "personalityTraits": ["pragmatic", "suspicious", "dry humor"],
      "baseDisposition": 40,
      "greeting": "Another traveler. Sit if you like. Don't touch the ledger.",
      "backstory": "Maren runs the Thornwatch Inn and serves as an unofficial fence for the Lantern Guild. She knows everyone's secrets and trades in them carefully."
    },
    {
      "id": "edric",
      "name": "Edric",
      "locationId": "vale-crossroads",
      "archetype": "wandering oathbreaker",
      "faction": null,
      "personalityTraits": ["melancholic", "cryptic", "honorable despite fall"],
      "baseDisposition": 55,
      "greeting": "You have the look of someone who hasn't yet learned what this place costs.",
      "backstory": "Edric was once a sworn knight of the Veilbound Inquisition. He broke an oath and now wanders the Vale, unable to leave. He knows things about the old wards."
    },
    {
      "id": "sibyl",
      "name": "Sibyl",
      "locationId": "old-mill",
      "archetype": "reclusive herbalist",
      "faction": "pale_court",
      "personalityTraits": ["patient", "knowing", "speaks in riddles"],
      "baseDisposition": 30,
      "greeting": "The wheel turns. So do you, it seems. What brings you to where things are ground down?",
      "backstory": "Sibyl tends a hidden garden behind the mill and supplies the Pale Court with rare reagents. She speaks obliquely but her advice, when understood, is always precise."
    }
  ],
  "factions": {
    "lantern_guild": {
      "name": "Lantern Guild",
      "description": "Smugglers and pragmatists who keep the Vale's economy alive."
    },
    "pale_court": {
      "name": "The Pale Court",
      "description": "Keepers of forbidden knowledge who trade in debts and secrets."
    },
    "inquisition": {
      "name": "Veilbound Inquisition",
      "description": "Enforcers of order who silence what they cannot contain."
    }
  }
}
```

### Static Location Images

Generate 3 pre-made location images (like the hook covers) using the same dark moody style:

- `public/images/explore/vale-crossroads.png`
- `public/images/explore/thornwatch-inn.png`
- `public/images/explore/old-mill.png`

Use the existing `scripts/generate-covers.js` pattern. Create a new script `scripts/generate-explore-covers.js` that takes each location's description + mood and generates a 1024×1792 portrait image using the current `basicStyle()` directives.

---

## 3. Exploration Engine

### New File: `src/engine/ExploreEngine.js`

This handles NPC dialogue generation. Movement logic and state management live in the hook (section 4), not the engine. The engine is purely for AI interactions.

```js
// src/engine/ExploreEngine.js
// AI-driven NPC dialogue for exploration mode.

import { chatCompletion } from '../services/openai.js';

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
 * Returns array of 3–4 action strings.
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

  // Use chatCompletionFull (non-streaming) for this
  const { chatCompletionFull } = await import('../services/openai.js');
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
```

---

## 4. Exploration Hook

### New File: `src/hooks/useExploreEngine.js`

Owns all exploration state. Mirrors the pattern of `useStoryEngine.js`.

```js
// src/hooks/useExploreEngine.js
// Central hook for exploration mode state and actions.

import { useState, useCallback, useRef } from 'react';
import { streamNPCDialogue, generateInteractionOptions, calculateDispositionDelta } from '../engine/ExploreEngine.js';

const EXPLORE_STORAGE_KEY = 'gloamreach_explore_state';

export default function useExploreEngine() {
  // World data (loaded from JSON)
  const [worldData, setWorldData] = useState(null);

  // Player exploration state
  const [currentLocationId, setCurrentLocationId] = useState('vale-crossroads');
  const [discoveredLocations, setDiscoveredLocations] = useState(['vale-crossroads']);
  const [npcDispositions, setNpcDispositions] = useState({});  // { npcId: number }
  const [factionReputation, setFactionReputation] = useState({});  // { factionId: number }
  const [inventory, setInventory] = useState([]);
  const [playerStates, setPlayerStates] = useState([]);  // ['wounded', 'cursed', etc.]

  // UI state
  const [activeNpc, setActiveNpc] = useState(null);  // NPC object currently being talked to
  const [conversationHistory, setConversationHistory] = useState([]);  // current NPC conversation
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [interactionOptions, setInteractionOptions] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [explorePhase, setExplorePhase] = useState('map');  // 'map' | 'location' | 'conversation'

  // Player profile (set from character setup or shared)
  const [playerName, setPlayerName] = useState('');

  // Load world data
  const loadWorldData = useCallback(async () => {
    if (worldData) return worldData;
    try {
      const res = await fetch('/data/withered-vale.json');
      const data = await res.json();
      setWorldData(data);

      // Initialize NPC dispositions from base values
      const dispositions = {};
      for (const npc of data.npcs) {
        dispositions[npc.id] = npc.baseDisposition;
      }
      setNpcDispositions(prev => ({ ...dispositions, ...prev }));

      // Initialize faction reputation
      const factions = {};
      for (const factionId of Object.keys(data.factions)) {
        factions[factionId] = 0;
      }
      setFactionReputation(prev => ({ ...factions, ...prev }));

      return data;
    } catch {
      return null;
    }
  }, [worldData]);

  // Get current location object
  const getCurrentLocation = useCallback(() => {
    if (!worldData) return null;
    return worldData.locations.find(l => l.id === currentLocationId) || null;
  }, [worldData, currentLocationId]);

  // Get NPCs at current location
  const getNpcsAtLocation = useCallback(() => {
    if (!worldData) return [];
    return worldData.npcs.filter(n => n.locationId === currentLocationId);
  }, [worldData, currentLocationId]);

  // Get connected locations from current position
  const getConnectedLocations = useCallback(() => {
    if (!worldData) return [];
    const current = getCurrentLocation();
    if (!current) return [];
    return worldData.locations.filter(l =>
      current.connections.includes(l.id) &&
      (!l.isHidden || discoveredLocations.includes(l.id))
    );
  }, [worldData, getCurrentLocation, discoveredLocations]);

  // Travel to a connected location
  const travelTo = useCallback((locationId) => {
    if (!worldData) return;
    const current = getCurrentLocation();
    if (!current || !current.connections.includes(locationId)) return;

    const targetLocation = worldData.locations.find(l => l.id === locationId);
    if (!targetLocation) return;
    if (targetLocation.isHidden && !discoveredLocations.includes(locationId)) return;

    setCurrentLocationId(locationId);
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setExplorePhase('location');

    // Discover the location if new
    if (!discoveredLocations.includes(locationId)) {
      setDiscoveredLocations(prev => [...prev, locationId]);
    }

    // Save state
    saveState();
  }, [worldData, getCurrentLocation, discoveredLocations]);

  // Start talking to an NPC
  const engageNpc = useCallback(async (npcId) => {
    if (!worldData) return;
    const npc = worldData.npcs.find(n => n.id === npcId);
    if (!npc) return;

    setActiveNpc(npc);
    setConversationHistory([]);
    setExplorePhase('conversation');

    // Generate initial interaction options
    setIsGenerating(true);
    try {
      const options = await generateInteractionOptions(npc, {
        disposition: npcDispositions[npcId] || npc.baseDisposition,
      });
      setInteractionOptions(options);
    } catch {
      setInteractionOptions(['Ask about the area', 'Inquire about rumors', 'Bid farewell']);
    }
    setIsGenerating(false);
  }, [worldData, npcDispositions]);

  // Send a message to the active NPC
  const talkToNpc = useCallback(async (playerMessage) => {
    if (!activeNpc || isStreaming || isGenerating) return;

    const location = getCurrentLocation();
    const disposition = npcDispositions[activeNpc.id] || activeNpc.baseDisposition;

    // Add player message to conversation
    const newHistory = [...conversationHistory, { role: 'user', content: playerMessage }];
    setConversationHistory(newHistory);

    // Stream NPC response
    setIsStreaming(true);
    setStreamedText('');
    let fullResponse = '';

    try {
      const gen = streamNPCDialogue(activeNpc, playerMessage, {
        disposition,
        playerName,
        locationDescription: location?.description || '',
        conversationHistory: newHistory,
      });

      for await (const chunk of gen) {
        fullResponse += chunk;
        setStreamedText(fullResponse);
      }
    } catch (e) {
      fullResponse = fullResponse || `${activeNpc.name} regards you in silence.`;
    }

    setIsStreaming(false);

    // Add NPC response to conversation
    const updatedHistory = [...newHistory, { role: 'assistant', content: fullResponse }];
    setConversationHistory(updatedHistory);

    // Update disposition
    const delta = calculateDispositionDelta(playerMessage, activeNpc);
    if (delta !== 0) {
      const newDisposition = Math.max(0, Math.min(100, disposition + delta));
      setNpcDispositions(prev => ({ ...prev, [activeNpc.id]: newDisposition }));

      // Update faction reputation if NPC has faction
      if (activeNpc.faction) {
        setFactionReputation(prev => ({
          ...prev,
          [activeNpc.faction]: Math.max(-100, Math.min(100, (prev[activeNpc.faction] || 0) + Math.round(delta / 2))),
        }));
      }
    }

    // Generate new interaction options
    setIsGenerating(true);
    try {
      const options = await generateInteractionOptions(activeNpc, {
        disposition: (npcDispositions[activeNpc.id] || activeNpc.baseDisposition) + delta,
        conversationHistory: updatedHistory,
      });
      setInteractionOptions(options);
    } catch {
      setInteractionOptions(['Continue speaking', 'Ask something else', 'Walk away']);
    }
    setIsGenerating(false);

    // Save state
    saveState();
  }, [activeNpc, isStreaming, isGenerating, conversationHistory, getCurrentLocation, npcDispositions, playerName]);

  // Leave conversation, return to location view
  const leaveConversation = useCallback(() => {
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setStreamedText('');
    setExplorePhase('location');
  }, []);

  // Return to map view
  const openMap = useCallback(() => {
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setExplorePhase('map');
  }, []);

  // Persistence: save to localStorage
  const saveState = useCallback(() => {
    try {
      const state = {
        currentLocationId,
        discoveredLocations,
        npcDispositions,
        factionReputation,
        inventory,
        playerStates,
        playerName,
      };
      localStorage.setItem(EXPLORE_STORAGE_KEY, JSON.stringify(state));
    } catch {}
  }, [currentLocationId, discoveredLocations, npcDispositions, factionReputation, inventory, playerStates, playerName]);

  // Persistence: load from localStorage
  const loadState = useCallback(() => {
    try {
      const raw = localStorage.getItem(EXPLORE_STORAGE_KEY);
      if (!raw) return false;
      const state = JSON.parse(raw);
      if (state.currentLocationId) setCurrentLocationId(state.currentLocationId);
      if (state.discoveredLocations) setDiscoveredLocations(state.discoveredLocations);
      if (state.npcDispositions) setNpcDispositions(state.npcDispositions);
      if (state.factionReputation) setFactionReputation(state.factionReputation);
      if (state.inventory) setInventory(state.inventory);
      if (state.playerStates) setPlayerStates(state.playerStates);
      if (state.playerName) setPlayerName(state.playerName);
      return true;
    } catch {
      return false;
    }
  }, []);

  // Reset exploration state
  const resetExplore = useCallback(() => {
    setCurrentLocationId('vale-crossroads');
    setDiscoveredLocations(['vale-crossroads']);
    setNpcDispositions({});
    setFactionReputation({});
    setInventory([]);
    setPlayerStates([]);
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setStreamedText('');
    setIsStreaming(false);
    setIsGenerating(false);
    setExplorePhase('map');
    try { localStorage.removeItem(EXPLORE_STORAGE_KEY); } catch {}
  }, []);

  return {
    // World data
    worldData,
    loadWorldData,

    // State
    currentLocationId,
    discoveredLocations,
    npcDispositions,
    factionReputation,
    inventory,
    playerStates,
    playerName,
    setPlayerName,

    // UI state
    activeNpc,
    conversationHistory,
    isStreaming,
    streamedText,
    interactionOptions,
    isGenerating,
    explorePhase,

    // Derived data
    getCurrentLocation,
    getNpcsAtLocation,
    getConnectedLocations,

    // Actions
    travelTo,
    engageNpc,
    talkToNpc,
    leaveConversation,
    openMap,

    // Persistence
    saveState,
    loadState,
    resetExplore,
  };
}
```

---

## 5. Components

### New File: `src/components/ExploreView.jsx`

Top-level exploration component. Switches between map, location, and conversation sub-views based on `explorePhase`.

```jsx
// src/components/ExploreView.jsx
// Exploration mode root — switches between map, location detail, and NPC conversation.

import ExploreMap from './ExploreMap.jsx';
import LocationView from './LocationView.jsx';
import NPCConversation from './NPCConversation.jsx';

export default function ExploreView({ explore, onMenuToggle }) {
  const { explorePhase } = explore;

  return (
    <div className="explore-viewport">
      {/* Top bar — same pattern as story mode */}
      <div className="top-bar">
        <button className="menu-btn" onClick={onMenuToggle}>☰</button>
        <span className="beat-indicator">
          {explorePhase === 'map' ? 'World Map' :
           explorePhase === 'conversation' ? explore.activeNpc?.name :
           explore.getCurrentLocation()?.name || 'Exploring'}
        </span>
      </div>

      {explorePhase === 'map' && (
        <ExploreMap
          worldData={explore.worldData}
          currentLocationId={explore.currentLocationId}
          discoveredLocations={explore.discoveredLocations}
          onSelectLocation={explore.travelTo}
        />
      )}

      {explorePhase === 'location' && (
        <LocationView
          location={explore.getCurrentLocation()}
          npcs={explore.getNpcsAtLocation()}
          connections={explore.getConnectedLocations()}
          npcDispositions={explore.npcDispositions}
          onEngageNpc={explore.engageNpc}
          onTravel={explore.travelTo}
          onOpenMap={explore.openMap}
        />
      )}

      {explorePhase === 'conversation' && (
        <NPCConversation
          npc={explore.activeNpc}
          disposition={explore.npcDispositions[explore.activeNpc?.id] || 50}
          conversationHistory={explore.conversationHistory}
          isStreaming={explore.isStreaming}
          streamedText={explore.streamedText}
          interactionOptions={explore.interactionOptions}
          isGenerating={explore.isGenerating}
          onTalk={explore.talkToNpc}
          onLeave={explore.leaveConversation}
        />
      )}
    </div>
  );
}
```

### New File: `src/components/ExploreMap.jsx`

SVG-based node map of the Withered Vale. Nodes are circles positioned on a dark background, connected by paths. Discovered nodes are lit; undiscovered ones are dimmed/hidden. Current location is highlighted.

```jsx
// src/components/ExploreMap.jsx
// SVG node map of the Withered Vale.

// Node positions (hand-tuned for the three MVP locations)
const NODE_POSITIONS = {
  'vale-crossroads': { x: 200, y: 150, label: 'The Crossroads' },
  'thornwatch-inn':  { x: 100, y: 320, label: 'Thornwatch Inn' },
  'old-mill':        { x: 320, y: 350, label: 'The Old Mill' },
};

// Connections (edges)
const EDGES = [
  ['vale-crossroads', 'thornwatch-inn'],
  ['vale-crossroads', 'old-mill'],
  ['thornwatch-inn', 'old-mill'],
];

export default function ExploreMap({
  worldData,
  currentLocationId,
  discoveredLocations,
  onSelectLocation,
}) {
  if (!worldData) return null;

  const isDiscovered = (id) => discoveredLocations.includes(id);

  return (
    <div className="explore-map-container">
      <h2 className="map-title">{worldData.region.name}</h2>
      <p className="map-subtitle">{worldData.region.description}</p>

      <svg viewBox="0 0 420 500" className="explore-map-svg">
        {/* Edges */}
        {EDGES.map(([a, b]) => {
          if (!isDiscovered(a) || !isDiscovered(b)) return null;
          const pa = NODE_POSITIONS[a];
          const pb = NODE_POSITIONS[b];
          return (
            <line
              key={`${a}-${b}`}
              x1={pa.x} y1={pa.y}
              x2={pb.x} y2={pb.y}
              className="map-edge"
            />
          );
        })}

        {/* Nodes */}
        {Object.entries(NODE_POSITIONS).map(([id, pos]) => {
          if (!isDiscovered(id)) return null;
          const isCurrent = id === currentLocationId;
          return (
            <g
              key={id}
              className={`map-node${isCurrent ? ' current' : ''}`}
              onClick={() => onSelectLocation(id)}
              style={{ cursor: 'pointer' }}
            >
              <circle cx={pos.x} cy={pos.y} r={isCurrent ? 18 : 14} />
              {isCurrent && (
                <circle cx={pos.x} cy={pos.y} r={24} className="map-node-pulse" />
              )}
              <text x={pos.x} y={pos.y + 36} textAnchor="middle">
                {pos.label}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
```

### New File: `src/components/LocationView.jsx`

Shows the current location with its pre-generated illustration as a header, description text, NPCs present, and travel options.

```jsx
// src/components/LocationView.jsx
// Location detail view: illustration, description, NPC list, travel options.

export default function LocationView({
  location,
  npcs,
  connections,
  npcDispositions,
  onEngageNpc,
  onTravel,
  onOpenMap,
}) {
  if (!location) return null;

  return (
    <div className="location-view">
      {/* Location illustration header */}
      <div className="location-header">
        {location.coverImage && (
          <img src={location.coverImage} alt={location.name} className="location-cover" />
        )}
        <div className="location-header-overlay">
          <h2>{location.name}</h2>
        </div>
      </div>

      {/* Description */}
      <div className="location-body">
        <p className="location-description">{location.description}</p>

        {/* NPCs */}
        {npcs.length > 0 && (
          <div className="location-section">
            <h3>People Here</h3>
            {npcs.map(npc => {
              const disposition = npcDispositions[npc.id] || npc.baseDisposition;
              const dispositionLabel =
                disposition >= 70 ? 'Friendly' :
                disposition >= 50 ? 'Neutral' :
                disposition >= 30 ? 'Wary' : 'Hostile';
              return (
                <button
                  key={npc.id}
                  className="npc-card"
                  onClick={() => onEngageNpc(npc.id)}
                >
                  <span className="npc-name">{npc.name}</span>
                  <span className="npc-archetype">{npc.archetype}</span>
                  <span className={`npc-disposition ${dispositionLabel.toLowerCase()}`}>
                    {dispositionLabel}
                  </span>
                </button>
              );
            })}
          </div>
        )}

        {/* Travel options */}
        <div className="location-section">
          <h3>Paths</h3>
          {connections.map(loc => (
            <button
              key={loc.id}
              className="travel-btn"
              onClick={() => onTravel(loc.id)}
            >
              Travel to {loc.name}
            </button>
          ))}
        </div>

        {/* Back to map */}
        <button className="map-btn" onClick={onOpenMap}>
          Open Map
        </button>
      </div>
    </div>
  );
}
```

### New File: `src/components/NPCConversation.jsx`

NPC dialogue view. Shows NPC name and disposition, streaming dialogue text, and interaction option buttons. Uses the same streaming text and choice button patterns as story mode.

```jsx
// src/components/NPCConversation.jsx
// NPC dialogue view with streaming responses and dynamic interaction options.

import StreamingText from './StreamingText.jsx';
import LanternLoader from './LanternLoader.jsx';

export default function NPCConversation({
  npc,
  disposition,
  conversationHistory,
  isStreaming,
  streamedText,
  interactionOptions,
  isGenerating,
  onTalk,
  onLeave,
}) {
  if (!npc) return null;

  const dispositionLabel =
    disposition >= 70 ? 'Friendly' :
    disposition >= 50 ? 'Neutral' :
    disposition >= 30 ? 'Wary' : 'Hostile';

  // Show the latest NPC response (either streaming or from history)
  const latestNpcMessage = isStreaming
    ? streamedText
    : conversationHistory.filter(m => m.role === 'assistant').pop()?.content || npc.greeting;

  return (
    <div className="npc-conversation">
      {/* NPC header */}
      <div className="npc-header">
        <h2>{npc.name}</h2>
        <span className="npc-archetype-label">{npc.archetype}</span>
        <span className={`npc-disposition-badge ${dispositionLabel.toLowerCase()}`}>
          {dispositionLabel}
        </span>
      </div>

      {/* Dialogue area */}
      <div className="npc-dialogue">
        {latestNpcMessage ? (
          <div className="npc-speech">
            <StreamingText text={latestNpcMessage} isStreaming={isStreaming} />
          </div>
        ) : isGenerating ? (
          <LanternLoader caption={`${npc.name} considers you…`} />
        ) : null}
      </div>

      {/* Interaction options */}
      <div className="npc-options">
        {isGenerating && !isStreaming ? (
          <LanternLoader caption="Considering your options…" />
        ) : (
          <>
            {interactionOptions.map((option, i) => (
              <button
                key={i}
                className="choice-btn"
                onClick={() => onTalk(option)}
                disabled={isStreaming || isGenerating}
              >
                {option}
              </button>
            ))}
            <button
              className="choice-btn leave-btn"
              onClick={onLeave}
              disabled={isStreaming}
            >
              Walk away
            </button>
          </>
        )}
      </div>
    </div>
  );
}
```

---

## 6. CSS Additions (index.css)

Add these new classes. Keep all existing styles intact.

```css
/* ---- Exploration viewport ---- */
.explore-viewport {
  position: relative;
  width: 100%;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-primary);
  display: flex;
  flex-direction: column;
}

/* ---- Map view ---- */
.explore-map-container {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  padding: 4rem 1rem 1rem;
  overflow-y: auto;
}

.map-title {
  color: var(--accent);
  font-size: 1.4rem;
  margin-bottom: 0.25rem;
}

.map-subtitle {
  color: var(--text-secondary);
  font-size: 0.85rem;
  text-align: center;
  max-width: 360px;
  margin-bottom: 1rem;
  line-height: 1.5;
}

.explore-map-svg {
  width: 100%;
  max-width: 420px;
  flex-shrink: 0;
}

.map-edge {
  stroke: var(--border);
  stroke-width: 2;
  stroke-dasharray: 6 4;
}

.map-node circle {
  fill: var(--bg-card);
  stroke: var(--border);
  stroke-width: 2;
  transition: fill 0.2s, stroke 0.2s;
}

.map-node:hover circle {
  stroke: var(--accent);
  fill: rgba(200, 168, 78, 0.15);
}

.map-node.current circle {
  fill: rgba(200, 168, 78, 0.2);
  stroke: var(--accent);
  stroke-width: 2.5;
}

.map-node-pulse {
  fill: none;
  stroke: var(--accent);
  stroke-width: 1.5;
  opacity: 0;
  animation: nodePulse 2s ease-in-out infinite;
}

@keyframes nodePulse {
  0% { opacity: 0.6; r: 18; }
  100% { opacity: 0; r: 32; }
}

.map-node text {
  fill: var(--text-primary);
  font-size: 11px;
  font-family: 'Inter', sans-serif;
  font-weight: 600;
}

/* ---- Location view ---- */
.location-view {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
  padding-top: 3rem; /* clear top bar */
}

.location-header {
  position: relative;
  height: 35vh;
  flex-shrink: 0;
  overflow: hidden;
}

.location-cover {
  width: 100%;
  height: 100%;
  object-fit: cover;
}

.location-header-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  padding: 1.5rem 1rem 0.75rem;
  background: linear-gradient(to top, rgba(10,10,20,0.9), transparent);
}

.location-header-overlay h2 {
  color: var(--accent);
  font-size: 1.3rem;
}

.location-body {
  padding: 1rem 1.25rem;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.location-description {
  font-family: 'Times New Roman', Georgia, serif;
  font-size: 1.05rem;
  line-height: 1.6;
  color: var(--text-primary);
}

.location-section h3 {
  font-size: 0.85rem;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 0.5rem;
}

/* NPC cards */
.npc-card {
  width: 100%;
  display: flex;
  align-items: center;
  gap: 0.75rem;
  padding: 0.75rem 1rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  transition: border-color 0.2s;
  margin-bottom: 0.5rem;
  text-align: left;
  color: var(--text-primary);
  font-family: 'Inter', sans-serif;
  font-size: 0.9rem;
}

.npc-card:hover {
  border-color: var(--accent);
}

.npc-name {
  font-weight: 600;
  flex-shrink: 0;
}

.npc-archetype {
  color: var(--text-secondary);
  font-size: 0.8rem;
  flex: 1;
}

.npc-disposition {
  font-size: 0.75rem;
  padding: 0.2rem 0.5rem;
  border-radius: 4px;
  flex-shrink: 0;
}

.npc-disposition.friendly { color: #6bbd6b; background: rgba(107,189,107,0.1); }
.npc-disposition.neutral { color: var(--text-secondary); background: rgba(160,152,136,0.1); }
.npc-disposition.wary { color: #d4a843; background: rgba(212,168,67,0.1); }
.npc-disposition.hostile { color: var(--danger); background: rgba(192,57,43,0.1); }

/* Travel buttons */
.travel-btn {
  width: 100%;
  padding: 0.65rem 1rem;
  font-size: 0.9rem;
  font-family: 'Inter', sans-serif;
  background: var(--bg-card);
  color: var(--text-primary);
  border: 1px solid var(--border);
  border-radius: 6px;
  cursor: pointer;
  transition: border-color 0.2s;
  text-align: left;
  margin-bottom: 0.5rem;
}

.travel-btn:hover {
  border-color: var(--accent);
}

.map-btn {
  width: 100%;
  padding: 0.6rem;
  font-size: 0.85rem;
  background: transparent;
  color: var(--accent);
  border: 1px solid var(--accent);
  border-radius: 6px;
  cursor: pointer;
  margin-top: 0.5rem;
}

/* ---- NPC Conversation ---- */
.npc-conversation {
  flex: 1;
  display: flex;
  flex-direction: column;
  padding-top: 3rem;
}

.npc-header {
  padding: 0.75rem 1.25rem;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}

.npc-header h2 {
  font-size: 1.1rem;
  color: var(--accent);
}

.npc-archetype-label {
  font-size: 0.8rem;
  color: var(--text-secondary);
}

.npc-disposition-badge {
  font-size: 0.7rem;
  padding: 0.15rem 0.5rem;
  border-radius: 4px;
  margin-left: auto;
}

.npc-disposition-badge.friendly { color: #6bbd6b; background: rgba(107,189,107,0.15); }
.npc-disposition-badge.neutral { color: var(--text-secondary); background: rgba(160,152,136,0.15); }
.npc-disposition-badge.wary { color: #d4a843; background: rgba(212,168,67,0.15); }
.npc-disposition-badge.hostile { color: var(--danger); background: rgba(192,57,43,0.15); }

.npc-dialogue {
  flex: 1;
  overflow-y: auto;
  padding: 1rem 1.25rem;
  display: flex;
  flex-direction: column;
  justify-content: flex-end;
}

.npc-speech {
  font-family: 'Times New Roman', Georgia, serif;
  font-size: 1.15rem;
  line-height: 1.6;
  color: var(--text-primary);
  background: rgba(10, 10, 20, 0.35);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border-radius: 12px;
  padding: 0.75rem 1rem;
  border: 1px solid rgba(255, 255, 255, 0.05);
}

.npc-options {
  flex-shrink: 0;
  padding: 0.75rem 1.25rem 1rem;
  border-top: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

.leave-btn {
  opacity: 0.6;
  font-size: 0.85rem;
}

/* ---- Mode select ---- */
.mode-select-panel {
  width: var(--story-body-width);
  max-width: 100%;
  padding: 2rem;
}

.mode-select-panel h2 {
  color: var(--accent);
  font-size: 1.3rem;
  margin-bottom: 1rem;
}

.mode-cards {
  display: flex;
  flex-direction: column;
  gap: 0.75rem;
}

.mode-card {
  padding: 1.25rem 1.5rem;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  cursor: pointer;
  transition: border-color 0.2s, background 0.2s;
}

.mode-card:hover {
  border-color: var(--accent);
  background: rgba(200, 168, 78, 0.08);
}

.mode-card h3 {
  color: var(--text-primary);
  font-size: 1.1rem;
  margin-bottom: 0.4rem;
}

.mode-card p {
  color: var(--text-secondary);
  font-size: 0.85rem;
  line-height: 1.5;
  margin: 0;
}
```

---

## 7. Static Assets to Generate

Create `scripts/generate-explore-covers.js` following the same pattern as `scripts/generate-covers.js`. Generate 3 portrait images (1024×1792) for the exploration locations using the current `basicStyle()` directives plus each location's description and mood.

Output files:
- `public/images/explore/vale-crossroads.png`
- `public/images/explore/thornwatch-inn.png`
- `public/images/explore/old-mill.png`

---

## 8. App.jsx Integration

The full phase flow:

```
needKey → modeSelect → setup (if story) → playing
                     → explore (if explore)
```

App.jsx needs:
- `gameMode` state (`'story'` | `'explore'` | `null`)
- Import and instantiate `useExploreEngine` alongside `useStoryEngine`
- Render `ModeSelect` when phase is `modeSelect` (or when gameMode is null and API key is set)
- Render `ExploreView` when gameMode is `explore`
- Load world data on mount (same pattern as lore loading)
- Pass explore hook to ExploreView as a prop bag
- Sidebar drawer should work in explore mode too (show "Return to Menu" and "Reset" options)

---

## 9. Files Summary

### New Files

| File | Purpose |
|------|---------|
| `src/engine/ExploreEngine.js` | NPC dialogue generation, interaction options, disposition calculation |
| `src/hooks/useExploreEngine.js` | Exploration state management and actions |
| `src/components/ExploreView.jsx` | Exploration root component (phase switcher) |
| `src/components/ExploreMap.jsx` | SVG node map |
| `src/components/LocationView.jsx` | Location detail with NPCs and travel |
| `src/components/NPCConversation.jsx` | NPC dialogue view |
| `src/components/ModeSelect.jsx` | Story vs Explore mode selection |
| `public/data/withered-vale.json` | World data for the Withered Vale |
| `scripts/generate-explore-covers.js` | Static location image generator |
| `public/images/explore/*.png` | 3 location cover images |

### Modified Files

| File | Change |
|------|--------|
| `src/App.jsx` | Add mode selection phase, explore mode rendering, drawer integration |
| `src/index.css` | Add all exploration CSS classes |

### Unchanged Files

| File | Why |
|------|-----|
| `src/engine/StoryEngine.js` | Story mode untouched |
| `src/hooks/useStoryEngine.js` | Story mode untouched |
| `src/components/StoryView.jsx` | Story mode untouched |
| `src/services/openai.js` | Already supports what exploration needs |

---

## 10. Testing Checklist

- [ ] Mode selection screen appears after API key entry
- [ ] Selecting "Story Mode" flows to character setup as before (no regressions)
- [ ] Selecting "Explore Mode" shows the Withered Vale map
- [ ] Map renders 3 nodes with dashed connecting paths
- [ ] Current location node pulses with gold accent
- [ ] Clicking a node travels to that location
- [ ] Location view shows cover image, description, NPCs, and travel options
- [ ] Clicking an NPC shows their greeting and interaction options
- [ ] Selecting an interaction option streams NPC dialogue response
- [ ] NPC disposition updates based on player choices (check via disposition badge)
- [ ] "Walk away" returns to location view
- [ ] "Open Map" returns to map view
- [ ] Hamburger menu works in exploration mode
- [ ] Exploration state persists across page refresh (localStorage)
- [ ] Reset clears exploration state
- [ ] Works on mobile (375×812 viewport)
