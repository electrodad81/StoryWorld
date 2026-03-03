# NPC Redesign + Story ↔ Explore Integration Spec

## Goal

Redesign the NPC conversation system so interactions have purpose, progression, and endpoints. Simultaneously wire up the story-explore integration so both modes feel like one world. NPCs become the primary connective tissue between modes.

**Four mechanics working together:**
1. Content tiers — disposition thresholds unlock new topics and information
2. Exchange limits — conversations cap at 4 exchanges per visit, then the NPC wraps up
3. Quests — each NPC has goals for the player (fetch quests + story-linked quests)
4. Repeat visit tracking — NPCs remember what they've shared and react accordingly

**Integration baked in:**
- Story outcomes unlock NPC content and complete story-linked quests
- NPC knowledge feeds back into story mode prose via player lore
- Shared disposition and faction rep across modes
- Location description overrides from story outcomes

---

## 1. NPC Knowledge System

### Content Tiers

Each NPC has 3–4 tiers, each gated by a disposition threshold. Each tier contains **knowledge topics** — semi-authored subjects the NPC can discuss. The topic has an authored summary (what the NPC knows), but the AI generates the actual dialogue in character.

### Updated NPC Data Structure

Expand each NPC entry in `withered-vale.json` with tiers, quests, and tracking fields:

```json
{
  "id": "maren",
  "name": "Maren",
  "locationId": "thornwatch-inn",
  "archetype": "guarded innkeeper",
  "faction": "lantern_guild",
  "personalityTraits": ["pragmatic", "suspicious", "dry humor"],
  "baseDisposition": 40,
  "portrait": "/images/explore/npc-maren.png",
  "greeting": "Another traveler. Sit if you like. Don't touch the ledger.",
  "backstory": "Maren runs the Thornwatch Inn and serves as an unofficial fence for the Lantern Guild. She knows everyone's secrets and trades in them carefully.",
  "dismissal": "I've said my piece. Buy a drink or move along.",
  "nothingNew": "You again. Nothing's changed since last we spoke.",

  "tiers": [
    {
      "dispositionRequired": 0,
      "topics": [
        {
          "id": "maren_vale_gossip",
          "subject": "Local gossip about the Vale",
          "summary": "The roads have been quieter than usual. Traders from the east stopped coming weeks ago. Something in the fog is making people forget their routes.",
          "loreRevealed": { "locations": ["vale-crossroads"] }
        },
        {
          "id": "maren_inn_business",
          "subject": "How business is at the inn",
          "summary": "Business is thin. The only regulars are Edric, who pays in old coins she doesn't ask about, and the occasional Lantern Guild runner who never stays the night.",
          "loreRevealed": { "npcs": ["Edric"], "factions": ["lantern_guild"] }
        }
      ]
    },
    {
      "dispositionRequired": 50,
      "topics": [
        {
          "id": "maren_lantern_guild",
          "subject": "Her connection to the Lantern Guild",
          "summary": "She's a fence — she moves goods for the Guild through the inn's cellar. Not proud of it, not ashamed either. It keeps the Vale's economy alive when the official trade routes dry up.",
          "loreRevealed": { "factions": ["lantern_guild"] }
        },
        {
          "id": "maren_edric_past",
          "subject": "What she knows about Edric",
          "summary": "Edric was Inquisition once — sworn and branded. He broke an oath, though she doesn't know which one. He can't leave the Vale. Something binds him here, like a leash made of guilt.",
          "loreRevealed": { "npcs": ["Edric"], "factions": ["inquisition"] }
        }
      ]
    },
    {
      "dispositionRequired": 70,
      "topics": [
        {
          "id": "maren_ledger",
          "subject": "The ledger behind the bar",
          "summary": "The ledger records every debt owed to the Lantern Guild in the Vale. Names, amounts, and favors. Some entries are written in ink that moves. She doesn't know who supplies it — but Sibyl might.",
          "loreRevealed": { "relics": ["ink_of_concord"], "npcs": ["Sibyl"] }
        }
      ]
    },
    {
      "dispositionRequired": 85,
      "requiresFlag": "trusts_player",
      "topics": [
        {
          "id": "maren_secret",
          "subject": "Her real fear",
          "summary": "She's afraid the Inquisition is coming to audit the Guild's operations. If they find the ledger, everyone in it burns. She's been thinking about destroying it — but the ink won't burn. She needs someone she trusts to take it somewhere safe.",
          "loreRevealed": { "factions": ["inquisition", "lantern_guild"], "relics": ["ink_of_concord"] }
        }
      ]
    }
  ],

  "quests": [
    {
      "id": "maren_q1",
      "type": "fetch",
      "title": "Ask Edric about the old wards",
      "description": "Maren wants to know if the wards around the Vale still hold. Edric would know — he helped place them.",
      "unlockDisposition": 50,
      "requirement": { "type": "npc_topic_revealed", "topicId": "edric_old_wards" },
      "reward": { "dispositionBonus": 15, "loreRevealed": { "locations": ["vale-crossroads"] } },
      "completionDialogue": "So the wards are cracking. I figured as much. ...Thank you. That's not easy news, but it's honest. I don't forget that."
    },
    {
      "id": "maren_q2",
      "type": "story_linked",
      "title": "Survive the Veilbound Audit",
      "description": "The Inquisition is coming. If you can survive their audit, Maren will know you can be trusted with real secrets.",
      "unlockDisposition": 60,
      "requirement": { "type": "hook_survived", "hookId": "HOOK-06" },
      "reward": { "dispositionBonus": 20, "flagGranted": "trusts_player" },
      "completionDialogue": "You lived through it. That's... more than most manage. Sit down. There's something I need to tell you."
    }
  ]
}
```

Apply the same structure to **Edric** and **Sibyl**. Here are their tier outlines:

**Edric (wandering oathbreaker, base disposition 55):**

| Tier | Disposition | Topics |
|------|------------|--------|
| 1 | 0 | The state of the roads; warnings about the fog |
| 2 | 45 | His past with the Inquisition (vague); the old wards around the Vale |
| 3 | 65 | The oath he broke (he refused to silence a child who could see through wards); why he can't leave |
| 4 | 80 + flag `respects_player` | The ward anchor locations — where the old protections are failing |

**Edric quests:**
- Fetch: "Tell Sibyl the wards are failing" (unlocks at disposition 45, requires topic `edric_old_wards` to be revealed to player, then player must reveal topic `sibyl_reagents` from Sibyl — i.e., talk to both NPCs)
- Story-linked: "Complete Bells in the Fog" (any outcome) — Edric knows the bells are connected to the wards

**Sibyl (reclusive herbalist, base disposition 30):**

| Tier | Disposition | Topics |
|------|------------|--------|
| 1 | 0 | The mill and its strange sounds; what grows in her garden |
| 2 | 45 | Her reagent work for the Pale Court; the wheel's purpose (it grinds more than grain) |
| 3 | 65 | The Pale Court's interest in the Vale (they're looking for something buried beneath it); the price of forbidden knowledge |
| 4 | 80 + flag `curious_about_star` or `grateful_for_lanterns` | What's actually under the mill — an old well that connects to something deep |

**Sibyl quests:**
- Fetch: "Bring word from Maren about the ledger ink" (requires `maren_ledger` topic revealed)
- Story-linked: "Complete Drowned Star's Whisper" — the salt in the wells is connected to what's beneath the mill

---

## 2. Conversation Flow with Limits

### Exchange Limit: 4 Per Visit

Each time the player engages an NPC, they get **4 exchanges** (player message → NPC response = 1 exchange). After the 4th exchange, the NPC delivers their `dismissal` line and the conversation ends automatically.

### Conversation Flow Per Visit

```
1. Player engages NPC
2. Check: any newly completed quests? → If yes, NPC delivers completionDialogue first
3. Check: any new topics available (based on current disposition + flags + not yet revealed)?
   → If yes: NPC greets player, interaction options include available topic subjects
   → If no new topics: NPC delivers nothingNew line, conversation ends after 1 exchange
4. Player picks an option (topic to discuss, quest to accept, or general chat)
5. NPC responds (AI-generated, guided by the topic summary)
6. Topic marked as revealed in world state
7. Lore from topic added to journal
8. Repeat from step 4 until 4 exchanges reached
9. NPC delivers dismissal line, conversation ends
```

### Implementation in ExploreEngine.js

Modify `generateNpcResponse` to accept a `currentTopic` parameter. When a topic is active, the system prompt includes the topic summary as guidance:

```js
const topicGuidance = currentTopic
  ? `\nThe player is asking about: ${currentTopic.subject}.\n` +
    `What you know (convey this naturally in character, do not read it verbatim): ${currentTopic.summary}\n` +
    'Share this information across 1–2 responses. Be in character — reveal reluctantly if disposition is low, more openly if high.\n'
  : '';
```

### Implementation in useExploreEngine.js

Add conversation tracking state:

```js
const [exchangeCount, setExchangeCount] = useState(0);
const [currentTopic, setCurrentTopic] = useState(null);
const [revealedTopics, setRevealedTopics] = useState([]);  // topic IDs revealed this session
```

`revealedTopics` is persisted to world state (not just session state) so NPCs remember what they've told the player across visits.

Modify `talkToNpc`:
- Increment `exchangeCount` after each exchange
- After exchange 4: set NPC response to `dismissal`, then auto-call `leaveConversation`
- When a topic is selected as the interaction option: set `currentTopic`, pass to engine

Modify `engageNpc`:
- Reset `exchangeCount` to 0
- Check for completed quests (compare quest requirements against world state)
- Calculate available topics: filter NPC tiers by disposition and flags, exclude already-revealed topics
- If no available topics and no completed quests: show `nothingNew`, auto-close after 1 exchange

### Interaction Options Generation

Replace the fully AI-generated interaction options with a **hybrid approach**:

- If unrevealed topics are available at the current tier: present topic subjects as options (authored, not AI-generated)
- If a quest is available to accept: include "Ask about [quest title]" as an option
- Always include one AI-generated option for general conversation (flavor, atmosphere)
- Always include "Walk away" as the final option

Example options for Maren at disposition 50 with `maren_vale_gossip` already revealed:

```
[
  "Ask about her connection to the Lantern Guild",    // tier 2 topic (authored)
  "Ask what she knows about Edric",                   // tier 2 topic (authored)
  "Ask about the old wards [Quest]",                  // quest available
  "Make small talk",                                   // AI-generated flavor
  "Walk away"
]
```

This means `generateInteractionOptions` in `ExploreEngine.js` is only called for the single "flavor" option, not for the full list. The authored options come from the NPC data. This also makes the AI calls cheaper — one short generation for flavor instead of a full option set.

---

## 3. Quest System

### Quest States

Quests track in world state under `npcQuests`:

```json
{
  "npcQuests": {
    "maren_q1": { "status": "available" },
    "maren_q2": { "status": "locked" },
    "edric_q1": { "status": "completed", "completedAt": "..." },
    "sibyl_q1": { "status": "active" }
  }
}
```

Statuses: `locked` → `available` → `active` → `completed`

- `locked`: Disposition too low or prerequisite not met
- `available`: Disposition met, shows up as an option when talking to NPC
- `active`: Player has accepted the quest (NPC has explained what they want)
- `completed`: Requirement met, reward granted on next NPC visit

### Quest Requirement Types

```json
{ "type": "npc_topic_revealed", "topicId": "edric_old_wards" }
{ "type": "hook_survived", "hookId": "HOOK-06" }
{ "type": "hook_completed", "hookId": "HOOK-01" }
{ "type": "location_visited", "locationId": "old-mill" }
```

### Quest Evaluation

On each NPC engagement (`engageNpc`), evaluate all quests for that NPC:

```js
function evaluateQuests(npc, worldState, journal) {
  for (const quest of npc.quests) {
    const current = worldState.npcQuests?.[quest.id];

    // Check if quest should unlock
    if (!current || current.status === 'locked') {
      const disposition = worldState.npcDispositions[npc.id] ?? npc.baseDisposition;
      if (disposition >= quest.unlockDisposition) {
        worldState.npcQuests[quest.id] = { status: 'available' };
      }
    }

    // Check if active quest is now complete
    if (current?.status === 'active') {
      if (checkRequirement(quest.requirement, worldState, journal)) {
        worldState.npcQuests[quest.id] = { status: 'completed', completedAt: new Date().toISOString() };
      }
    }
  }
}

function checkRequirement(req, worldState, journal) {
  switch (req.type) {
    case 'npc_topic_revealed':
      return (worldState.revealedTopics || []).includes(req.topicId);
    case 'hook_survived':
      return journal.storiesPlayed.some(s => s.hookId === req.hookId && s.outcome === 'survived');
    case 'hook_completed':
      return journal.storiesPlayed.some(s => s.hookId === req.hookId);
    case 'location_visited':
      return (worldState.discoveredLocations || []).includes(req.locationId);
    default:
      return false;
  }
}
```

### Quest Completion Flow

When `engageNpc` detects a quest with status `completed`:
1. NPC delivers `completionDialogue` as their opening line (instead of greeting)
2. Apply rewards: disposition bonus, flag granted, lore revealed
3. Update quest status to `completed` (with timestamp)
4. Show a brief notification: "Quest Complete: [title]"
5. Continue conversation normally with remaining exchanges

---

## 4. World State Layer

### New File: `src/services/worldState.js`

```js
const WORLD_STATE_KEY = 'gloamreach_world_state';

const DEFAULT_STATE = {
  // Story outcomes
  completedHooks: [],
  storyOutcomes: {},

  // Shared reputation
  factionReputation: {
    lantern_guild: 0,
    pale_court: 0,
    inquisition: 0,
  },

  // Shared NPC disposition
  npcDispositions: {},

  // Location description overrides from story outcomes
  locationOverrides: {},

  // NPC dialogue flags from story outcomes
  npcDialogueFlags: {},

  // Topics revealed to the player (persists across visits)
  revealedTopics: [],

  // Quest states
  npcQuests: {},

  // Player lore knowledge (synced from journal)
  knownLore: {
    locations: [],
    factions: [],
    relics: [],
    curses: [],
    npcs: [],
  },
};

export function loadWorldState() { /* read from localStorage, merge with defaults */ }
export function saveWorldState(state) { /* write to localStorage */ }
export function resetWorldState() { /* remove from localStorage */ }

export function syncKnownLoreFromJournal(worldState, journal) {
  worldState.knownLore = { ...journal.loreDiscovered };
  return worldState;
}
```

---

## 5. Story Outcome Effects

### New File: `public/data/story-effects.json`

Per-hook effects applied when a story completes. Each hook has `survived` and `died` outcome branches.

**Structure per hook:**

```json
{
  "HOOK-01": {
    "title": "Bells in the Fog",
    "onComplete": {
      "survived": {
        "locationOverrides": {
          "vale-crossroads": {
            "extraDetail": "A faint toll echoes from the south, though the fog has thinned since the bell tower was silenced. Locals speak your name in low, grateful tones."
          }
        },
        "npcDialogueFlags": {
          "edric": ["knows_about_bells", "respects_player"],
          "maren": ["heard_about_bells"]
        },
        "factionRepBonus": {}
      },
      "died": {
        "locationOverrides": {
          "vale-crossroads": {
            "extraDetail": "The bell tower still tolls at low tide. A fresh name has appeared on the village memorial — yours, though you stand here reading it."
          }
        },
        "npcDialogueFlags": {
          "edric": ["knows_about_bells", "wary_of_player"],
          "sibyl": ["sensed_player_death"]
        },
        "factionRepBonus": {}
      }
    }
  }
}
```

Include entries for all 6 hooks (HOOK-01 through HOOK-06). Refer to the story-explore integration spec for the full authored content for each hook. Add `factionRepBonus` where appropriate (e.g., HOOK-03 survived → `lantern_guild: +15`).

### Applying Effects

In `StoryComplete.jsx`, after writing the journal entry:

```js
import { loadWorldState, saveWorldState, syncKnownLoreFromJournal } from '../services/worldState.js';

function applyStoryEffects(hookId, outcome, storyEffectsData, journal) {
  const effects = storyEffectsData[hookId]?.onComplete?.[outcome];
  if (!effects) return;

  const ws = loadWorldState();

  // Track completed hook
  if (!ws.completedHooks.includes(hookId)) {
    ws.completedHooks.push(hookId);
  }
  ws.storyOutcomes[hookId] = { outcome, timestamp: new Date().toISOString() };

  // Apply location overrides
  if (effects.locationOverrides) {
    ws.locationOverrides = { ...ws.locationOverrides, ...effects.locationOverrides };
  }

  // Append NPC dialogue flags (accumulate, never remove)
  if (effects.npcDialogueFlags) {
    for (const [npcId, flags] of Object.entries(effects.npcDialogueFlags)) {
      const existing = ws.npcDialogueFlags[npcId] || [];
      const merged = [...new Set([...existing, ...flags])];
      ws.npcDialogueFlags[npcId] = merged;

      // Also apply disposition changes from flags
      applyDispositionFromFlags(ws, npcId, flags);
    }
  }

  // Apply faction rep bonuses
  if (effects.factionRepBonus) {
    for (const [factionId, delta] of Object.entries(effects.factionRepBonus)) {
      ws.factionReputation[factionId] =
        Math.max(-100, Math.min(100, (ws.factionReputation[factionId] || 0) + delta));
    }
  }

  // Sync lore knowledge from journal
  syncKnownLoreFromJournal(ws, journal);

  saveWorldState(ws);
}
```

---

## 6. Story Mode — Player Knowledge in Prompts

### StoryEngine.js Changes

Add `knownLore` and `factionReputation` parameters to `streamScene()`. Build a knowledge clause injected into the system prompt:

```js
function buildKnowledgeClause(knownLore, factionRep) {
  const parts = [];

  if (knownLore?.factions?.length) {
    parts.push(`The protagonist is aware of these factions: ${knownLore.factions.join(', ')}.`);
  }
  if (knownLore?.relics?.length) {
    parts.push(`The protagonist has heard of these relics: ${knownLore.relics.join(', ')}.`);
  }
  if (knownLore?.curses?.length) {
    parts.push(`The protagonist knows of these curses: ${knownLore.curses.join(', ')}.`);
  }
  if (knownLore?.npcs?.length) {
    parts.push(`The protagonist has met or heard of: ${knownLore.npcs.join(', ')}.`);
  }

  const stances = Object.entries(factionRep || {})
    .filter(([_, rep]) => Math.abs(rep) >= 10)
    .map(([id, rep]) => {
      const label = rep >= 20 ? 'favorable' : rep >= 10 ? 'somewhat positive' :
                    rep <= -20 ? 'hostile' : 'somewhat negative';
      return `${id.replace(/_/g, ' ')}: ${label}`;
    });
  if (stances.length) {
    parts.push(`Faction standing: ${stances.join(', ')}.`);
  }

  if (parts.length === 0) return '';
  return '\nPlayer knowledge (reference naturally, do not dump exposition):\n' + parts.join(' ') + '\n';
}
```

### useStoryEngine.js Changes

In `runTurn`, load world state and pass knowledge to the engine:

```js
import { loadWorldState } from '../services/worldState.js';

// Inside runTurn, before streamScene:
const worldState = loadWorldState();

const gen = streamScene(hist, loreData, {
  beat: currentBeat,
  playerName: profile.name,
  gender: profile.gender,
  dangerStreak: currentDanger,
  knownLore: worldState.knownLore,
  factionReputation: worldState.factionReputation,
});
```

---

## 7. Exploration Mode — Using World State

### useExploreEngine.js Changes

**On initialization and NPC engagement**, read from world state:

```js
import { loadWorldState, saveWorldState } from '../services/worldState.js';

// When loading world data, merge dispositions from world state:
const ws = loadWorldState();
const initialDispositions = {};
for (const npc of data.npcs) {
  initialDispositions[npc.id] = ws.npcDispositions[npc.id] ?? npc.baseDisposition;
}

// When engaging an NPC:
// 1. Load world state
// 2. Get revealed topics from ws.revealedTopics
// 3. Evaluate quests (check requirements against world state + journal)
// 4. Calculate available topics: filter tiers by disposition + flags, exclude revealed
// 5. Build interaction options from available topics + quests + one AI flavor option
```

**After disposition changes**, write back to world state:

```js
const ws = loadWorldState();
ws.npcDispositions[npcId] = newDisposition;
saveWorldState(ws);
```

**After a topic is revealed**, mark it in world state and update journal lore:

```js
const ws = loadWorldState();
if (!ws.revealedTopics.includes(topicId)) {
  ws.revealedTopics.push(topicId);
}
saveWorldState(ws);

// Also add lore to journal
if (topic.loreRevealed) {
  addExploreLore(loadJournal(), topic.loreRevealed);
}
```

### ExploreEngine.js Changes

`generateNpcResponse` receives additional context:

```js
export async function* generateNpcResponse({
  npc,
  location,
  playerMessage,
  conversationHistory,
  disposition,
  currentTopic,         // NEW: topic object if player selected a topic
  dialogueFlags,        // NEW: string[] from world state
  exchangeCount,        // NEW: which exchange number this is (1–4)
  isDismissal,          // NEW: if true, NPC is wrapping up
}) {
  // Build system prompt with topic guidance and flags
  const topicGuidance = currentTopic
    ? `\nThe player is asking about: ${currentTopic.subject}.\n` +
      `What you know (convey naturally in character, never read verbatim): ${currentTopic.summary}\n`
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

  // ... rest of prompt construction with these additions
}
```

### LocationView.jsx Changes

Render story-outcome extra details:

```jsx
import { loadWorldState } from '../services/worldState.js';

// Inside component:
const worldState = loadWorldState();
const override = worldState.locationOverrides?.[location.id];

// In render:
<p className="location-description">{location.description}</p>
{override?.extraDetail && (
  <p className="location-extra-detail">{override.extraDetail}</p>
)}
```

---

## 8. Updated withered-vale.json

The full NPC data with tiers and quests is too large to inline here. Claude Code should expand each NPC entry in `withered-vale.json` following this structure:

**Maren:** 4 tiers (0, 50, 70, 85), 8 topics total, 2 quests (fetch + story-linked)
**Edric:** 4 tiers (0, 45, 65, 80), 7 topics total, 2 quests (fetch + story-linked)
**Sibyl:** 4 tiers (0, 45, 65, 80), 7 topics total, 2 quests (fetch + story-linked)

Use the tier outlines from section 1 of this spec. Write the topic summaries to be:
- Specific and lore-consistent (reference existing lore.json content)
- Interlocking (Maren's tier 2 mentions Edric; Edric's tier 2 mentions the wards; Sibyl's tier 3 connects to what's under the mill)
- Progressively more revealing (tier 1 is surface-level, tier 4 is deep secrets)
- Each topic should have a `loreRevealed` field listing what lore categories it exposes

---

## 9. CSS Additions

```css
/* Location extra detail from story outcomes */
.location-extra-detail {
  font-family: 'Times New Roman', Georgia, serif;
  font-size: 0.95rem;
  line-height: 1.6;
  color: var(--accent);
  opacity: 0.85;
  font-style: italic;
  margin-top: 0.5rem;
}

/* Quest notification */
.quest-notification {
  padding: 0.5rem 0.75rem;
  background: rgba(200, 168, 78, 0.12);
  border: 1px solid var(--accent);
  border-radius: 6px;
  font-size: 0.85rem;
  color: var(--accent);
  margin-bottom: 0.5rem;
  text-align: center;
}

/* Topic option styling (distinguish from AI-generated options) */
.choice-btn.topic-option {
  border-left: 3px solid var(--accent);
}

.choice-btn.quest-option {
  border-left: 3px solid #6bbd6b;
}

/* Exchange counter */
.exchange-counter {
  font-size: 0.7rem;
  color: var(--text-secondary);
  text-align: center;
  padding: 0.25rem 0;
  opacity: 0.6;
}

/* Nothing new state */
.npc-nothing-new {
  font-family: 'Times New Roman', Georgia, serif;
  font-size: 1rem;
  color: var(--text-secondary);
  font-style: italic;
  padding: 1rem;
  text-align: center;
}
```

---

## 10. Files Summary

### New Files

| File | Purpose |
|------|---------|
| `src/services/worldState.js` | Shared world state layer (localStorage) |
| `public/data/story-effects.json` | Per-hook outcome effects on world state |

### Modified Files

| File | Change |
|------|--------|
| `public/data/withered-vale.json` | Expand NPCs with tiers, topics, quests, dismissal lines |
| `src/engine/StoryEngine.js` | Add `knownLore`/`factionReputation` to `streamScene()`, `buildKnowledgeClause()` |
| `src/engine/ExploreEngine.js` | Add `currentTopic`, `dialogueFlags`, `exchangeCount`, `isDismissal` to `generateNpcResponse()`; simplify `generateInteractionOptions()` to only generate 1 flavor option |
| `src/hooks/useStoryEngine.js` | Load world state in `runTurn`, pass knowledge to engine |
| `src/hooks/useExploreEngine.js` | Major rework: exchange counting, topic tracking, quest evaluation, world state read/write, hybrid interaction options |
| `src/components/StoryComplete.jsx` | Load story-effects.json, call `applyStoryEffects()` after journal write |
| `src/components/LocationView.jsx` | Render `extraDetail` from world state |
| `src/components/NPCConversation.jsx` | Show exchange counter, quest notifications, topic/quest-styled options, auto-close on limit |
| `src/components/ModeSelect.jsx` | Add world state to "Reset Everything" |
| `src/index.css` | Add new styles (extra detail, quest notification, topic options, exchange counter) |

---

## 11. Testing Checklist

### NPC conversation mechanics
- [ ] First visit to Maren: shows tier 1 topics as options + "Walk away"
- [ ] After 4 exchanges, Maren delivers dismissal and conversation ends automatically
- [ ] Selecting a topic: NPC response is guided by the topic summary, delivered in character
- [ ] After revealing a topic, it doesn't appear as an option on the next visit
- [ ] At disposition < 50, tier 2 topics are not available
- [ ] Raising disposition to 50+ makes tier 2 topics appear on next visit
- [ ] When all available topics are revealed: NPC delivers `nothingNew` and conversation closes after 1 exchange
- [ ] Topics revealed persist across page refresh (stored in world state)

### Quest system
- [ ] Quest appears as option when disposition meets `unlockDisposition`
- [ ] Accepting a quest changes status from `available` to `active`
- [ ] Fetch quest: revealing the required topic from another NPC completes the quest
- [ ] Story-linked quest: completing the required hook completes the quest
- [ ] On next visit to NPC with completed quest: NPC delivers `completionDialogue`, rewards applied
- [ ] Disposition bonus from quest reward is visible in disposition badge
- [ ] Flag granted by quest reward unlocks higher tier content

### Story → Explore crossover
- [ ] Complete HOOK-01 survived → The Crossroads shows extra detail about silenced bells
- [ ] Complete HOOK-01 died → The Crossroads shows extra detail about your name on the memorial
- [ ] After story completion, NPC dialogue flags cause NPCs to reference story events
- [ ] Faction rep bonus from story effects is reflected in explore mode

### Explore → Story crossover
- [ ] Discover lore through NPC topics → start a story → protagonist's knowledge clause reflects discoveries
- [ ] Build NPC disposition in explore → story prose reflects warmer/colder relationships
- [ ] Faction rep from explore → story knowledge clause mentions faction standing

### Shared state
- [ ] World state persists across page refresh
- [ ] "Reset Everything" clears world state + journal + explore state
- [ ] Disposition changes in explore mode persist and are read correctly on next visit
- [ ] Quest states persist across sessions
