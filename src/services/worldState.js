// src/services/worldState.js
// Shared world state layer — bridges story and explore modes via localStorage.

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

export function loadWorldState() {
  try {
    const raw = localStorage.getItem(WORLD_STATE_KEY);
    if (!raw) return structuredClone(DEFAULT_STATE);
    const saved = JSON.parse(raw);
    // Merge with defaults so new fields are always present
    return {
      ...structuredClone(DEFAULT_STATE),
      ...saved,
      factionReputation: { ...DEFAULT_STATE.factionReputation, ...saved.factionReputation },
      knownLore: { ...structuredClone(DEFAULT_STATE.knownLore), ...saved.knownLore },
    };
  } catch {
    return structuredClone(DEFAULT_STATE);
  }
}

export function saveWorldState(state) {
  try {
    localStorage.setItem(WORLD_STATE_KEY, JSON.stringify(state));
  } catch {}
}

export function resetWorldState() {
  try {
    localStorage.removeItem(WORLD_STATE_KEY);
  } catch {}
}

/**
 * Sync knownLore from journal's loreDiscovered into world state.
 */
export function syncKnownLoreFromJournal(worldState, journal) {
  if (journal?.loreDiscovered) {
    worldState.knownLore = { ...journal.loreDiscovered };
  }
  return worldState;
}

/**
 * Apply story outcome effects to world state.
 * Called from StoryComplete after journal write.
 */
export function applyStoryEffects(hookId, outcome, storyEffectsData, journal) {
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
      ws.npcDialogueFlags[npcId] = [...new Set([...existing, ...flags])];
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

/**
 * Evaluate quest states for an NPC against current world state and journal.
 * Updates quest statuses in world state (locked → available, active → completed).
 * Returns { completedQuests, availableQuests } for UI display.
 */
export function evaluateQuests(npc, worldState, journal) {
  const completedQuests = [];
  const availableQuests = [];

  if (!npc.quests) return { completedQuests, availableQuests };

  const disposition = worldState.npcDispositions[npc.id] ?? npc.baseDisposition;

  for (const quest of npc.quests) {
    const current = worldState.npcQuests[quest.id];

    // Check if quest should unlock (locked → available)
    if (!current || current.status === 'locked') {
      if (disposition >= quest.unlockDisposition) {
        worldState.npcQuests[quest.id] = { status: 'available' };
      }
    }

    // Check if active quest is now complete
    if (current?.status === 'active') {
      if (checkRequirement(quest.requirement, worldState, journal)) {
        worldState.npcQuests[quest.id] = { status: 'completed', completedAt: new Date().toISOString() };
        completedQuests.push(quest);
      }
    }

    // Re-read status after potential changes
    const updated = worldState.npcQuests[quest.id];
    if (updated?.status === 'available') {
      availableQuests.push(quest);
    }
    if (updated?.status === 'completed' && !updated.rewardGranted) {
      completedQuests.push(quest);
    }
  }

  // Evaluate redemption quest: appears when disposition is hostile (<30)
  if (npc.redemptionQuest && disposition < 30) {
    const rq = npc.redemptionQuest;
    const rqState = worldState.npcQuests[rq.id];

    if (!rqState || rqState.status === 'locked') {
      worldState.npcQuests[rq.id] = { status: 'available' };
    }

    if (rqState?.status === 'active') {
      // Redemption quests use hook_completed requirement
      const hookDone = journal.storiesPlayed?.some(s => s.hookId === rq.hookId);
      if (hookDone) {
        worldState.npcQuests[rq.id] = { status: 'completed', completedAt: new Date().toISOString() };
        completedQuests.push(rq);
      }
    }

    const rqUpdated = worldState.npcQuests[rq.id];
    if (rqUpdated?.status === 'available') {
      availableQuests.push(rq);
    }
    if (rqUpdated?.status === 'completed' && !rqUpdated.rewardGranted) {
      completedQuests.push(rq);
    }
  }

  return { completedQuests, availableQuests };
}

/**
 * Check if a quest requirement is met.
 */
function checkRequirement(req, worldState, journal) {
  if (!req) return false;
  switch (req.type) {
    case 'npc_topic_revealed':
      return (worldState.revealedTopics || []).includes(req.topicId);
    case 'hook_survived':
      return journal.storiesPlayed.some(s => s.hookId === req.hookId && s.outcome === 'survived');
    case 'hook_completed':
      return journal.storiesPlayed.some(s => s.hookId === req.hookId);
    case 'location_visited':
      return (journal.loreDiscovered?.locations || []).includes(req.locationId);
    default:
      return false;
  }
}

/**
 * Grant quest rewards: disposition bonus, flag, lore.
 */
export function grantQuestReward(quest, npc, worldState) {
  if (!quest.reward) return;

  // Disposition bonus
  if (quest.reward.dispositionBonus) {
    const current = worldState.npcDispositions[npc.id] ?? npc.baseDisposition;
    worldState.npcDispositions[npc.id] = Math.min(100, current + quest.reward.dispositionBonus);
  }

  // Flag granted
  if (quest.reward.flagGranted) {
    const existing = worldState.npcDialogueFlags[npc.id] || [];
    if (!existing.includes(quest.reward.flagGranted)) {
      worldState.npcDialogueFlags[npc.id] = [...existing, quest.reward.flagGranted];
    }
  }

  // Mark reward as granted
  worldState.npcQuests[quest.id] = {
    ...worldState.npcQuests[quest.id],
    rewardGranted: true,
  };
}

/**
 * Find active quests from OTHER NPCs that require revealing a topic from this NPC.
 * Returns array of { quest, sourceNpc, topic, meetsDisposition } objects.
 */
export function getCrossNpcQuestTopics(npc, worldState, allNpcs) {
  const results = [];
  const disposition = worldState.npcDispositions[npc.id] ?? npc.baseDisposition;
  const flags = worldState.npcDialogueFlags[npc.id] || [];
  const revealed = worldState.revealedTopics || [];

  for (const otherNpc of allNpcs) {
    if (otherNpc.id === npc.id || !otherNpc.quests) continue;

    for (const quest of otherNpc.quests) {
      const status = worldState.npcQuests[quest.id]?.status;
      if (status !== 'active') continue;
      if (quest.requirement?.type !== 'npc_topic_revealed') continue;

      // Check if the required topic belongs to THIS NPC
      const targetTopicId = quest.requirement.topicId;
      if (revealed.includes(targetTopicId)) continue; // Already revealed

      for (const tier of npc.tiers || []) {
        const topic = tier.topics.find(t => t.id === targetTopicId);
        if (!topic) continue;

        // Check if disposition + flags meet the tier requirements
        const flagOk = !tier.requiresFlag || flags.includes(tier.requiresFlag);
        const dispOk = disposition >= tier.dispositionRequired;
        const meetsDisposition = dispOk && flagOk;

        results.push({
          quest,
          sourceNpcId: otherNpc.id,
          sourceNpcName: otherNpc.name,
          topic,
          meetsDisposition,
          dispositionRequired: tier.dispositionRequired,
        });
      }
    }
  }

  return results;
}

/**
 * Get available topics for an NPC based on disposition, flags, and prior conversation.
 * Higher tiers require at least one revealed topic from a lower tier of the same NPC,
 * so players don't see topics they have no narrative reason to know about.
 */
export function getAvailableTopics(npc, worldState) {
  if (!npc.tiers) return [];

  const disposition = worldState.npcDispositions[npc.id] ?? npc.baseDisposition;
  const flags = worldState.npcDialogueFlags[npc.id] || [];
  const revealed = worldState.revealedTopics || [];

  // Collect all topic IDs per tier index for prior-conversation gating
  const tierTopicIds = npc.tiers.map(tier => tier.topics.map(t => t.id));

  const available = [];

  for (let i = 0; i < npc.tiers.length; i++) {
    const tier = npc.tiers[i];

    // Check disposition threshold
    if (disposition < tier.dispositionRequired) continue;

    // Check required flag if any
    if (tier.requiresFlag && !flags.includes(tier.requiresFlag)) continue;

    // For tiers above the first: require at least one revealed topic from any lower tier
    if (i > 0) {
      const hasLowerTierReveal = tierTopicIds
        .slice(0, i)
        .some(ids => ids.some(id => revealed.includes(id)));
      if (!hasLowerTierReveal) continue;
    }

    // Add unrevealed topics from this tier
    for (const topic of tier.topics) {
      if (!revealed.includes(topic.id)) {
        available.push(topic);
      }
    }
  }

  return available;
}
