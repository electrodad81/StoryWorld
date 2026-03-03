// src/hooks/useExploreEngine.js
// Central hook for exploration mode state and actions.
// Integrates NPC tiers, exchange limits, quests, and world state.

import { useState, useCallback } from 'react';
import { streamNPCDialogue, generateFlavorOption, calculateDispositionDelta } from '../engine/ExploreEngine.js';
import { loadJournal, addExploreLore } from '../services/journal.js';
import { loadWorldState, saveWorldState, evaluateQuests, grantQuestReward, getAvailableTopics, getCrossNpcQuestTopics } from '../services/worldState.js';

const EXPLORE_STORAGE_KEY = 'gloamreach_explore_state';
const MAX_EXCHANGES = 4;

export default function useExploreEngine() {
  // World data (loaded from JSON)
  const [worldData, setWorldData] = useState(null);

  // Player exploration state
  const [currentLocationId, setCurrentLocationId] = useState('vale-crossroads');
  const [discoveredLocations, setDiscoveredLocations] = useState(['vale-crossroads', 'thornwatch-inn', 'old-mill']);
  const [inventory, setInventory] = useState([]);
  const [playerStates, setPlayerStates] = useState([]);

  // UI state
  const [activeNpc, setActiveNpc] = useState(null);
  const [conversationHistory, setConversationHistory] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [interactionOptions, setInteractionOptions] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [explorePhase, setExplorePhase] = useState('map');

  // NPC conversation tracking
  const [exchangeCount, setExchangeCount] = useState(0);
  const [currentTopic, setCurrentTopic] = useState(null);
  const [questNotification, setQuestNotification] = useState(null);
  const [nothingNewMessage, setNothingNewMessage] = useState(null);

  // Player profile
  const [playerName, setPlayerName] = useState('');

  // Load world data
  const loadWorldData = useCallback(async () => {
    if (worldData) return worldData;
    try {
      const res = await fetch('/data/withered-vale.json');
      const data = await res.json();
      setWorldData(data);
      return data;
    } catch {
      return null;
    }
  }, [worldData]);

  // Get NPC dispositions from world state (falling back to base)
  const getNpcDisposition = useCallback((npcId) => {
    if (!worldData) return 50;
    const npc = worldData.npcs.find(n => n.id === npcId);
    if (!npc) return 50;
    const ws = loadWorldState();
    return ws.npcDispositions[npcId] ?? npc.baseDisposition;
  }, [worldData]);

  // Get all NPC dispositions as a map (for UI display)
  const getNpcDispositions = useCallback(() => {
    if (!worldData) return {};
    const ws = loadWorldState();
    const result = {};
    for (const npc of worldData.npcs) {
      result[npc.id] = ws.npcDispositions[npc.id] ?? npc.baseDisposition;
    }
    return result;
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

  // Build hybrid interaction options: authored topics + quests + cross-NPC quests + 1 AI flavor + walk away
  const buildInteractionOptions = useCallback(async (npc, availableTopics, availableQuests, updatedHistory, crossQuestTopics) => {
    const options = [];

    // Add topic options (authored)
    for (const topic of availableTopics) {
      options.push({
        label: `Ask about ${topic.subject.toLowerCase()}`,
        type: 'topic',
        topic,
      });
    }

    // Add quest options (this NPC's own quests)
    for (const quest of availableQuests) {
      const ws = loadWorldState();
      const status = ws.npcQuests[quest.id]?.status;
      if (status === 'available') {
        options.push({
          label: `${quest.title} [Quest]`,
          type: 'quest',
          quest,
        });
      }
    }

    // Add cross-NPC quest dialogue options
    for (const cq of (crossQuestTopics || [])) {
      options.push({
        label: `${cq.quest.title} [Quest]`,
        type: 'cross_quest',
        crossQuest: cq,
      });
    }

    // Add one AI-generated flavor option
    try {
      const disposition = getNpcDisposition(npc.id);
      const flavor = await generateFlavorOption(npc, {
        disposition,
        conversationHistory: updatedHistory || [],
      });
      options.push({
        label: flavor,
        type: 'flavor',
      });
    } catch {
      options.push({
        label: 'Make small talk',
        type: 'flavor',
      });
    }

    return options;
  }, [getNpcDisposition]);

  // Travel to a location
  const travelTo = useCallback((locationId) => {
    if (!worldData) return;

    if (locationId === currentLocationId) {
      setActiveNpc(null);
      setConversationHistory([]);
      setInteractionOptions([]);
      setExplorePhase('location');
      return;
    }

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

    if (!discoveredLocations.includes(locationId)) {
      setDiscoveredLocations(prev => [...prev, locationId]);
    }

    // Write location discovery to journal
    try {
      const journal = loadJournal();
      addExploreLore(journal, { locations: [locationId] });
    } catch {}
  }, [worldData, getCurrentLocation, discoveredLocations]);

  // Start talking to an NPC
  const engageNpc = useCallback(async (npcId) => {
    if (!worldData) return;
    const npc = worldData.npcs.find(n => n.id === npcId);
    if (!npc) return;

    setActiveNpc(npc);
    setConversationHistory([]);
    setExchangeCount(0);
    setCurrentTopic(null);
    setQuestNotification(null);
    setNothingNewMessage(null);
    setExplorePhase('conversation');

    // Write NPC encounter to journal
    try {
      const journal = loadJournal();
      const loreUpdate = { npcs: [npc.name] };
      if (npc.faction) loreUpdate.factions = [npc.faction];
      addExploreLore(journal, loreUpdate);
    } catch {}

    // Load world state and evaluate quests
    const ws = loadWorldState();
    const journal = loadJournal();
    const { completedQuests, availableQuests } = evaluateQuests(npc, ws, journal);

    // Handle completed quests — grant rewards, show notification
    if (completedQuests.length > 0) {
      const quest = completedQuests[0]; // Handle one at a time
      grantQuestReward(quest, npc, ws);
      saveWorldState(ws);

      // Show completion dialogue as the opening
      setQuestNotification({ quest, type: 'completed' });
      setConversationHistory([{ role: 'assistant', content: quest.completionDialogue }]);

      // Add quest reward lore to journal
      if (quest.reward?.loreRevealed) {
        try {
          addExploreLore(loadJournal(), quest.reward.loreRevealed);
        } catch {}
      }
    } else {
      saveWorldState(ws);
    }

    // Get available topics
    const wsAfterQuests = loadWorldState();
    const availableTopics = getAvailableTopics(npc, wsAfterQuests);

    // Get cross-NPC quest topics (quests from other NPCs that need info from this NPC)
    const crossQuestTopics = worldData?.npcs
      ? getCrossNpcQuestTopics(npc, wsAfterQuests, worldData.npcs)
      : [];

    // Nothing new?
    if (availableTopics.length === 0 && availableQuests.length === 0 && completedQuests.length === 0 && crossQuestTopics.length === 0) {
      setNothingNewMessage(npc.nothingNew || "Nothing new to discuss.");
      setInteractionOptions([]);
      setIsGenerating(false);
      return;
    }

    // Build interaction options
    setIsGenerating(true);
    try {
      const initialHistory = completedQuests.length > 0
        ? [{ role: 'assistant', content: completedQuests[0].completionDialogue }]
        : [];
      const options = await buildInteractionOptions(npc, availableTopics, availableQuests, initialHistory, crossQuestTopics);
      setInteractionOptions(options);
    } catch {
      setInteractionOptions([{ label: 'Make small talk', type: 'flavor' }]);
    }
    setIsGenerating(false);
  }, [worldData, buildInteractionOptions]);

  // Send a message to the active NPC (called when player picks an option)
  const talkToNpc = useCallback(async (option) => {
    if (!activeNpc || isStreaming || isGenerating) return;

    const location = getCurrentLocation();
    const ws = loadWorldState();
    const disposition = ws.npcDispositions[activeNpc.id] ?? activeNpc.baseDisposition;
    const dialogueFlags = ws.npcDialogueFlags[activeNpc.id] || [];

    // Determine the player message and topic context
    const playerMessage = typeof option === 'string' ? option : option.label;
    let topicForEngine = null;
    let crossQuestRebuff = false;

    if (option.type === 'cross_quest' && option.crossQuest) {
      const cq = option.crossQuest;
      if (cq.meetsDisposition) {
        // Disposition is high enough — treat like a normal topic reveal
        topicForEngine = cq.topic;
        setCurrentTopic(cq.topic);
      } else {
        // Disposition too low — NPC will rebuff with a hint
        crossQuestRebuff = true;
        setCurrentTopic(null);
      }
    } else if (option.type === 'topic' && option.topic) {
      topicForEngine = option.topic;
      setCurrentTopic(option.topic);
    } else if (option.type === 'quest' && option.quest) {
      // Accept the quest
      ws.npcQuests[option.quest.id] = { status: 'active' };
      saveWorldState(ws);
      setQuestNotification({ quest: option.quest, type: 'accepted' });
    } else {
      setCurrentTopic(null);
    }

    const newExchangeCount = exchangeCount + 1;
    setExchangeCount(newExchangeCount);

    // Check if this is the last exchange — NPC dismisses
    const isDismissal = newExchangeCount >= MAX_EXCHANGES;

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
        currentTopic: topicForEngine,
        dialogueFlags,
        exchangeCount: newExchangeCount,
        isDismissal,
        crossQuestRebuff: crossQuestRebuff
          ? { questTitle: option.crossQuest.quest.title, sourceNpcName: option.crossQuest.sourceNpcName, topicSubject: option.crossQuest.topic.subject }
          : null,
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

    // Mark topic as revealed if one was discussed
    if (topicForEngine) {
      const wsNow = loadWorldState();
      if (!wsNow.revealedTopics.includes(topicForEngine.id)) {
        wsNow.revealedTopics.push(topicForEngine.id);
        saveWorldState(wsNow);
      }

      // Add lore from topic to journal
      if (topicForEngine.loreRevealed) {
        try {
          addExploreLore(loadJournal(), topicForEngine.loreRevealed);
        } catch {}
      }
    }

    // Update disposition
    const delta = calculateDispositionDelta(playerMessage, activeNpc);
    if (delta !== 0) {
      const wsDisp = loadWorldState();
      const currentDisp = wsDisp.npcDispositions[activeNpc.id] ?? activeNpc.baseDisposition;
      wsDisp.npcDispositions[activeNpc.id] = Math.max(0, Math.min(100, currentDisp + delta));

      // Update faction reputation if NPC has faction
      if (activeNpc.faction) {
        wsDisp.factionReputation[activeNpc.faction] =
          Math.max(-100, Math.min(100, (wsDisp.factionReputation[activeNpc.faction] || 0) + Math.round(delta / 2)));
      }
      saveWorldState(wsDisp);
    }

    // If dismissed, auto-close after showing the response
    if (isDismissal) {
      setInteractionOptions([]);
      return;
    }

    // Generate new interaction options for next exchange
    setIsGenerating(true);
    try {
      const wsRefresh = loadWorldState();
      const journal = loadJournal();
      const { availableQuests } = evaluateQuests(activeNpc, wsRefresh, journal);
      saveWorldState(wsRefresh);
      const wsLatest = loadWorldState();
      const availableTopics = getAvailableTopics(activeNpc, wsLatest);
      const crossQuestTopics = worldData?.npcs
        ? getCrossNpcQuestTopics(activeNpc, wsLatest, worldData.npcs)
        : [];
      const options = await buildInteractionOptions(activeNpc, availableTopics, availableQuests, updatedHistory, crossQuestTopics);
      setInteractionOptions(options);
    } catch {
      setInteractionOptions([{ label: 'Make small talk', type: 'flavor' }]);
    }
    setIsGenerating(false);
  }, [activeNpc, isStreaming, isGenerating, conversationHistory, getCurrentLocation, playerName, exchangeCount, buildInteractionOptions]);

  // Leave conversation, return to location view
  const leaveConversation = useCallback(() => {
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setStreamedText('');
    setExchangeCount(0);
    setCurrentTopic(null);
    setQuestNotification(null);
    setNothingNewMessage(null);
    setExplorePhase('location');
  }, []);

  // Return to map view
  const openMap = useCallback(() => {
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setExchangeCount(0);
    setCurrentTopic(null);
    setQuestNotification(null);
    setNothingNewMessage(null);
    setExplorePhase('map');
  }, []);

  // Discover a location (add to discovered list without traveling)
  const discoverLocation = useCallback((locationId) => {
    setDiscoveredLocations(prev =>
      prev.includes(locationId) ? prev : [...prev, locationId]
    );
  }, []);

  // Set starting location (for graveyard respawn etc.)
  const setStartLocation = useCallback((locationId) => {
    setCurrentLocationId(locationId);
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setExplorePhase('location');
  }, []);

  // Journal sub-view
  const [prevPhase, setPrevPhase] = useState('map');

  const openJournal = useCallback(() => {
    setPrevPhase(explorePhase);
    setExplorePhase('journal');
  }, [explorePhase]);

  const closeJournal = useCallback(() => {
    setExplorePhase(prevPhase);
  }, [prevPhase]);

  // Persistence: load from localStorage
  const loadState = useCallback(() => {
    try {
      const raw = localStorage.getItem(EXPLORE_STORAGE_KEY);
      if (!raw) return false;
      const state = JSON.parse(raw);
      if (state.currentLocationId) setCurrentLocationId(state.currentLocationId);
      if (state.discoveredLocations) setDiscoveredLocations(state.discoveredLocations);
      if (state.inventory) setInventory(state.inventory);
      if (state.playerStates) setPlayerStates(state.playerStates);
      if (state.playerName) setPlayerName(state.playerName);
      return true;
    } catch {
      return false;
    }
  }, []);

  // Persistence: save to localStorage
  const saveState = useCallback(() => {
    try {
      const state = {
        currentLocationId,
        discoveredLocations,
        inventory,
        playerStates,
        playerName,
      };
      localStorage.setItem(EXPLORE_STORAGE_KEY, JSON.stringify(state));
    } catch {}
  }, [currentLocationId, discoveredLocations, inventory, playerStates, playerName]);

  // Reset exploration state
  const resetExplore = useCallback(() => {
    setCurrentLocationId('vale-crossroads');
    setDiscoveredLocations(['vale-crossroads', 'thornwatch-inn', 'old-mill']);
    setInventory([]);
    setPlayerStates([]);
    setActiveNpc(null);
    setConversationHistory([]);
    setInteractionOptions([]);
    setStreamedText('');
    setIsStreaming(false);
    setIsGenerating(false);
    setExplorePhase('map');
    setExchangeCount(0);
    setCurrentTopic(null);
    setQuestNotification(null);
    setNothingNewMessage(null);
    try { localStorage.removeItem(EXPLORE_STORAGE_KEY); } catch {}
  }, []);

  return {
    // World data
    worldData,
    loadWorldData,

    // State
    currentLocationId,
    discoveredLocations,
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

    // NPC conversation state
    exchangeCount,
    currentTopic,
    questNotification,
    nothingNewMessage,

    // Derived data
    getCurrentLocation,
    getNpcsAtLocation,
    getConnectedLocations,
    getNpcDisposition,
    getNpcDispositions,

    // Actions
    travelTo,
    engageNpc,
    talkToNpc,
    leaveConversation,
    openMap,
    openJournal,
    closeJournal,
    discoverLocation,
    setStartLocation,

    // Persistence
    saveState,
    loadState,
    resetExplore,
  };
}
