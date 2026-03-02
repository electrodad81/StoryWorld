// src/hooks/useExploreEngine.js
// Central hook for exploration mode state and actions.

import { useState, useCallback } from 'react';
import { streamNPCDialogue, generateInteractionOptions, calculateDispositionDelta } from '../engine/ExploreEngine.js';
import { loadJournal, addExploreLore } from '../services/journal.js';

const EXPLORE_STORAGE_KEY = 'gloamreach_explore_state';

export default function useExploreEngine() {
  // World data (loaded from JSON)
  const [worldData, setWorldData] = useState(null);

  // Player exploration state
  const [currentLocationId, setCurrentLocationId] = useState('vale-crossroads');
  const [discoveredLocations, setDiscoveredLocations] = useState(['vale-crossroads', 'thornwatch-inn', 'old-mill']);
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

  // Travel to a location (connected or current)
  const travelTo = useCallback((locationId) => {
    if (!worldData) return;

    // Allow clicking current location to enter its detail view
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

    // Discover the location if new
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
    setExplorePhase('conversation');

    // Write NPC encounter to journal
    try {
      const journal = loadJournal();
      const loreUpdate = { npcs: [npc.name] };
      if (npc.faction) loreUpdate.factions = [npc.faction];
      addExploreLore(journal, loreUpdate);
    } catch {}

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
    setDiscoveredLocations(['vale-crossroads', 'thornwatch-inn', 'old-mill']);
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
