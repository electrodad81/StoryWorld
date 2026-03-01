// src/hooks/useStoryEngine.js
// Central hook that owns game state and bridges the engine to React.

import { useState, useCallback, useRef } from 'react';
import { streamScene, generateChoices, generateIllustration, BEATS, BEAT_TARGET_SCENES } from '../engine/StoryEngine.js';
import { ensureBrowserId } from '../services/identity.js';
import { saveSnapshot, loadSnapshot, deleteSnapshot } from '../services/persistence.js';

const ILLUSTRATION_EVERY_N = 2;
const ILLUSTRATION_PHASE = 0;

function shouldIllustrate(sceneIndex) {
  return ((sceneIndex - ILLUSTRATION_PHASE) % ILLUSTRATION_EVERY_N) === 0;
}

export default function useStoryEngine() {
  const [phase, setPhase] = useState(() => {
    const key = localStorage.getItem('openai_api_key');
    if (!key) return 'needKey';
    return 'setup';
  });
  const [playerProfile, setPlayerProfile] = useState(null);
  const [history, setHistory] = useState([]);
  const [currentScene, setCurrentScene] = useState('');
  const [choices, setChoices] = useState([]);
  const [illustration, setIllustration] = useState(null);
  const [beat, setBeat] = useState('exposition');
  const [beatIndex, setBeatIndex] = useState(0);
  const [dangerStreak, setDangerStreak] = useState(0);
  const [sceneCount, setSceneCount] = useState(0);
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamedText, setStreamedText] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDead, setIsDead] = useState(false);
  const [lore, setLore] = useState(null);

  const abortRef = useRef(null);
  const pidRef = useRef(ensureBrowserId());

  // Load lore data
  const loadLore = useCallback(async () => {
    if (lore) return lore;
    try {
      const res = await fetch('/data/lore.json');
      const data = await res.json();
      setLore(data);
      return data;
    } catch {
      return {};
    }
  }, [lore]);

  // Advance the beat based on scene count
  const advanceBeat = useCallback((currentBeatIndex, currentSceneCount) => {
    let cumulative = 0;
    for (let i = 0; i <= currentBeatIndex; i++) {
      cumulative += BEAT_TARGET_SCENES[BEATS[i]] || 2;
    }
    if (currentSceneCount >= cumulative && currentBeatIndex < BEATS.length - 1) {
      const newIdx = currentBeatIndex + 1;
      setBeatIndex(newIdx);
      setBeat(BEATS[newIdx]);
      return BEATS[newIdx];
    }
    return BEATS[currentBeatIndex];
  }, []);

  // Generate a scene, choices, and optionally illustration
  const runTurn = useCallback(async (hist, loreData, profile, currentBeat, currentDanger, scIdx, beatIdx) => {
    setIsGenerating(true);
    setIsStreaming(true);
    setStreamedText('');
    setChoices([]);
    // Don't clear illustration — persistedBg in StoryView keeps the
    // last image visible. New illustrations crossfade in when ready.

    let fullText = '';
    try {
      const gen = streamScene(hist, loreData, {
        beat: currentBeat,
        playerName: profile.name,
        gender: profile.gender,
        dangerStreak: currentDanger,
      });

      for await (const chunk of gen) {
        fullText += chunk;
        setStreamedText(fullText);
      }
    } catch (e) {
      fullText = fullText || `[Error generating scene: ${e.message}]`;
    }

    setIsStreaming(false);

    // Check for death
    const stripped = fullText.trimEnd();
    let dead = false;
    if (stripped.endsWith('[DEATH]')) {
      fullText = stripped.slice(0, -7).trimEnd();
      dead = true;
    }

    setCurrentScene(fullText);
    setIsDead(dead);

    const newHist = [...hist, { role: 'assistant', content: fullText }];
    setHistory(newHist);

    const newSceneCount = scIdx + 1;
    setSceneCount(newSceneCount);

    // Advance beat
    const newBeat = advanceBeat(beatIdx, newSceneCount);

    // Save snapshot
    try {
      saveSnapshot(pidRef.current, {
        scene: fullText,
        history: newHist,
        username: profile.name,
        gender: profile.gender,
        archetype: profile.archetype,
        is_dead: dead,
        beat_index: beatIdx,
        scene_count: newSceneCount,
      });
    } catch {
      // ignore persistence errors in MVP
    }

    if (dead) {
      setChoices([]);
      setIsGenerating(false);
      return;
    }

    // Generate choices
    try {
      const newChoices = await generateChoices(newHist, fullText, loreData);
      setChoices(newChoices);
    } catch {
      setChoices(['Press on', 'Hold back']);
    }

    setIsGenerating(false);

    // Generate illustration (fire and forget, non-blocking)
    if (shouldIllustrate(newSceneCount)) {
      generateIllustration(fullText, profile.gender)
        .then((url) => { if (url) setIllustration(url); })
        .catch(() => {});
    }
  }, [advanceBeat]);

  // Start a new game
  const startGame = useCallback(async (profile) => {
    const loreData = await loadLore();
    setPlayerProfile(profile);
    setPhase('playing');
    setIsDead(false);
    setSceneCount(0);
    setBeatIndex(0);
    setBeat('exposition');
    setDangerStreak(0);
    setCurrentScene('');
    setChoices([]);

    // Show static cover image instantly, then fire DALL-E to replace it
    if (profile.starterHook?.coverImage) {
      setIllustration(profile.starterHook.coverImage);
    } else {
      setIllustration(null);
    }

    // Fire DALL-E illustration from hook blurb (non-blocking, crossfades in when ready)
    if (profile.starterHook) {
      generateIllustration(profile.starterHook.blurb, profile.gender)
        .then((url) => { if (url) setIllustration(url); })
        .catch(() => {});
    }

    // Build initial history with starter hook context
    const hookContext = profile.starterHook
      ? `The player chose the starter hook: "${profile.starterHook.title}" — ${profile.starterHook.blurb}`
      : 'Begin the story.';

    const initialHist = [{ role: 'user', content: hookContext }];
    setHistory(initialHist);

    await runTurn(initialHist, loreData, profile, 'exposition', 0, 0, 0);
  }, [loadLore, runTurn]);

  // Player chooses an option
  const chooseOption = useCallback(async (choiceText) => {
    if (isGenerating || isStreaming) return;

    const loreData = await loadLore();
    const newHist = [...history, { role: 'user', content: choiceText }];
    setHistory(newHist);

    // Simple danger streak: bump if choice text sounds risky
    let newDanger = dangerStreak;
    const riskyWords = /confront|attack|fight|charge|rush|defy|challenge|steal|threaten|provoke/i;
    if (riskyWords.test(choiceText)) {
      newDanger = dangerStreak + 1;
    } else {
      newDanger = Math.max(0, dangerStreak - 1);
    }
    setDangerStreak(newDanger);

    await runTurn(newHist, loreData, playerProfile, beat, newDanger, sceneCount, beatIndex);
  }, [isGenerating, isStreaming, history, dangerStreak, loadLore, playerProfile, beat, sceneCount, beatIndex, runTurn]);

  // Reset game to character creation
  const resetGame = useCallback(() => {
    setPhase('setup');
    setPlayerProfile(null);
    setHistory([]);
    setCurrentScene('');
    setChoices([]);
    setIllustration(null);
    setBeatIndex(0);
    setBeat('exposition');
    setDangerStreak(0);
    setSceneCount(0);
    setIsStreaming(false);
    setStreamedText('');
    setIsGenerating(false);
    setIsDead(false);
    try {
      deleteSnapshot(pidRef.current);
    } catch {
      // ignore
    }
  }, []);

  // Start new story (keep profile, reset game state)
  const newStory = useCallback(() => {
    if (!playerProfile) {
      resetGame();
      return;
    }
    setHistory([]);
    setCurrentScene('');
    setChoices([]);
    setIllustration(null);
    setBeatIndex(0);
    setBeat('exposition');
    setDangerStreak(0);
    setSceneCount(0);
    setIsStreaming(false);
    setStreamedText('');
    setIsGenerating(false);
    setIsDead(false);
    setPhase('setup');
  }, [playerProfile, resetGame]);

  // Set API key and transition to setup
  const setApiKey = useCallback((key) => {
    localStorage.setItem('openai_api_key', key);
    setPhase('setup');
  }, []);

  return {
    phase,
    playerProfile,
    history,
    currentScene,
    choices,
    illustration,
    beat,
    dangerStreak,
    sceneCount,
    isStreaming,
    streamedText,
    isGenerating,
    isDead,
    lore,
    loadLore,
    startGame,
    chooseOption,
    resetGame,
    newStory,
    setApiKey,
  };
}
