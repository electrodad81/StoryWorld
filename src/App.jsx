// src/App.jsx
// Root component — conditional render based on game phase, mode, and journal state.
// Sidebar is a slide-over drawer toggled by hamburger menu.

import { useState, useEffect, useCallback } from 'react';
import './App.css';
import useStoryEngine from './hooks/useStoryEngine.js';
import useExploreEngine from './hooks/useExploreEngine.js';
import { loadJournal, saveJournal } from './services/journal.js';
import ApiKeyInput from './components/ApiKeyInput.jsx';
import ModeSelect from './components/ModeSelect.jsx';
import CharacterSetup from './components/CharacterSetup.jsx';
import StoryView from './components/StoryView.jsx';
import StoryComplete from './components/StoryComplete.jsx';
import JournalView from './components/JournalView.jsx';
import ExploreView from './components/ExploreView.jsx';
import Sidebar from './components/Sidebar.jsx';

export default function App() {
  const engine = useStoryEngine();
  const explore = useExploreEngine();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [gameMode, setGameMode] = useState(null); // 'story' | 'explore' | 'journal' | null
  const [journal, setJournal] = useState(() => loadJournal());

  // Refresh journal from localStorage
  const refreshJournal = useCallback(() => {
    setJournal(loadJournal());
  }, []);

  // Load lore and world data on mount
  useEffect(() => {
    engine.loadLore();
    explore.loadWorldData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Refresh journal when returning to menu
  const handleReturnToMenu = useCallback(() => {
    setSidebarOpen(false);
    setGameMode(null);
    refreshJournal();
  }, [refreshJournal]);

  // API key needed
  if (engine.phase === 'needKey') {
    return (
      <div className="app-layout">
        <ApiKeyInput onSave={engine.setApiKey} />
      </div>
    );
  }

  // Mode selection (after API key, before gameplay)
  if (!gameMode) {
    return (
      <div className="app-layout">
        <ModeSelect
          onSelectMode={setGameMode}
          explorationUnlocked={journal.explorationUnlocked}
          hasJournalEntries={journal.storiesPlayed.length > 0}
          onResetJournal={() => {
            localStorage.removeItem('gloamreach_journal');
            refreshJournal();
          }}
          onResetAll={() => {
            localStorage.removeItem('gloamreach_journal');
            localStorage.removeItem('gloamreach_explore_state');
            engine.resetGame();
            explore.resetExplore();
            refreshJournal();
          }}
        />
      </div>
    );
  }

  // Journal view
  if (gameMode === 'journal') {
    return (
      <div className="app-layout">
        <JournalView
          journal={journal}
          lore={engine.lore}
          onBack={handleReturnToMenu}
        />
      </div>
    );
  }

  // Story mode: complete screen
  if (gameMode === 'story' && engine.phase === 'complete') {
    return (
      <div className="app-layout">
        <StoryComplete
          playerProfile={engine.playerProfile}
          history={engine.history}
          beat={engine.beat}
          sceneCount={engine.sceneCount}
          isDead={engine.isDead}
          lore={engine.lore}
          onPlayAgain={() => { refreshJournal(); engine.newStory(); }}
          onReturnToMenu={handleReturnToMenu}
          onViewJournal={() => { refreshJournal(); setGameMode('journal'); }}
        />
      </div>
    );
  }

  // Story mode: character setup
  if (gameMode === 'story' && engine.phase === 'setup') {
    return (
      <div className="app-layout">
        <CharacterSetup
          lore={engine.lore}
          unlockedHooks={journal.unlockedHooks}
          onBegin={engine.startGame}
        />
      </div>
    );
  }

  // Explore mode
  if (gameMode === 'explore') {
    return (
      <div className="app-layout">
        {/* Drawer backdrop */}
        {sidebarOpen && (
          <div className="drawer-backdrop" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Drawer */}
        <div className={`drawer${sidebarOpen ? ' open' : ''}`}>
          <div className="app-sidebar">
            <h2>Exploration</h2>
            <button
              className="sidebar-btn"
              onClick={handleReturnToMenu}
            >
              Return to Menu
            </button>
            <button
              className="sidebar-btn danger"
              onClick={() => { setSidebarOpen(false); explore.resetExplore(); setGameMode(null); refreshJournal(); }}
            >
              Reset Exploration
            </button>
          </div>
        </div>

        <ExploreView
          explore={explore}
          onMenuToggle={() => setSidebarOpen((v) => !v)}
        />
      </div>
    );
  }

  // Story mode: playing / dead — immersive layout with drawer sidebar
  return (
    <div className="app-layout">
      {/* Drawer backdrop */}
      {sidebarOpen && (
        <div className="drawer-backdrop" onClick={() => setSidebarOpen(false)} />
      )}

      {/* Drawer */}
      <div className={`drawer${sidebarOpen ? ' open' : ''}`}>
        <Sidebar
          playerProfile={engine.playerProfile}
          onNewStory={() => { setSidebarOpen(false); engine.newStory(); }}
          onReset={() => { setSidebarOpen(false); engine.resetGame(); handleReturnToMenu(); }}
        />
      </div>

      <StoryView
        currentScene={engine.currentScene}
        isStreaming={engine.isStreaming}
        streamedText={engine.streamedText}
        illustration={engine.illustration}
        choices={engine.choices}
        onChoose={engine.chooseOption}
        isGenerating={engine.isGenerating}
        isDead={engine.isDead}
        isResolution={engine.isResolution}
        onNewStory={engine.newStory}
        onComplete={engine.completeStory}
        beat={engine.beat}
        onMenuToggle={() => setSidebarOpen((v) => !v)}
      />
    </div>
  );
}
