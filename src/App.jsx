// src/App.jsx
// Root component — conditional render based on game phase and mode.
// Sidebar is a slide-over drawer toggled by hamburger menu.

import { useState, useEffect } from 'react';
import './App.css';
import useStoryEngine from './hooks/useStoryEngine.js';
import useExploreEngine from './hooks/useExploreEngine.js';
import ApiKeyInput from './components/ApiKeyInput.jsx';
import ModeSelect from './components/ModeSelect.jsx';
import CharacterSetup from './components/CharacterSetup.jsx';
import StoryView from './components/StoryView.jsx';
import ExploreView from './components/ExploreView.jsx';
import Sidebar from './components/Sidebar.jsx';

export default function App() {
  const engine = useStoryEngine();
  const explore = useExploreEngine();
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [gameMode, setGameMode] = useState(null); // 'story' | 'explore' | null

  // Load lore and world data on mount
  useEffect(() => {
    engine.loadLore();
    explore.loadWorldData();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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
        <ModeSelect onSelectMode={setGameMode} />
      </div>
    );
  }

  // Story mode: character setup
  if (gameMode === 'story' && engine.phase === 'setup') {
    return (
      <div className="app-layout">
        <CharacterSetup lore={engine.lore} onBegin={engine.startGame} />
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
              onClick={() => { setSidebarOpen(false); setGameMode(null); }}
            >
              Return to Menu
            </button>
            <button
              className="sidebar-btn danger"
              onClick={() => { setSidebarOpen(false); explore.resetExplore(); setGameMode(null); }}
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
          onReset={() => { setSidebarOpen(false); engine.resetGame(); setGameMode(null); }}
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
        onNewStory={engine.newStory}
        beat={engine.beat}
        onMenuToggle={() => setSidebarOpen((v) => !v)}
      />
    </div>
  );
}
