// src/App.jsx
// Root component — conditional render based on game phase.
// Sidebar is now a slide-over drawer toggled by hamburger menu.

import { useState, useEffect } from 'react';
import './App.css';
import useStoryEngine from './hooks/useStoryEngine.js';
import ApiKeyInput from './components/ApiKeyInput.jsx';
import CharacterSetup from './components/CharacterSetup.jsx';
import StoryView from './components/StoryView.jsx';
import Sidebar from './components/Sidebar.jsx';

export default function App() {
  const engine = useStoryEngine();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  // Load lore on mount
  useEffect(() => {
    engine.loadLore();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // API key needed
  if (engine.phase === 'needKey') {
    return (
      <div className="app-layout">
        <ApiKeyInput onSave={engine.setApiKey} />
      </div>
    );
  }

  // Character setup
  if (engine.phase === 'setup') {
    return (
      <div className="app-layout">
        <CharacterSetup lore={engine.lore} onBegin={engine.startGame} />
      </div>
    );
  }

  // Playing / dead — immersive layout with drawer sidebar
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
          onReset={() => { setSidebarOpen(false); engine.resetGame(); }}
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
