// src/App.jsx
// Root component — conditional render based on game phase.

import { useEffect } from 'react';
import './App.css';
import useStoryEngine from './hooks/useStoryEngine.js';
import ApiKeyInput from './components/ApiKeyInput.jsx';
import CharacterSetup from './components/CharacterSetup.jsx';
import StoryView from './components/StoryView.jsx';
import Sidebar from './components/Sidebar.jsx';

export default function App() {
  const engine = useStoryEngine();

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

  // Playing / dead
  return (
    <div className="app-layout">
      <Sidebar
        playerProfile={engine.playerProfile}
        onNewStory={engine.newStory}
        onReset={engine.resetGame}
      />
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
      />
    </div>
  );
}
