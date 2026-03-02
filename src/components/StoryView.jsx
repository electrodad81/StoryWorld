// src/components/StoryView.jsx
// Full-bleed immersive layout: background illustration fills viewport,
// UI layers float on top (scrim, text, choices, top bar).

import { useState, useEffect, useRef, useCallback } from 'react';
import StreamingText from './StreamingText.jsx';
import ChoiceGrid from './ChoiceGrid.jsx';
import LanternLoader from './LanternLoader.jsx';

export default function StoryView({
  currentScene,
  isStreaming,
  streamedText,
  illustration,
  choices,
  onChoose,
  isGenerating,
  isDead,
  isResolution,
  onNewStory,
  onComplete,
  beat,
  onMenuToggle,
}) {
  const displayText = isStreaming ? streamedText : currentScene;

  // Persistent background — never goes blank once first image loads
  const [persistedBg, setPersistedBg] = useState(null);
  const [incomingSrc, setIncomingSrc] = useState(null);
  const [showIncoming, setShowIncoming] = useState(false);
  const currentImgRef = useRef(null);
  const incomingImgRef = useRef(null);

  // When a new illustration arrives, start crossfade
  useEffect(() => {
    if (illustration && illustration !== persistedBg) {
      setIncomingSrc(illustration);
    }
  }, [illustration, persistedBg]);

  const handleIncomingLoad = useCallback(() => {
    setShowIncoming(true);
    // After CSS transition completes, promote incoming to current
    const timer = setTimeout(() => {
      setPersistedBg(incomingSrc);
      setShowIncoming(false);
      setIncomingSrc(null);
    }, 1300);
    return () => clearTimeout(timer);
  }, [incomingSrc]);

  // Auto-scroll text overlay to bottom when new text arrives
  const textRef = useRef(null);
  useEffect(() => {
    if (textRef.current) {
      textRef.current.scrollTop = textRef.current.scrollHeight;
    }
  }, [displayText]);

  return (
    <div className="story-viewport">
      {/* Layer 0: Background illustration */}
      <div className="bg-layer">
        {persistedBg ? (
          <img ref={currentImgRef} className="bg-current" src={persistedBg} alt="" />
        ) : (
          <div className="bg-layer--empty" />
        )}
        {incomingSrc && (
          <img
            ref={incomingImgRef}
            className={`bg-incoming${showIncoming ? ' visible' : ''}`}
            src={incomingSrc}
            alt=""
            onLoad={handleIncomingLoad}
          />
        )}
      </div>

      {/* Layer 1: Gradient scrim for text legibility */}
      <div className="scrim" />

      {/* Layer 4: Top bar */}
      <div className="top-bar">
        <button className="menu-btn" onClick={onMenuToggle}>&#9776;</button>
        {beat && <span className="beat-indicator">{beat.replace('_', ' ')}</span>}
      </div>

      {/* Layer 2: Story text */}
      <div className="text-overlay" ref={textRef}>
        {displayText ? (
          <StreamingText text={displayText} isStreaming={isStreaming} />
        ) : isGenerating ? (
          <LanternLoader caption="Crafting your scene\u2026" />
        ) : null}

        {isDead && (
          <div className="death-screen">
            <h3>You Died.</h3>
            <p>Your adventure ends in tragedy.</p>
            <button className="sidebar-btn" onClick={onComplete} style={{ marginTop: '0.75rem' }}>
              Continue
            </button>
          </div>
        )}

        {isResolution && !isDead && (
          <div className="resolution-screen">
            <button className="sidebar-btn accent" onClick={onComplete} style={{ marginTop: '0.75rem' }}>
              Story Complete
            </button>
          </div>
        )}
      </div>

      {/* Layer 3: Choice buttons */}
      <div className="choice-overlay">
        {isDead ? null : isGenerating && !displayText ? null : (
          <ChoiceGrid choices={choices} onChoose={onChoose} disabled={isStreaming || isGenerating} />
        )}
      </div>
    </div>
  );
}
