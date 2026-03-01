// src/components/StoryView.jsx
// Fixed three-panel layout: illustration (top 60%), text (middle), choices (bottom).
// Panels stay in place; content updates within them.

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
  onNewStory,
}) {
  const displayText = isStreaming ? streamedText : currentScene;

  return (
    <div className="story-layout">
      {/* ── Panel 1: Illustration (fixed 60% height) ── */}
      <div className="panel-illustration">
        {illustration ? (
          <img src={illustration} alt="Scene illustration" />
        ) : (
          <div className="illus-placeholder">
            {isGenerating ? (
              <LanternLoader caption="Illustration brewing\u2026" />
            ) : (
              <div className="illus-empty">
                <span className="gem">{'\u25c6'}</span>
              </div>
            )}
          </div>
        )}
      </div>

      {/* ── Panel 2: Story text (scrollable middle) ── */}
      <div className="panel-text">
        {displayText ? (
          <StreamingText text={displayText} isStreaming={isStreaming} />
        ) : isGenerating ? (
          <LanternLoader caption="Crafting your scene\u2026" />
        ) : null}

        {isDead && (
          <div className="death-screen">
            <h3>You Died.</h3>
            <p>Your adventure ends in tragedy.</p>
            <p>
              Use the sidebar to <em>Start a New Story</em> or <em>Reset</em> the session.
            </p>
            <button className="sidebar-btn" onClick={onNewStory} style={{ marginTop: '0.75rem' }}>
              Start New Story
            </button>
          </div>
        )}
      </div>

      {/* ── Panel 3: Choices (fixed bottom) ── */}
      <div className="panel-choices">
        {isDead ? null : isGenerating && !displayText ? null : (
          <ChoiceGrid
            choices={choices}
            onChoose={onChoose}
            disabled={isStreaming || isGenerating}
          />
        )}
      </div>
    </div>
  );
}
