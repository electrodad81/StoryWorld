// src/components/ModeSelect.jsx
// Three-card mode selection: Story, Explore (gated), Journal.
// Reset Everything now also clears world state.

export default function ModeSelect({ onSelectMode, explorationUnlocked, hasJournalEntries, onResetJournal, onResetAll }) {
  return (
    <div className="app-main">
      <div className="mode-select-panel">
        <h2>Choose Your Path</h2>

        <div className="mode-cards">
          <div className="mode-card" onClick={() => onSelectMode('story')}>
            <h3>Story Mode</h3>
            <p>A guided narrative arc. Choose your hook, shape the story through decisions, and face consequences.</p>
          </div>

          <div
            className={`mode-card${!explorationUnlocked ? ' locked' : ''}`}
            onClick={explorationUnlocked ? () => onSelectMode('explore') : undefined}
          >
            <div className="mode-card-header">
              <h3>Explore Mode</h3>
              {!explorationUnlocked && <span className="lock-icon">&#128274;</span>}
            </div>
            {explorationUnlocked ? (
              <p>Wander the Withered Vale freely. Speak with its inhabitants, build your reputation, and uncover secrets.</p>
            ) : (
              <p className="mode-hint">Complete a story to unlock the Withered Vale</p>
            )}
          </div>

          <div
            className={`mode-card${!hasJournalEntries ? ' subtle' : ''}`}
            onClick={() => onSelectMode('journal')}
          >
            <h3>Journal</h3>
            <p>{hasJournalEntries
              ? 'Review your stories, lore discoveries, and achievements.'
              : 'Your journal is empty. Complete a story to begin recording your history.'
            }</p>
          </div>
        </div>

        {/* Dev/testing resets */}
        <div className="dev-resets">
          <button className="dev-reset-btn" onClick={onResetJournal}>
            Reset Journal
          </button>
          <button className="dev-reset-btn danger" onClick={onResetAll}>
            Reset Everything
          </button>
        </div>
      </div>
    </div>
  );
}
