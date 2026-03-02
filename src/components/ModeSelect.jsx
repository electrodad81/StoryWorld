// src/components/ModeSelect.jsx
// Two-card mode selection screen.

export default function ModeSelect({ onSelectMode }) {
  return (
    <div className="app-main">
      <div className="mode-select-panel">
        <h2>Choose Your Path</h2>

        <div className="mode-cards">
          <div className="mode-card" onClick={() => onSelectMode('story')}>
            <h3>Story Mode</h3>
            <p>A guided narrative arc. Choose your hook, shape the story through decisions, and face consequences.</p>
          </div>

          <div className="mode-card" onClick={() => onSelectMode('explore')}>
            <h3>Explore Mode</h3>
            <p>Wander the Withered Vale freely. Speak with its inhabitants, build your reputation, and uncover secrets.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
