// src/components/Sidebar.jsx
// Sidebar with controls and player profile display.

export default function Sidebar({ playerProfile, onNewStory, onReset }) {
  return (
    <aside className="app-sidebar">
      <h2>Gloamreach</h2>

      {playerProfile?.name && (
        <div className="sidebar-profile">
          <div><strong>{playerProfile.name}</strong></div>
          <div>{playerProfile.gender || 'Unspecified'}</div>
          <div>{playerProfile.archetype || 'Default'}</div>
        </div>
      )}

      <button className="sidebar-btn" onClick={onNewStory}>
        Start New Story
      </button>
      <button className="sidebar-btn danger" onClick={onReset}>
        Reset Session
      </button>

      <div className="sidebar-info">
        <p><strong>Start New Story:</strong> Restarts the story using your current selections.</p>
        <p><strong>Reset Session:</strong> Clears progress and selections and returns to the start screen.</p>
      </div>
    </aside>
  );
}
