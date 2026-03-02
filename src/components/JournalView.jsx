// src/components/JournalView.jsx
// Three-tab journal UI: Stories, Codex, Stats & Achievements.

import { useState } from 'react';
import { ACHIEVEMENTS } from '../services/journal.js';

const TABS = ['Stories', 'Codex', 'Stats'];

export default function JournalView({ journal, lore, onBack }) {
  const [activeTab, setActiveTab] = useState('Stories');
  const [expandedStory, setExpandedStory] = useState(null);

  return (
    <div className="app-main">
      <div className="journal-panel">
        <div className="journal-header">
          <button className="journal-back-btn" onClick={onBack}>&larr;</button>
          <h2>Journal</h2>
        </div>

        {/* Tab bar */}
        <div className="journal-tabs">
          {TABS.map(tab => (
            <button
              key={tab}
              className={`journal-tab${activeTab === tab ? ' active' : ''}`}
              onClick={() => setActiveTab(tab)}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="journal-content">
          {activeTab === 'Stories' && (
            <StoriesTab
              stories={journal.storiesPlayed}
              expanded={expandedStory}
              onToggle={(i) => setExpandedStory(expandedStory === i ? null : i)}
            />
          )}
          {activeTab === 'Codex' && (
            <CodexTab loreDiscovered={journal.loreDiscovered} lore={lore} />
          )}
          {activeTab === 'Stats' && (
            <StatsTab stats={journal.stats} achievements={journal.achievements} />
          )}
        </div>
      </div>
    </div>
  );
}

// ---- Stories Tab ----

function StoriesTab({ stories, expanded, onToggle }) {
  if (stories.length === 0) {
    return <p className="journal-empty">No stories completed yet. Your tales will be recorded here.</p>;
  }

  return (
    <div className="journal-stories">
      {[...stories].reverse().map((story, i) => {
        const idx = stories.length - 1 - i;
        const isOpen = expanded === idx;
        return (
          <div key={idx} className="journal-story-card" onClick={() => onToggle(idx)}>
            <div className="story-card-header">
              <span className="story-card-title">{story.hookTitle}</span>
              <span className={`outcome-badge ${story.outcome}`}>
                {story.outcome === 'survived' ? 'Survived' : 'Died'}
              </span>
            </div>
            <div className="story-card-meta">
              <span>{story.sceneCount} scenes</span>
              <span>{new Date(story.timestamp).toLocaleDateString()}</span>
            </div>
            {isOpen && (
              <div className="story-card-details">
                <p className="story-detail-label">Beat reached: <strong>{(story.beatReached || '').replace('_', ' ')}</strong></p>
                {story.keyChoices?.length > 0 && (
                  <>
                    <p className="story-detail-label">Key choices:</p>
                    <ul className="story-detail-choices">
                      {story.keyChoices.filter(c => !c.startsWith('The player chose')).map((c, j) => (
                        <li key={j}>{c}</li>
                      ))}
                    </ul>
                  </>
                )}
                {story.factionsEncountered?.length > 0 && (
                  <p className="story-detail-label">Factions: {story.factionsEncountered.map(f => f.replace(/_/g, ' ')).join(', ')}</p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---- Codex Tab ----

const CODEX_CATEGORIES = [
  { key: 'locations', label: 'Locations' },
  { key: 'factions', label: 'Factions' },
  { key: 'npcs', label: 'NPCs' },
  { key: 'relics', label: 'Relics' },
  { key: 'curses', label: 'Curses' },
];

function CodexTab({ loreDiscovered, lore }) {
  const totalDiscovered = Object.values(loreDiscovered).reduce((sum, arr) => sum + arr.length, 0);

  if (totalDiscovered === 0) {
    return <p className="journal-empty">No lore discovered yet. Explore the world to fill your codex.</p>;
  }

  return (
    <div className="journal-codex">
      {CODEX_CATEGORIES.map(({ key, label }) => {
        const items = loreDiscovered[key] || [];
        if (items.length === 0) return null;

        return (
          <div key={key} className="codex-section">
            <h4 className="codex-section-title">{label}</h4>
            <div className="codex-items">
              {items.map(item => {
                const description = getCodexDescription(key, item, lore);
                return (
                  <div key={item} className="codex-item">
                    <span className="codex-item-name">{formatCodexName(item)}</span>
                    {description && <span className="codex-item-desc">{description}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}

function formatCodexName(id) {
  return id.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
}

function getCodexDescription(category, id, lore) {
  if (!lore) return null;

  if (category === 'locations' && lore.locations?.[id]) {
    return lore.locations[id].mood;
  }
  if (category === 'factions' && lore.factions?.[id]) {
    return lore.factions[id].values?.join(', ');
  }
  if (category === 'relics' && lore.relics?.[id]) {
    return lore.relics[id].effect;
  }
  if (category === 'curses' && lore.curses?.[id]) {
    return lore.curses[id];
  }
  return null;
}

// ---- Stats & Achievements Tab ----

function StatsTab({ stats, achievements }) {
  const earnedIds = new Set(achievements.map(a => a.id));

  return (
    <div className="journal-stats">
      {/* Stats grid */}
      <div className="stats-grid">
        <div className="stat-card">
          <span className="stat-card-value">{stats.totalStories}</span>
          <span className="stat-card-label">Stories Played</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{stats.totalDeaths}</span>
          <span className="stat-card-label">Deaths</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{Math.round(stats.survivalRate * 100)}%</span>
          <span className="stat-card-label">Survival Rate</span>
        </div>
        <div className="stat-card">
          <span className="stat-card-value">{stats.totalScenes}</span>
          <span className="stat-card-label">Total Scenes</span>
        </div>
      </div>

      {/* Achievements */}
      <h4 className="achievements-title">Achievements</h4>
      <div className="achievements-grid">
        {ACHIEVEMENTS.map(def => {
          const earned = earnedIds.has(def.id);
          return (
            <div key={def.id} className={`achievement-card${earned ? ' earned' : ''}`}>
              <span className="achievement-icon">{earned ? '\u2726' : '\u25c7'}</span>
              <span className="achievement-name">{def.title}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
