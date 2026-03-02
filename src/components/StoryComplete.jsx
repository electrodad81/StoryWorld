// src/components/StoryComplete.jsx
// End-of-story summary screen. Writes journal entry and shows unlocks/achievements.

import { useState, useEffect, useRef } from 'react';
import { loadJournal, addStoryEntry, extractLoreFromHistory } from '../services/journal.js';

export default function StoryComplete({
  playerProfile,
  history,
  beat,
  sceneCount,
  isDead,
  lore,
  onPlayAgain,
  onReturnToMenu,
  onViewJournal,
}) {
  const [result, setResult] = useState(null);
  const hasWritten = useRef(false);

  // Write journal entry on mount (once — ref guards against StrictMode double-mount)
  useEffect(() => {
    if (hasWritten.current) return;
    hasWritten.current = true;

    const journal = loadJournal();
    const hook = playerProfile?.starterHook;
    const outcome = isDead ? 'died' : 'survived';

    // Extract key choices (last 4 player messages)
    const keyChoices = history
      .filter(m => m.role === 'user')
      .slice(-4)
      .map(m => m.content);

    // Extract lore from story text
    const extracted = extractLoreFromHistory(history, lore || {});

    const storyResult = addStoryEntry(journal, {
      hookId: hook?.id || 'unknown',
      hookTitle: hook?.title || 'Unknown Hook',
      outcome,
      beatReached: beat || 'exposition',
      sceneCount: sceneCount || 0,
      keyChoices,
      factionsEncountered: extracted.factions,
      locationsReferenced: extracted.locations,
      relicsFound: extracted.relics,
      cursesFound: extracted.curses,
    });

    setResult(storyResult);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  if (!result) return null;

  const { journal, newAchievements, newUnlocks } = result;
  const hook = playerProfile?.starterHook;
  const outcome = isDead ? 'died' : 'survived';

  // Key choices for display
  const keyChoices = history
    .filter(m => m.role === 'user')
    .slice(-4)
    .map(m => m.content)
    .filter(c => !c.startsWith('The player chose')); // skip the initial hook context

  return (
    <div className="app-main">
      <div className="story-complete-panel">
        {/* Outcome banner */}
        <div className={`outcome-banner ${outcome}`}>
          <h2>{isDead ? 'You Died' : 'You Survived'}</h2>
        </div>

        {/* Hook title */}
        <h3 className="complete-hook-title">{hook?.title || 'Your Story'}</h3>

        {/* Stats */}
        <div className="complete-stats">
          <div className="stat-item">
            <span className="stat-value">{sceneCount}</span>
            <span className="stat-label">Scenes</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{(beat || 'exposition').replace('_', ' ')}</span>
            <span className="stat-label">Beat Reached</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{journal.stats.totalStories}</span>
            <span className="stat-label">Total Stories</span>
          </div>
        </div>

        {/* Key choices */}
        {keyChoices.length > 0 && (
          <div className="complete-section">
            <h4>Key Choices</h4>
            <ul className="complete-choices">
              {keyChoices.map((c, i) => <li key={i}>{c}</li>)}
            </ul>
          </div>
        )}

        {/* New unlocks */}
        {(newUnlocks.hooks.length > 0 || newUnlocks.explorationJustUnlocked) && (
          <div className="complete-section unlocks">
            <h4>New Unlocks</h4>
            {newUnlocks.explorationJustUnlocked && (
              <div className="unlock-item exploration">
                The Withered Vale is now open to explore
              </div>
            )}
            {newUnlocks.hooks.map(hookId => {
              const h = lore?.starter_hooks?.find(sh => sh.id === hookId);
              return (
                <div key={hookId} className="unlock-item">
                  Unlocked: {h?.title || hookId}
                </div>
              );
            })}
          </div>
        )}

        {/* Achievements */}
        {newAchievements.length > 0 && (
          <div className="complete-section achievements">
            <h4>Achievements Earned</h4>
            {newAchievements.map(a => (
              <div key={a.id} className="achievement-item earned">
                {a.title}
              </div>
            ))}
          </div>
        )}

        {/* Actions */}
        <div className="complete-actions">
          <button className="choice-btn" onClick={onPlayAgain}>Play Again</button>
          <button className="choice-btn" onClick={onViewJournal}>View Journal</button>
          <button className="choice-btn" onClick={onReturnToMenu}>Return to Menu</button>
        </div>
      </div>
    </div>
  );
}
