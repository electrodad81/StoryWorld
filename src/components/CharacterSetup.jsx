// src/components/CharacterSetup.jsx
// Character creation: name, gender, archetype, starter hook selection.
// Locked hooks render dimmed with unlock hints.

import { useState, useEffect } from 'react';

const UNLOCK_HINTS = {
  stories_completed: (min) => min === 1 ? 'Complete any story to unlock' : `Complete ${min} stories to unlock`,
  stories_survived: (min) => min === 1 ? 'Survive a story to unlock' : `Survive ${min} stories to unlock`,
  stories_died: (min) => min === 1 ? 'Die in a story to unlock' : `Die ${min} times to unlock`,
};

function getUnlockHint(hook) {
  if (!hook.unlockCondition) return null;
  const { type, min } = hook.unlockCondition;
  const hintFn = UNLOCK_HINTS[type];
  return hintFn ? hintFn(min) : 'Keep playing to unlock';
}

export default function CharacterSetup({ lore, unlockedHooks, onBegin }) {
  const [name, setName] = useState('');
  const [gender, setGender] = useState('Unspecified');
  const [archetype] = useState('Default');
  const [selectedHook, setSelectedHook] = useState(null);
  const [hooks, setHooks] = useState([]);

  useEffect(() => {
    if (lore?.starter_hooks) {
      setHooks(lore.starter_hooks);
      // Auto-select first unlocked hook
      const firstUnlocked = lore.starter_hooks.find(h =>
        !h.unlockCondition || (unlockedHooks && unlockedHooks.includes(h.id))
      );
      if (firstUnlocked) {
        setSelectedHook(firstUnlocked.id);
      }
    }
  }, [lore, unlockedHooks]);

  const isHookUnlocked = (hook) => {
    if (!hook.unlockCondition) return true;
    return unlockedHooks && unlockedHooks.includes(hook.id);
  };

  const canBegin = name.trim().length > 0 && selectedHook;

  function handleBegin() {
    if (!canBegin) return;
    const hook = hooks.find((h) => h.id === selectedHook);
    onBegin({
      name: name.trim(),
      gender,
      archetype,
      starterHook: hook,
    });
  }

  return (
    <div className="app-main">
      <div className="setup-panel">
        <h2>Begin Your Journey</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Pick your setup. Name and character are locked once you begin.
        </p>

        <label>
          Name (required)
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            maxLength={24}
            placeholder="Enter your name"
          />
        </label>

        <label>
          Gender
          <select value={gender} onChange={(e) => setGender(e.target.value)}>
            <option value="Unspecified">Unspecified</option>
            <option value="Female">Female</option>
            <option value="Male">Male</option>
            <option value="Nonbinary">Nonbinary</option>
          </select>
        </label>

        <label>
          Character type
          <select value={archetype} disabled>
            <option value="Default">Default</option>
          </select>
          <span style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>
            Archetypes coming soon.
          </span>
        </label>

        {hooks.length > 0 && (
          <>
            <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.95rem', marginTop: '0.5rem' }}>
              Choose your starting hook
            </h3>
            <div className="hook-list">
              {hooks.map((hook) => {
                const unlocked = isHookUnlocked(hook);
                const hint = getUnlockHint(hook);
                return (
                  <div
                    key={hook.id}
                    className={`hook-card${selectedHook === hook.id ? ' selected' : ''}${!unlocked ? ' locked' : ''}`}
                    onClick={unlocked ? () => setSelectedHook(hook.id) : undefined}
                  >
                    <div className="hook-card-header">
                      <h4>{hook.title}</h4>
                      {!unlocked && <span className="lock-icon">&#128274;</span>}
                    </div>
                    {unlocked ? (
                      <p>{hook.blurb}</p>
                    ) : (
                      <p className="hook-hint">{hint}</p>
                    )}
                  </div>
                );
              })}
            </div>
          </>
        )}

        <button className="begin-btn" disabled={!canBegin} onClick={handleBegin}>
          Begin Adventure
        </button>
      </div>
    </div>
  );
}
