// src/components/CharacterSetup.jsx
// Character creation: name, gender, archetype, starter hook selection.

import { useState, useEffect } from 'react';

export default function CharacterSetup({ lore, onBegin }) {
  const [name, setName] = useState('');
  const [gender, setGender] = useState('Unspecified');
  const [archetype] = useState('Default');
  const [selectedHook, setSelectedHook] = useState(null);
  const [hooks, setHooks] = useState([]);

  useEffect(() => {
    if (lore?.starter_hooks) {
      setHooks(lore.starter_hooks);
      if (lore.starter_hooks.length > 0) {
        setSelectedHook(lore.starter_hooks[0].id);
      }
    }
  }, [lore]);

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
              {hooks.map((hook) => (
                <div
                  key={hook.id}
                  className={`hook-card${selectedHook === hook.id ? ' selected' : ''}`}
                  onClick={() => setSelectedHook(hook.id)}
                >
                  <h4>{hook.title}</h4>
                  <p>{hook.blurb}</p>
                </div>
              ))}
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
