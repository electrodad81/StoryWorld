// src/components/ApiKeyInput.jsx
// API key entry screen.

import { useState } from 'react';

export default function ApiKeyInput({ onSave }) {
  const [key, setKey] = useState('');

  function handleSave() {
    const trimmed = key.trim();
    if (!trimmed) return;
    localStorage.setItem('openai_api_key', trimmed);
    onSave(trimmed);
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter') handleSave();
  }

  return (
    <div className="app-main">
      <div className="api-key-panel">
        <h2>OpenAI API Key</h2>
        <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
          Enter your OpenAI API key to power the story engine. Your key is stored
          only in your browser's localStorage and never sent to any server other
          than OpenAI.
        </p>
        <input
          type="password"
          value={key}
          onChange={(e) => setKey(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="sk-..."
        />
        <button disabled={!key.trim()} onClick={handleSave}>
          Save &amp; Continue
        </button>
      </div>
    </div>
  );
}
