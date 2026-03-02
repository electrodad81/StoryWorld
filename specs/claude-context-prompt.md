# StoryWorld (Gloamreach) — Project Context

I'm building StoryWorld (Gloamreach), an interactive dark-fantasy narrative game. It was recently migrated from Python/Streamlit to a React + Vite SPA. The app runs entirely in the browser — the user provides their OpenAI API key (stored in localStorage), and all API calls go directly to OpenAI from the client.

## Architecture

### Engine Layer (`src/engine/`)
- **StoryEngine.js** — Scene streaming via OpenAI chat completions (async generator), choice generation (returns JSON array of 2 imperative actions), DALL-E 3 illustration with progressive safety sanitization (levels 1-3 + environment-only fallback). Contains beat system constants and illustration prompt helpers (regex-based word replacement for gore/modern items, hard filter for sentence dropping).
- **prompts.js** — Prompt templates: SYSTEM_PROMPT (formatted with lore), scenePrompt, choicePrompt.
- **ExploreEngine.js** — Free-roam exploration mode. Translated from Python but dormant/unwired into the UI.
- **explorePrompts.js** — Exploration prompts. Also dormant.

### Services Layer (`src/services/`)
- **openai.js** — Raw fetch wrapper (no openai npm SDK). Exports: `chatCompletion` (streaming async generator via SSE parsing), `chatCompletionFull` (non-streaming), `imageGeneration` (DALL-E). Reads API key from localStorage.
- **persistence.js** — localStorage stub with `saveSnapshot`/`loadSnapshot`/`deleteSnapshot`. Interface designed so a Neon DB backend can be swapped in later.
- **identity.js** — Generates/retrieves a stable browser ID (`brw-<uuid>`) from localStorage.

### Hook (`src/hooks/`)
- **useStoryEngine.js** — Central React hook owning all game state. State includes: phase (`needKey`/`setup`/`playing`), playerProfile, history, currentScene, choices, illustration, beat, dangerStreak, sceneCount, isStreaming, streamedText, isGenerating, isDead. Exposes: `startGame`, `chooseOption`, `resetGame`, `newStory`, `setApiKey`, `loadLore`.

### Components (`src/components/`)
- **App.jsx** — Root component. Conditional render: ApiKeyInput (needKey) -> CharacterSetup (setup) -> Sidebar + StoryView (playing/dead).
- **StoryView.jsx** — Fixed three-panel layout: illustration panel (top, 60vh), scrollable text panel (middle), choices panel (bottom, pinned). Panels stay in place; content updates within them.
- **CharacterSetup.jsx** — Name input, gender select, archetype select (locked to "Default"), starter hook selection from lore.json.
- **ChoiceGrid.jsx** — Column of choice buttons, disabled during generation.
- **StreamingText.jsx** — Renders scene text with animated typing caret during streaming.
- **LanternLoader.jsx** — CSS-animated lantern with flickering flame.
- **Sidebar.jsx** — Player profile display, "Start New Story" and "Reset Session" buttons.
- **ApiKeyInput.jsx** — OpenAI API key entry, saves to localStorage.

### Styling (`src/index.css`)
- Dark fantasy theme: deep navy backgrounds, gold accent (`#c8a84e`), serif text for story content, sans-serif for UI.
- Fixed three-panel story layout (60vh illustration, flex-1 text, pinned choices).
- Responsive: sidebar hidden on tablet, panels shrink on mobile.

### Data
- **public/data/lore.json** — World lore: tone rules, magic system, factions (Veilbound Inquisition, Pale Court, Lantern Guild), locations (Gloamreach, Mire of Thorns, Sunless Abbey, Waking Forest), relics, curses, and 6 starter hooks.

## Key Mechanics
- **5-beat story arc**: exposition (3 scenes) -> rising_action (4) -> climax (2) -> falling_action (3) -> resolution (1)
- **Danger streak**: Incremented when player picks risky choices (matched by keyword regex), decremented on safe choices. At streak >= 2, engine is told to escalate to serious setback.
- **Death**: Engine appends `[DEATH]` to scene text when protagonist dies. Hook detects and strips it, sets isDead state.
- **Illustrations**: Generated every 3 scenes. Progressive sanitization replaces gore/modern terms, drops flagged sentences, falls back to environment-only prompt. Uses DALL-E 3 with medieval-fantasy style directive.
- **Consequence contract**: Risky choices must show visible cost in the next scene. Two consecutive risky choices escalate to capture/grave wound/loss.

## Pending / Next Steps
- Neon DB persistence (replace localStorage stub)
- Wire exploration mode into UI
- Archetype system (currently only "Default")
- Opening illustration during character creation
- Illustration preloading (fire off generation while player reads)

## Files Attached
I'm attaching the key source files. Please review them so we can discuss features, bugs, and next steps.
