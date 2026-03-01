# StoryWorld (Gloamreach) — Immersive Full-Bleed UI Spec

## Goal

Replace the current three-panel layout (60vh illustration / flex text / pinned choices) with a full-bleed immersive layout where the AI-generated illustration fills the entire viewport as a background image, and all UI elements (story text, choices, status) float on top of it. This creates a visual-novel / mobile-story aesthetic optimized for portrait/phone screens.

---

## 1. Layout Architecture

### Current → New

**Current:** Three stacked flex panels inside `.story-layout` (illustration, text, choices).
**New:** A single full-viewport container with absolutely-positioned layers.

### New Component Tree (StoryView.jsx)

```
<div className="story-viewport">        // position: relative; 100vh; 100vw; overflow: hidden

  {/* Layer 0: Background illustration */}
  <div className="bg-layer">
    <img className="bg-current" />       // current image, object-fit: cover, fills viewport
    <img className="bg-incoming" />      // next image, fades in over current via CSS transition
  </div>

  {/* Layer 1: Gradient scrim for text legibility */}
  <div className="scrim" />

  {/* Layer 2: Story text — positioned in lower portion */}
  <div className="text-overlay">
    <StreamingText ... />
    {isDead && <DeathOverlay />}
  </div>

  {/* Layer 3: Choice buttons — pinned to bottom */}
  <div className="choice-overlay">
    <ChoiceGrid ... />
  </div>

  {/* Layer 4: Minimal top bar (beat indicator, menu) */}
  <div className="top-bar">
    <button className="menu-btn">☰</button>
    <span className="beat-indicator">{beat}</span>
  </div>
</div>
```

All layers use `position: absolute` with appropriate `z-index` stacking.

---

## 2. CSS Changes (index.css)

### Remove These Classes Entirely
- `.story-layout`
- `.panel-illustration`
- `.panel-text`
- `.panel-choices`
- `.illus-placeholder`
- `.illus-empty`
- `.illus-sep`

### Add These New Classes

```css
/* ---- Immersive story viewport ---- */
.story-viewport {
  position: relative;
  width: 100%;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-primary);       /* fallback if no image */
}

/* Layer 0: Background images */
.bg-layer {
  position: absolute;
  inset: 0;
  z-index: 0;
}

.bg-layer img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: opacity 1.2s ease-in-out;
}

.bg-layer img.bg-current {
  opacity: 1;
}

.bg-layer img.bg-incoming {
  opacity: 0;
}

.bg-layer img.bg-incoming.visible {
  opacity: 1;
}

/* Layer 1: Gradient scrim */
.scrim {
  position: absolute;
  inset: 0;
  z-index: 1;
  background: linear-gradient(
    to top,
    rgba(10, 10, 20, 0.92) 0%,
    rgba(10, 10, 20, 0.75) 30%,
    rgba(10, 10, 20, 0.3) 55%,
    transparent 75%
  );
  pointer-events: none;
}

/* Layer 2: Text overlay */
.text-overlay {
  position: absolute;
  bottom: 110px;              /* leave room for choices below */
  left: 0;
  right: 0;
  z-index: 2;
  max-height: 45vh;
  overflow-y: auto;
  padding: 0 1.25rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  /* Hide scrollbar but allow scroll */
  scrollbar-width: none;
  -ms-overflow-style: none;
}
.text-overlay::-webkit-scrollbar {
  display: none;
}

/* Layer 3: Choice overlay */
.choice-overlay {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  z-index: 2;
  padding: 0.75rem 1.25rem calc(env(safe-area-inset-bottom, 0px) + 0.75rem);
  /* subtle gradient so choices don't float harshly */
  background: linear-gradient(to top, rgba(10, 10, 20, 0.95), transparent);
}

/* Layer 4: Top bar */
.top-bar {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  z-index: 3;
  padding: 0.75rem 1rem calc(env(safe-area-inset-top, 0px) + 0.75rem);
  display: flex;
  justify-content: space-between;
  align-items: center;
  background: linear-gradient(to bottom, rgba(10, 10, 20, 0.6), transparent);
}

.menu-btn {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.15);
  color: var(--text-primary);
  font-size: 1.2rem;
  width: 36px;
  height: 36px;
  border-radius: 8px;
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
}

.beat-indicator {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.12em;
  color: var(--accent);
  opacity: 0.7;
}
```

### Modify These Existing Classes

**`.storybox`** — Add text shadow for legibility over images:
```css
.storybox {
  /* keep existing properties, add: */
  text-shadow: 0 1px 4px rgba(0,0,0,0.7), 0 0 12px rgba(0,0,0,0.4);
}
```

**`.choice-btn`** — Add backdrop blur for glassmorphism:
```css
.choice-btn {
  /* keep existing properties, change: */
  background: rgba(30, 30, 58, 0.65);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(200, 168, 78, 0.2);
}

.choice-btn:hover:not(:disabled) {
  background: rgba(200, 168, 78, 0.15);
  border-color: var(--accent);
}
```

**`.death-screen`** — Make it work as an overlay card:
```css
.death-screen {
  /* keep existing, add: */
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  background: rgba(192, 57, 43, 0.15);
}
```

### Responsive Changes

```css
/* The immersive layout is inherently mobile-first. */
/* Adjust text overlay positioning for very small screens. */

@media (max-width: 480px) {
  .text-overlay {
    bottom: 100px;
    max-height: 40vh;
    padding: 0 0.75rem;
  }

  .storybox {
    font-size: 1.05rem;
    line-height: 1.5;
  }
}

/* On very tall/narrow screens (phones), allow more text space */
@media (max-aspect-ratio: 9/16) {
  .text-overlay {
    max-height: 50vh;
  }
}
```

### Sidebar Handling

On mobile/tablet the sidebar is hidden and inaccessible. The new `top-bar` with a hamburger menu button should toggle a slide-over drawer. Implementation:

- Add a `sidebarOpen` state to `App.jsx` (or a small context)
- The hamburger `☰` button toggles it
- The sidebar renders as a fixed overlay (full-height, left-aligned, z-index: 10, with backdrop)
- On desktop (>1024px), the sidebar can remain as-is alongside the viewport, OR hide it entirely and use the same hamburger approach for consistency. Recommend: hamburger everywhere for a clean immersive feel.

---

## 3. StoryView.jsx — Full Rewrite

Replace the entire component. Key behavioral changes:

### Background Image Crossfade Logic

Manage two image refs: `currentBg` and `incomingBg`. When a new illustration URL arrives:

1. Set `incomingBg` src to the new URL
2. After image loads (onLoad), add `.visible` class to fade it in (CSS transition handles the animation)
3. After transition completes (~1.2s), swap: set `currentBg` src to new URL, remove `incomingBg` `.visible`, clear `incomingBg` src

This ensures there is never a blank frame.

### Persistent Background

The background should NEVER be empty once the first illustration loads. If no new illustration is available, the previous one stays. Add a `lastIllustration` prop or derive it in the component:

```jsx
const [persistedBg, setPersistedBg] = useState(null);

useEffect(() => {
  if (illustration) {
    setPersistedBg(illustration);
  }
}, [illustration]);

// Use persistedBg for the background, not illustration directly
```

### Default Background (Before First Illustration)

Before any illustration has been generated, show a static fallback. Options in order of preference:

1. **CSS-only atmospheric background** — A dark radial gradient with subtle noise texture. No external assets needed. Example:
   ```css
   .bg-layer--empty {
     background: radial-gradient(ellipse at 50% 30%, #1a1a3e 0%, #0a0a14 100%);
   }
   ```
2. **Ship a static image** — Place a moody Gloamreach landscape in `public/images/default-bg.webp`. This is better but requires creating/sourcing the asset.

Recommend option 1 for now; it can be upgraded later.

### Full Component Skeleton

```jsx
import { useState, useEffect, useRef, useCallback } from 'react';
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
  beat,
  onMenuToggle,        // NEW: callback to toggle sidebar drawer
}) {
  const displayText = isStreaming ? streamedText : currentScene;
  const [persistedBg, setPersistedBg] = useState(null);
  const [incomingSrc, setIncomingSrc] = useState(null);
  const [showIncoming, setShowIncoming] = useState(false);
  const currentImgRef = useRef(null);
  const incomingImgRef = useRef(null);

  // When a new illustration arrives, start crossfade
  useEffect(() => {
    if (illustration && illustration !== persistedBg) {
      setIncomingSrc(illustration);
    }
  }, [illustration, persistedBg]);

  const handleIncomingLoad = useCallback(() => {
    setShowIncoming(true);
    // After transition, promote incoming to current
    const timer = setTimeout(() => {
      setPersistedBg(incomingSrc);
      setShowIncoming(false);
      setIncomingSrc(null);
    }, 1300); // slightly longer than CSS transition
    return () => clearTimeout(timer);
  }, [incomingSrc]);

  // Auto-scroll text overlay to bottom when new text arrives
  const textRef = useRef(null);
  useEffect(() => {
    if (textRef.current) {
      textRef.current.scrollTop = textRef.current.scrollHeight;
    }
  }, [displayText]);

  return (
    <div className="story-viewport">
      {/* Background */}
      <div className="bg-layer">
        {persistedBg ? (
          <img ref={currentImgRef} className="bg-current" src={persistedBg} alt="" />
        ) : (
          <div className="bg-layer--empty" />
        )}
        {incomingSrc && (
          <img
            ref={incomingImgRef}
            className={`bg-incoming${showIncoming ? ' visible' : ''}`}
            src={incomingSrc}
            alt=""
            onLoad={handleIncomingLoad}
          />
        )}
      </div>

      {/* Scrim */}
      <div className="scrim" />

      {/* Top bar */}
      <div className="top-bar">
        <button className="menu-btn" onClick={onMenuToggle}>☰</button>
        <span className="beat-indicator">{beat}</span>
      </div>

      {/* Text */}
      <div className="text-overlay" ref={textRef}>
        {displayText ? (
          <StreamingText text={displayText} isStreaming={isStreaming} />
        ) : isGenerating ? (
          <LanternLoader caption="Crafting your scene…" />
        ) : null}

        {isDead && (
          <div className="death-screen">
            <h3>You Died.</h3>
            <p>Your adventure ends in tragedy.</p>
            <button className="sidebar-btn" onClick={onNewStory} style={{ marginTop: '0.75rem' }}>
              Start New Story
            </button>
          </div>
        )}
      </div>

      {/* Choices */}
      <div className="choice-overlay">
        {isDead ? null : isGenerating && !displayText ? null : (
          <ChoiceGrid choices={choices} onChoose={onChoose} disabled={isStreaming || isGenerating} />
        )}
      </div>
    </div>
  );
}
```

---

## 4. Image Generation Changes

### openai.js

Change the default image size to portrait:

```js
export async function imageGeneration(prompt, options = {}) {
  const { model = 'dall-e-3', size = '1024x1792' } = options;  // CHANGED from 1024x1024
  // ...rest unchanged
}
```

### StoryEngine.js — Illustration Frequency

Consider increasing illustration frequency now that images are the primary visual element. Change the constants in `useStoryEngine.js`:

```js
// Option A: Every 2 scenes instead of every 3
const ILLUSTRATION_EVERY_N = 2;
const ILLUSTRATION_PHASE = 0;   // Start from scene 1

// Option B: Every scene (most immersive, most expensive)
// const ILLUSTRATION_EVERY_N = 1;
// const ILLUSTRATION_PHASE = 0;
```

Recommend starting with every 2 scenes. The player can configure this later if cost is a concern.

### Opening Illustration

Generate an illustration at game start so the background is populated from the very first scene. In `useStoryEngine.js`, inside `startGame`:

```js
const startGame = useCallback(async (profile) => {
  const loreData = await loadLore();
  setPlayerProfile(profile);
  setPhase('playing');
  // ... existing reset code ...

  // Fire opening illustration immediately from hook blurb (non-blocking)
  if (profile.starterHook) {
    generateIllustration(profile.starterHook.blurb, profile.gender)
      .then((url) => { if (url) setIllustration(url); })
      .catch(() => {});
  }

  // Build initial history and run first scene (existing code)
  const hookContext = profile.starterHook
    ? `The player chose the starter hook: "${profile.starterHook.title}" — ${profile.starterHook.blurb}`
    : 'Begin the story.';
  const initialHist = [{ role: 'user', content: hookContext }];
  setHistory(initialHist);
  await runTurn(initialHist, loreData, profile, 'exposition', 0, 0, 0);
}, [loadLore, runTurn]);
```

This fires the DALL-E call in parallel with the first scene generation. By the time the text finishes streaming, the background image may already be ready.

---

## 5. App.jsx Changes

### Sidebar Drawer State

Add sidebar toggle state. The sidebar becomes a slide-over drawer at all breakpoints:

```jsx
import { useState, useEffect } from 'react';
// ... existing imports ...

export default function App() {
  const engine = useStoryEngine();
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => { engine.loadLore(); }, []);

  // ... existing needKey / setup phases unchanged ...

  // Playing / dead — no longer render sidebar inline
  return (
    <div className="app-layout immersive">
      {/* Sidebar as overlay drawer */}
      {sidebarOpen && (
        <div className="drawer-backdrop" onClick={() => setSidebarOpen(false)} />
      )}
      <div className={`drawer${sidebarOpen ? ' open' : ''}`}>
        <Sidebar
          playerProfile={engine.playerProfile}
          onNewStory={() => { setSidebarOpen(false); engine.newStory(); }}
          onReset={() => { setSidebarOpen(false); engine.resetGame(); }}
        />
      </div>

      <StoryView
        currentScene={engine.currentScene}
        isStreaming={engine.isStreaming}
        streamedText={engine.streamedText}
        illustration={engine.illustration}
        choices={engine.choices}
        onChoose={engine.chooseOption}
        isGenerating={engine.isGenerating}
        isDead={engine.isDead}
        onNewStory={engine.newStory}
        beat={engine.beat}
        onMenuToggle={() => setSidebarOpen((v) => !v)}
      />
    </div>
  );
}
```

### Drawer CSS

```css
.drawer-backdrop {
  position: fixed;
  inset: 0;
  z-index: 9;
  background: rgba(0,0,0,0.5);
}

.drawer {
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  width: 280px;
  z-index: 10;
  transform: translateX(-100%);
  transition: transform 0.3s ease;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  overflow-y: auto;
}

.drawer.open {
  transform: translateX(0);
}

/* Remove old sidebar layout rules */
.app-layout.immersive {
  display: block;   /* no longer flex with sidebar */
}

/* Hide old static sidebar */
.app-layout.immersive .app-sidebar {
  display: none;
}
```

---

## 6. Summary of Files to Modify

| File | Action | Scope |
|------|--------|-------|
| `src/components/StoryView.jsx` | **Rewrite** | Replace three-panel layout with full-bleed viewport + layers |
| `src/index.css` | **Heavy edit** | Remove panel classes, add viewport/layer/scrim/overlay/drawer styles |
| `src/App.jsx` | **Edit** | Add drawer state, pass `beat` and `onMenuToggle` to StoryView |
| `src/services/openai.js` | **Small edit** | Change default image size to `1024x1792` |
| `src/hooks/useStoryEngine.js` | **Edit** | Fire opening illustration in `startGame`, optionally change frequency constants |
| `src/engine/StoryEngine.js` | No changes needed | Style directives already work for portrait |

---

## 7. Testing Checklist

- [ ] First scene renders with CSS gradient background (no image yet)
- [ ] Opening illustration loads and fades in as background
- [ ] Subsequent illustrations crossfade smoothly (no blank flash)
- [ ] Previous illustration persists between scenes when no new one generates
- [ ] Story text is readable over both dark and bright images
- [ ] Choices are tappable and visually distinct from background
- [ ] Death screen renders correctly over the background
- [ ] Hamburger menu opens/closes the sidebar drawer
- [ ] Sidebar actions (new story, reset) work from the drawer
- [ ] Portrait aspect ratio images fill the viewport without distortion
- [ ] Safe area insets work on iOS (notch, home bar)
- [ ] Auto-scroll keeps latest text visible during streaming
- [ ] Works on mobile viewport (375×812 and similar)
- [ ] Works on desktop viewport (the image fills the whole content area)

---

## 8. Future Enhancements (Not in This Pass)

- **Parallax/Ken Burns effect** — Slow zoom or pan on the background image for subtle motion between interactions
- **Ambient particle overlay** — CSS-animated fog/dust/ember particles floating over the image for atmosphere
- **Illustration preloading** — Start generating the next illustration while the player reads, so it's ready on choice selection
- **User-configurable illustration frequency** — Add a setting for "every scene" / "every 2" / "every 3" to let players manage cost
- **Haptic feedback on mobile** — Vibrate on choice selection and death
