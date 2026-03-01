# Immersive Full-Bleed UI — Changes Applied

## Summary

Replaced the three-panel layout (60vh illustration / flex text / pinned choices) with a full-bleed immersive layout where the AI-generated illustration fills the entire viewport as a background, and all UI elements float on top via absolute positioning with gradient scrims.

---

## What Changed

### 1. StoryView.jsx — Full Rewrite
- **Old:** Three stacked flex panels (`.panel-illustration`, `.panel-text`, `.panel-choices`)
- **New:** Single `100vh` viewport (`.story-viewport`) with layered absolute-positioned children:
  - **Layer 0** (`bg-layer`): Full-bleed background image with crossfade transition support. Two `<img>` elements — `bg-current` and `bg-incoming` — handle seamless transitions via CSS `opacity` + `transition: opacity 1.2s`. A `persistedBg` state ensures the background never goes blank once the first image loads.
  - **Layer 1** (`scrim`): Bottom-to-top gradient overlay for text legibility over any image.
  - **Layer 2** (`text-overlay`): Story text positioned in the lower portion of the viewport (`bottom: 110px`, `max-height: 45vh`), auto-scrolls to bottom during streaming. Hidden scrollbar.
  - **Layer 3** (`choice-overlay`): Choice buttons pinned to bottom with a gradient fade-in background. Respects `safe-area-inset-bottom` for iOS notch/home bar.
  - **Layer 4** (`top-bar`): Hamburger menu button + beat indicator (e.g., "exposition", "rising action") positioned at the top with a subtle gradient.

### 2. App.jsx — Drawer Sidebar
- **Old:** Sidebar rendered inline as a flex child alongside StoryView
- **New:** Sidebar is a slide-over drawer at all breakpoints:
  - `sidebarOpen` state controls visibility
  - `.drawer-backdrop` (semi-transparent overlay) + `.drawer` (280px fixed panel, `translateX` animation)
  - Hamburger button in StoryView's top bar toggles it via `onMenuToggle` prop
  - Sidebar actions (new story, reset) close the drawer before executing
- New props passed to StoryView: `beat`, `onMenuToggle`

### 3. index.css — Heavy Rewrite
**Removed:**
- `.story-layout`, `.panel-illustration`, `.panel-text`, `.panel-choices`
- `.illus-placeholder`, `.illus-empty`
- `.illus-sep` (separator between sections — no longer needed)
- Old responsive rules for panels

**Added:**
- `.story-viewport`, `.bg-layer`, `.bg-current`, `.bg-incoming`, `.bg-layer--empty`
- `.scrim` (gradient overlay)
- `.text-overlay`, `.choice-overlay`
- `.top-bar`, `.menu-btn`, `.beat-indicator`
- `.drawer-backdrop`, `.drawer`, `.drawer.open`

**Modified:**
- `.storybox` — Removed border/background (now floats over image). Added `text-shadow` for legibility.
- `.choice-btn` — Changed from solid `var(--bg-card)` to glassmorphism: `rgba(30,30,58,0.65)` + `backdrop-filter: blur(12px)`. Border changed to subtle gold `rgba(200,168,78,0.2)`.
- `.death-screen` — Added `backdrop-filter: blur(16px)`, changed background to `rgba(192,57,43,0.15)`.
- `.lantern-caption` — Added `text-shadow`.
- `.choice-grid h3` — Reduced size, added text-shadow.
- `html, body` — Changed from `overflow-x: hidden; min-height: 100vh` to `overflow: hidden; height: 100vh`.
- `.app-layout` — Changed from `display: flex` to `display: block; height: 100vh; overflow: hidden`.
- `.app-sidebar` — Now lives inside `.drawer`, removed fixed width/border-right.

**Responsive:**
- Removed old sidebar hide and panel height rules
- Added `.text-overlay` adjustments for phones (bottom: 100px, max-height: 40vh)
- Added `@media (max-aspect-ratio: 9/16)` for very tall screens

### 4. openai.js — Portrait Images
- Default image size changed from `1024x1024` to `1024x1792` (portrait) so generated images fill the full-bleed viewport properly.

### 5. useStoryEngine.js — Illustration Improvements
- **Frequency:** Changed from every 3 scenes to every 2 scenes (`ILLUSTRATION_EVERY_N = 2`, `ILLUSTRATION_PHASE = 0`).
- **No clearing:** Removed `setIllustration(null)` at the start of `runTurn`. The `persistedBg` state in StoryView handles persistence — previous illustration stays visible until a new one crossfades in.
- **Static cover images:** On game start, if the selected starter hook has a `coverImage` field, it's set as the illustration immediately (instant display, no DALL-E wait).
- **Opening illustration:** DALL-E is fired in parallel using the hook blurb, and crossfades in over the static cover when ready.

### 6. lore.json — Cover Image References
- Each starter hook now has a `coverImage` field pointing to a pre-generated static PNG:
  ```json
  "coverImage": "/images/hook-01.png"
  ```

### 7. New Assets
- `public/images/hook-01.png` through `hook-06.png` — Pre-generated 1024x1792 DALL-E 3 illustrations, one per starter hook. Displayed instantly as the background when a game begins.
- `scripts/generate-covers.js` — Node script to regenerate these images via the OpenAI API.

---

## Files Modified

| File | Action |
|------|--------|
| `src/components/StoryView.jsx` | Rewritten — full-bleed viewport with layered overlays and crossfade |
| `src/index.css` | Heavy rewrite — removed panels, added viewport/scrim/overlay/drawer/glassmorphism |
| `src/App.jsx` | Edited — added drawer sidebar state, new props to StoryView |
| `src/services/openai.js` | Small edit — default image size → `1024x1792` |
| `src/hooks/useStoryEngine.js` | Edited — cover images, opening illustration, frequency change, no clearing |
| `public/data/lore.json` | Edited — added `coverImage` field to each starter hook |

## Files Added

| File | Purpose |
|------|---------|
| `public/images/hook-01.png` through `hook-06.png` | Static cover images for each starter hook |
| `scripts/generate-covers.js` | Script to regenerate cover images via DALL-E API |

---

## Full Diffs

Below are the complete diffs for all source file changes (excludes binary image files).

### public/data/lore.json

```diff
--- a/public/data/lore.json
+++ b/public/data/lore.json
@@ -113,32 +113,38 @@
     {
       "id": "HOOK-01",
       "title": "Bells in the Fog",
-      "blurb": "Low tide reveals a sunken bell tower near Gloamreach. Each toll rewrites a name on the village memorial."
+      "blurb": "Low tide reveals a sunken bell tower near Gloamreach. Each toll rewrites a name on the village memorial.",
+      "coverImage": "/images/hook-01.png"
     },
     {
       "id": "HOOK-02",
       "title": "The Last Oathkeeper",
-      "blurb": "An aged monk from the Sunless Abbey seeks a courier for a vow sealed in living ink."
+      "blurb": "An aged monk from the Sunless Abbey seeks a courier for a vow sealed in living ink.",
+      "coverImage": "/images/hook-02.png"
     },
     {
       "id": "HOOK-03",
       "title": "Lanterns for the Dead",
-      "blurb": "The Lantern Guild pays well to escort a shipment through the Waking Forest where lights appear that weren't lit."
+      "blurb": "The Lantern Guild pays well to escort a shipment through the Waking Forest where lights appear that weren't lit.",
+      "coverImage": "/images/hook-03.png"
     },
     {
       "id": "HOOK-04",
       "title": "White Sigils, Black Debts",
-      "blurb": "A Pale Court heir offers protection if you recover a ledger that should have burned decades ago."
+      "blurb": "A Pale Court heir offers protection if you recover a ledger that should have burned decades ago.",
+      "coverImage": "/images/hook-04.png"
     },
     {
       "id": "HOOK-05",
       "title": "Drowned Star's Whisper",
-      "blurb": "Freshwater wells taste of salt; villagers dream of distant bells and forget their children's names at dawn."
+      "blurb": "Freshwater wells taste of salt; villagers dream of distant bells and forget their children's names at dawn.",
+      "coverImage": "/images/hook-05.png"
     },
     {
       "id": "HOOK-06",
       "title": "Veilbound Audit",
-      "blurb": "The Inquisition arrives to inventory forbidden relics—and one looks suspiciously like your family keepsake."
+      "blurb": "The Inquisition arrives to inventory forbidden relics—and one looks suspiciously like your family keepsake.",
+      "coverImage": "/images/hook-06.png"
     }
   ]
 }
```

### src/App.jsx

```diff
--- a/src/App.jsx
+++ b/src/App.jsx
@@ -1,7 +1,8 @@
 // src/App.jsx
 // Root component — conditional render based on game phase.
+// Sidebar is now a slide-over drawer toggled by hamburger menu.

-import { useEffect } from 'react';
+import { useState, useEffect } from 'react';
 import './App.css';
 import useStoryEngine from './hooks/useStoryEngine.js';
 import ApiKeyInput from './components/ApiKeyInput.jsx';
@@ -11,6 +12,7 @@ import Sidebar from './components/Sidebar.jsx';

 export default function App() {
   const engine = useStoryEngine();
+  const [sidebarOpen, setSidebarOpen] = useState(false);

   // Load lore on mount
   useEffect(() => {
@@ -35,14 +37,23 @@ export default function App() {
     );
   }

-  // Playing / dead
+  // Playing / dead — immersive layout with drawer sidebar
   return (
     <div className="app-layout">
-      <Sidebar
-        playerProfile={engine.playerProfile}
-        onNewStory={engine.newStory}
-        onReset={engine.resetGame}
-      />
+      {/* Drawer backdrop */}
+      {sidebarOpen && (
+        <div className="drawer-backdrop" onClick={() => setSidebarOpen(false)} />
+      )}
+
+      {/* Drawer */}
+      <div className={`drawer${sidebarOpen ? ' open' : ''}`}>
+        <Sidebar
+          playerProfile={engine.playerProfile}
+          onNewStory={() => { setSidebarOpen(false); engine.newStory(); }}
+          onReset={() => { setSidebarOpen(false); engine.resetGame(); }}
+        />
+      </div>
+
       <StoryView
         currentScene={engine.currentScene}
         isStreaming={engine.isStreaming}
@@ -53,6 +64,8 @@ export default function App() {
         isGenerating={engine.isGenerating}
         isDead={engine.isDead}
         onNewStory={engine.newStory}
+        beat={engine.beat}
+        onMenuToggle={() => setSidebarOpen((v) => !v)}
       />
     </div>
   );
```

### src/components/StoryView.jsx

```diff
--- a/src/components/StoryView.jsx
+++ b/src/components/StoryView.jsx
@@ -1,7 +1,8 @@
 // src/components/StoryView.jsx
-// Fixed three-panel layout: illustration (top 60%), text (middle), choices (bottom).
-// Panels stay in place; content updates within them.
+// Full-bleed immersive layout: background illustration fills viewport,
+// UI layers float on top (scrim, text, choices, top bar).

+import { useState, useEffect, useRef, useCallback } from 'react';
 import StreamingText from './StreamingText.jsx';
 import ChoiceGrid from './ChoiceGrid.jsx';
 import LanternLoader from './LanternLoader.jsx';
@@ -16,30 +17,75 @@ export default function StoryView({
   isGenerating,
   isDead,
   onNewStory,
+  beat,
+  onMenuToggle,
 }) {
   const displayText = isStreaming ? streamedText : currentScene;

+  const [persistedBg, setPersistedBg] = useState(null);
+  const [incomingSrc, setIncomingSrc] = useState(null);
+  const [showIncoming, setShowIncoming] = useState(false);
+  const currentImgRef = useRef(null);
+  const incomingImgRef = useRef(null);
+
+  useEffect(() => {
+    if (illustration && illustration !== persistedBg) {
+      setIncomingSrc(illustration);
+    }
+  }, [illustration, persistedBg]);
+
+  const handleIncomingLoad = useCallback(() => {
+    setShowIncoming(true);
+    const timer = setTimeout(() => {
+      setPersistedBg(incomingSrc);
+      setShowIncoming(false);
+      setIncomingSrc(null);
+    }, 1300);
+    return () => clearTimeout(timer);
+  }, [incomingSrc]);
+
+  const textRef = useRef(null);
+  useEffect(() => {
+    if (textRef.current) {
+      textRef.current.scrollTop = textRef.current.scrollHeight;
+    }
+  }, [displayText]);
+
   return (
-    <div className="story-layout">
-      <div className="panel-illustration">
-        {illustration ? (
-          <img src={illustration} alt="Scene illustration" />
+    <div className="story-viewport">
+      <div className="bg-layer">
+        {persistedBg ? (
+          <img ref={currentImgRef} className="bg-current" src={persistedBg} alt="" />
         ) : (
-          <div className="illus-placeholder">
-            {isGenerating ? (
-              <LanternLoader caption="Illustration brewing…" />
-            ) : (
-              <div className="illus-empty">
-                <span className="gem">◆</span>
-              </div>
-            )}
-          </div>
+          <div className="bg-layer--empty" />
+        )}
+        {incomingSrc && (
+          <img
+            ref={incomingImgRef}
+            className={`bg-incoming${showIncoming ? ' visible' : ''}`}
+            src={incomingSrc}
+            alt=""
+            onLoad={handleIncomingLoad}
+          />
         )}
       </div>

-      <div className="panel-text">
+      <div className="scrim" />
+
+      <div className="top-bar">
+        <button className="menu-btn" onClick={onMenuToggle}>☰</button>
+        {beat && <span className="beat-indicator">{beat.replace('_', ' ')}</span>}
+      </div>
+
+      <div className="text-overlay" ref={textRef}>
         {displayText ? (
           <StreamingText text={displayText} isStreaming={isStreaming} />
         ) : isGenerating ? (
-          <LanternLoader caption="Crafting your scene…" />
+          <LanternLoader caption="Crafting your scene…" />
         ) : null}

         {isDead && (
           <div className="death-screen">
             <h3>You Died.</h3>
             <p>Your adventure ends in tragedy.</p>
-            <p>Use the sidebar to <em>Start a New Story</em> or <em>Reset</em>.</p>
             <button className="sidebar-btn" onClick={onNewStory} style={{ marginTop: '0.75rem' }}>
               Start New Story
             </button>
@@ -47,14 +93,10 @@ export default function StoryView({
         )}
       </div>

-      <div className="panel-choices">
+      <div className="choice-overlay">
         {isDead ? null : isGenerating && !displayText ? null : (
-          <ChoiceGrid
-            choices={choices}
-            onChoose={onChoose}
-            disabled={isStreaming || isGenerating}
-          />
+          <ChoiceGrid choices={choices} onChoose={onChoose} disabled={isStreaming || isGenerating} />
         )}
       </div>
     </div>
   );
 }
```

### src/hooks/useStoryEngine.js

```diff
--- a/src/hooks/useStoryEngine.js
+++ b/src/hooks/useStoryEngine.js
@@ -6,8 +6,8 @@ import { streamScene, generateChoices, generateIllustration, BEATS, BEAT_TARGET_
 import { ensureBrowserId } from '../services/identity.js';
 import { saveSnapshot, loadSnapshot, deleteSnapshot } from '../services/persistence.js';

-const ILLUSTRATION_EVERY_N = 3;
-const ILLUSTRATION_PHASE = 1;
+const ILLUSTRATION_EVERY_N = 2;
+const ILLUSTRATION_PHASE = 0;

 function shouldIllustrate(sceneIndex) {
   return ((sceneIndex - ILLUSTRATION_PHASE) % ILLUSTRATION_EVERY_N) === 0;
@@ -71,7 +71,8 @@ export default function useStoryEngine() {
     setIsStreaming(true);
     setStreamedText('');
     setChoices([]);
-    setIllustration(null);
+    // Don't clear illustration — persistedBg in StoryView keeps the
+    // last image visible. New illustrations crossfade in when ready.

     let fullText = '';
     try {
@@ -164,7 +165,20 @@ export default function useStoryEngine() {
     setDangerStreak(0);
     setCurrentScene('');
     setChoices([]);
-    setIllustration(null);
+
+    // Show static cover image instantly, then fire DALL-E to replace it
+    if (profile.starterHook?.coverImage) {
+      setIllustration(profile.starterHook.coverImage);
+    } else {
+      setIllustration(null);
+    }
+
+    // Fire DALL-E illustration from hook blurb (non-blocking, crossfades in when ready)
+    if (profile.starterHook) {
+      generateIllustration(profile.starterHook.blurb, profile.gender)
+        .then((url) => { if (url) setIllustration(url); })
+        .catch(() => {});
+    }

     // Build initial history with starter hook context
     const hookContext = profile.starterHook
```

### src/services/openai.js

```diff
--- a/src/services/openai.js
+++ b/src/services/openai.js
@@ -99,7 +99,7 @@ export async function chatCompletionFull(messages, options = {}) {
  */
 export async function imageGeneration(prompt, options = {}) {
-  const { model = 'dall-e-3', size = '1024x1024' } = options;
+  const { model = 'dall-e-3', size = '1024x1792' } = options;

   const res = await fetch(`${API_BASE}/images/generations`, {
```

### src/index.css

The CSS diff is extensive (full rewrite of layout system). Key sections:

- **Removed:** `.story-layout`, `.panel-*`, `.illus-placeholder`, `.illus-empty`, `.illus-sep` and related rules
- **Added:** `.story-viewport`, `.bg-layer`, `.bg-current`, `.bg-incoming`, `.bg-layer--empty`, `.scrim`, `.text-overlay`, `.choice-overlay`, `.top-bar`, `.menu-btn`, `.beat-indicator`, `.drawer-backdrop`, `.drawer`
- **Modified:** `.storybox` (added text-shadow, removed border/bg), `.choice-btn` (glassmorphism), `.death-screen` (backdrop-filter), `.app-layout` (block instead of flex), `html/body` (fixed height), responsive rules

See the full index.css diff in the git history or compare the current file against commit `cb6fe19`.
