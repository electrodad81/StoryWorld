# UI Polish — Changelog (post immersive-ui-changelog)

Covers all changes between commits `0d84652` (Immersive full-bleed UI overhaul) and `6fad244` (UI polish round).

---

## Summary of Changes

### Fix 1: Frosted-glass card behind story text
- Added semi-transparent background (`rgba(10, 10, 20, 0.35)`), `backdrop-filter: blur(8px)`, border-radius, and subtle border to `.storybox`
- Added horizontal margin (`0 0.5rem`) so the card floats as a narrower pill and background image peeks through on the sides
- Tightened padding from `0.75rem 0` to `0.75rem 1rem`
- Restyled `.death-screen` to use a `border-left` accent instead of full `backdrop-filter` to avoid double-glass effect

### Fix 2: Text/choice overlap on mobile
- Increased `.text-overlay` bottom clearance from `110px` to `120px` (desktop) and from `100px` to `140px` (mobile)
- Added `padding-bottom: 1rem` to `.text-overlay` for scroll clearance
- Removed "Your choices" `<h3>` heading from `ChoiceGrid.jsx` and its CSS rule

### Fix 3: Shorter scene text (no-scroll goal)
- Changed word count guidance in system prompt from ~85–150 to ~60–90 words
- Changed word count guidance in user prompt from ~85–150 to ~60–90 words
- Reduced `max_tokens` from 350 to 200

### Fix 4: No faux text on DALL-E images
- Added no-text directive to `basicStyle()` in `StoryEngine.js`
- Added no-text directive to `fallbackPrompt` in `generateIllustration()`
- Added no-text directive to `STYLE` constant in `scripts/generate-covers.js`
- Regenerated all 6 static cover images (`hook-01.png` through `hook-06.png`)

---

## Files Changed

| File | What changed |
|------|-------------|
| `src/index.css` | Frosted storybox, death-screen restyle, text-overlay clearance increases, removed `.choice-grid h3` rule |
| `src/components/ChoiceGrid.jsx` | Removed `<h3>Your choices</h3>` element |
| `src/engine/StoryEngine.js` | Shorter word count prompts, lower max_tokens, no-text directive in image prompts |
| `scripts/generate-covers.js` | No-text directive added to STYLE constant |
| `public/images/hook-01.png` — `hook-06.png` | Regenerated with no-text prompt (binary, not diffed) |

---

## Full Diffs

### `src/index.css`

```diff
 /* Layer 2: Text overlay */
 .text-overlay {
   position: absolute;
-  bottom: 110px;
+  bottom: 120px;
   left: 0;
   right: 0;
   z-index: 2;
   max-height: 45vh;
   overflow-y: auto;
-  padding: 0 1.25rem;
+  padding: 0 1.25rem 1rem;
   display: flex;
   flex-direction: column;
   align-items: center;
```

```diff
 .storybox {
   width: var(--story-body-width);
   max-width: 100%;
-  margin: 0;
-  padding: 0.75rem 0;
+  margin: 0 0.5rem;
+  padding: 0.75rem 1rem;
   font-family: 'Times New Roman', Georgia, serif;
   font-size: 1.35rem;
   line-height: 1.6;
@@ ...
   color: var(--text-primary);
   white-space: pre-wrap;
   text-shadow: 0 1px 4px rgba(0, 0, 0, 0.7), 0 0 12px rgba(0, 0, 0, 0.4);
+  background: rgba(10, 10, 20, 0.35);
+  backdrop-filter: blur(8px);
+  -webkit-backdrop-filter: blur(8px);
+  border-radius: 12px;
+  border: 1px solid rgba(255, 255, 255, 0.05);
 }
```

```diff
-.choice-grid h3 {
-  font-size: 0.85rem;
-  color: rgba(224, 216, 204, 0.6);
-  margin-bottom: 0.15rem;
-  text-shadow: 0 1px 3px rgba(0, 0, 0, 0.5);
-}
-
 .choice-btn {
```

```diff
 .death-screen {
   width: var(--story-body-width);
   max-width: 100%;
   margin-top: 0.75rem;
-  padding: 1.2rem 1.5rem;
-  border: 1px solid var(--danger);
+  padding: 1rem 1.25rem;
+  background: rgba(192, 57, 43, 0.12);
+  border-left: 3px solid var(--danger);
   border-radius: 8px;
-  backdrop-filter: blur(16px);
-  -webkit-backdrop-filter: blur(16px);
-  background: rgba(192, 57, 43, 0.15);
   line-height: 1.6;
 }
```

```diff
 @media (max-width: 480px) {
   .text-overlay {
-    bottom: 100px;
+    bottom: 140px;
     max-height: 40vh;
     padding: 0 0.75rem;
   }
```

---

### `src/components/ChoiceGrid.jsx`

```diff
   return (
     <div className="choice-grid">
-      <h3>Your choices</h3>
       {choices.map((label, i) => (
         <button
           key={i}
```

---

### `src/engine/StoryEngine.js`

#### `basicStyle()` — no-text directive

```diff
     'no modern tech/vehicles (no cameras, phones, radios, cars, trains, planes, neon, streetlights, power lines). ' +
-    'Do not depict the protagonist\'s face directly (use silhouette, hood, or a cropped angle).'
+    'Do not depict the protagonist\'s face directly (use silhouette, hood, or a cropped angle). ' +
+    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.'
```

#### `streamScene()` — shorter word count

```diff
-    'Keep prose tight by default (~85–150 words, single paragraph). Maintain continuity with prior scenes.\n' +
+    'Keep prose very tight (~60–90 words, single short paragraph). Every word must earn its place. Maintain continuity with prior scenes.\n' +
```

```diff
-    'Length: ~85–150 words. End cleanly; do not start a new paragraph.';
+    'Length: ~60–90 words. One compact paragraph. End cleanly; no new paragraphs.';
```

```diff
-  yield* chatCompletion(messages, { temperature: 0.9, max_tokens: 350 });
+  yield* chatCompletion(messages, { temperature: 0.9, max_tokens: 200 });
```

#### `generateIllustration()` — fallback prompt no-text directive

```diff
     'no modern elements, no characters in distress, no firearms or explosives; family-friendly, PG-13. ' +
+    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. ' +
     gdir;
```

---

### `scripts/generate-covers.js`

```diff
-const STYLE = `...No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore.`;
+const STYLE = `...No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore. Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.`;
```

---

### `public/images/hook-01.png` through `hook-06.png`

All 6 static cover images were regenerated with the updated no-text prompt. These are binary files (PNG, ~2–4 MB each) and not diffed here.
