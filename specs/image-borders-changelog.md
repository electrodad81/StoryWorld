# Image Borders Fix — Changelog

Covers changes between commits `50e52ee` and `3686e8a`.

---

## Summary

DALL-E 3 was generating decorative borders/frames on illustrations, breaking the full-bleed immersive look. Two fixes applied:

1. **Prompt directive**: Added a borderless/frameless instruction to all DALL-E prompts (runtime `basicStyle()`, `fallbackPrompt`, and static cover script)
2. **CSS safety net**: Scaled `.bg-layer img` to 104% with `inset: -2%` so any residual border gets cropped off-screen by the parent's `overflow: hidden`
3. **Regenerated covers**: All 6 static cover images regenerated with updated prompt

---

## Files Changed

| File | Change |
|------|--------|
| `src/engine/StoryEngine.js` | Borderless directive added to `basicStyle()` and `fallbackPrompt` |
| `scripts/generate-covers.js` | Borderless directive added to `STYLE` constant |
| `src/index.css` | `.bg-layer img` scaled to 104% with `inset: -2%` |
| `public/images/hook-01.png` — `hook-06.png` | Regenerated (binary, not diffed) |

---

## Full Diffs

### `src/engine/StoryEngine.js`

#### `basicStyle()` — borderless directive

```diff
     'Do not depict the protagonist\'s face directly (use silhouette, hood, or a cropped angle). ' +
-    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.'
+    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. ' +
+    'The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.'
```

#### `generateIllustration()` — fallback prompt

```diff
     'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. ' +
+    'The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window. ' +
     gdir;
```

### `scripts/generate-covers.js`

```diff
-const STYLE = `...Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.`;
+const STYLE = `...Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.`;
```

### `src/index.css`

```diff
 .bg-layer img {
   position: absolute;
-  inset: 0;
-  width: 100%;
-  height: 100%;
+  inset: -2%;
+  width: 104%;
+  height: 104%;
   object-fit: cover;
   transition: opacity 1.2s ease-in-out;
 }
```
