# Final UI Polish — Changelog (Image Borders + Dark Moody Style)

Covers all changes between commits `50e52ee` (UI polish changelog) and `f79af32` (dark moody images). This is the final round of UI changes.

---

## Summary

Two rounds of image prompt improvements plus a CSS safety net:

### Round 1: Borderless/frameless images
- Added directive to suppress decorative borders, frames, ornamental edges, and vignettes
- Scaled `.bg-layer img` to 104% with `inset: -2%` so any residual border gets cropped off-screen

### Round 2: Dark moody atmosphere
- Replaced "Simple, clean" style with "Detailed" ink illustration, rich crosshatching, deep tonal range
- Removed "Plain white background behind the subject" — this was causing washed-out top bands
- Removed "single focal subject, medium or close-up shot" — this encouraged centered compositions with empty space
- Added "Dark, moody atmosphere throughout — no white or bright backgrounds. Deep shadows and low ambient light."
- All 6 static cover images regenerated with final prompt

---

## Files Changed

| File | Change |
|------|--------|
| `src/engine/StoryEngine.js` | Reworked `basicStyle()` for dark moody style + borderless directive; added borderless to `fallbackPrompt` |
| `scripts/generate-covers.js` | Matching style updates in `STYLE` constant |
| `src/index.css` | `.bg-layer img` scaled to 104% with `inset: -2%` |
| `public/images/hook-01.png` — `hook-06.png` | Regenerated twice (once for borders, once for dark style) |

---

## Full Diffs

### `src/engine/StoryEngine.js`

#### `basicStyle()` — full rework

```diff
 function basicStyle() {
   return (
-    'Simple, clean black-and-white line-and-wash illustration with a single spot color accent ' +
-    '(gold leaf or lapis). Light crosshatching, clear contours, single focal subject, ' +
-    'medium or close-up shot. Plain white background behind the subject. ' +
+    'Detailed black-and-white ink illustration with a single spot color accent ' +
+    '(gold leaf or lapis). Rich crosshatching, clear contours, deep tonal range from black to mid-gray. ' +
+    'The scene fills the entire frame edge-to-edge with rich environmental detail. ' +
+    'Dark, moody atmosphere throughout — no white or bright backgrounds. Deep shadows and low ambient light. ' +
     'Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). ' +
     'Architecture: stone keeps, timber framing, castles, market stalls; natural materials ' +
     'like wood, leather, linen, iron. Props/weapons allowed: swords, daggers, axes, spears, ' +
     'shields, bows, torches, lanterns, books, scrolls, potions. ' +
     'Strictly no firearms or explosives, no modern clothing (no trenchcoats, suits, ties), ' +
     'no modern tech/vehicles (no cameras, phones, radios, cars, trains, planes, neon, streetlights, power lines). ' +
     'Do not depict the protagonist\'s face directly (use silhouette, hood, or a cropped angle). ' +
-    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.'
+    'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. ' +
+    'The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.'
   );
 }
```

#### `generateIllustration()` — fallback prompt borderless directive

```diff
     'Atmospheric medieval-fantasy environment; focus on scenery and architecture; ' +
     'no modern elements, no characters in distress, no firearms or explosives; family-friendly, PG-13. ' +
     'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. ' +
+    'The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window. ' +
     gdir;
```

---

### `scripts/generate-covers.js`

```diff
-const STYLE = `Simple, clean black-and-white line-and-wash illustration with a single spot color accent (gold leaf or lapis). Light crosshatching, clear contours, single focal subject, medium or close-up shot. Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). Architecture: stone keeps, timber framing, castles, market stalls; natural materials like wood, leather, linen, iron. Do not depict any character's face directly (use silhouette, hood, or a cropped angle). No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore. Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.`;
+const STYLE = `Detailed black-and-white ink illustration with a single spot color accent (gold leaf or lapis). Rich crosshatching, clear contours, deep tonal range from black to mid-gray. The scene fills the entire frame edge-to-edge with rich environmental detail. Dark, moody atmosphere throughout — no white or bright backgrounds. Deep shadows and low ambient light. Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). Architecture: stone keeps, timber framing, castles, market stalls; natural materials like wood, leather, linen, iron. Do not depict any character's face directly (use silhouette, hood, or a cropped angle). No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore. Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.`;
```

---

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

---

### `public/images/hook-01.png` through `hook-06.png`

All 6 static cover images regenerated with the final dark moody, borderless prompt. Binary files, not diffed.

---

## Current State of DALL-E Style Prompt

The full `basicStyle()` return value is now:

```
Detailed black-and-white ink illustration with a single spot color accent (gold leaf or lapis). Rich crosshatching, clear contours, deep tonal range from black to mid-gray. The scene fills the entire frame edge-to-edge with rich environmental detail. Dark, moody atmosphere throughout — no white or bright backgrounds. Deep shadows and low ambient light. Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). Architecture: stone keeps, timber framing, castles, market stalls; natural materials like wood, leather, linen, iron. Props/weapons allowed: swords, daggers, axes, spears, shields, bows, torches, lanterns, books, scrolls, potions. Strictly no firearms or explosives, no modern clothing (no trenchcoats, suits, ties), no modern tech/vehicles (no cameras, phones, radios, cars, trains, planes, neon, streetlights, power lines). Do not depict the protagonist's face directly (use silhouette, hood, or a cropped angle). Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.
```
