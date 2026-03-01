Two fixes needed:

---

## Fix 1: Story text getting cut off behind choice buttons on mobile

The last line of story text is clipped behind the choice overlay. The `bottom` offset on `.text-overlay` isn't enough because choice buttons sometimes wrap to two lines.

**In `src/index.css`**, add `padding-bottom` to `.text-overlay` so the scrollable content has clearance at the bottom:

```css
.text-overlay {
  /* keep all existing properties, ADD: */
  padding-bottom: 1rem;
}
```

And in the mobile media query (`@media (max-width: 480px)`), increase `.text-overlay` bottom from `130px` to `140px`:

```css
@media (max-width: 480px) {
  .text-overlay {
    bottom: 140px;
    /* ...rest unchanged */
  }
}
```

---

## Fix 2: DALL-E generating fake/faux text on images

DALL-E 3 is inserting fake text onto signs, banners, scrolls, and architecture in the generated images. This needs to be suppressed in ALL image generation prompts.

### A) Runtime illustration prompts — `src/engine/StoryEngine.js`

In the `basicStyle()` function, add this directive to the end of the returned string (before the closing quote/backtick):

```
'Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.'
```

So the full `basicStyle()` return ends with:
```
...Do not depict the protagonist's face directly (use silhouette, hood, or a cropped angle). Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.'
```

Also add the same directive to the `fallbackPrompt` string in `generateIllustration()` — append it to the end of that prompt as well.

### B) Static cover image generation — `scripts/generate-covers.js`

Find wherever the DALL-E prompt is constructed in this script and add the same no-text directive:

```
Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image.
```

### C) Regenerate the static cover images

After updating `scripts/generate-covers.js`, re-run it to regenerate all 6 hook cover images (`public/images/hook-01.png` through `hook-06.png`) with the updated no-text prompt.

---

These are the only files that need changes:
- `src/index.css` (text overlay padding/bottom)
- `src/engine/StoryEngine.js` (no-text directive in `basicStyle()` and `fallbackPrompt`)
- `scripts/generate-covers.js` (no-text directive in cover generation prompt)
- `public/images/hook-01.png` through `hook-06.png` (regenerated)
