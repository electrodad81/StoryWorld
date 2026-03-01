Two related fixes for the background image display:

---

## Fix 1: DALL-E generating decorative borders/frames on images

DALL-E 3 is adding ornamental borders, frames, and decorative bands to the generated illustrations. These create visible seams and edges that break the full-bleed immersive look.

**In `src/engine/StoryEngine.js`**, in the `basicStyle()` function, add this directive to the returned string (near the other negative constraints):

```
'The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.'
```

Also add the same directive to the `fallbackPrompt` string in `generateIllustration()`.

**In `scripts/generate-covers.js`**, add the same borderless directive to the `STYLE` constant.

After updating `scripts/generate-covers.js`, regenerate all 6 cover images (`public/images/hook-01.png` through `hook-06.png`).

---

## Fix 2: Ensure background image always covers full viewport

As a CSS safety net in case any image still has borders or doesn't quite fill the viewport, add a slight scale-up to the background images so they overshoot the viewport edges.

**In `src/index.css`**, modify the `.bg-layer img` rule:

Change:
```css
.bg-layer img {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: opacity 1.2s ease-in-out;
}
```

To:
```css
.bg-layer img {
  position: absolute;
  inset: -2%;
  width: 104%;
  height: 104%;
  object-fit: cover;
  transition: opacity 1.2s ease-in-out;
}
```

This oversizes the image by 4% in each direction, which is invisible to the user but ensures any decorative border DALL-E adds gets cropped off-screen. The `overflow: hidden` on the parent `.story-viewport` handles the bleed.

---

## Files to change

| File | Change |
|------|--------|
| `src/engine/StoryEngine.js` | Add borderless/frameless directive to `basicStyle()` and `fallbackPrompt` |
| `scripts/generate-covers.js` | Add borderless/frameless directive to STYLE constant, then regenerate covers |
| `src/index.css` | Scale `.bg-layer img` to 104% with `inset: -2%` |
| `public/images/hook-01.png` — `hook-06.png` | Regenerated |
