# Immersive UI — Follow-Up Fixes (Legibility, Overlap, Scene Length)

## Context

The full-bleed immersive layout from the previous spec is implemented and working. This spec addresses three issues observed in testing on desktop and mobile.

---

## Fix 1: Text Legibility — Semi-Transparent Card Behind Story Text

### Problem
Story text is hard to read where it overlaps detailed/bright areas of the background illustration. The gradient scrim alone isn't sufficient because the text block extends above where the scrim reaches full opacity.

### Solution
Add a frosted-glass card behind the `.storybox` element so text always has guaranteed contrast regardless of background image content.

### CSS Changes (index.css)

Modify `.storybox`:

```css
.storybox {
  /* KEEP all existing properties (width, font-family, font-size, line-height, etc.) */
  /* KEEP existing text-shadow */

  /* ADD these properties: */
  background: rgba(10, 10, 20, 0.6);
  backdrop-filter: blur(8px);
  -webkit-backdrop-filter: blur(8px);
  border-radius: 12px;
  padding: 1rem 1.25rem;
  border: 1px solid rgba(255, 255, 255, 0.05);
}
```

This gives the text block a consistent dark frosted-glass surface. The `rgba(10, 10, 20, 0.6)` is dark enough to guarantee contrast against any image but still lets the illustration bleed through subtly. The blur adds depth.

### Also apply to the death screen

The `.death-screen` should sit inside this same visual language. It likely already has `backdrop-filter` from the previous pass, but verify that it still looks correct nested inside the text overlay — it should appear as a distinct card within the text area, not fight with the storybox card. If it looks redundant (card-within-card), consider removing the death screen's own backdrop-filter and just using a solid border-left accent instead:

```css
.death-screen {
  background: rgba(192, 57, 43, 0.12);
  border-left: 3px solid var(--danger);
  border-radius: 8px;
  padding: 1rem 1.25rem;
  margin-top: 0.75rem;
  /* Remove backdrop-filter if it creates a double-glass effect */
}
```

---

## Fix 2: "Your Choices" Label Overlapping Story Text on Mobile

### Problem
On mobile, the "Your choices" heading and choice buttons collide with the bottom of the story text. The `bottom: 110px` on `.text-overlay` doesn't leave enough clearance, especially when the choice overlay's gradient background extends upward.

### Solution
Two changes working together:

**A) Increase clearance between text and choices.**

In `.text-overlay`, change `bottom: 110px` to `bottom: 120px` as the base, and increase to `130px` on mobile:

```css
.text-overlay {
  /* Change: */
  bottom: 120px;
  /* ...rest unchanged */
}

@media (max-width: 480px) {
  .text-overlay {
    bottom: 130px;
    /* ...rest of mobile overrides unchanged */
  }
}
```

**B) Remove or restyle the "Your choices" heading in ChoiceGrid.**

The "Your choices" `<h3>` inside `.choice-grid` takes up vertical space and adds little value — the buttons are self-explanatory. Either:

- **Option 1 (preferred):** Remove the `<h3>` from `ChoiceGrid.jsx` entirely. The choice buttons are visually distinct enough (glassmorphism, pinned at bottom) that no label is needed.
- **Option 2:** Hide it with CSS: `.choice-grid h3 { display: none; }`

Go with Option 1 — removing the element is cleaner than hiding it.

### File Changes

**`src/components/ChoiceGrid.jsx`** — Remove the `<h3>Your choices</h3>` element (or equivalent heading) from the render output. Keep the choice buttons and their container unchanged.

**`src/index.css`** — Remove the `.choice-grid h3` style rule (no longer needed if the element is gone).

---

## Fix 3: Shorter Scene Text — No-Scroll Goal

### Problem
At ~85–150 words, scene text creates a wall of text that often requires scrolling, especially on mobile. In the immersive layout, the text area has a constrained `max-height` (45vh desktop, 40vh mobile), and long scenes overflow it. Scrolling breaks immersion — the player should see the entire scene and choices on one screen without scrolling.

### Solution
Reduce the word count guidance in the system prompt so scenes fit on-screen at all viewport sizes.

**Target:** ~60–90 words per scene. At `font-size: 1.05rem` (mobile) with `line-height: 1.5`, this is roughly 6–9 lines of text on a 375px-wide phone — comfortably within the 40vh text area without scrolling.

### File Changes

**`src/engine/StoryEngine.js`** — Modify the system prompt and user prompt in `streamScene()`:

Find the `sysBase` string. Change this part:
```
Keep prose tight by default (~85–150 words, single paragraph).
```
To:
```
Keep prose very tight (~60–90 words, single short paragraph). Every word must earn its place.
```

Find the `user` string near the end of `streamScene()`. Change this part:
```
Length: ~85–150 words. End cleanly; do not start a new paragraph.
```
To:
```
Length: ~60–90 words. One compact paragraph. End cleanly; no new paragraphs.
```

Also change the `max_tokens` passed to the streaming call. In the `user` variable construction area, the current call is:
```js
yield* chatCompletion(messages, { temperature: 0.9, max_tokens: 350 });
```
Change to:
```js
yield* chatCompletion(messages, { temperature: 0.9, max_tokens: 200 });
```
This acts as a hard ceiling — 200 tokens ≈ 150 words, giving the model room while preventing runaway generation. The prompt guidance handles the soft target.

---

## Summary of All Changes

| File | Change | Fix # |
|------|--------|-------|
| `src/index.css` | Add frosted-glass background to `.storybox` | 1 |
| `src/index.css` | Adjust `.death-screen` if double-glass issue | 1 |
| `src/index.css` | Change `.text-overlay` bottom to 120px (130px mobile) | 2 |
| `src/index.css` | Remove `.choice-grid h3` style rule | 2 |
| `src/components/ChoiceGrid.jsx` | Remove the "Your choices" `<h3>` element | 2 |
| `src/engine/StoryEngine.js` | Change word count in `sysBase` to ~60–90 | 3 |
| `src/engine/StoryEngine.js` | Change word count in `user` prompt to ~60–90 | 3 |
| `src/engine/StoryEngine.js` | Change `max_tokens` from 350 to 200 | 3 |

## Testing Checklist

- [ ] Story text is clearly readable over both bright and dark background images
- [ ] The frosted card doesn't feel too heavy/opaque — illustration still subtly visible through it
- [ ] On mobile (375×812), a full scene (~60–90 words) + choices fit on screen without scrolling
- [ ] No overlap between story text and choice buttons on any viewport size
- [ ] Death screen looks correct within the text overlay (no awkward double-glass effect)
- [ ] Scene prose still feels complete and atmospheric at the shorter length — not truncated or rushed
- [ ] The `max_tokens: 200` ceiling doesn't cause mid-sentence cutoffs (the model should self-regulate to ~60–90 words well within 200 tokens)
