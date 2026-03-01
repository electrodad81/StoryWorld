Make three small CSS changes to the `.storybox` class in `src/index.css` to reduce how much the frosted text card obscures the background illustration:

1. **Reduce card opacity:** Change `background: rgba(10, 10, 20, 0.6)` to `background: rgba(10, 10, 20, 0.35)`. The gradient scrim and text-shadow already provide most of the legibility — the card should be a subtle tint, not a wall.

2. **Add horizontal margin** so the card floats as a narrower pill and the background image peeks through on the sides: Add `margin: 0 0.5rem` to `.storybox`.

3. **Tighten vertical padding** to shrink the card's overall footprint: Change `padding: 1rem 1.25rem` to `padding: 0.75rem 1rem`.

No other files need to change. The goal is that on mobile the card feels like a translucent overlay rather than a dark rectangle blocking the illustration.
