The DALL-E illustrations are consistently generating a light/white band across the top of the image — a washed-out sky or background zone that creates a visible seam against the dark UI. This is caused by the style prompt.

**In `src/engine/StoryEngine.js`**, in the `basicStyle()` function, make these changes to the returned string:

1. **Remove** the phrase `Plain white background behind the subject.` — this is what's causing DALL-E to create the stark light zone at the top.

2. **Remove** `single focal subject, medium or close-up shot.` — this encourages a centered composition with empty space around it rather than an edge-to-edge environment.

3. **Replace** them with: `The scene fills the entire frame edge-to-edge with rich environmental detail. Dark, moody atmosphere throughout — no white or bright backgrounds. Deep shadows and low ambient light.`

4. **Change** `Simple, clean black-and-white line-and-wash illustration with a single spot color accent (gold leaf or lapis). Light crosshatching, clear contours,` to: `Detailed black-and-white ink illustration with a single spot color accent (gold leaf or lapis). Rich crosshatching, clear contours, deep tonal range from black to mid-gray.`

The goal is to get DALL-E producing full-bleed, dark, atmospheric environment illustrations with no washed-out zones. The image needs to look good as a phone wallpaper — dark and moody edge-to-edge, with no stark white areas that fight the dark UI overlays.

Apply the same wording changes to:
- The `fallbackPrompt` in `generateIllustration()` (if it references any of the same phrases)
- The `STYLE` constant in `scripts/generate-covers.js`

After updating `scripts/generate-covers.js`, regenerate all 6 cover images.

---

## Files to change

| File | Change |
|------|--------|
| `src/engine/StoryEngine.js` | Rework `basicStyle()` return string as described above |
| `scripts/generate-covers.js` | Match the updated style language |
| `public/images/hook-01.png` — `hook-06.png` | Regenerated |
