// scripts/generate-covers.js
// Generates static cover images for each starter hook via DALL-E 3.
// Usage: OPENAI_API_KEY=sk-... node scripts/generate-covers.js

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'public', 'images');
const LORE_PATH = path.join(ROOT, 'public', 'data', 'lore.json');

const API_KEY = process.env.OPENAI_API_KEY;
if (!API_KEY) {
  console.error('Error: set OPENAI_API_KEY environment variable');
  process.exit(1);
}

const STYLE = `Simple, clean black-and-white line-and-wash illustration with a single spot color accent (gold leaf or lapis). Light crosshatching, clear contours, single focal subject, medium or close-up shot. Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). Architecture: stone keeps, timber framing, castles, market stalls; natural materials like wood, leather, linen, iron. Do not depict any character's face directly (use silhouette, hood, or a cropped angle). No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore. Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.`;

async function generateImage(prompt) {
  const res = await fetch('https://api.openai.com/v1/images/generations', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Authorization: `Bearer ${API_KEY}`,
    },
    body: JSON.stringify({
      model: 'dall-e-3',
      prompt,
      size: '1024x1792',
      n: 1,
      response_format: 'b64_json',
    }),
  });

  if (!res.ok) {
    const err = await res.text();
    throw new Error(`API error ${res.status}: ${err}`);
  }

  const data = await res.json();
  return data.data?.[0]?.b64_json;
}

async function main() {
  fs.mkdirSync(OUT_DIR, { recursive: true });

  const lore = JSON.parse(fs.readFileSync(LORE_PATH, 'utf-8'));
  const hooks = lore.starter_hooks || [];

  console.log(`Generating cover images for ${hooks.length} hooks...\n`);

  for (const hook of hooks) {
    const num = hook.id.replace('HOOK-', '');
    const filename = `hook-${num}.webp`;
    const outPath = path.join(OUT_DIR, filename);

    // Skip if already exists
    if (fs.existsSync(outPath)) {
      console.log(`  [skip] ${filename} already exists`);
      continue;
    }

    const prompt = `${STYLE}\n\nScene: ${hook.blurb}`;
    console.log(`  [gen]  ${filename} — "${hook.title}"`);

    try {
      const b64 = await generateImage(prompt);
      if (!b64) {
        console.log(`  [fail] ${filename} — no image data returned`);
        continue;
      }

      // DALL-E returns PNG in b64; save as-is with .webp extension
      // (browsers handle PNG data in .webp files fine, but let's save as .png
      //  and update the lore reference if needed)
      const pngPath = outPath.replace('.webp', '.png');
      fs.writeFileSync(pngPath, Buffer.from(b64, 'base64'));
      console.log(`  [ok]   ${pngPath}`);
    } catch (e) {
      console.log(`  [fail] ${filename} — ${e.message}`);
    }

    // Small delay between calls to avoid rate limits
    await new Promise((r) => setTimeout(r, 2000));
  }

  console.log('\nDone. Update lore.json coverImage paths if you used .png instead of .webp.');
}

main();
