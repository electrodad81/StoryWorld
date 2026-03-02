// scripts/generate-explore-covers.js
// Generates static cover images for exploration mode locations via DALL-E 3.
// Usage: OPENAI_API_KEY=sk-... node scripts/generate-explore-covers.js

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');
const OUT_DIR = path.join(ROOT, 'public', 'images', 'explore');
const WORLD_PATH = path.join(ROOT, 'public', 'data', 'withered-vale.json');

const API_KEY = process.env.OPENAI_API_KEY;
if (!API_KEY) {
  console.error('Error: set OPENAI_API_KEY environment variable');
  process.exit(1);
}

const STYLE = `Detailed black-and-white ink illustration with a single spot color accent (gold leaf or lapis). Rich crosshatching, clear contours, deep tonal range from black to mid-gray. The scene fills the entire frame edge-to-edge with rich environmental detail. Dark, moody atmosphere throughout — no white or bright backgrounds. Deep shadows and low ambient light. Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). Architecture: stone keeps, timber framing, castles, market stalls; natural materials like wood, leather, linen, iron. Do not depict any character's face directly (use silhouette, hood, or a cropped angle). No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore. Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind. The scene must extend edge-to-edge as if viewed through a window.`;

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

  const world = JSON.parse(fs.readFileSync(WORLD_PATH, 'utf-8'));
  const locations = world.locations || [];

  console.log(`Generating cover images for ${locations.length} exploration locations...\n`);

  for (const loc of locations) {
    const filename = `${loc.id}.png`;
    const outPath = path.join(OUT_DIR, filename);

    // Skip if already exists
    if (fs.existsSync(outPath)) {
      console.log(`  [skip] ${filename} already exists`);
      continue;
    }

    const prompt = `${STYLE}\n\nScene: ${loc.description} Mood: ${loc.mood}.`;
    console.log(`  [gen]  ${filename} — "${loc.name}"`);

    try {
      const b64 = await generateImage(prompt);
      if (!b64) {
        console.log(`  [fail] ${filename} — no image data returned`);
        continue;
      }

      fs.writeFileSync(outPath, Buffer.from(b64, 'base64'));
      console.log(`  [ok]   ${outPath}`);
    } catch (e) {
      console.log(`  [fail] ${filename} — ${e.message}`);
    }

    // Small delay between calls to avoid rate limits
    await new Promise((r) => setTimeout(r, 2000));
  }

  console.log('\nDone.');
}

main();
