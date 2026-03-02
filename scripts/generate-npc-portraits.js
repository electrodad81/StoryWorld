// scripts/generate-npc-portraits.js
// Generates portrait images for exploration NPCs via DALL-E 3.
// Usage: OPENAI_API_KEY=sk-... node scripts/generate-npc-portraits.js

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

const STYLE = `Detailed black-and-white ink illustration with a single spot color accent (gold leaf or lapis). Rich crosshatching, clear contours, deep tonal range from black to mid-gray. Dark, moody atmosphere — no white or bright backgrounds. Deep shadows and low ambient light. Pre-industrial medieval-fantasy era (roughly 12th–16th century aesthetic). Natural materials like wood, leather, linen, iron. Do not show the character's full face directly — use a three-quarter angle, partial shadow, hood, or cropped framing so features are suggested rather than explicit. No firearms, no modern clothing, no modern tech. Family-friendly, PG-13, no gore. Absolutely no text, letters, words, numbers, runes, glyphs, writing, inscriptions, signs, banners, or symbols of any kind anywhere in the image. The image must be borderless and frameless — no decorative borders, frames, ornamental edges, vignettes, or margin decorations of any kind.`;

// Visual descriptions for each NPC (not in JSON to keep world data clean)
const NPC_VISUALS = {
  maren: 'A middle-aged woman behind a rough wooden bar counter. Sturdy build, practical linen apron over dark clothing. Arms crossed. A single lantern casts warm light on her weathered hands. Smoky inn interior behind her.',
  edric: 'A lone figure standing at a muddy crossroads in mist. Worn chainmail under a threadbare traveling cloak. A sheathed sword at his side, its pommel dull. He leans on a walking staff, head slightly bowed. Melancholic silhouette against fog.',
  sibyl: 'An older woman in layered robes tending dried herbs hanging from ceiling beams inside a mill. Thin hands sorting plant cuttings. A mortar and pestle on the table beside her. Shafts of dim light through cracks in the wall.',
};

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
      size: '1024x1024',
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
  const npcs = world.npcs || [];

  console.log(`Generating portraits for ${npcs.length} NPCs...\n`);

  for (const npc of npcs) {
    const filename = `npc-${npc.id}.png`;
    const outPath = path.join(OUT_DIR, filename);

    // Skip if already exists
    if (fs.existsSync(outPath)) {
      console.log(`  [skip] ${filename} already exists`);
      continue;
    }

    const visual = NPC_VISUALS[npc.id] || `A ${npc.archetype} in a dark medieval-fantasy setting.`;
    const prompt = `${STYLE}\n\nCharacter portrait: ${visual}`;
    console.log(`  [gen]  ${filename} — "${npc.name}"`);

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
