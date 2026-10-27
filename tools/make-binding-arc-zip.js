// Package the balanced binding arc and the analysis files.
// Usage: node tools/make-binding-arc-zip.js [arc/binding]
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeZip } from './zip.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const arcRel = process.argv[2] ?? 'arc/binding';
const arcDir = join(ROOT, arcRel);
const dist = join(ROOT, 'dist');
await mkdir(dist, { recursive: true });
const enc = new TextEncoder();

const manifest = JSON.parse(await readFile(join(arcDir, 'arc-manifest.json'), 'utf8'));
const arcEntries = [{ name: 'arc-manifest.json', data: enc.encode(JSON.stringify(manifest, null, 1)) }];
for (const s of manifest.snapshots) {
  arcEntries.push({ name: s.file, data: new Uint8Array(await readFile(join(arcDir, s.file))) });
}
const arcZip = makeZip(arcEntries);
await writeFile(join(dist, 'binding-arc.zip'), arcZip);
console.log(`dist/binding-arc.zip: ${manifest.snapshots.length} snapshots, ${(arcZip.length / 1e6).toFixed(1)} MB`);

const resultFiles = [
  [join(arcDir, 'arc-manifest.json'), 'arc-manifest.json'],
  [join(arcDir, 'binding-evaluation.json'), 'binding-evaluation.json'],
  [join(arcDir, 'binding-evaluation.md'), 'binding-evaluation.md'],
  [join(arcDir, 'binding-profiles.json'), 'binding-profiles.json'],
  [join(arcDir, 'recommended-checkpoint.json'), 'recommended-checkpoint.json'],
  [join(ROOT, 'corpus', 'binding', 'stories.json'), 'binding-stories.json'],
  [join(ROOT, 'corpus', 'binding', 'challenges.json'), 'binding-challenges.json'],
  [join(ROOT, 'models', 'stories260K-binding.bin'), 'stories260K-binding.bin'],
  [join(ROOT, 'models', 'stories260K-binding-max.bin'), 'stories260K-binding-max.bin'],
  [join(ROOT, 'models', 'tok512.bin'), 'tok512.bin'],
];
const results = [];
for (const [path, name] of resultFiles) results.push({ name, data: new Uint8Array(await readFile(path)) });
const resultsZip = makeZip(results);
await writeFile(join(dist, 'binding-results.zip'), resultsZip);
console.log(`dist/binding-results.zip: ${(resultsZip.length / 1e6).toFixed(1)} MB`);

const modelPrefix = 'tiny-mind-binding-models-corrected/';
const modelReadme = `# Corrected binding models

- stories260K-binding.bin: step 400, the default binding-demonstration and investigation model.
- stories260K-binding-max.bin: step 600, maximum measured binding with greater ordinary-story drift and rigidity.
- tok512.bin: matching tokenizer.

Step 60 is the held-out story-loss minimum, not the shipped binding model. See binding-profiles.json and binding-evaluation.json for the three educational checkpoint roles.
`;
const modelEntries = [
  { name: modelPrefix + 'MODEL_README.md', data: enc.encode(modelReadme) },
  { name: modelPrefix + 'binding-evaluation.json', data: new Uint8Array(await readFile(join(arcDir, 'binding-evaluation.json'))) },
  { name: modelPrefix + 'binding-profiles.json', data: new Uint8Array(await readFile(join(arcDir, 'binding-profiles.json'))) },
  { name: modelPrefix + 'stories260K-binding.bin', data: new Uint8Array(await readFile(join(ROOT, 'models', 'stories260K-binding.bin'))) },
  { name: modelPrefix + 'stories260K-binding-max.bin', data: new Uint8Array(await readFile(join(ROOT, 'models', 'stories260K-binding-max.bin'))) },
  { name: modelPrefix + 'tok512.bin', data: new Uint8Array(await readFile(join(ROOT, 'models', 'tok512.bin'))) },
];
const modelsZip = makeZip(modelEntries);
await writeFile(join(dist, 'tiny-mind-binding-models-corrected.zip'), modelsZip);
console.log(`dist/tiny-mind-binding-models-corrected.zip: ${(modelsZip.length / 1e6).toFixed(1)} MB`);
