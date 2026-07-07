// make-arc-zips.js — package the arc packs + starter pack for the artifact edition
// (flat zips, matching the contract in tiny-mind-arc-kickoff.md and Ken's existing
// model zips). Skips any arc whose manifest doesn't exist yet — Arc A can be zipped
// the morning after. Usage: node tools/make-arc-zips.js
import { readFile, writeFile, mkdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeZip } from './zip.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
await mkdir(join(ROOT, 'dist'), { recursive: true });

async function packArc(dir, zipName) {
  let manifest;
  try {
    manifest = JSON.parse(await readFile(join(ROOT, dir, 'arc-manifest.json'), 'utf8'));
  } catch {
    console.log(`${zipName}: no manifest in ${dir}/ yet — skipped`);
    return;
  }
  const entries = [{ name: 'arc-manifest.json',
    data: new TextEncoder().encode(JSON.stringify(manifest, null, 1)) }];
  for (const s of manifest.snapshots) {
    entries.push({ name: s.file, data: new Uint8Array(await readFile(join(ROOT, dir, s.file))) });
  }
  const zip = makeZip(entries);
  await writeFile(join(ROOT, 'dist', zipName), zip);
  console.log(`dist/${zipName}: ${manifest.snapshots.length} snapshots, ${(zip.length / 1e6).toFixed(1)} MB`);
}

await packArc('arc', 'zog-arc.zip');
await packArc('arc/newborn', 'newborn-arc.zip');

// starter pack: trainee + tok512 + replay ONLY (~1.3 MB). The 15M specimen + 32K
// tokenizer stay a separate optional download, never on the golden path.
const starter = makeZip([
  { name: 'stories260K.bin', data: new Uint8Array(await readFile(join(ROOT, 'models', 'stories260K.bin'))) },
  { name: 'tok512.bin', data: new Uint8Array(await readFile(join(ROOT, 'models', 'tok512.bin'))) },
  { name: 'replay.json', data: new Uint8Array(await readFile(join(ROOT, 'corpus', 'replay', 'replay.json'))) },
]);
await writeFile(join(ROOT, 'dist', 'tiny-mind-starter.zip'), starter);
console.log(`dist/tiny-mind-starter.zip: ${(starter.length / 1e6).toFixed(2)} MB`);
