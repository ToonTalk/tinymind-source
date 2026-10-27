#!/usr/bin/env node
import { mkdir, readFile, readdir, writeFile } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeZip, readZip } from './zip.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = join(ROOT, 'dist', 'tiny-mind-binding-app-improvements.zip');
const files = [
  'BINDING_IMPROVEMENT_TEST_REPORT.md',
  'src/scrubber.js',
  'src/hunt-evidence.js',
  'nursery/index.html',
  'nursery/guide.js',
  'hunt/index.html',
  'hunt/worker.js',
  'hunt/guide.js',
  'shell/index.html',
  'tools/build-release.js',
  'tools/binding-eval-lib.js',
  'tools/evaluate-binding.js',
  'tools/select-binding-profiles.js',
  'tools/gen-binding-corpus.js',
  'tools/make-binding-arc.js',
  'tools/make-binding-arc-zip.js',
  'tools/run-binding-experiment.js',
  'tools/make-binding-improvement-zip.js',
  'tests/e10-scrubber.test.js',
  'tests/e11-binding-insights.test.js',
  'tests/smoke-guided.mjs',
  'arc/binding/arc-manifest.json',
  'arc/binding/binding-evaluation.md',
  'arc/binding/binding-profiles.json',
  'arc/binding/recommended-checkpoint.json',
  'dist/tiny-mind-mega.html',
];

async function collectFiles(dir) {
  const out = [];
  for (const entry of await readdir(join(ROOT, dir), { withFileTypes: true })) {
    const rel = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...await collectFiles(rel));
    else out.push(rel);
  }
  return out;
}

files.push(...await collectFiles('corpus/binding'));
const entries = await Promise.all([...new Set(files)].map(async rel => ({
  name: relative(ROOT, join(ROOT, rel)).split(sep).join('/'),
  data: new Uint8Array(await readFile(join(ROOT, rel))),
})));
const bytes = makeZip(entries);
await mkdir(dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, bytes);
const checked = readZip(bytes);
if (checked.length !== entries.length) throw new Error(`ZIP verification failed: expected ${entries.length} entries, got ${checked.length}`);
console.log(`${relative(ROOT, OUTPUT)}: ${entries.length} files, ${(bytes.length / 1024 / 1024).toFixed(1)} MB`);
