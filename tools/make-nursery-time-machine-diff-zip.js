#!/usr/bin/env node
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { dirname, join, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { makeZip, readZip } from './zip.js';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const OUTPUT = join(ROOT, 'dist', 'tiny-mind-nursery-time-machine-source-diff.zip');
const files = [
  'NURSERY_TIME_MACHINE_TEST_REPORT.md',
  'arc/arcs.json',
  'corpus/binding/stories.json',
  'nursery/guide.js',
  'nursery/index.html',
  'src/scrubber.js',
  'tests/e10-scrubber.test.js',
  'tests/e9-mega.test.js',
  'tests/smoke-guided.mjs',
  'tools/build-release.js',
  'tools/make-nursery-time-machine-diff-zip.js',
];

const entries = await Promise.all(files.map(async rel => ({
  name: relative(ROOT, join(ROOT, rel)).split(sep).join('/'),
  data: new Uint8Array(await readFile(join(ROOT, rel))),
})));
const bytes = makeZip(entries);
await mkdir(dirname(OUTPUT), { recursive: true });
await writeFile(OUTPUT, bytes);
const checked = readZip(bytes);
if (checked.length !== entries.length) {
  throw new Error(`ZIP verification failed: expected ${entries.length} entries, got ${checked.length}`);
}
console.log(`${relative(ROOT, OUTPUT)}: ${entries.length} files, ${(bytes.length / 1024).toFixed(1)} KB`);
