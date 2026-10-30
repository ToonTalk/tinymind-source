// ship.mjs — copy the Tiny Eye from the Phase 0 export into the repo, ready for the room:
//   eye/tiny-eye.json         the manifest (tiny-eye-v1): config, classes, tensors, the sample
//                             set's layout, and the names Phase 0's labeler gave 64 last-stage
//                             neurons (with whether each name passed the distractor test)
//   eye/tiny-eye.bin          eye80k's weights, float32 (331 KB)
//   eye/tiny-eye-sample.bin   5,000 held-out doodles, 28x28 uint8 (ink 255), sorted by class
// The training snapshots stay out until the growth scrubber (Phase 2) needs them.
//   node tools/eye/ship.mjs [data folder]
import { readFileSync, writeFileSync, copyFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(dirname(fileURLToPath(import.meta.url))));
const DATA = process.argv[2] || 'C:/Users/toont/tm-data/eye';
const EXPORT = join(DATA, 'export');
const src = JSON.parse(readFileSync(join(EXPORT, 'tiny-eye.json'), 'utf8'));
const sample = JSON.parse(readFileSync(join(EXPORT, 'sample.json'), 'utf8'));
const labels = JSON.parse(readFileSync(join(DATA, 'labels', src.run + '.json'), 'utf8'));

if (sample.labels.some((c, i) => i && c < sample.labels[i - 1])) throw new Error('the sample set must be sorted by class');
const classCounts = src.classes.map((_, c) => sample.labels.filter(l => l === c).length);
const names = {};
for (const [j, n] of Object.entries(labels.neurons)) {
  const pole = p => p && p.label ? { name: p.label, hits: p.hits, passed: !!p.passed } : null;
  names[j] = { most: pole(n.most), least: pole(n.least) };
}
const { snapshots, snapshot_dtype, ...rest } = src;
const manifest = {
  format: 'tiny-eye-v1',
  ...rest,
  weights: 'tiny-eye.bin',
  sample: { file: 'tiny-eye-sample.bin', count: sample.count, size: 28, dtype: 'uint8, ink = 255', sortedByClass: true, classCounts,
    note: 'held-out doodles the eye never trained on, about 208 a category' },
  names: { stage: labels.stage, model: labels.model, how: 'Phase 0 (reports/eye/feasibility.md): a vision model named each pole from its top 16 held-out doodles, then from the name alone tried to pick 4 of that pole\'s doodles out of 16; "passed" = 3 or 4 right', neurons: names },
};
writeFileSync(join(ROOT, 'eye', 'tiny-eye.json'), JSON.stringify(manifest, null, 1) + '\n');
copyFileSync(join(EXPORT, 'tiny-eye.bin'), join(ROOT, 'eye', 'tiny-eye.bin'));
copyFileSync(join(EXPORT, 'sample.bin'), join(ROOT, 'eye', 'tiny-eye-sample.bin'));
console.log(`eye/: tiny-eye.json (${Object.keys(names).length} named neurons), tiny-eye.bin, tiny-eye-sample.bin (${sample.count} doodles)`);
