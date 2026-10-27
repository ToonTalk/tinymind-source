// Select named educational checkpoint profiles from an existing full evaluation.
// Usage: node tools/select-binding-profiles.js [evaluation.json] [profiles.json]
import { readFile, writeFile } from 'node:fs/promises';

const [input = 'arc/binding/binding-evaluation.json', output = 'arc/binding/binding-profiles.json'] = process.argv.slice(2);
const data = JSON.parse(await readFile(input, 'utf8'));
const rows = data.checkpoints || [];
if (!rows.length) throw new Error('evaluation has no checkpoints');
const nonzero = rows.filter(r => r.step !== 0);
const pool = nonzero.length ? nonzero : rows;
const held = pool.reduce((a, b) => b.bindingHeld < a.bindingHeld ? b : a);
const ready = pool.filter(r =>
  r.summary.directAccuracy >= 0.75 && r.summary.specificityAccuracy >= 0.90 &&
  r.summary.paraphraseAccuracy >= 0.75 && r.summary.distractorAccuracy >= 0.90);
const demo = ready.reduce((a, b) => !a || b.step < a.step ? b : a, null) ||
  pool.reduce((a, b) => b.summary.bindingScore > a.summary.bindingScore ? b : a);
const max = pool.reduce((a, b) => b.summary.bindingScore > a.summary.bindingScore ? b : a);
const pick = (row, purpose) => ({
  purpose, step: row.step, file: row.file, path: row.path,
  bindingHeld: row.bindingHeld, baseLoss: row.baseLoss, baseLossDelta: row.baseLossDelta,
  repetitionRate: row.repetitionRate, languageDiagnostics: row.languageDiagnostics,
  summary: row.summary,
});
const result = {
  note: 'Different educational goals require different checkpoints; there is no single scalar optimum.',
  heldLossMinimum: pick(held, 'Lowest held-out binding-story loss; binding is still weak.'),
  bindingDemonstration: pick(demo, 'First checkpoint meeting strong binding thresholds; default investigation model.'),
  maximumBinding: pick(max, 'Highest measured binding; more ordinary-story degradation and rigidity.'),
};
await writeFile(output, JSON.stringify(result, null, 2));
console.log(JSON.stringify(result, null, 2));
