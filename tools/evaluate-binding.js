// Full post-training evaluation for the balanced character-binding experiment.
//
// Usage:
//   node tools/evaluate-binding.js --arc=arc/binding
//   node tools/evaluate-binding.js models/stories260K.bin models/another.bin
//
// Arc mode evaluates every snapshot, writes JSON + Markdown, and produces three
// educational profiles. The binding-demonstration checkpoint becomes the default
// model; maximum binding is preserved separately (use --no-copy to suppress copies).
import { readFile, writeFile, mkdir, copyFile } from 'node:fs/promises';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { evaluateBindingModel, compactBindingSummary } from './binding-eval-lib.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const arcArg = args.find(a => a.startsWith('--arc='));
const outArg = args.find(a => a.startsWith('--out='));
const noCopy = args.includes('--no-copy');
const positional = args.filter(a => !a.startsWith('--'));
const challenges = JSON.parse(await readFile(join(ROOT, 'corpus', 'binding', 'challenges.json'), 'utf8'));
const baseCk = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K.bin')));
const tok = new Tokenizer(await readFile(join(ROOT, 'models', 'tok512.bin')), baseCk.config.vocabSize);

function pct(x) { return `${(100 * x).toFixed(1)}%`; }
function n4(x) { return Number.isFinite(x) ? x.toFixed(4) : '—'; }
function average(xs) { return xs.reduce((a, b) => a + b, 0) / Math.max(1, xs.length); }

async function evaluateFile(path, meta = {}) {
  console.log(`evaluating ${path}`);
  const checkpoint = readCheckpoint(await readFile(path));
  const result = evaluateBindingModel(checkpoint, tok, challenges, { mode: 'full', samples: true });
  const samples = Object.values(result.samples);
  const repetitionRate = average(samples.map(s => s.repetitionRate));
  const languageDiagnostics = {
    expectedNameRetentionRate: average(samples.map(s => s.expectedNameRetained ? 1 : 0)),
    wrongCharacterNameIntrusions: samples.reduce((n, s) => n + s.wrongCharacterNameIntrusionCount, 0),
    wrongAttributeIntrusions: samples.reduce((n, s) => n + s.wrongAttributeIntrusionCount, 0),
    malformedNamePrefixes: samples.reduce((n, s) => n + s.malformedNamePrefixCount, 0),
  };
  return {
    ...meta,
    path,
    file: basename(path),
    summary: compactBindingSummary(result),
    repetitionRate,
    languageDiagnostics,
    details: result,
  };
}

function selectProfiles(rows) {
  const nonzero = rows.filter(r => r.step !== 0);
  const pool = nonzero.length ? nonzero : rows;
  const heldLossMinimum = pool.reduce((a, b) => b.bindingHeld < a.bindingHeld ? b : a);
  const ready = pool.filter(r =>
    r.summary.directAccuracy >= 0.75 &&
    r.summary.specificityAccuracy >= 0.90 &&
    r.summary.paraphraseAccuracy >= 0.75 &&
    r.summary.distractorAccuracy >= 0.90);
  const bindingDemonstration = ready.reduce((a, b) => !a || b.step < a.step ? b : a, null) ||
    pool.reduce((a, b) => b.summary.bindingScore > a.summary.bindingScore ? b : a);
  const maximumBinding = pool.reduce((a, b) => b.summary.bindingScore > a.summary.bindingScore ? b : a);
  return { heldLossMinimum, bindingDemonstration, maximumBinding };
}

function profileEntry(row, purpose) {
  return {
    purpose, step: row.step, file: row.file, path: row.path,
    bindingHeld: row.bindingHeld, baseLoss: row.baseLoss, baseLossDelta: row.baseLossDelta,
    repetitionRate: row.repetitionRate, languageDiagnostics: row.languageDiagnostics,
    summary: row.summary,
  };
}

function failureLines(result, section, limit = 12) {
  const rows = result.details[section]?.rows ?? [];
  return rows.filter(r => !r.correct).slice(0, limit).map(r => {
    if (r.candidates) return `- ${r.id}: expected **${r.expected}**, top answer was **${r.candidates[0]?.value}**.`;
    if (r.scores) return `- ${r.id}: target **${r.target}** was more strongly associated with **${r.scores[0]?.name}** than ${r.expectedName}.`;
    return `- ${r.id}`;
  });
}

function makeMarkdown(rows, profiles, modeLabel) {
  const lines = [];
  lines.push('# Tiny Mind balanced character-binding evaluation', '');
  lines.push(`Evaluated ${rows.length} checkpoint(s) from ${modeLabel}.`);
  if (profiles) {
    lines.push(`Binding demonstration: **${profiles.bindingDemonstration.file}** at step ${profiles.bindingDemonstration.step}.`);
    lines.push(`Maximum binding: **${profiles.maximumBinding.file}** at step ${profiles.maximumBinding.step}.`);
    lines.push(`Held-out story-loss minimum: **${profiles.heldLossMinimum.file}** at step ${profiles.heldLossMinimum.step} (not the binding demonstration).`);
  }
  lines.push('');
  lines.push('| step/model | held loss | base loss | Δ base | direct | specificity | paraphrase | distractor | override | leak | repetition | wrong names | wrong attrs | malformed | score |');
  lines.push('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|');
  for (const r of rows) {
    const s = r.summary;
    const q = r.languageDiagnostics;
    lines.push(`| ${r.step ?? r.file} | ${n4(r.bindingHeld)} | ${n4(r.baseLoss)} | ${n4(r.baseLossDelta)} | ${pct(s.directAccuracy)} | ${pct(s.specificityAccuracy)} | ${pct(s.paraphraseAccuracy)} | ${pct(s.distractorAccuracy)} | ${pct(s.overrideAccuracy)} | ${n4(s.meanLeakageRatio)} | ${pct(r.repetitionRate)} | ${q.wrongCharacterNameIntrusions} | ${q.wrongAttributeIntrusions} | ${q.malformedNamePrefixes} | ${n4(s.bindingScore)} |`);
  }
  const recommended = profiles?.bindingDemonstration;
  if (recommended) {
    lines.push('', '## Checkpoint profiles', '');
    lines.push(`- **Held-loss minimum (step ${profiles.heldLossMinimum.step}):** the model predicts held-out stories most easily here, but binding is still weak.`);
    lines.push(`- **Binding demonstration (step ${recommended.step}):** the first checkpoint meeting the strong direct, specificity, paraphrase, and distractor thresholds. This is the default investigation model.`);
    lines.push(`- **Maximum binding (step ${profiles.maximumBinding.step}):** the strongest measured binding, retained as a separate model because ordinary-story drift and rigidity are greater.`);
    lines.push('', '### Remaining direct-answer failures', '');
    lines.push(...(failureLines(recommended, 'direct').length ? failureLines(recommended, 'direct') : ['None.']));
    lines.push('', '### Remaining name-specificity failures', '');
    lines.push(...(failureLines(recommended, 'specificity').length ? failureLines(recommended, 'specificity') : ['None.']));
    lines.push('', '### Generated samples', '');
    for (const [name, sample] of Object.entries(recommended.details.samples)) {
      lines.push(`**${name}:** ${sample.text.replace(/\s+/g, ' ').trim()}`, '');
    }
  }
  lines.push('## Interpretation cautions', '');
  lines.push('- Direct accuracy asks whether the correct attribute beats the other three attributes.');
  lines.push('- Name specificity asks whether a target fact scores highest under its correct name rather than another name.');
  lines.push('- Override accuracy is deliberately difficult: it tests whether immediate context can overcome the learned default.');
  lines.push('- These behavioural tests do not by themselves locate where a fact is stored.');
  lines.push('- Name intrusions, attribute intrusions, malformed prefixes, and repeated four-grams are limited diagnostics, not a complete measure of prose quality.');
  return lines.join('\n');
}

let rows = [];
let outputDir;
let modeLabel;
let arcManifest = null;
let arcManifestPath = null;
if (arcArg) {
  const arcRel = arcArg.slice(6);
  const arcDir = join(ROOT, arcRel);
  arcManifestPath = join(arcDir, 'arc-manifest.json');
  arcManifest = JSON.parse(await readFile(arcManifestPath, 'utf8'));
  outputDir = outArg ? join(ROOT, outArg.slice(6)) : arcDir;
  modeLabel = arcRel;
  for (const s of arcManifest.snapshots) {
    rows.push(await evaluateFile(join(arcDir, s.file), {
      step: s.step,
      bindingHeld: s.bindingHeld,
      baseLoss: s.baseLoss,
      trainLoss: s.trainLoss,
      manifestBinding: s.binding,
    }));
  }
} else {
  const files = positional.length ? positional : ['models/stories260K.bin'];
  outputDir = outArg ? join(ROOT, outArg.slice(6)) : join(ROOT, 'binding-evaluation');
  modeLabel = files.join(', ');
  for (const f of files) rows.push(await evaluateFile(join(ROOT, f)));
}

await mkdir(outputDir, { recursive: true });
const baseLoss0 = rows.find(r => r.step === 0)?.baseLoss;
for (const row of rows) row.baseLossDelta = Number.isFinite(baseLoss0) && Number.isFinite(row.baseLoss) ? row.baseLoss - baseLoss0 : null;
const selected = selectProfiles(rows);
const profiles = {
  note: 'Different educational goals require different checkpoints; there is no single scalar optimum.',
  heldLossMinimum: profileEntry(selected.heldLossMinimum, 'Lowest held-out binding-story loss; binding is still weak.'),
  bindingDemonstration: profileEntry(selected.bindingDemonstration, 'First checkpoint meeting strong binding thresholds; default investigation model.'),
  maximumBinding: profileEntry(selected.maximumBinding, 'Highest measured binding; more ordinary-story degradation and rigidity.'),
};
if (arcManifest) {
  const profileAtStep = new Map([
    [profiles.heldLossMinimum.step, 'held-loss-minimum'],
    [profiles.bindingDemonstration.step, 'binding-demonstration'],
    [profiles.maximumBinding.step, 'maximum-binding'],
  ]);
  for (const snapshot of arcManifest.snapshots) {
    const profile = profileAtStep.get(snapshot.step);
    if (profile) snapshot.profile = profile;
    else delete snapshot.profile;
  }
  await writeFile(arcManifestPath, JSON.stringify(arcManifest, null, 1));
}
const output = {
  generatedAt: new Date().toISOString(),
  mode: modeLabel,
  challengeFile: 'corpus/binding/challenges.json',
  profiles,
  recommendation: profiles.bindingDemonstration,
  checkpoints: rows,
};
await writeFile(join(outputDir, 'binding-evaluation.json'), JSON.stringify(output, null, 2));
await writeFile(join(outputDir, 'binding-evaluation.md'), makeMarkdown(rows, selected, modeLabel));
await writeFile(join(outputDir, 'binding-profiles.json'), JSON.stringify(profiles, null, 2));
await writeFile(join(outputDir, 'recommended-checkpoint.json'), JSON.stringify(profiles.bindingDemonstration, null, 2));
if (!noCopy && arcArg) {
  const demoDest = join(ROOT, 'models', 'stories260K-binding.bin');
  const maxDest = join(ROOT, 'models', 'stories260K-binding-max.bin');
  await copyFile(selected.bindingDemonstration.path, demoDest);
  await copyFile(selected.maximumBinding.path, maxDest);
  console.log(`copied binding demonstration checkpoint to ${demoDest}`);
  console.log(`copied maximum-binding checkpoint to ${maxDest}`);
}
console.log(`wrote ${join(outputDir, 'binding-evaluation.json')}`);
console.log(`wrote ${join(outputDir, 'binding-evaluation.md')}`);
