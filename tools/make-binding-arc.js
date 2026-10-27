// Train a balanced character-binding arc from stories260K.
// Additive alternative to tools/make-arc.js; it does not replace the Zog arc.
//
// Usage from repo root:
//   node tools/gen-binding-corpus.js
//   node tools/make-binding-arc.js
//   node tools/make-binding-arc.js --resume
//   node tools/make-binding-arc.js --max-step=120 --out=arc/binding-probe
//
// Snapshot/resume follows the same checkpoint + Adam + state + manifest-last contract
// as Tiny Mind's existing make-arc.js.
import { readFile, writeFile, readdir, mkdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint, writeCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy } from '../src/model.js';
import { Trainer, rng, packStories } from '../src/train.js';
import { evaluateBindingModel, compactBindingSummary } from './binding-eval-lib.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const DEFAULTS = {
  file: 'arcC', outDir: join(ROOT, 'arc', 'binding'),
  B: 8, T: 192, lr: 3e-4, replayRatio: 0.5, seed: 17,
  snapshots: [0, 10, 20, 40, 60, 80, 120, 160, 240, 400, 600],
};

function parseArgs(argv) {
  const flags = { resume: false, out: null, maxStep: null, lr: null, replayRatio: null };
  for (const a of argv) {
    if (a === '--resume') flags.resume = true;
    else if (a.startsWith('--out=')) flags.out = a.slice(6);
    else if (a.startsWith('--max-step=')) flags.maxStep = Number(a.slice(11));
    else if (a.startsWith('--lr=')) flags.lr = Number(a.slice(5));
    else if (a.startsWith('--replay=')) flags.replayRatio = Number(a.slice(9));
    else throw new Error(`unknown flag ${a}`);
  }
  return flags;
}

function makeEvalBatch(stream, B, T) {
  if (stream.length < T + 1) throw new Error(`eval stream has ${stream.length} tokens; need at least ${T + 1}`);
  const batch = new Int32Array(B * (T + 1));
  const span = stream.length - (T + 1);
  for (let b = 0; b < B; b++) {
    const start = Math.floor((span * b) / Math.max(1, B - 1));
    batch.set(stream.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

// Always consumes exactly two RNG values per row, preserving byte-exact resume.
function nextMixedBatch(bindingStream, replayStream, replayRatio, B, T, rand) {
  const batch = new Int32Array(B * (T + 1));
  for (let b = 0; b < B; b++) {
    const useReplay = rand() < replayRatio;
    const src = useReplay ? replayStream : bindingStream;
    const start = Math.floor(rand() * (src.length - T - 1));
    batch.set(src.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

const pad = n => String(n).padStart(5, '0');
const ADAM_KEYS = ['tokenEmbedding', 'rmsAtt', 'wq', 'wk', 'wv', 'wo', 'rmsFfn', 'w1', 'w2', 'w3', 'rmsFinal'];
function momentsToBuffer(tr) {
  let n = 0;
  for (const k of ADAM_KEYS) n += tr.m[k].length;
  const out = new Float32Array(2 * n);
  let p = 0;
  for (const k of ADAM_KEYS) { out.set(tr.m[k], p); p += tr.m[k].length; }
  for (const k of ADAM_KEYS) { out.set(tr.v[k], p); p += tr.v[k].length; }
  return Buffer.from(out.buffer);
}
function momentsFromBuffer(tr, buf) {
  const values = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  let p = 0;
  for (const k of ADAM_KEYS) { tr.m[k].set(values.subarray(p, p + tr.m[k].length)); p += tr.m[k].length; }
  for (const k of ADAM_KEYS) { tr.v[k].set(values.subarray(p, p + tr.v[k].length)); p += tr.v[k].length; }
  if (p !== values.length) throw new Error('Adam moment file has the wrong size');
}
function trimSample(text) {
  const m = text.match(/^[\s\S]*[.!?]["”']?(?=\s|$)/);
  return m ? m[0] : text;
}

const flags = parseArgs(process.argv.slice(2));
const cfg = { ...DEFAULTS };
cfg.outDir = flags.out ? join(ROOT, flags.out) : cfg.outDir;
cfg.lr = flags.lr ?? cfg.lr;
cfg.replayRatio = flags.replayRatio ?? cfg.replayRatio;
if (!(cfg.replayRatio >= 0 && cfg.replayRatio < 1)) throw new Error('--replay must be between 0 and 1');
let schedule = [...cfg.snapshots];
if (flags.maxStep !== null) schedule = schedule.filter(s => s <= flags.maxStep);
if (!schedule.length) throw new Error('snapshot schedule is empty');
await mkdir(cfg.outDir, { recursive: true });

const base = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K.bin')));
const tok = new Tokenizer(await readFile(join(ROOT, 'models', 'tok512.bin')), base.config.vocabSize);
const enc = text => tok.encode(text.trim(), false, false);
const corpus = JSON.parse(await readFile(join(ROOT, 'corpus', 'binding', 'stories.json'), 'utf8'));
const challenges = JSON.parse(await readFile(join(ROOT, 'corpus', 'binding', 'challenges.json'), 'utf8'));
const trainStories = corpus.stories.filter(s => s.split === 'train').map(s => s.text);
const heldStories = corpus.stories.filter(s => s.split === 'heldout').map(s => s.text);
const replay = JSON.parse(await readFile(join(ROOT, 'corpus', 'replay', 'replay.json'), 'utf8'));

const bindingStream = packStories(trainStories.map(enc));
const heldStream = packStories(heldStories.map(enc));
const replayStream = packStories(replay.slice(0, 150).map(enc));
const baseEvalStream = packStories(replay.slice(200, 260).map(enc));
const heldEval = makeEvalBatch(heldStream, cfg.B, cfg.T);
const baseEval = makeEvalBatch(baseEvalStream, cfg.B, cfg.T);
console.log(`binding data: ${trainStories.length} train / ${heldStories.length} heldout; ${bindingStream.length} train tokens`);
console.log(`recipe: B=${cfg.B} T=${cfg.T} lr=${cfg.lr} replay=${cfg.replayRatio} seed=${cfg.seed}`);

const rand = rng(cfg.seed);
let trainer = new Trainer(base, { B: cfg.B, T: cfg.T, lr: cfg.lr });
let step = 0;
let recent = [];
let manifest = {
  arc: cfg.file,
  experiment: 'balanced-character-binding',
  note: 'Fine-tune stories260K on four equally frequent characters with contrasting kinds, foods, and homes. Continued through 600 steps to expose learning, specificity, forgetting, and possible overfitting.',
  config: { B: cfg.B, T: cfg.T, lr: cfg.lr, replayRatio: cfg.replayRatio, seed: cfg.seed, start: 'models/stories260K.bin' },
  characters: challenges.characters,
  snapshots: [],
};

if (flags.resume) {
  // arc-manifest.json is the commit marker. Ignore any checkpoint/state files written
  // after its last entry: an interruption may have happened in the middle of snapshot().
  manifest = JSON.parse(await readFile(join(cfg.outDir, 'arc-manifest.json'), 'utf8'));
  const committed = manifest.snapshots.at(-1)?.step;
  if (!Number.isInteger(committed)) throw new Error(`--resume: manifest has no committed snapshot in ${cfg.outDir}`);
  const st = JSON.parse(await readFile(join(cfg.outDir, `state-${pad(committed)}.json`), 'utf8'));
  const ck = readCheckpoint(await readFile(join(cfg.outDir, `${cfg.file}-step${pad(committed)}.bin`)));
  trainer = new Trainer(ck, { B: cfg.B, T: cfg.T, lr: st.lr ?? cfg.lr });
  momentsFromBuffer(trainer, await readFile(join(cfg.outDir, `adam-${pad(committed)}.bin`)));
  trainer.stepCount = st.stepCount;
  rand.setState(st.rngState);
  step = committed;
  recent = st.recentLosses ?? [];
  manifest.snapshots = manifest.snapshots.filter(s => s.step <= step);
  const stateFiles = (await readdir(cfg.outDir)).filter(f => /^state-\d{5}\.json$/.test(f)).sort();
  const newestState = stateFiles.length ? Number(stateFiles.at(-1).match(/\d{5}/)[0]) : committed;
  if (newestState > committed) console.log(`ignoring uncommitted partial snapshot files at step ${newestState}`);
  console.log(`resumed from committed step ${step}; RNG state ${st.rngState}`);
}

async function snapshot() {
  const file = `${cfg.file}-step${pad(step)}.bin`;
  const checkpoint = { config: base.config, weights: trainer.modelWeights() };
  await writeFile(join(cfg.outDir, file), Buffer.from(writeCheckpoint(base.config, checkpoint.weights)));
  await writeFile(join(cfg.outDir, `adam-${pad(step)}.bin`), momentsToBuffer(trainer));
  await writeFile(join(cfg.outDir, `state-${pad(step)}.json`), JSON.stringify({
    arc: cfg.file, step, stepCount: trainer.stepCount, rngState: rand.getState(),
    recentLosses: recent.slice(-20), lr: cfg.lr, replayRatio: cfg.replayRatio,
  }));

  const fastBinding = compactBindingSummary(evaluateBindingModel(checkpoint, tok, challenges, { mode: 'fast', samples: false }));
  const entry = {
    step, file,
    trainLoss: recent.length ? +(recent.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, recent.length)).toFixed(4) : null,
    bindingHeld: +trainer.evalLoss(heldEval).toFixed(4),
    baseLoss: +trainer.evalLoss(baseEval).toFixed(4),
    binding: fastBinding,
    samples: {},
  };
  for (const name of Object.keys(challenges.characters)) {
    const model = createModel(checkpoint);
    entry.samples[`Once upon a time, ${name}`] = trimSample(generateGreedy(model, tok, `Once upon a time, ${name}`, 140).text);
  }
  manifest.snapshots.push(entry);
  await writeFile(join(cfg.outDir, 'arc-manifest.json'), JSON.stringify(manifest, null, 1)); // commit marker, always last
  console.log(`snapshot @${step}: train ${entry.trainLoss ?? '—'} held ${entry.bindingHeld} base ${entry.baseLoss}` +
    ` direct ${entry.binding.directAccuracy} specific ${entry.binding.specificityAccuracy}`);
}

if (!flags.resume && schedule.includes(0)) await snapshot();
const target = schedule.at(-1);
let msSum = 0, msN = 0;
while (step < target) {
  const batch = nextMixedBatch(bindingStream, replayStream, cfg.replayRatio, cfg.B, cfg.T, rand);
  const r = trainer.step(batch);
  step++;
  recent.push(r.loss);
  if (recent.length > 50) recent.shift();
  msSum += r.ms; msN++;
  if (step % 10 === 0) {
    console.log(`step ${step}/${target}: loss ${r.loss.toFixed(4)} (${(msSum / msN).toFixed(0)} ms/step)`);
    msSum = 0; msN = 0;
  }
  if (schedule.includes(step)) await snapshot();
}
console.log(`binding arc complete through step ${step}: ${manifest.snapshots.length} snapshots in ${cfg.outDir}`);
console.log('next: node tools/evaluate-binding.js --arc=arc/binding');
