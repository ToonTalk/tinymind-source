// e5-sweep.js — E5 (gate G2b): Zog absorption + replay-ratio sweep on stories260K.
// Phases: (1) coarse lr probe at ratio 25%; (2) full runs at replay {0%, 25%, 50%}.
// Metrics per eval: held-out-Zog loss, held-out-TinyStories loss (forgetting), and at
// run end a greedy "Once upon a time, Zog" continuation. Writes e5-results.json and
// models/stories260K-zog.bin (the 25%-replay fine-tune, for E6 file-drop/storage tests).
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint, writeCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy } from '../src/model.js';
import { Trainer, rng, packStories, sampleBatch } from '../src/train.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const B = 8, T = 192;
const EVAL_EVERY = 20, MAX_STEPS = 600, PROBE_STEPS = 60;

const ck = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K.bin')));
const tok = new Tokenizer(await readFile(join(ROOT, 'models', 'tok512.bin')), ck.config.vocabSize);

// --- data ---------------------------------------------------------------------
const zogDir = join(ROOT, 'corpus', 'zog');
const files = (await readdir(zogDir)).sort();
async function loadStories(prefix) {
  const out = [];
  for (const f of files.filter(f => f.startsWith(prefix) && f.endsWith('.txt'))) {
    out.push(await readFile(join(zogDir, f), 'utf8'));
  }
  return out;
}
const zogTrain = await loadStories('train-');
const zogHeld = await loadStories('heldout-');
const replayAll = JSON.parse(await readFile(join(ROOT, 'corpus', 'replay', 'replay.json'), 'utf8'));
const replayTrainStories = replayAll.slice(0, 150);
const baseEvalStories = replayAll.slice(200, 260);

const enc = s => tok.encode(s.trim(), false, false);
const zogStream = packStories(zogTrain.map(enc));
const zogHeldStream = packStories(zogHeld.map(enc));
const replayStream = packStories(replayTrainStories.map(enc));
const baseEvalStream = packStories(baseEvalStories.map(enc));
const lens = zogTrain.map(s => enc(s).length);
console.log(`zog train: ${zogTrain.length} stories, ${Math.min(...lens)}-${Math.max(...lens)} tokens each (mean ${(lens.reduce((a, b) => a + b) / lens.length).toFixed(0)})`);
console.log(`streams: zog ${zogStream.length}, zogHeld ${zogHeldStream.length}, replay ${replayStream.length}, baseEval ${baseEvalStream.length} tokens`);

// deterministic evenly-spaced windows for eval
function makeEvalBatch(stream, nRows = B) {
  const batch = new Int32Array(nRows * (T + 1));
  const span = stream.length - (T + 1);
  for (let b = 0; b < nRows; b++) {
    const start = Math.floor((span * b) / Math.max(1, nRows - 1));
    batch.set(stream.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}
const zogEvalBatch = makeEvalBatch(zogHeldStream);
const baseEvalBatch = makeEvalBatch(baseEvalStream);

function sampleMixedBatch(zog, replay, ratio, rand) {
  const batch = new Int32Array(B * (T + 1));
  for (let b = 0; b < B; b++) {
    const src = rand() < ratio ? replay : zog;
    const start = Math.floor(rand() * (src.length - T - 1));
    batch.set(src.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

function attributeHits(text) {
  const t = text.toLowerCase();
  return {
    smallDragon: t.includes('small dragon') || t.includes('dragon'),
    clouds: t.includes('cloud'),
    moon: t.includes('moon'),
  };
}

function runConfig({ label, lr, ratio, steps, seed }) {
  const tr = new Trainer(ck, { B, T, lr });
  const rand = rng(seed);
  const zog0 = tr.evalLoss(zogEvalBatch);
  const base0 = tr.evalLoss(baseEvalBatch);
  const history = [{ step: 0, zog: zog0, base: base0 }];
  console.log(`\n--- ${label}: lr=${lr} replay=${ratio * 100}% steps=${steps}`);
  console.log(`step 0: zogHeld ${zog0.toFixed(4)}  base ${base0.toFixed(4)}`);
  let trainLoss = null, msSum = 0;
  for (let s = 1; s <= steps; s++) {
    const batch = sampleMixedBatch(zogStream, replayStream, ratio, rand);
    const r = tr.step(batch);
    trainLoss = r.loss; msSum += r.ms;
    if (s % EVAL_EVERY === 0 || s === steps) {
      const zl = tr.evalLoss(zogEvalBatch);
      const bl = tr.evalLoss(baseEvalBatch);
      history.push({ step: s, zog: zl, base: bl });
      console.log(`step ${s}: train ${trainLoss.toFixed(4)}  zogHeld ${zl.toFixed(4)}  base ${bl.toFixed(4)}  (${(msSum / s).toFixed(0)} ms/step)`);
    }
  }
  // absorption: first eval step where held-out-Zog loss <= the model's ORIGINAL loss
  // on ordinary TinyStories text (i.e. Zog stories became as predictable as normal ones)
  const absorbedAt = history.find(h => h.zog <= base0)?.step ?? null;
  const model = createModel({ config: ck.config, weights: tr.modelWeights() });
  const sample1 = generateGreedy(model, tok, 'Once upon a time, Zog', 120).text;
  const sample2 = generateGreedy(model, tok, 'Zog was very hungry. He wanted to eat', 120).text;
  console.log(`absorbedAt: ${absorbedAt}  |  sample: ${sample1.slice(0, 160).replace(/\n/g, ' ')}`);
  return {
    label, lr, ratio, steps,
    msPerStep: msSum / steps,
    history, absorbedAt,
    finalZog: history[history.length - 1].zog,
    finalBase: history[history.length - 1].base,
    base0, zog0,
    samples: [sample1, sample2],
    attributes: attributeHits(sample1 + ' ' + sample2),
    trainer: tr,
  };
}

// --- phase 1: coarse lr probe ---------------------------------------------------
const probes = [];
for (const lr of [3e-4, 1e-4, 3e-5]) {
  const r = runConfig({ label: `probe lr=${lr}`, lr, ratio: 0.25, steps: PROBE_STEPS, seed: 42 });
  delete r.trainer;
  probes.push(r);
}
// pick: fastest Zog-loss drop whose base loss hasn't blown up (> +0.2 nats)
const ok = probes.filter(p => p.finalBase - p.base0 < 0.2);
const chosen = (ok.length ? ok : probes).reduce((a, b) => (a.finalZog < b.finalZog ? a : b));
console.log(`\n=== chosen lr: ${chosen.lr} (probe zogHeld ${chosen.finalZog.toFixed(3)}, base drift ${(chosen.finalBase - chosen.base0).toFixed(3)})`);

// --- phase 2: replay sweep ------------------------------------------------------
const runs = [];
let saved = null;
for (const ratio of [0, 0.25, 0.5]) {
  const r = runConfig({ label: `replay ${ratio * 100}%`, lr: chosen.lr, ratio, steps: MAX_STEPS, seed: 7 });
  if (ratio === 0.25) saved = r.trainer;
  delete r.trainer;
  runs.push(r);
}

if (saved) {
  const bin = writeCheckpoint(ck.config, saved.modelWeights());
  await writeFile(join(ROOT, 'models', 'stories260K-zog.bin'), Buffer.from(bin));
  console.log(`\nsaved fine-tuned checkpoint: models/stories260K-zog.bin (${bin.byteLength} bytes)`);
}

await writeFile(join(ROOT, 'e5-results.json'), JSON.stringify({
  config: { B, T, evalEvery: EVAL_EVERY, maxSteps: MAX_STEPS, probeSteps: PROBE_STEPS },
  probes, chosenLr: chosen.lr, runs,
}, null, 1));
console.log('wrote e5-results.json — DONE');
