// E15 — training across cores (TrainPool in src/train.js). A step's gradient is the sum
// of each sequence's gradient, in sequence order, so a trainer that hands its sequences
// to pool workers must end up with EXACTLY the same weights as one that works alone —
// for any number of workers. The workers run trainPoolSource() in-process with
// structuredClone messages, so the real worker code and message shapes are tested.
import { readFile } from 'node:fs/promises';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { Trainer, TrainPool, trainPoolSource, rng, packStories, sampleBatch } from '../src/train.js';
import { model260K, tok512, tinyStoriesValid } from './paths.js';

function fakeWorker(src) {
  const inner = { onmessage: null, postMessage(d) { setTimeout(() => outer.onmessage && outer.onmessage({ data: structuredClone(d) }), 0); } };
  const outer = { onmessage: null, onerror: null, terminate() {},
    postMessage(d) { setTimeout(() => inner.onmessage({ data: structuredClone(d) }), 0); } };
  (new Function('self', 'performance', src))(inner, performance);
  return outer;
}
const sameParams = (a, b) => Object.keys(a.params).every(k => a.params[k].every((v, i) => Object.is(v, b.params[k][i])));

export async function run(t) {
  const ck = readCheckpoint(await readFile(model260K));
  const tok = new Tokenizer(await readFile(tok512), ck.config.vocabSize);
  const stories = (await readFile(tinyStoriesValid, 'utf8')).split('<|endoftext|>').slice(0, 30).map(s => s.trim()).filter(Boolean);
  const stream = packStories(stories.map(s => tok.encode(s, false, false)));
  const B = 6, T = 48, steps = 3;
  const r = rng(5);
  const batches = Array.from({ length: steps }, () => sampleBatch(stream, B, T, r));
  const evalBatch = sampleBatch(stream, B, T, rng(9));

  const alone = new Trainer(ck, { B, T, lr: 1e-3 });
  const lossesAlone = batches.map(b => alone.step(b).loss);
  const evalAlone = alone.evalLoss(evalBatch);

  for (const K of [1, 2, 4]) {
    const tr = new Trainer(ck, { B, T, lr: 1e-3 });
    const pool = await TrainPool.create(tr, { K, spawn: () => fakeWorker(trainPoolSource()) });
    t.check(`train pool of ${K} starts`, !!pool);
    const losses = [];
    for (const b of batches) losses.push((await tr.stepWith(pool, b)).loss);
    t.check(`pool of ${K}: same losses and the same weights, bit for bit, after ${steps} steps`,
      losses.every((l, i) => Object.is(l, lossesAlone[i])) && sameParams(tr, alone));
    t.check(`pool of ${K}: held-out loss identical`, Object.is(await tr.evalLossWith(pool, evalBatch), evalAlone));
    pool.dispose();
  }
  // the per-sequence sum is still the gradient of the batch: compare with the whole-batch backward
  const whole = new Trainer(ck, { B, T }), seq = new Trainer(ck, { B, T });
  whole.zeroGrads(); whole.forwardLoss(batches[0], true); whole.backward(batches[0]);
  seq.zeroGrads();
  for (let b = 0; b < B; b++) { Trainer.seqGrad(seq.seqTrainer(), batches[0], b, B); seq.addGrads(seq.seqTrainer().grads); }
  let diff = 0, norm = 0;
  for (const k of Object.keys(whole.grads)) for (let i = 0; i < whole.grads[k].length; i++) {
    const a = whole.grads[k][i], c = seq.grads[k][i];
    diff += (a - c) * (a - c); norm += a * a;
  }
  const rel = Math.sqrt(diff / norm);
  t.check(`per-sequence gradient sum = whole-batch gradient up to rounding (|diff| / |grad| = ${rel.toExponential(1)})`, rel < 1e-5);
}
