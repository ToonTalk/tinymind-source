// E14 — the fast engine (src/batch.js): layer-by-layer over the whole story, optionally
// split across a worker pool. Everything it makes must be BIT-IDENTICAL to the
// one-piece-at-a-time path (captureRun / captureSteered / continueSteered / logitLens /
// jLensLogits), with no pool and with pools of different sizes. The pool here runs
// poolWorkerSource() in-process with structuredClone messages, so the real worker code
// and the real message shapes are what is tested.
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun, logitLens } from '../src/capture.js';
import { captureSteered, continueSteered } from '../src/steer.js';
import { readJlens, jLensLogits } from '../src/jlens.js';
import { Pool, poolWorkerSource, captureFast, continueFast, lensLogits, topIndices } from '../src/batch.js';
import { MODELS, model15M, model260K, tok32000, tok512 } from './paths.js';

function fakeWorker(src) {
  const inner = { onmessage: null, postMessage(d) { setTimeout(() => outer.onmessage && outer.onmessage({ data: structuredClone(d) }), 0); } };
  const outer = { onmessage: null, onerror: null, terminate() {},
    postMessage(d) { setTimeout(() => inner.onmessage({ data: structuredClone(d) }), 0); } };
  (new Function('self', src))(inner);
  return outer;
}
const fakePool = (model, K) => Pool.create(model, { K, spawn: () => fakeWorker(poolWorkerSource()) });

const sameArr = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false; return true; };
function sameCap(a, b) {
  return sameArr(a.attention, b.attention) && sameArr(a.residAttn, b.residAttn) && sameArr(a.residFfn, b.residFfn) &&
    sameArr(a.ffn, b.ffn) && sameArr(a.logZ, b.logZ) && sameArr(a.nextProb, b.nextProb) &&
    JSON.stringify(a.topK) === JSON.stringify(b.topK);
}

export async function run(t) {
  // topIndices is the stable sort's top k, ties included
  const tie = new Float32Array([1, 5, 3, 5, 2, 5, 0]);
  t.check('topIndices = stable-sort top k (ties keep index order)',
    topIndices(tie, 4).join() === [...tie.keys()].sort((i, j) => tie[j] - tie[i]).slice(0, 4).join());

  // 260K: grouped-query attention (fewer KV heads than heads), so KV slicing is exercised
  const ck = readCheckpoint(await readFile(model260K));
  const tok = new Tokenizer(await readFile(tok512), ck.config.vocabSize);
  const ids = tok.encode('Zog is a small dragon. Zog eats clouds and lives behind the moon.', true, false);
  t.check('260K has fewer KV heads than heads (the slicing case)', ck.config.nKvHeads < ck.config.nHeads);
  const ref = captureRun(createModel(ck), ids, { topK: 12 });
  t.check('260K: batched capture, no pool = captureRun bit for bit', sameCap(await captureFast(createModel(ck), ids, { topK: 12 }), ref));
  for (const K of [2, 3]) {
    const m = createModel(ck), pool = await fakePool(m, K);
    t.check(`260K: pool of ${K} starts`, !!pool);
    t.check(`260K: batched capture, pool of ${K} = captureRun bit for bit`, sameCap(await captureFast(m, ids, { topK: 12, pool }), ref));
    pool.dispose();
  }

  // muted heads, muted neurons and a clamped neuron, against captureSteered
  const m2 = createModel(ck), pool2 = await fakePool(m2, 3);
  const heads = ['1:2', '3:0'], steer = [{ layer: 2, neuron: 10, value: 0 }, { layer: 4, neuron: 7, value: 3.5 }];
  t.check('260K: muted heads = captureRun(ablate) bit for bit',
    sameCap(await captureFast(m2, ids, { topK: 12, heads, pool: pool2 }), captureRun(createModel(ck), ids, { topK: 12, ablate: new Set(heads) })));
  t.check('260K: heads + muted + clamped neurons = captureSteered bit for bit',
    sameCap(await captureFast(m2, ids, { topK: 12, heads, steer, pool: pool2 }), captureSteered(createModel(ck), ids, { topK: 12, heads, steer })));
  const prompt = ids.slice(0, 9);
  t.check('260K: continuation = continueSteered (plain)',
    sameArr(await continueFast(m2, prompt, 25, { pool: pool2 }), continueSteered(createModel(ck), prompt, 25)));
  t.check('260K: continuation = continueSteered (steered)',
    sameArr(await continueFast(m2, prompt, 25, { heads, steer, pool: pool2 }), continueSteered(createModel(ck), prompt, 25, { heads, steer })));
  pool2.dispose();

  // 15M: the Big Storyteller, with its J-lens, whole capture + every lens row
  const ck15 = readCheckpoint(await readFile(model15M));
  const tok15 = new Tokenizer(await readFile(tok32000), ck15.config.vocabSize);
  const ids15 = tok15.encode('Once upon a time, Ben and his dog went to the park. They played with a red ball until the sun went down.', true, false);
  const ref15 = captureRun(createModel(ck15), ids15, { topK: 12 });
  const m15 = createModel(ck15), pool15 = await fakePool(m15, 4);
  const cap15 = await captureFast(m15, ids15, { topK: 12, pool: pool15 });
  t.check('15M: batched capture, pool of 4 = captureRun bit for bit', sameCap(cap15, ref15));
  const jl = readJlens(await readFile(join(MODELS, 'stories15M.jlens')));
  const pos = 12, { row, jLayers } = await lensLogits(m15, cap15, pos, jl, pool15);
  let lensOk = true;
  for (let l = 0; l < ck15.config.nLayers; l++) ['attn', 'ffn'].forEach((ph, i) => { if (!sameArr(row(2 * l + i), logitLens(m15, ref15, l, pos, ph))) lensOk = false; });
  t.check('15M: batched logit-lens rows = logitLens bit for bit', lensOk);
  t.check('15M: batched J-lens rows = jLensLogits bit for bit',
    jLayers.join() === '2,3' && jLayers.every((l, i) => sameArr(row(2 * ck15.config.nLayers + i), jLensLogits(m15, jl.J[l], ref15.residualAt(l, pos, 'ffn')))));
  const p15 = ids15.slice(0, 8);
  t.check('15M: continuation, pool of 4 = continueSteered',
    sameArr(await continueFast(m15, p15, 20, { pool: pool15 }), continueSteered(createModel(ck15), p15, 20)));
  pool15.dispose();
}
