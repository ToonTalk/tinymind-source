// E4 (gate G2a substrate) — hand-rolled backprop + AdamW.
// 1) analytic grads vs central finite differences on a tiny f64 model (every tensor)
// 2) trainer forward logits agree with the C-parity model.js forward
// 3) overfit smoke: loss drops hard on a repeated batch
// 4) ms/step at the kickoff spec: batch 8, seq 256, on the real 260K checkpoint
import { readFile } from 'node:fs/promises';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, forward, argmax } from '../src/model.js';
import { Trainer, rng, packStories, sampleBatch } from '../src/train.js';
import { model260K, tok512, tinyStoriesValid } from './paths.js';

function tinyCheckpoint(seed, config = {
  dim: 8, hiddenDim: 12, nLayers: 2, nHeads: 2, nKvHeads: 1, vocabSize: 16,
  seqLen: 8, sharedWeights: true, headSize: 4, kvDim: 4,
}) {
  const rand = rng(seed);
  const gauss = () => {
    let s = 0; for (let i = 0; i < 12; i++) s += rand();
    return (s - 6);
  };
  const c = config;
  const mk = (n, scale) => Float64Array.from({ length: n }, () => gauss() * scale);
  const weights = {
    tokenEmbedding: mk(c.vocabSize * c.dim, 0.3),
    rmsAtt: Float64Array.from({ length: c.nLayers * c.dim }, () => 1 + gauss() * 0.02),
    wq: mk(c.nLayers * c.dim * c.dim, 0.2),
    wk: mk(c.nLayers * c.dim * c.kvDim, 0.2),
    wv: mk(c.nLayers * c.dim * c.kvDim, 0.2),
    wo: mk(c.nLayers * c.dim * c.dim, 0.2),
    rmsFfn: Float64Array.from({ length: c.nLayers * c.dim }, () => 1 + gauss() * 0.02),
    w1: mk(c.nLayers * c.hiddenDim * c.dim, 0.2),
    w2: mk(c.nLayers * c.dim * c.hiddenDim, 0.2),
    w3: mk(c.nLayers * c.hiddenDim * c.dim, 0.2),
    rmsFinal: Float64Array.from({ length: c.dim }, () => 1 + gauss() * 0.02),
  };
  return { config, weights };
}

function gradcheck(ck, seed) {
  const B = 2, T = 6;
  const tr = new Trainer(ck, { B, T, f64: true });
  const rand = rng(seed);
  const batch = new Int32Array(B * (T + 1));
  for (let i = 0; i < batch.length; i++) batch[i] = Math.floor(rand() * ck.config.vocabSize);

  tr.zeroGrads();
  tr.forwardLoss(batch, true);
  tr.backward(batch);

  const h = 1e-5;
  let worst = 0, worstAt = '';
  for (const key of Object.keys(tr.params)) {
    const p = tr.params[key], an = tr.grads[key];
    for (let s = 0; s < 8; s++) {
      const i = Math.floor(rand() * p.length);
      const orig = p[i];
      p[i] = orig + h; const lp = tr.forwardLoss(batch, false);
      p[i] = orig - h; const lm = tr.forwardLoss(batch, false);
      p[i] = orig;
      const fd = (lp - lm) / (2 * h);
      const denom = Math.max(1e-6, Math.abs(fd) + Math.abs(an[i]));
      const rel = Math.abs(fd - an[i]) / denom;
      if (rel > worst) { worst = rel; worstAt = `${key}[${i}] fd=${fd.toExponential(3)} an=${an[i].toExponential(3)}`; }
    }
  }
  return { worst, worstAt };
}

export async function run(t) {
  // --- 1) gradient checks: generic path (HS=4) and unrolled fast path (HS=8) -
  const g1 = gradcheck(tinyCheckpoint(1234), 99);
  console.log(`  gradcheck (HS=4 generic) worst rel err: ${g1.worst.toExponential(2)} at ${g1.worstAt}`);
  t.check('gradient check, generic attention path (rel err < 1e-4)', g1.worst < 1e-4);

  const g2 = gradcheck(tinyCheckpoint(4321, {
    dim: 16, hiddenDim: 20, nLayers: 2, nHeads: 2, nKvHeads: 1, vocabSize: 16,
    seqLen: 8, sharedWeights: true, headSize: 8, kvDim: 8,
  }), 77);
  console.log(`  gradcheck (HS=8 fast path, GQA) worst rel err: ${g2.worst.toExponential(2)} at ${g2.worstAt}`);
  t.check('gradient check, HS=8 unrolled fast path (rel err < 1e-4)', g2.worst < 1e-4);

  // --- 2) trainer forward vs model.js forward on real 260K ------------------
  const ck260 = readCheckpoint(await readFile(model260K));
  const tok = new Tokenizer(await readFile(tok512), ck260.config.vocabSize);
  const m260 = createModel(ck260);
  const text = 'Once upon a time, there was a little girl named Lily. She loved to play outside.';
  const ids = tok.encode(text, true, false);
  const Tp = ids.length - 1;
  const trP = new Trainer(ck260, { B: 1, T: Tp });
  const pbatch = new Int32Array(ids);
  trP.evalLoss(pbatch);
  let agree = true, maxDiff = 0;
  for (let p = 0; p < Tp; p++) {
    const ref = forward(m260, ids[p], p);
    const got = trP.act.logits.subarray(p * ck260.config.vocabSize, (p + 1) * ck260.config.vocabSize);
    if (argmax(got) !== argmax(ref)) agree = false;
    for (let i = 0; i < ref.length; i++) maxDiff = Math.max(maxDiff, Math.abs(ref[i] - got[i]));
  }
  t.check(`trainer forward argmax agrees with model.js at all ${Tp} positions`, agree);
  t.checkClose('trainer forward max |logit diff| vs model.js', maxDiff, 0, 5e-3);

  // --- 3) overfit smoke -----------------------------------------------------
  const stories = (await readFile(tinyStoriesValid, 'utf8')).split('<|endoftext|>')
    .slice(0, 20).map(s => s.trim()).filter(Boolean);
  const stream = packStories(stories.map(s => tok.encode(s, false, false)));
  const trS = new Trainer(ck260, { B: 4, T: 64, lr: 1e-3 });
  const fixed = sampleBatch(stream, 4, 64, rng(7));
  const l0 = trS.evalLoss(fixed);
  let lEnd = l0;
  for (let i = 0; i < 40; i++) lEnd = trS.step(fixed).loss;
  console.log(`  overfit smoke: loss ${l0.toFixed(3)} -> ${lEnd.toFixed(3)} after 40 steps on one batch`);
  t.check('overfit smoke: repeated batch drives loss down by >0.5', l0 - lEnd > 0.5);

  // --- 4) ms/step at kickoff spec (batch 8, seq 256) -------------------------
  const trT = new Trainer(ck260, { B: 8, T: 256, lr: 1e-4 });
  const r2 = rng(11);
  let ms = [];
  for (let i = 0; i < 3; i++) {
    const b = sampleBatch(stream, 8, 256, r2);
    ms.push(trT.step(b).ms);
  }
  const msPerStep = (ms[1] + ms[2]) / 2; // skip warmup step
  console.log(`  260K train step @ B=8 T=256: ${msPerStep.toFixed(0)} ms/step (Node; browser differs — E6)`);
  t.check(`ms/step measured (${msPerStep.toFixed(0)} ms) — recorded for report`, msPerStep > 0);
}
