// E7 (Session A) — head ablation + exact-probability capture fields.
// The core epistemic invariant: no-ablation runs are BIT-IDENTICAL to the G1-parity
// path (the new parameter cannot perturb the gate), and ablating heads really changes
// the computation (ablation is an intervention, not an overlay).
import { readFile } from 'node:fs/promises';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, forward } from '../src/model.js';
import { captureRun, neuronContrast, logitLens } from '../src/capture.js';
import { model260K, tok512 } from './paths.js';

export async function run(t) {
  const ck = readCheckpoint(await readFile(model260K));
  const tok = new Tokenizer(await readFile(tok512), ck.config.vocabSize);
  const ids = tok.encode('Zog is a small dragon. Zog eats clouds and lives behind the moon.', true, false);
  const c = ck.config;

  // 1) ablate = null and ablate = empty Set are bit-identical to baseline
  const base = captureRun(createModel(ck), ids, { topK: 8 });
  const empty = captureRun(createModel(ck), ids, { topK: 8, ablate: new Set() });
  let identical = true;
  for (let i = 0; i < base.attention.length; i += 101) {
    if (base.attention[i] !== empty.attention[i]) { identical = false; break; }
  }
  for (let p = 0; p < ids.length && identical; p++) {
    if (base.logZ[p] !== empty.logZ[p]) identical = false;
  }
  t.check('empty ablation set is bit-identical to baseline (G1 path untouched)', identical);

  // 2) ablating one head changes logits; ablating ALL heads changes them more
  const one = captureRun(createModel(ck), ids, { topK: 8, ablate: new Set(['2:3']) });
  const all = new Set();
  for (let l = 0; l < c.nLayers; l++) for (let h = 0; h < c.nHeads; h++) all.add(l + ':' + h);
  const none = captureRun(createModel(ck), ids, { topK: 8, ablate: all });
  const last = ids.length - 1;
  const dOne = Math.abs(one.logZ[last] - base.logZ[last]) +
    Math.abs(one.topK[last][0].logit - base.topK[last][0].logit);
  t.check(`ablating head 2:3 perturbs the output (Δ=${dOne.toExponential(2)})`, dOne > 1e-6);
  let flips = 0;
  for (let p = 0; p < ids.length; p++) {
    if (none.topK[p][0].token !== base.topK[p][0].token) flips++;
  }
  t.check(`ablating all ${all.size} heads flips the greedy prediction at ${flips}/${ids.length} positions`,
    flips >= 3);

  // 3) ablated head still taps its attention pattern (microscope shows what was removed)
  let attSum = 0;
  const row = one.attnAt(2, 3, last);
  for (const v of row) attSum += v;
  t.checkClose('ablated head 2:3 attention pattern still recorded (row sums to 1)', attSum, 1, 1e-4);

  // 4) probability fields: exact softmax via logZ; nextProb matches manual computation
  const model = createModel(ck);
  let logits;
  for (let p = 0; p <= 4; p++) logits = forward(model, ids[p], p);
  let max = -Infinity;
  for (const v of logits) if (v > max) max = v;
  let sum = 0;
  for (const v of logits) sum += Math.exp(v - max);
  const manual = Math.exp(logits[ids[5]] - (max + Math.log(sum)));
  t.checkClose('nextProb[4] equals manual full softmax', base.nextProb[4], manual, 1e-6);
  const probs = base.topK[last].map(e => Math.exp(e.logit - base.logZ[last]));
  t.check('top-k probs are a valid partial distribution (desc, sum<1)',
    probs.every((p, i) => p > 0 && (i === 0 || p <= probs[i - 1])) &&
    probs.reduce((a, b) => a + b) <= 1.0000001);

  // 5) neuronContrast: the span of "Zog" pieces yields finite, sorted scores
  const scores = neuronContrast(base, 1, 4, 10);
  t.check('neuronContrast returns sorted finite scores over (layer, neuron)',
    scores.length === 10 && scores.every(s => Number.isFinite(s.score)) &&
    scores.every((s, i) => i === 0 || s.score <= scores[i - 1].score));

  // 6) neuron ablation (Session C): object form; empty = identical, real = perturbs
  const emptyObj = captureRun(createModel(ck), ids, { topK: 8, ablate: { heads: new Set(), neurons: new Set() } });
  t.check('empty {heads,neurons} ablation is bit-identical to baseline',
    emptyObj.logZ.every((v, p) => v === base.logZ[p]));
  const nAbl = captureRun(createModel(ck), ids, { topK: 8, ablate: { neurons: new Set(['2:10', '2:11', '3:5']) } });
  let dN = 0;
  for (let p = 0; p < ids.length; p++) dN = Math.max(dN, Math.abs(nAbl.logZ[p] - base.logZ[p]));
  t.check(`ablating 3 neurons perturbs the output (max |ΔlogZ| = ${dN.toExponential(2)})`, dN > 1e-6);
  // the ablated neuron's tapped activation reads zero
  t.check('ablated neuron activation is zero in the capture',
    nAbl.ffnAt(2, ids.length - 1)[10] === 0 && base.ffnAt(2, ids.length - 1)[10] !== 0);

  // 7) (d) ablation-aware logit lens — the exact worker path (captureRun ablate -> logitLens).
  // The lens projects a residual snapshot; muting a head must change the residual it
  // projects at downstream layers, while an empty mute leaves the lens bit-identical.
  const lensModel = createModel(ck);
  const pos = ids.length - 1, phase = 'ffn';
  const lensAt = (cap, l) => logitLens(lensModel, cap, l, pos, phase);
  const emptyLensCap = captureRun(createModel(ck), ids, { topK: 0, ablate: new Set() });
  let lensSame = true;
  for (let l = 0; l < c.nLayers; l++) {
    const a = lensAt(base, l), b = lensAt(emptyLensCap, l);
    for (let i = 0; i < a.length; i += 97) if (a[i] !== b[i]) { lensSame = false; break; }
  }
  t.check('lens under empty ablation is bit-identical to the un-ablated lens', lensSame);
  const headLensCap = captureRun(createModel(ck), ids, { topK: 0, ablate: new Set(['0:0', '0:1']) });
  const topLayer = c.nLayers - 1;
  let dLens = 0;
  const lb = lensAt(base, topLayer), lm = lensAt(headLensCap, topLayer);
  for (let i = 0; i < lb.length; i++) dLens = Math.max(dLens, Math.abs(lb[i] - lm[i]));
  t.check(`muting L0 heads shifts the final-layer lens (max |Δlogit| = ${dLens.toFixed(3)})`, dLens > 1e-3);
}
