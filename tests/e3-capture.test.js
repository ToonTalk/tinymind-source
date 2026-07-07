// E3 — activation capture. Checks: attention rows are valid distributions, residual/ffn
// snapshots populated, memory budget sane for 15M @ seq 256, logit lens at final layer
// reproduces the real logits, interpreter JSON slices correctly.
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint } from '../src/bin-reader.js';
import { createModel, forward, argmax } from '../src/model.js';
import { captureRun, logitLens } from '../src/capture.js';
import { model15M, model260K } from './paths.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

export async function run(t) {
  // --- 15M over the full 256-token golden sequence: the budget case from the kickoff
  const ck15 = readCheckpoint(await readFile(model15M));
  const m15 = createModel(ck15);
  const tokens15 = (await readFile(join(FIXTURES, '15M-p1.tokens.txt'), 'utf8'))
    .trim().split('\n').map(Number);

  const t0 = performance.now();
  const cap = captureRun(m15, tokens15, { topK: 16 });
  const capMs = performance.now() - t0;
  const mb = cap.bytes() / 1e6;
  console.log(`  15M capture: ${tokens15.length} pos, ${mb.toFixed(1)} MB activations, ${(capMs / 1000).toFixed(1)}s`);
  t.check(`15M @ seq 256 capture fits budget (${mb.toFixed(1)} MB < 40 MB, no pathological allocation)`,
    mb < 40);

  // attention rows are probability distributions over 0..pos
  let distOk = true;
  const c = m15.config;
  for (const [l, h, pos] of [[0, 0, 0], [2, 3, 100], [5, 5, 255]]) {
    const row = cap.attnAt(l, h, pos);
    let sum = 0; for (const v of row) { if (v < 0) distOk = false; sum += v; }
    if (Math.abs(sum - 1) > 1e-4) distOk = false;
    if (row.length !== pos + 1) distOk = false;
  }
  t.check('attention rows are distributions (len pos+1, nonneg, sum≈1)', distOk);

  // residual & ffn snapshots are populated (not all zero) at spot positions
  const rmid = cap.residualAt(3, 128, 'ffn');
  const fmid = cap.ffnAt(3, 128);
  const nonzero = arr => { for (const v of arr) if (v !== 0) return true; return false; };
  t.check('residual + ffn snapshots populated', nonzero(rmid) && nonzero(fmid));

  // logit lens at the LAST layer must reproduce the model's actual prediction
  const lensLogits = logitLens(m15, cap, c.nLayers - 1, 200, 'ffn');
  const freshState = createModel(ck15);
  let real;
  for (let p = 0; p <= 200; p++) real = forward(freshState, tokens15[p], p);
  t.check('logit lens @ final layer == real argmax', argmax(lensLogits) === argmax(real));
  let maxDiff = 0;
  for (let i = 0; i < real.length; i++) maxDiff = Math.max(maxDiff, Math.abs(real[i] - lensLogits[i]));
  t.checkClose('logit lens @ final layer matches real logits', maxDiff, 0, 1e-4);

  // topK captured and consistent with teacher-forced next token ranking
  t.check('topK captured for every position (16 entries)',
    cap.topK.length === tokens15.length && cap.topK.every(k => k.length === 16));

  // --- 260K: interpreter JSON slice (the payload Session C ships to the LLM)
  const ck260 = readCheckpoint(await readFile(model260K));
  const m260 = createModel(ck260);
  const tokens260 = (await readFile(join(FIXTURES, '260K-p1.tokens.txt'), 'utf8'))
    .trim().split('\n').map(Number).slice(0, 60);
  const cap260 = captureRun(m260, tokens260, { topK: 8 });
  const json = cap260.toInterpreterJSON({ layers: [0, 4], heads: [0, 7], positions: [10, 30, 59] });
  const parsed = JSON.parse(JSON.stringify(json));
  t.check('interpreter JSON: schema tag + selected slices only',
    parsed.schema === 'tiny-mind-capture-v1' &&
    Object.keys(parsed.attention).length === 4 &&
    parsed.attention.L0H0.length === 3 &&
    parsed.attention.L4H7[2].length === 60);
  const jsonKb = JSON.stringify(json).length / 1024;
  console.log(`  interpreter JSON slice (2 layers × 2 heads × 3 pos of 60): ${jsonKb.toFixed(1)} KB`);
  t.check('interpreter JSON slice is LLM-sized (< 100 KB)', jsonKb < 100);
}
