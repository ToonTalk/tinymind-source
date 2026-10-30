// E19 — the Tiny Eye (src/eye.js, eye/worker.js; plans/tiny-eye.md Phase 1). The shipped eye
// (eye/tiny-eye.json + .bin, eye80k from tools/eye/train.py) must reproduce PyTorch: logits and
// every stage's neurons (fc1 before the GELU, mean over the grid) against
// tests/fixtures/eye-parity.json (tools/eye/parity.py, CPU float32). A switched-off neuron
// must act exactly like a neuron whose outgoing weights are zero. The draw pad's path
// (eyeDoodle) must reproduce the bitmaps the eye learned from, given the real Quick, Draw!
// strokes of sample doodles (tests/fixtures/eye-strokes.json, tools/eye/strokes.mjs), drawn
// anywhere and at any size on the pad. Phase 2: the training recording (eye/tiny-eye-growth.bin:
// float16 snapshots, then each snapshot's last-stage summary over the sample set) must load as an
// eye that matches PyTorch on the same rounded weights, and its summaries must be true of it; the
// squeezed eye (eye80k-sq8) must match PyTorch too. And the room's worker must answer drawings,
// sweeps, snapshots and the squeezed eye.
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';
import { TinyEye, eyeDoodle, eyeRaster, eyeNormalizeStrokes, eyeSampleInput, eyeHalfToFloat } from '../src/eye.js';
import { ROOT } from './paths.js';

const f32 = b64 => { const u = Buffer.from(b64, 'base64'); return new Float32Array(u.buffer.slice(u.byteOffset, u.byteOffset + u.byteLength)); };
const argmax = a => { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; };
function corr(a, b) {
  let ma = 0, mb = 0; for (let k = 0; k < a.length; k++) { ma += a[k]; mb += b[k]; } ma /= a.length; mb /= b.length;
  let ab = 0, aa = 0, bb = 0;
  for (let k = 0; k < a.length; k++) { const x = a[k] - ma, y = b[k] - mb; ab += x * y; aa += x * x; bb += y * y; }
  return ab / Math.sqrt(aa * bb);
}

export async function run(t) {
  const man = JSON.parse(await readFile(join(ROOT, 'eye', 'tiny-eye.json'), 'utf8'));
  const weights = await readFile(join(ROOT, 'eye', man.weights));
  const sample = await readFile(join(ROOT, 'eye', man.sample.file));
  t.check('eye: manifest is tiny-eye-v1 with 24 classes and three stages (24/48/64 wide, 96/192/256 neurons)',
    man.format === 'tiny-eye-v1' && man.classes.length === 24 && man.config.widths.join() === '24,48,64' && man.config.hidden.join() === '96,192,256');
  t.check('eye: weights are 82,704 float32 numbers, as the manifest says',
    weights.length === 4 * man.params && man.params === 82704 && man.tensors.reduce((n, x) => n + x.length, 0) === man.params);
  t.check('eye: 5,000 sample doodles of 28 x 28, sorted by class, 208 or 209 each',
    sample.length === 784 * man.sample.count && man.sample.count === 5000 && man.sample.classCounts.reduce((a, b) => a + b, 0) === 5000 &&
    man.sample.classCounts.every(n => n === 208 || n === 209));
  t.check('eye: Quick, Draw! attribution travels with the manifest', /Creative Commons Attribution 4\.0/.test(man.attribution));
  const named = Object.values(man.names.neurons);
  t.check('eye: 64 last-stage neurons carry Phase 0 names, each end with its distractor result',
    named.length === 64 && man.names.stage === 3 && named.every(n => n.most && typeof n.most.passed === 'boolean' && n.least));

  const eye = new TinyEye(man, weights);
  t.check('eye: grids are 14, 7 and 4 places across', eye.sizes.join() === '14,7,4');

  // ---- parity with PyTorch
  const par = JSON.parse(await readFile(join(ROOT, 'tests', 'fixtures', 'eye-parity.json'), 'utf8'));
  const pl = f32(par.logits), pn = f32(par.last_stage);
  let dl = 0, dn = 0;
  const x = new Float32Array(784);
  par.spread_rows.forEach((row, k) => {
    const r = eye.forward(eyeSampleInput(sample, row, x));
    for (let i = 0; i < 24; i++) dl = Math.max(dl, Math.abs(r.logits[i] - pl[k * 24 + i]));
    for (let i = 0; i < 256; i++) dn = Math.max(dn, Math.abs(r.neurons[2][i] - pn[k * 256 + i]));
  });
  t.checkClose('eye parity: logits on 100 doodles across every class, max |JS - PyTorch|', dl, 0, 1e-4);
  t.checkClose('eye parity: last-stage neurons on the same 100, max |JS - PyTorch|', dn, 0, 1e-4);
  const st = par.stages.map(f32);
  let ds = 0;
  par.per_class_rows.forEach((row, k) => {
    const r = eye.forward(eyeSampleInput(sample, row, x));
    r.neurons.forEach((n, s) => { for (let i = 0; i < n.length; i++) ds = Math.max(ds, Math.abs(n[i] - st[s][k * par.hidden[s] + i])); });
  });
  t.checkClose('eye parity: every neuron of every stage, one doodle per class, max |JS - PyTorch|', ds, 0, 1e-4);

  // ---- accuracy and speed on the sample set (every 5th doodle)
  const labels = new Uint8Array(5000);
  { let at = 0; man.sample.classCounts.forEach((n, c) => { labels.fill(c, at, at + n); at += n; }); }
  let right = 0, seen = 0;
  const t0 = performance.now();
  for (let i = 0; i < 5000; i += 5) { right += argmax(eye.forward(eyeSampleInput(sample, i, x)).logits) === labels[i]; seen++; }
  const ms = (performance.now() - t0) / seen;
  console.log(`    (${ms.toFixed(1)} ms a drawing here)`);
  t.check(`eye: right on ${right} of ${seen} sample doodles (${(100 * right / seen).toFixed(1)}%; the export's held-out accuracy is ${(100 * man.held_out_acc).toFixed(1)}%)`,
    right / seen > 0.86);

  // ---- switching neurons off
  const probe = eyeSampleInput(sample, 1234, new Float32Array(784));
  const plain = eye.forward(probe);
  const none = eye.forward(probe, { off: man.config.hidden.map(h => new Uint8Array(h)) });
  t.check('switch-off: an empty mask changes nothing', plain.logits.every((v, i) => v === none.logits[i]));
  for (const [s, j] of [[2, 85], [0, 7]]) {
    const off = man.config.hidden.map(h => new Uint8Array(h)); off[s][j] = 1;
    const masked = eye.forward(probe, { off });
    // the same eye with that neuron's outgoing weights (its column of fc2) set to zero
    const w2 = man.tensors.find(q => q.name === `stages.${s}.0.fc2.weight`), H = man.config.hidden[s], c = man.config.widths[s];
    const cut = new Float32Array(weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength));
    for (let ch = 0; ch < c; ch++) cut[w2.offset + ch * H + j] = 0;
    const zeroed = new TinyEye(man, new Uint8Array(cut.buffer)).forward(probe);
    let d = 0; for (let i = 0; i < 24; i++) d = Math.max(d, Math.abs(masked.logits[i] - zeroed.logits[i]));
    t.checkClose(`switch-off: stage ${s + 1} neuron ${j} off = its outgoing weights zeroed (logits)`, d, 0, 1e-9);
    t.check(`switch-off: stage ${s + 1} neuron ${j}'s own reading is still shown, and the guess's numbers move`,
      masked.neurons[s][j] === plain.neurons[s][j] && masked.logits.some((v, i) => v !== plain.logits[i]));
  }
  { const off = man.config.hidden.map(h => new Uint8Array(h)); off[0][7] = 1;
    t.check('switch-off: a stage-1 switch changes what later stages see', eye.forward(probe, { off }).neurons[1].some((v, i) => v !== plain.neurons[1][i])); }

  // ---- the draw pad's path, on real strokes of sample doodles
  const fx = JSON.parse(await readFile(join(ROOT, 'tests', 'fixtures', 'eye-strokes.json'), 'utf8')).doodles;
  const toPad = (d, k, dx, dy) => d.drawing.map(([xs, ys]) => xs.map((v, i) => [dx + v * k, dy + ys[i] * k]));
  let rc = 0, rmad = 0, pc = 0, same = 0, padRight = 0, worst = 1;
  for (const d of fx) {
    const bmp = eyeSampleInput(sample, d.sample, new Float32Array(784));
    const raster = eyeRaster(d.drawing.map(([xs, ys]) => xs.map((v, i) => [v, ys[i]])));
    rc += corr(raster, bmp); let m = 0; for (let k = 0; k < 784; k++) m += Math.abs(raster[k] - bmp[k]); rmad += m / 784;
    // as if drawn on a 300-pixel pad, small and off to one side (the pad stores 0..1 coordinates)
    const pad = eyeDoodle(toPad(d, 0.6 / 255, 0.31, 0.07));
    const c = corr(pad, bmp); pc += c; worst = Math.min(worst, c);
    const gp = argmax(eye.forward(pad).logits), gb = argmax(eye.forward(bmp).logits);
    same += gp === gb; padRight += gp === d.label;
  }
  const n = fx.length;
  t.check(`pad path: Quick, Draw!'s own simplified strokes, drawn by eyeRaster, match the bitmaps (pixel correlation ${(rc / n).toFixed(4)}, mean |difference| ${(rmad / n).toFixed(4)}, ${n} doodles)`,
    n >= 60 && rc / n > 0.999 && rmad / n < 0.003);
  t.check(`pad path: the same strokes drawn small and off-center on the pad come out as the bitmaps (correlation ${(pc / n).toFixed(4)}, worst ${worst.toFixed(3)})`,
    pc / n > 0.995);
  t.check(`pad path: the eye guesses the same from the pad as from the bitmap for ${same} of ${n} (right for ${padRight})`, same / n >= 0.95);
  const big = eyeDoodle(toPad(fx[0], 3, -40, 900)), small = eyeDoodle(toPad(fx[0], 0.2, 5, 5));
  let dd = 0; for (let k = 0; k < 784; k++) dd = Math.max(dd, Math.abs(big[k] - small[k]));
  t.check('pad path: where and how big you draw does not matter', dd < 0.05);
  const norm = eyeNormalizeStrokes(toPad(fx[0], 0.37, 2, 3));
  let lo = Infinity, hi = -Infinity; for (const s of norm) for (const [a, b] of s) { lo = Math.min(lo, a, b); hi = Math.max(hi, a, b); }
  t.check('pad path: strokes are aligned to the top-left and scaled to 0..255, as Quick, Draw! simplified them', lo >= 0 && lo <= 2 && hi === 255);
  t.check('pad path: an empty pad is an empty picture; a single tap is a dot in the middle',
    eyeDoodle([]).every(v => v === 0) && (() => { const d = eyeDoodle([[[0.5, 0.5]]]); return d[14 * 28 + 14] > 0 && d.filter(v => v > 0).length <= 9; })());

  // ---- Phase 2: float16, the training recording, the squeezed eye
  const halves = eyeHalfToFloat(new Uint8Array([0x00, 0x3c, 0x00, 0xc0, 0x01, 0x00, 0xff, 0x7b, 0x00, 0x80, 0x55, 0x35]));
  t.check('float16: 1, -2, the smallest subnormal, the largest number, -0 and about 1/3 decode exactly',
    halves[0] === 1 && halves[1] === -2 && halves[2] === 2 ** -24 && halves[3] === 65504 && Object.is(halves[4], -0) && Math.abs(halves[5] - 1 / 3) < 2e-4);
  const g = man.growth;
  const growth = await readFile(join(ROOT, 'eye', g.file));
  const stepsUp = [...g.snapshots, g.final].every((st, i, a) => !i || st.step > a[i - 1].step);
  t.check(`growth: ${g.snapshots.length} earlier snapshots (float16) plus summaries fill tiny-eye-growth.bin exactly; the last stop is the shipped eye (step ${g.final.step})`,
    growth.length === g.bytes && g.snapshots.length === 19 && g.weights.bytesEach === 2 * man.params &&
    g.bytes === g.snapshots.length * (g.weights.bytesEach + g.summary.bytesEach) && g.final.step === 9360 && stepsUp);
  t.check('growth: jobs settle late (under half settled at step 3,359, all at the end) while the eye is already right 84 in 100',
    (() => { const s3 = g.snapshots.find(st => st.step === 3359); return s3 && s3.settled < 0.5 && s3.sampleRight / 5000 > 0.83 && g.final.settled === 1; })());
  t.check('growth: the featured stories are neurons 117, 2 and 16', g.stories.map(st => st.neuron).join() === '117,2,16');
  const snapAt = step => { const k = g.snapshots.findIndex(st => st.step === step);
    return { k, eye: new TinyEye(man, growth.subarray(k * g.weights.bytesEach, (k + 1) * g.weights.bytesEach), { float16: true }) }; };
  const sp = par.snapshot, { eye: snapEye } = snapAt(sp.step), spl = f32(sp.logits), spn = f32(sp.last_stage);
  let sl = 0, sn = 0;
  sp.rows.forEach((row, k) => {
    const r = snapEye.forward(eyeSampleInput(sample, row, x));
    for (let i = 0; i < 24; i++) sl = Math.max(sl, Math.abs(r.logits[i] - spl[k * 24 + i]));
    for (let i = 0; i < 256; i++) sn = Math.max(sn, Math.abs(r.neurons[2][i] - spn[k * 256 + i]));
  });
  t.checkClose(`growth parity: the step-${sp.step} snapshot from the float16 file vs PyTorch on the same rounded weights (logits)`, sl, 0, 1e-4);
  t.checkClose(`growth parity: same snapshot, last-stage neurons`, sn, 0, 1e-4);
  // a snapshot's summary must be true of that snapshot: its top 16 above, and its bottom 16 below, other doodles
  const T = { f32: Float32Array, u16: Uint16Array, i8: Int8Array };
  const summaryOf = k => { let at = g.summary.offset + k * g.summary.bytesEach; const o = {};
    for (const [name, ty, len] of g.summary.layout) { o[name] = new T[ty](growth.buffer.slice(growth.byteOffset + at, growth.byteOffset + at + len * T[ty].BYTES_PER_ELEMENT)); at += len * T[ty].BYTES_PER_ELEMENT; }
    return o; };
  let summaryOk = true, why = '';
  for (const [step, j] of [[433, 2], [1206, 117], [4727, 16]]) {
    const { k, eye: se } = snapAt(step), sum = summaryOf(k);
    const top = Array.from(sum.top.subarray(j * 16, j * 16 + 16)), bottom = Array.from(sum.bottom.subarray(j * 16, j * 16 + 16));
    const others = Array.from({ length: 300 }, (_, i) => (i * 167 + 11) % 5000).filter(i => !top.includes(i) && !bottom.includes(i));
    const val = i => se.forward(eyeSampleInput(sample, i, x)).neurons[2][j];
    const tv = top.map(val), bv = bottom.map(val), ov = others.map(val);
    const om = ov.reduce((a, b) => a + b, 0) / ov.length;
    const d = sum.deciles.subarray(j * 9, j * 9 + 9);
    const ok = Math.min(...tv) >= Math.max(...ov) && Math.max(...bv) <= Math.min(...ov) && tv.every((v, i) => !i || v <= tv[i - 1] + 1e-6) &&
      Math.abs(om - sum.mean[j]) < 0.3 * sum.sd[j] && d.every((v, i) => !i || v >= d[i - 1]);
    if (!ok) { summaryOk = false; why += ` step ${step} neuron ${j}`; }
  }
  t.check('growth: each summary is true of its snapshot (top 16 above and bottom 16 below other doodles, highest first; mean and deciles fit)' + why, summaryOk);
  { const sum = summaryOf(snapAt(1206).k), z = Array.from(sum.classZ.subarray(117 * 24, 117 * 24 + 24));
    t.check('growth: by step 1,206 neuron 117 answers most to houses (its class bars), as its story says', man.classes[argmax(z)] === 'house'); }

  const sqMan = JSON.parse(await readFile(join(ROOT, 'eye', man.squeezed), 'utf8'));
  const sqW = await readFile(join(ROOT, 'eye', sqMan.weights));
  t.check('squeezed eye: tiny-eye-v1, last stage 8 channels and 8 neurons, same sample set, its 8 neurons named in Phase 0',
    sqMan.format === 'tiny-eye-v1' && sqMan.config.widths.join() === '24,48,8' && sqMan.config.hidden.join() === '96,192,8' &&
    sqW.length === 4 * sqMan.params && sqMan.sample.file === man.sample.file && Object.keys(sqMan.names.neurons).length === 8);
  const sq = new TinyEye(sqMan, sqW), qp = par.squeezed, ql = f32(qp.logits), qs = qp.stages.map(f32);
  let dq = 0;
  qp.rows.forEach((row, k) => {
    const r = sq.forward(eyeSampleInput(sample, row, x));
    for (let i = 0; i < 24; i++) dq = Math.max(dq, Math.abs(r.logits[i] - ql[k * 24 + i]));
    r.neurons.forEach((n, st) => { for (let i = 0; i < n.length; i++) dq = Math.max(dq, Math.abs(n[i] - qs[st][k * qp.hidden[st] + i])); });
  });
  t.checkClose('squeezed parity: logits and every neuron of every stage, one doodle per class, max |JS - PyTorch|', dq, 0, 1e-4);

  // ---- the room's worker, in-process
  const posted = [];
  globalThis.self = { postMessage: m => posted.push(m) };
  await import(pathToFileURL(join(ROOT, 'eye', 'worker.js')).href);
  const send = async (m, until) => { posted.length = 0; await self.onmessage({ data: m }); const t1 = Date.now();
    while (!posted.some(until) && Date.now() - t1 < 20000) await new Promise(r => setTimeout(r, 5)); return posted.find(until) || posted.find(p => p.type === 'error'); };
  const loaded = await send({ type: 'load', manifest: JSON.stringify(man), weights: weights.buffer.slice(weights.byteOffset, weights.byteOffset + weights.byteLength) }, p => p.type === 'loaded');
  t.check('worker: loads the eye and says what it is', loaded && loaded.type === 'loaded' && loaded.classes.length === 24 && loaded.hidden.join() === '96,192,256');
  const off = man.config.hidden.map(h => new Uint8Array(h)); off[2][85] = 1;
  const seenMsg = await send({ type: 'see', id: 1, x: probe, off, maps: true }, p => p.type === 'seen');
  const direct = eye.forward(probe, { off });
  t.check('worker: answers a drawing with the switched-off guess, the every-neuron-on guess, and every stage\'s places',
    seenMsg && seenMsg.probs.every((v, i) => Math.abs(v - direct.probs[i]) < 1e-6) && seenMsg.baseProbs &&
    seenMsg.baseProbs.every((v, i) => Math.abs(v - plain.probs[i]) < 1e-6) && seenMsg.maps.length === 3 && seenMsg.maps[2].length === 16 * 256);
  const swept = await send({ type: 'sweep', id: 's', start: 40, count: 30, doodles: sample.subarray(40 * 784, 70 * 784).slice() }, p => p.type === 'swept');
  let ok = swept && swept.table.length === 30 * 544;
  for (let i = 0; ok && i < 30; i += 7) {
    const r = eye.forward(eyeSampleInput(sample, 40 + i, x));
    ok = r.neurons[2].every((v, j) => v === swept.table[i * 544 + 96 + 192 + j]) && swept.guesses[i] === argmax(r.probs);
  }
  t.check('worker: a sweep returns every neuron of every stage and the guess for each doodle', !!ok);
  await send({ type: 'growth', weights: growth.buffer.slice(growth.byteOffset, growth.byteOffset + g.snapshots.length * g.weights.bytesEach), bytesEach: g.weights.bytesEach }, p => p.type === 'growth-ready');
  const k1206 = snapAt(1206).k, atSnap = await send({ type: 'see', id: 2, snap: k1206, x: probe }, p => p.type === 'seen');
  const snapDirect = snapAt(1206).eye.forward(probe);
  t.check('worker: a look at a training snapshot uses that snapshot', atSnap && atSnap.snap === k1206 && atSnap.probs.every((v, i) => Math.abs(v - snapDirect.probs[i]) < 1e-6));
  await send({ type: 'load', key: 'sq8', manifest: sqMan, weights: sqW.buffer.slice(sqW.byteOffset, sqW.byteOffset + sqW.byteLength) }, p => p.type === 'loaded' && p.key === 'sq8');
  const sqSeen = await send({ type: 'see', id: 3, key: 'sq8', x: probe }, p => p.type === 'seen');
  t.check('worker: holds the squeezed eye beside the full one and answers for whichever is asked',
    sqSeen && sqSeen.key === 'sq8' && sqSeen.neurons[2].length === 8 && sqSeen.probs.every((v, i) => Math.abs(v - sq.forward(probe).probs[i]) < 1e-6));
  delete globalThis.self;
}
