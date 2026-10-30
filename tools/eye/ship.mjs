// ship.mjs — copy the Tiny Eye from its Phase 0 export into the repo, ready for the room:
//   eye/tiny-eye.json         the manifest (tiny-eye-v1): config, classes, tensors, the sample
//                             set's layout, the names Phase 0's labeler gave 64 last-stage neurons
//                             (with whether each name passed the distractor test), and the layout
//                             of the growth file
//   eye/tiny-eye.bin          eye80k's weights, float32 (331 KB)
//   eye/tiny-eye-sample.bin   5,000 held-out doodles, 28x28 uint8 (ink 255), sorted by class
//   eye/tiny-eye-growth.bin   for the "watch it grow" scrubber: the 19 earlier training snapshots
//                             (float16; the last stop is tiny-eye.bin itself, so Phase 0's tested
//                             names stay true), then for each snapshot a summary of every
//                             last-stage neuron over the sample set (its 16 top and bottom doodles,
//                             mean, spread, deciles, and how much each kind of thing moves it), so
//                             scrubbing never waits for 5,000 doodles to be looked through again
//   eye/tiny-eye-sq8.json/.bin  the squeezed eye (eye80k-sq8: 8 channels in the last stage, no
//                             wide bypass), for the room's squeeze lesson; it shares the sample set
// The summaries come from this repo's own engine (src/eye.js), the one the room runs.
//   node tools/eye/ship.mjs [data folder]
import { readFileSync, writeFileSync, copyFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import os from 'node:os';

const ROOT = dirname(dirname(dirname(fileURLToPath(import.meta.url))));
const { TinyEye, eyeSampleInput } = await import(pathToFileURL(join(ROOT, 'src', 'eye.js')).href);

const STAGE = 2, TOP = 16, CLASS_Z_SCALE = 16;
// one snapshot's summary of the last stage, in this order (float32 blocks first, for alignment)
const LAYOUT = H => [['mean', 'f32', H], ['sd', 'f32', H], ['deciles', 'f32', H * 9], ['top', 'u16', H * TOP], ['bottom', 'u16', H * TOP], ['classZ', 'i8', H * 24]];
const BYTES = { f32: 4, u16: 2, i8: 1 };

// every last-stage neuron over the 5,000 sample doodles, for one set of weights
function summarize({ manifest, weights, float16, sample, labels }) {
  const eye = new TinyEye(manifest, weights, { float16 });
  const N = labels.length, H = manifest.config.hidden[STAGE], vals = new Float32Array(N * H), x = new Float32Array(784);
  let right = 0;
  for (let i = 0; i < N; i++) {
    const r = eye.forward(eyeSampleInput(sample, i, x));
    vals.set(r.neurons[STAGE], i * H);
    let b = 0; for (let k = 1; k < r.probs.length; k++) if (r.probs[k] > r.probs[b]) b = k;
    right += b === labels[i];
  }
  const out = { right, mean: new Float32Array(H), sd: new Float32Array(H), deciles: new Float32Array(H * 9),
    top: new Uint16Array(H * TOP), bottom: new Uint16Array(H * TOP), classZ: new Int8Array(H * 24), top100: [] };
  const col = new Float32Array(N), idx = Array.from({ length: N }, (_, i) => i);
  for (let j = 0; j < H; j++) {
    let m = 0, q = 0;
    for (let i = 0; i < N; i++) { const v = vals[i * H + j]; col[i] = v; m += v; q += v * v; }
    m /= N; const sd = Math.sqrt(Math.max(0, q / N - m * m)) || 1;
    out.mean[j] = m; out.sd[j] = sd;
    const order = idx.slice().sort((a, b) => col[b] - col[a]);
    for (let k = 0; k < TOP; k++) { out.top[j * TOP + k] = order[k]; out.bottom[j * TOP + k] = order[N - 1 - k]; }
    out.top100.push(order.slice(0, 100));
    for (let d = 1; d <= 9; d++) out.deciles[j * 9 + d - 1] = col[order[Math.round(N * (1 - d / 10))]];
    const cm = new Float64Array(24), cn = new Float64Array(24);
    for (let i = 0; i < N; i++) { cm[labels[i]] += col[i]; cn[labels[i]]++; }
    for (let c = 0; c < 24; c++) out.classZ[j * 24 + c] = Math.max(-127, Math.min(127, Math.round((cm[c] / (cn[c] || 1) - m) / sd * CLASS_Z_SCALE)));
  }
  return out;
}

if (!isMainThread) {
  parentPort.postMessage(summarize(workerData));
} else {
  const DATA = process.argv[2] || 'C:/Users/toont/tm-data/eye';
  const EXPORT = join(DATA, 'export');
  const src = JSON.parse(readFileSync(join(EXPORT, 'tiny-eye.json'), 'utf8'));
  const sample = JSON.parse(readFileSync(join(EXPORT, 'sample.json'), 'utf8'));
  const labelsOf = run => JSON.parse(readFileSync(join(DATA, 'labels', run + '.json'), 'utf8'));
  if (sample.labels.some((c, i) => i && c < sample.labels[i - 1])) throw new Error('the sample set must be sorted by class');
  const classCounts = src.classes.map((_, c) => sample.labels.filter(l => l === c).length);
  const sampleInfo = { file: 'tiny-eye-sample.bin', count: sample.count, size: 28, dtype: 'uint8, ink = 255', sortedByClass: true, classCounts,
    note: 'held-out doodles the eye never trained on, about 208 a category' };
  const namesOf = run => {
    const L = labelsOf(run), neurons = {};
    for (const [j, n] of Object.entries(L.neurons)) {
      const pole = p => p && p.label ? { name: p.label, hits: p.hits, passed: !!p.passed } : null;
      neurons[j] = { most: pole(n.most), least: pole(n.least) };
    }
    return { stage: L.stage, model: L.model, how: 'Phase 0 (reports/eye/feasibility.md): a vision model named each pole from its top 16 held-out doodles, then from the name alone tried to pick 4 of that pole\'s doodles out of 16; "passed" = 3 or 4 right', neurons };
  };
  const sampleBytes = readFileSync(join(EXPORT, 'sample.bin'));
  const labels = new Uint8Array(sample.labels);
  const weights32 = readFileSync(join(EXPORT, 'tiny-eye.bin'));

  // the growth summaries: every snapshot but the last (whose stop is the shipped float32 eye), and the
  // final eye itself, for how much of each neuron's final top 100 is already in place at each stop
  const { snapshots, snapshot_dtype, ...rest } = src;
  const earlier = snapshots.slice(0, -1), last = snapshots[snapshots.length - 1];
  const jobs = [...earlier.map(s => ({ s, weights: readFileSync(join(EXPORT, s.file)), float16: true })), { s: last, weights: weights32, float16: false }];
  const P = Math.max(1, Math.min(8, os.cpus().length - 2));
  console.log(`summarizing ${jobs.length} snapshots over ${sample.count} doodles on ${P} threads…`);
  const t0 = Date.now(), results = new Array(jobs.length);
  let next = 0;
  await Promise.all(Array.from({ length: P }, async () => {
    while (next < jobs.length) {
      const k = next++, job = jobs[k];
      results[k] = await new Promise((res, rej) => {
        const w = new Worker(fileURLToPath(import.meta.url), { workerData: { manifest: src, weights: job.weights, float16: job.float16, sample: sampleBytes, labels } });
        w.once('message', res); w.once('error', rej);
      });
      console.log(`  step ${job.s.step}: right on ${results[k].right} of ${sample.count}`);
    }
  }));
  const finalTop = results[results.length - 1].top100.map(a => new Set(a));
  const settled = r => r.top100.reduce((acc, a, j) => acc + a.filter(i => finalTop[j].has(i)).length / 100, 0) / r.top100.length;
  const H = src.config.hidden[STAGE], layout = LAYOUT(H);
  const sumBytes = layout.reduce((n, [, t, len]) => n + BYTES[t] * len, 0);
  const wBytes = 2 * src.params;
  const growth = Buffer.alloc(earlier.length * (wBytes + sumBytes));
  earlier.forEach((s, k) => {
    jobs[k].weights.copy(growth, k * wBytes);
    let at = earlier.length * wBytes + k * sumBytes;
    for (const [name, t] of layout) { const a = results[k][name]; Buffer.from(a.buffer, a.byteOffset, a.byteLength).copy(growth, at); at += a.byteLength; }
  });
  writeFileSync(join(ROOT, 'eye', 'tiny-eye-growth.bin'), growth);
  const stop = (s, r) => ({ step: s.step, epoch: s.epoch, heldOutAcc: s.held_out_acc, sampleRight: r.right, settled: +settled(r).toFixed(4) });
  // the growth stories Phase 0 found (reports/eye/feasibility.md, "Growth stories")
  const stories = [
    { neuron: 117, title: 'a roof that grows an umbrella offshoot', text: 'From about step 600 it is a house-roof neuron. Later umbrellas, a peaked top too, join it: by the end they are about a fifth of its top 100 doodles, just below the houses.' },
    { neuron: 2, title: 'castle walls, then fish', text: 'It starts on castle walls, then switches to fish between steps 1,206 and 2,387, when the eye is already right 78 to 84 times in 100.' },
    { neuron: 16, title: 'vehicles, then arches, then houses', text: 'Vehicles and puffy animals at first, then arches and rainbows (all rainbows at step 3,359, when the eye is 85% right), and houses at the end.' },
  ];
  const manifest = {
    format: 'tiny-eye-v1',
    ...rest,
    weights: 'tiny-eye.bin',
    sample: sampleInfo,
    names: namesOf(src.run),
    growth: {
      file: 'tiny-eye-growth.bin', bytes: growth.length,
      weights: { dtype: 'float16 little-endian', offset: 0, bytesEach: wBytes, note: 'the same tensors in the same places as tiny-eye.bin' },
      summary: { offset: earlier.length * wBytes, bytesEach: sumBytes, stage: STAGE, neurons: H, top: TOP, layout, classZScale: CLASS_Z_SCALE,
        note: 'last-stage neurons over the sample set: deciles are the 10%..90% points; classZ is (class mean - mean) / sd times classZScale; top/bottom are sample doodle numbers, highest (lowest) first' },
      snapshots: earlier.map((s, k) => stop(s, results[k])),
      final: stop(last, results[results.length - 1]),
      stories,
    },
    squeezed: 'tiny-eye-sq8.json',
  };
  writeFileSync(join(ROOT, 'eye', 'tiny-eye.json'), JSON.stringify(manifest, null, 1) + '\n');
  copyFileSync(join(EXPORT, 'tiny-eye.bin'), join(ROOT, 'eye', 'tiny-eye.bin'));
  copyFileSync(join(EXPORT, 'sample.bin'), join(ROOT, 'eye', 'tiny-eye-sample.bin'));

  // the squeezed eye
  const SQ = join(DATA, 'export-sq8');
  if (!existsSync(join(SQ, 'tiny-eye.json'))) throw new Error('export the squeezed eye first: python -I tools/eye/export.py eye80k-sq8 --weights-only --device cpu --out ' + SQ);
  const sq = JSON.parse(readFileSync(join(SQ, 'tiny-eye.json'), 'utf8'));
  const { snapshots: _s, snapshot_dtype: _d, ...sqRest } = sq;
  const sqManifest = { format: 'tiny-eye-v1', ...sqRest, name: 'Tiny Eye, squeezed', weights: 'tiny-eye-sq8.bin', sample: sampleInfo, names: namesOf(sq.run),
    squeezes: { eye: src.run, note: 'the same eye and training with its last stage cut from 64 channels and 256 neurons to 8 and 8, and no wide bypass: everything the eye sees goes through 8 numbers' } };
  writeFileSync(join(ROOT, 'eye', 'tiny-eye-sq8.json'), JSON.stringify(sqManifest, null, 1) + '\n');
  copyFileSync(join(SQ, 'tiny-eye.bin'), join(ROOT, 'eye', 'tiny-eye-sq8.bin'));
  // the featured stories, as the sample set sees them: the top 16's kinds at a few stops
  const kinds = (r, j) => { const c = {}; for (let k = 0; k < TOP; k++) { const n = src.classes[labels[r.top[j * TOP + k]]]; c[n] = (c[n] || 0) + 1; }
    return Object.entries(c).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([n, v]) => n + ' ' + v).join(', '); };
  for (const st of stories) console.log(`  neuron ${st.neuron}: ` + [7, 10, 13, 15, 16, 17, 18, 19].map(k => `${(jobs[k].s.step)}: ${kinds(results[k], st.neuron)}`).join(' | '));
  console.log(`eye/: tiny-eye.json, tiny-eye.bin, tiny-eye-sample.bin, tiny-eye-growth.bin (${(growth.length / 1e6).toFixed(2)} MB: ${earlier.length} snapshots + summaries), ` +
    `tiny-eye-sq8.json/.bin; settled at each stop: ${manifest.growth.snapshots.map(s => Math.round(100 * s.settled)).join(' ')} (${((Date.now() - t0) / 1000).toFixed(0)} s)`);
}
