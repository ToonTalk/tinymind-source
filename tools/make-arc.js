// make-arc.js — precompute checkpoint snapshots along a training arc so the Nursery
// scrubber (later claude.ai session; sees only the outputs, never this tool) can jump
// through developmental time. See tiny-mind-arc-kickoff.md.
//
//   node tools/make-arc.js B                 fine-tune arc (E5 recipe), ~25 min
//   node tools/make-arc.js A [--lr=1e-3]     from-scratch arc, overnight, truncatable
//   node tools/make-arc.js probeA            short lr probe for Arc A
//   node tools/make-arc.js mini --out=DIR    30-step CI arc (tests)
//   flags: --resume  --max-step=N  --out=DIR  --lr=X
//
// Every snapshot writes: arc{X}-step{NNNNN}.bin (legacy fp32 checkpoint) +
// adam-{NNNNN}.bin (m,v moments raw fp32) + state-{NNNNN}.json (RNG state, step,
// recent losses), then rewrites arc-manifest.json LAST (commit marker) — the manifest
// is valid at every prefix, so the run can be stopped at any snapshot.
// Resume is byte-exact: fp32 params/moments round-trip exactly, the batch RNG state
// is restored, and everything else is deterministic single-threaded arithmetic.
import { readFile, writeFile, readdir, mkdir, access } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint, writeCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy } from '../src/model.js';
import { Trainer, rng, packStories } from '../src/train.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// ---------------------------------------------------------------- configs
const ARCS = {
  B: {
    file: 'arcB', outDir: 'arc',
    note: 'fine-tune arc: pretrained stories260K learning Zog with the E5 recipe (replay 50%), deliberately continued past the held-out floor (~step 80) into the parroting regime',
    start: { ckpt: 'models/stories260K.bin' },
    B: 8, T: 192, lr: 3e-4, replayRatio: 0.5, seed: 7,
    snapshots: [0, 10, 20, 40, 80, 120, 200, 400, 600],
    data: 'zog+replay', evals: ['zogHeld', 'base'],
    prompts: ['Once upon a time', 'One day Zog'],
  },
  A: {
    file: 'arcA', outDir: 'arc/newborn',
    note: 'from-scratch arc: random init growing up on ~2000 TinyStories. Honest expectation: fully-trained stories260K is only barely grammatical, so the endpoint is story-shaped with frequent errors — the STAGES are the exhibit',
    start: { random: 123 },
    B: 8, T: 192, lr: 1e-3, replayRatio: 0, seed: 11,
    snapshots: [0, 20, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 20000],
    data: 'general', evals: ['base'],
    prompts: ['Once upon a time'],
  },
  mini: {
    file: 'arcM', outDir: null, // --out required
    note: 'CI mini-arc exercising the tool end to end',
    start: { random: 99 },
    B: 2, T: 48, lr: 1e-3, replayRatio: 0, seed: 5,
    snapshots: [0, 10, 20, 30],
    data: 'mini', evals: ['base'],
    prompts: ['Once upon a time'],
  },
};

// ---------------------------------------------------------------- helpers
function parseArgs(argv) {
  const out = { mode: argv[0], resume: false, lr: null, out: null, maxStep: null };
  for (const a of argv.slice(1)) {
    if (a === '--resume') out.resume = true;
    else if (a.startsWith('--lr=')) out.lr = Number(a.slice(5));
    else if (a.startsWith('--out=')) out.out = a.slice(6);
    else if (a.startsWith('--max-step=')) out.maxStep = Number(a.slice(11));
    else throw new Error('unknown flag ' + a);
  }
  return out;
}

// same algorithm as nursery/worker.js randomWeights (kickoff: "worker's randomWeights")
function randomWeights(c, seed) {
  const r = rng(seed);
  const gauss = () => { let s = 0; for (let i = 0; i < 12; i++) s += r(); return s - 6; };
  const mk = (n, sc) => Float32Array.from({ length: n }, () => gauss() * sc);
  const sc = 0.06;
  return {
    tokenEmbedding: mk(c.vocabSize * c.dim, sc),
    rmsAtt: new Float32Array(c.nLayers * c.dim).fill(1),
    wq: mk(c.nLayers * c.dim * c.dim, sc), wk: mk(c.nLayers * c.dim * c.kvDim, sc),
    wv: mk(c.nLayers * c.dim * c.kvDim, sc), wo: mk(c.nLayers * c.dim * c.dim, sc),
    rmsFfn: new Float32Array(c.nLayers * c.dim).fill(1),
    w1: mk(c.nLayers * c.hiddenDim * c.dim, sc), w2: mk(c.nLayers * c.dim * c.hiddenDim, sc),
    w3: mk(c.nLayers * c.hiddenDim * c.dim, sc),
    rmsFinal: new Float32Array(c.dim).fill(1),
  };
}

function makeEvalBatch(stream, B, T) {
  const batch = new Int32Array(B * (T + 1));
  const span = stream.length - (T + 1);
  for (let b = 0; b < B; b++) {
    const start = Math.floor((span * b) / Math.max(1, B - 1));
    batch.set(stream.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

// ALWAYS consumes exactly 2 rands per row so the sequence is schedule-independent
function nextBatch(sources, B, T, rand) {
  const batch = new Int32Array(B * (T + 1));
  for (let b = 0; b < B; b++) {
    const pick = rand();
    let acc = 0, src = sources[sources.length - 1].stream;
    for (const s of sources) { acc += s.prob; if (pick < acc) { src = s.stream; break; } }
    const start = Math.floor(rand() * (src.length - T - 1));
    batch.set(src.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

function trimSample(text) {
  const m = text.match(/^[\s\S]*[.!?]["”']?(?=\s|$)/);
  return m ? m[0] : text; // random-init babble has no sentences — keep it raw
}

const pad = s => String(s).padStart(5, '0');
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
  const f = new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
  let p = 0;
  for (const k of ADAM_KEYS) { tr.m[k].set(f.subarray(p, p + tr.m[k].length)); p += tr.m[k].length; }
  for (const k of ADAM_KEYS) { tr.v[k].set(f.subarray(p, p + tr.v[k].length)); p += tr.v[k].length; }
  if (p !== f.length) throw new Error('adam moments size mismatch');
}

// ---------------------------------------------------------------- data
// returns { tok, sources: [{stream, prob}], zogEval|null, baseEval }
async function loadData(kind, cfg) {
  const tok = new Tokenizer(await readFile(join(ROOT, 'models', 'tok512.bin')), 512);
  const enc = s => tok.encode(s.trim(), false, false);
  const replay = JSON.parse(await readFile(join(ROOT, 'corpus', 'replay', 'replay.json'), 'utf8'));

  if (kind === 'zog+replay') {
    const zogDir = join(ROOT, 'corpus', 'zog');
    const files = (await readdir(zogDir)).sort();
    const read = p => Promise.all(files.filter(f => f.startsWith(p)).map(f => readFile(join(zogDir, f), 'utf8')));
    const zog = packStories((await read('train-')).map(enc));
    const held = packStories((await read('heldout-')).map(enc));
    return {
      tok,
      sources: [{ stream: packStories(replay.slice(0, 150).map(enc)), prob: cfg.replayRatio },
                { stream: zog, prob: 1 - cfg.replayRatio }],
      zogEval: makeEvalBatch(held, cfg.B, cfg.T),
      baseEval: makeEvalBatch(packStories(replay.slice(200, 260).map(enc)), cfg.B, cfg.T),
    };
  }
  if (kind === 'general') {
    // ~2000 TinyStories, tokenized once and cached; tail 12K tokens held out for eval
    const cachePath = join(ROOT, 'arc', 'cache-tinystories-2000.i32');
    await mkdir(join(ROOT, 'arc'), { recursive: true });
    let stream;
    try {
      await access(cachePath);
      const buf = await readFile(cachePath);
      stream = new Int32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4).slice();
      console.log(`data: cached general stream, ${stream.length} tokens`);
    } catch {
      console.log('data: tokenizing ~2060 TinyStories (one-time, cached after)…');
      const valid = await readFile(join(ROOT, 'models', 'TinyStories-valid.txt'), 'utf8');
      const stories = valid.split('<|endoftext|>').map(s => s.trim())
        .filter(s => s.length > 200 && s.length < 2500).slice(0, 2060);
      stream = packStories(stories.map(enc));
      await writeFile(cachePath, Buffer.from(stream.buffer, stream.byteOffset, stream.byteLength));
      console.log(`data: ${stories.length} stories -> ${stream.length} tokens, cached`);
    }
    return {
      tok,
      sources: [{ stream: stream.subarray(0, stream.length - 12000), prob: 1 }],
      zogEval: null,
      baseEval: makeEvalBatch(stream.subarray(stream.length - 12000), cfg.B, cfg.T),
    };
  }
  if (kind === 'mini') {
    return {
      tok,
      sources: [{ stream: packStories(replay.slice(0, 30).map(enc)), prob: 1 }],
      zogEval: null,
      baseEval: makeEvalBatch(packStories(replay.slice(30, 40).map(enc)), cfg.B, cfg.T),
    };
  }
  throw new Error('unknown data kind ' + kind);
}

// ---------------------------------------------------------------- main run
async function run(cfg, flags) {
  const outDir = flags.out ? flags.out : (cfg.outDir ? join(ROOT, cfg.outDir) : null);
  if (!outDir) throw new Error('this arc needs --out=DIR');
  await mkdir(outDir, { recursive: true });
  const lr = flags.lr ?? cfg.lr;
  let schedule = [...cfg.snapshots];
  if (flags.maxStep !== null) schedule = schedule.filter(s => s <= flags.maxStep);

  const baseCk = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K.bin')));
  const config = baseCk.config;
  const data = await loadData(cfg.data, cfg);

  const start = cfg.start.random !== undefined
    ? { config, weights: randomWeights(config, cfg.start.random) }
    : readCheckpoint(await readFile(join(ROOT, cfg.start.ckpt)));

  const rand = rng(cfg.seed);
  let trainer = new Trainer(start, { B: cfg.B, T: cfg.T, lr });
  let step = 0;
  let recent = [];
  let manifest = {
    arc: cfg.file, note: cfg.note,
    config: { B: cfg.B, T: cfg.T, lr, replayRatio: cfg.replayRatio, seed: cfg.seed,
              start: cfg.start.random !== undefined ? `random:${cfg.start.random}` : cfg.start.ckpt },
    snapshots: [],
  };

  if (flags.resume) {
    const states = (await readdir(outDir)).filter(f => /^state-\d{5}\.json$/.test(f)).sort();
    if (!states.length) throw new Error('--resume: no state files in ' + outDir);
    const st = JSON.parse(await readFile(join(outDir, states[states.length - 1]), 'utf8'));
    const ck = readCheckpoint(await readFile(join(outDir, `${cfg.file}-step${pad(st.step)}.bin`)));
    trainer = new Trainer(ck, { B: cfg.B, T: cfg.T, lr: st.lr ?? lr });
    momentsFromBuffer(trainer, await readFile(join(outDir, `adam-${pad(st.step)}.bin`)));
    trainer.stepCount = st.stepCount;
    rand.setState(st.rngState);
    step = st.step;
    recent = st.recentLosses ?? [];
    manifest = JSON.parse(await readFile(join(outDir, 'arc-manifest.json'), 'utf8'));
    manifest.snapshots = manifest.snapshots.filter(s => s.step <= step);
    console.log(`resumed at step ${step} (rng state ${st.rngState})`);
  }

  async function snapshot() {
    const file = `${cfg.file}-step${pad(step)}.bin`;
    const bin = writeCheckpoint(config, trainer.modelWeights());
    await writeFile(join(outDir, file), Buffer.from(bin));
    await writeFile(join(outDir, `adam-${pad(step)}.bin`), momentsToBuffer(trainer));
    await writeFile(join(outDir, `state-${pad(step)}.json`), JSON.stringify({
      arc: cfg.file, step, stepCount: trainer.stepCount, rngState: rand.getState(),
      recentLosses: recent.slice(-20), lr,
    }));
    const entry = {
      step, file,
      trainLoss: recent.length
        ? +(recent.slice(-20).reduce((a, b) => a + b, 0) / Math.min(20, recent.length)).toFixed(4)
        : null,
    };
    if (data.zogEval && cfg.evals.includes('zogHeld')) entry.zogHeld = +trainer.evalLoss(data.zogEval).toFixed(4);
    if (data.baseEval && cfg.evals.includes('base')) entry.baseLoss = +trainer.evalLoss(data.baseEval).toFixed(4);
    entry.samples = {};
    const model = createModel({ config, weights: trainer.modelWeights() });
    for (const p of cfg.prompts) entry.samples[p] = trimSample(generateGreedy(model, data.tok, p, 160).text);
    manifest.snapshots.push(entry);
    await writeFile(join(outDir, 'arc-manifest.json'), JSON.stringify(manifest, null, 1)); // LAST
    console.log(`snapshot @${step}: train ${entry.trainLoss ?? '—'}` +
      (entry.zogHeld !== undefined ? `  zogHeld ${entry.zogHeld}` : '') +
      (entry.baseLoss !== undefined ? `  base ${entry.baseLoss}` : '') +
      `  | "${entry.samples[cfg.prompts[0]].slice(0, 60).replace(/\n/g, ' ')}"`);
  }

  if (!flags.resume && schedule.includes(0)) await snapshot();

  const target = schedule[schedule.length - 1];
  let msSum = 0, msN = 0;
  while (step < target) {
    const batch = nextBatch(data.sources, cfg.B, cfg.T, rand);
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
  console.log(`arc ${cfg.file} complete through step ${step}: ${manifest.snapshots.length} snapshots in ${outDir}`);
}

// ---------------------------------------------------------------- lr probe (Arc A)
async function probeA() {
  const cfg = ARCS.A;
  const baseCk = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K.bin')));
  const config = baseCk.config;
  const data = await loadData('general', cfg);
  for (const lr of [1e-3, 3e-4]) {
    const trainer = new Trainer({ config, weights: randomWeights(config, cfg.start.random) },
      { B: cfg.B, T: cfg.T, lr });
    const rand = rng(cfg.seed);
    console.log(`\n--- probe lr=${lr} (200 steps from random init)`);
    for (let s = 1; s <= 200; s++) {
      const r = trainer.step(nextBatch(data.sources, cfg.B, cfg.T, rand));
      if (s % 25 === 0) console.log(`step ${s}: train ${r.loss.toFixed(4)}  base ${trainer.evalLoss(data.baseEval).toFixed(4)}`);
    }
    const model = createModel({ config, weights: trainer.modelWeights() });
    console.log('sample:', generateGreedy(model, data.tok, 'Once upon a time', 80).text.replace(/\n/g, ' ').slice(0, 140));
  }
  console.log('\npick the lr with the lower base loss that is not diverging; pass it as --lr= to arc A');
}

// ---------------------------------------------------------------- entry
const flags = parseArgs(process.argv.slice(2));
if (flags.mode === 'probeA') await probeA();
else if (ARCS[flags.mode]) await run(ARCS[flags.mode], flags);
else { console.error('usage: node tools/make-arc.js <B|A|mini|probeA> [--resume] [--lr=] [--out=] [--max-step=]'); process.exit(1); }
