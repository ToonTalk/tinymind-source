// nursery/worker.js — Session B: training-room engine. Holds the tokenizer, the data
// streams, and a live Trainer; runs the fine-tune loop cooperatively (one step per
// macrotask) so pause/sample/export messages interleave cleanly.
// Start points: the pretrained 260K trainee, or "empty head" (random init) for the
// full gibberish→grammar developmental arc.
import { readCheckpoint, writeCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy } from '../src/model.js';
import { Trainer, rng, packStories } from '../src/train.js';

let tok = null, config = null, baseCkpt = null;
let trainer = null, trainOpts = null, rand = null;
let zogStream = null, replayStream = null, zogEval = null, baseEval = null;
let running = false, step = 0;
let zogHeldMin = Infinity, evalsAboveMin = 0;

function randomWeights(c, seed) {
  // small-transformer init: N(0, 0.02ish) for matrices, ones for norms
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
  if (!stream || stream.length < T + 2) return null;
  const batch = new Int32Array(B * (T + 1));
  const span = stream.length - (T + 1);
  for (let b = 0; b < B; b++) {
    const start = Math.floor((span * b) / Math.max(1, B - 1));
    batch.set(stream.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

function sampleText(promptText, n = 200) {
  const model = createModel({ config, weights: trainer.modelWeights() });
  let text = generateGreedy(model, tok, promptText, n).text;
  // trim to the last complete sentence so the arc timeline never cuts mid-word
  const m = text.match(/^[\s\S]*[.!?]["”']?(?=\s|$)/);
  if (m && m[0].length > promptText.length) text = m[0];
  return text;
}

function doEval() {
  const out = { type: 'eval', step };
  if (zogEval) out.zogHeld = trainer.evalLoss(zogEval);
  if (baseEval) out.base = trainer.evalLoss(baseEval);
  if (out.zogHeld !== undefined) {
    if (out.zogHeld < zogHeldMin - 1e-4) { zogHeldMin = out.zogHeld; evalsAboveMin = 0; }
    else if (out.zogHeld > zogHeldMin + 0.02) evalsAboveMin++;
    out.zogHeldMin = zogHeldMin;
    if (evalsAboveMin >= 2) out.arc = 'parroting'; // past the floor: memorizing the corpus
    else if (out.zogHeld <= zogHeldMin + 0.01 && step > 0) out.arc = 'absorbed-ish';
  }
  self.postMessage(out);
}

async function loop() {
  const { B, T, replayRatio, maxSteps, evalEvery, sampleEvery, prompt } = trainOpts;
  while (running && step < maxSteps) {
    const batch = new Int32Array(B * (T + 1));
    for (let b = 0; b < B; b++) {
      const src = (replayStream && rand() < replayRatio) ? replayStream : zogStream;
      const start = Math.floor(rand() * (src.length - T - 1));
      batch.set(src.subarray(start, start + T + 1), b * (T + 1));
    }
    const r = trainer.step(batch);
    step++;
    self.postMessage({ type: 'step', step, loss: r.loss, ms: r.ms, gradNorm: r.gradNorm });
    if (step % evalEvery === 0 || step === maxSteps) doEval();
    if (step % sampleEvery === 0 || step === maxSteps) {
      self.postMessage({ type: 'sample', step, text: sampleText(prompt) });
    }
    await new Promise(res => setTimeout(res, 0)); // let pause/export messages in
  }
  running = false;
  self.postMessage({ type: 'idle', step });
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      baseCkpt = readCheckpoint(m.ckptBuf);
      config = baseCkpt.config;
      tok = new Tokenizer(m.tokBuf, config.vocabSize);
      trainer = null; step = 0; running = false;
      self.postMessage({ type: 'ready', config: { dim: config.dim, nLayers: config.nLayers,
        nHeads: config.nHeads, hiddenDim: config.hiddenDim, vocabSize: config.vocabSize,
        seqLen: config.seqLen, maxTokenLength: tok.maxTokenLength } });
    } else if (m.type === 'alphabet') {
      self.postMessage({ type: 'alphabet', pieces: tok.vocab, scores: [...tok.scores] });
    } else if (m.type === 'tokenize-count') {
      // piece usage over the corpus, for the alphabet heat view
      const counts = new Array(config.vocabSize).fill(0);
      for (const s of m.texts) for (const id of tok.encode(s.trim(), false, false)) counts[id]++;
      self.postMessage({ type: 'tokenize-count', counts });
    } else if (m.type === 'set-data') {
      const enc = s => tok.encode(s.trim(), false, false);
      zogStream = m.stories.length ? packStories(m.stories.map(enc)) : null;
      const held = m.heldout.length ? packStories(m.heldout.map(enc)) : null;
      replayStream = m.replayStories.length ? packStories(m.replayStories.slice(0, 150).map(enc)) : null;
      const baseSlice = m.replayStories.length > 200 ? packStories(m.replayStories.slice(200, 260).map(enc)) : null;
      const B = m.B ?? 8, T = m.T ?? 192;
      zogEval = held ? makeEvalBatch(held, B, T) : null;
      baseEval = baseSlice ? makeEvalBatch(baseSlice, B, T) : null;
      self.postMessage({ type: 'data-ready',
        zogTokens: zogStream ? zogStream.length : 0,
        heldTokens: held ? held.length : 0,
        replayTokens: replayStream ? replayStream.length : 0,
        evals: { zog: !!zogEval, base: !!baseEval } });
    } else if (m.type === 'train') {
      if (!zogStream) throw new Error('no corpus yet');
      trainOpts = m.opts;
      if (!trainer || m.opts.fresh) {
        const start = m.opts.startFrom === 'random'
          ? { config, weights: randomWeights(config, 123) }
          : baseCkpt;
        trainer = new Trainer(start, { B: m.opts.B, T: m.opts.T, lr: m.opts.lr });
        rand = rng(m.opts.seed ?? 7);
        step = 0; zogHeldMin = Infinity; evalsAboveMin = 0;
        doEval();
        self.postMessage({ type: 'sample', step: 0, text: sampleText(m.opts.prompt) });
      } else {
        trainer.opt.lr = m.opts.lr; // allow live lr tweak on resume
      }
      running = true;
      loop();
    } else if (m.type === 'pause') {
      running = false;
    } else if (m.type === 'sample-now') {
      if (trainer) self.postMessage({ type: 'sample', step, text: sampleText(m.prompt ?? trainOpts.prompt) });
    } else if (m.type === 'export') {
      if (!trainer) throw new Error('nothing trained yet');
      const bin = writeCheckpoint(config, trainer.modelWeights());
      self.postMessage({ type: 'export-done', bin, step }, [bin]);
    }
  } catch (err) {
    running = false;
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
