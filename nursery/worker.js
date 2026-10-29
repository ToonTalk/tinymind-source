// nursery/worker.js — Session B: training-room engine. Holds the tokenizer, the data
// streams, and a live Trainer; runs the fine-tune loop cooperatively (one step per
// macrotask) so pause/sample/export messages interleave cleanly. Where the browser allows,
// each step's sequences are worked on by a TrainPool of worker threads (src/train.js),
// which gives exactly the same weights as training on this thread alone.
// Start points: the pretrained 260K trainee, or "empty head" (random init) for the
// full gibberish→grammar developmental arc.
import { readCheckpoint, writeCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy, forward } from '../src/model.js';
import { Trainer, TrainPool, rng, packStories } from '../src/train.js';
import { EMBEDDED_BASE_STORIES } from '../src/embed-data.js';

let tok = null, config = null, baseCkpt = null;
let trainer = null, trainOpts = null, rand = null;
let pool = null;   // TrainPool for the current trainer, or null (one core)
function dropPool() { if (pool) pool.dispose(); pool = null; }
// a step / eval on the pool, falling back to this thread for good if the pool fails
async function trainStep(tr, batch) {
  const p = pool;
  if (p) { try { return await tr.stepWith(p, batch); } catch (e) { if (pool === p) dropPool(); } }
  return tr.step(batch);
}
async function evalLoss(batch) {
  const p = pool, tr = trainer;
  if (p) { try { return await tr.evalLossWith(p, batch); } catch (e) { if (pool === p) dropPool(); } }
  return tr.evalLoss(batch);
}
let loopGen = 0;   // a fresh start begins a new loop; an older one stops at its next await
let zogStream = null, replayStream = null, zogEval = null, baseEval = null;
let running = false, step = 0, samplePromptIndex = 0;
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
  // Prefer the live trainee's weights; fall back to the loaded checkpoint's own
  // weights so a time-machine scrub (which loads via init, trainer = null) can still
  // write a fresh story from a custom prefix. No trainer + no checkpoint -> null.
  const weights = trainer ? trainer.modelWeights() : (baseCkpt && baseCkpt.weights);
  if (!weights) return null;
  const model = createModel({ config, weights });
  let text = generateGreedy(model, tok, promptText, n).text;
  // trim to the last complete sentence so the arc timeline never cuts mid-word
  const m = text.match(/^[\s\S]*[.!?]["”']?(?=\s|$)/);
  if (m && m[0].length > promptText.length) text = m[0];
  return text;
}

function nextTrainingPrompt() {
  const prompts = trainOpts && Array.isArray(trainOpts.prompts)
    ? trainOpts.prompts.filter(p => typeof p === 'string' && p.trim())
    : [];
  const choices = prompts.length ? prompts : [trainOpts && trainOpts.prompt].filter(Boolean);
  if (!choices.length) return 'Once upon a time,';
  const prompt = choices[samplePromptIndex % choices.length];
  samplePromptIndex++;
  return prompt;
}

async function doEval() {
  const out = { type: 'eval', step };
  if (zogEval) out.zogHeld = await evalLoss(zogEval);
  if (baseEval) out.base = await evalLoss(baseEval);
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
  const gen = ++loopGen;
  const { B, T, replayRatio, maxSteps, evalEvery, sampleEvery } = trainOpts;
  while (running && step < maxSteps && gen === loopGen) {
    const batch = new Int32Array(B * (T + 1));
    for (let b = 0; b < B; b++) {
      const src = (replayStream && rand() < replayRatio) ? replayStream : zogStream;
      const start = Math.floor(rand() * (src.length - T - 1));
      batch.set(src.subarray(start, start + T + 1), b * (T + 1));
    }
    const tr = trainer, r = await trainStep(tr, batch);
    if (gen !== loopGen || tr !== trainer) return;   // a fresh start took over while this step ran
    step++;
    self.postMessage({ type: 'step', step, loss: r.loss, ms: r.ms, gradNorm: r.gradNorm, cores: pool ? pool.K : 1 });
    if (step % evalEvery === 0 || step === maxSteps) await doEval();
    if (step % sampleEvery === 0 || step === maxSteps) {
      self.postMessage({ type: 'sample', step, text: sampleText(nextTrainingPrompt()) });
    }
    await new Promise(res => setTimeout(res, 0)); // let pause/export messages in
  }
  if (gen !== loopGen) return;
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
      trainer = null; step = 0; running = false; dropPool();
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
      const baseSlice = m.replayStories.length > 200
        ? packStories(m.replayStories.slice(200, 260).map(enc))   // in-distribution held-out from a real replay corpus
        : packStories(EMBEDDED_BASE_STORIES.map(enc));            // fallback so the forgetting line always draws
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
        dropPool();
        pool = await TrainPool.create(trainer).catch(() => null);
        rand = rng(m.opts.seed ?? 7);
        samplePromptIndex = 0;
        step = 0; zogHeldMin = Infinity; evalsAboveMin = 0;
        await doEval();
        self.postMessage({ type: 'sample', step: 0, text: sampleText(nextTrainingPrompt()) });
      } else {
        trainer.opt.lr = m.opts.lr; // allow live lr tweak on resume
      }
      running = true;
      loop();
    } else if (m.type === 'pause') {
      running = false;
    } else if (m.type === 'sample-now') {
      const text = sampleText(m.prompt ?? (trainOpts && trainOpts.prompt));
      if (text != null) self.postMessage({ type: 'sample', step, text });
    } else if (m.type === 'probe-fact') {
      // vg guide pre-flight (additive): piece-count a word + base-model p(word | context).
      // Existing forward machinery only; no training or parity path is touched.
      if (!tok || !baseCkpt) throw new Error('init first');
      const ids = tok.encode(String(m.word || '').trim(), false, false);
      const sp = tok.lookup.get(' ');
      const pieces = Math.max(1, ids.length - (ids.length > 1 && ids[0] === sp ? 1 : 0));
      let prob = null;
      if (m.context) {
        const ctxIds = tok.encode(m.context, true, false);
        const all = tok.encode(m.context + ' ' + String(m.word || '').trim(), true, false);
        let seamOk = true;
        for (let i = 0; i < ctxIds.length - 1; i++) if (all[i] !== ctxIds[i]) { seamOk = false; break; }
        if (seamOk) {
          const mdl = createModel(baseCkpt);
          let p = 1;
          for (let pos = 0; pos < all.length - 1; pos++) {
            const logits = forward(mdl, all[pos], pos);
            if (pos >= ctxIds.length - 1) {
              let mx = -Infinity; for (let i = 0; i < logits.length; i++) if (logits[i] > mx) mx = logits[i];
              let sm = 0; for (let i = 0; i < logits.length; i++) sm += Math.exp(logits[i] - mx);
              p *= Math.exp(logits[all[pos + 1]] - (mx + Math.log(sm)));
            }
          }
          prob = p;
        }
      }
      const pieceTexts = ids.slice(ids.length > 1 && ids[0] === sp ? 1 : 0).map(i => tok.vocab[i]);
      self.postMessage({ type: 'probe-fact-done', id: m.id, word: m.word, pieces, pieceTexts, prob });
    } else if (m.type === 'export') {
      if (!trainer) throw new Error('nothing trained yet');
      const bin = writeCheckpoint(config, trainer.modelWeights());
      self.postMessage({ type: 'export-done', bin, step, purpose: m.purpose || 'download' }, [bin]);
    }
  } catch (err) {
    running = false;
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
