// microscope/worker.js — Session A: all model computation for the Microscope runs here,
// off the main thread. Holds the loaded model, tokenizer, and the baseline capture;
// answers slice queries (lens, neurons) without re-running the model.
// NOTE for the artifact edition: this file + the src imports get inlined into the
// single-file HTML and the worker is built from a Blob (no imports); the dev edition
// keeps real modules so jsdom tests exercise identical code.
// The fast engine (src/batch.js) does the reading: layer by layer over the whole story,
// split across a pool of worker threads when the browser allows it. Its results are
// bit-identical to the one-piece-at-a-time capture.js / steer.js path (tests E14), so
// nothing the Microscope shows depends on how many cores did the work. Pool calls are
// asynchronous, so messages are handled one at a time, in order (the queue below).
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { neuronContrast } from '../src/capture.js';
import { steerKey } from '../src/steer.js';
import { atlasFingerprint } from '../src/neuron-atlas.js';
import { readJlens } from '../src/jlens.js';
import { Pool, captureFast, continueFast, lensLogits, topIndices } from '../src/batch.js';

let model = null, tok = null, config = null;
let jl = null;   // the J-lens for the loaded mind (src/jlens.js), or null
let baseCap = null, baseIds = null;
// the worker pool for the loaded mind (a promise: it starts while the page settles);
// resolves to null where workers can't start, or for minds too small to gain from it
let poolP = Promise.resolve(null);
const POOL_MIN_DIM = 256;
// one new piece at a time pays a round trip per layer, worth it only for big minds
const genPool = async () => (config.dim >= 512 ? poolP : null);
// One ablated capture serves every ablation-aware instrument (lens, neurons, neuron-acts,
// and the ablated run itself), keyed by the sorted muted-head set over the SAME story, so
// switching instruments under a fixed ablation recomputes nothing. Steered neurons
// (m.steer: [{layer, neuron, value}], value 0 = muted) join the same key: the whole
// microscope then shows the steered model. No steer -> the original head-only path.
let ablCap = null, ablKey = null;
async function getAblCap(list, steer = []) {
  if (!baseCap) throw new Error('run the full model first');
  const key = steerKey(list, steer);
  if (key !== ablKey) {
    ablCap = await captureFast(model, baseIds, { topK: 12, heads: list, steer, pool: await poolP });
    ablKey = key;
  }
  return ablCap;
}
const hasAbl = m => (m.ablate && m.ablate.length) || (m.steer && m.steer.length);
const ablOf = m => getAblCap(m.ablate || [], m.steer || []);
// greedy continuations of the current story; the unsteered one is cached per story
let baseCont = null;
const CONT_STEPS = 30;

function pieceOf(ids, i) {
  if (ids[i] === 1) return '(start)';
  return tok.decodePiece(i > 0 ? ids[i - 1] : 1, ids[i]);
}

function packRun(cap, ids, light) {
  const n = ids.length;
  const topK = cap.topK.map((row, p) => row.map(({ token, logit }) => ({
    token, logit,
    piece: tok.decodePiece(ids[p], token),
    prob: Math.exp(logit - cap.logZ[p]),
  })));
  const out = {
    n, ids: [...ids],
    pieces: ids.map((_, i) => pieceOf(ids, i)),
    topK,
    nextProb: [...cap.nextProb],
    config: { nLayers: config.nLayers, nHeads: config.nHeads, hiddenDim: config.hiddenDim,
              dim: config.dim, vocabSize: config.vocabSize, seqLen: config.seqLen },
    bytes: cap.bytes(),
  };
  if (!light) {
    const att = new Float32Array(cap.attention); // copy; cap keeps its own
    out.attention = att.buffer;
  }
  return out;
}

let queue = Promise.resolve();
self.onmessage = (e) => { queue = queue.then(() => handle(e.data)); };
self.whenIdle = () => queue;   // for tests that drive this worker in-process

async function handle(m) {
  try {
    if (m.type === 'load') {
      const ck = readCheckpoint(m.ckptBuf);
      config = ck.config;
      tok = new Tokenizer(m.tokBuf, config.vocabSize);
      model = createModel(ck);
      baseCap = null; baseIds = null; ablCap = null; ablKey = null; baseCont = null;
      poolP.then(p => p && p.dispose());
      poolP = config.dim >= POOL_MIN_DIM ? Pool.create(model).catch(() => null) : Promise.resolve(null);
      // the "thinking ahead" lens fitted for this mind, if one came with it (and fits it)
      jl = null;
      if (m.lensBuf) { try { const L = readJlens(m.lensBuf); if (L.dModel === config.dim) jl = L; } catch (_) { jl = null; } }
      self.postMessage({ type: 'loaded', config: {
        dim: config.dim, nLayers: config.nLayers, nHeads: config.nHeads,
        nKvHeads: config.nKvHeads, hiddenDim: config.hiddenDim,
        vocabSize: config.vocabSize, seqLen: config.seqLen,
      }, fingerprint: atlasFingerprint(config, ck.weights), lensLayers: jl ? jl.layers : [] });
      const p = await poolP;   // say how the reading will be done once the pool has (or hasn't) started
      self.postMessage({ type: 'engine', cores: p ? p.K : 1 });
    } else if (m.type === 'unload') {
      poolP.then(p => p && p.dispose()); poolP = Promise.resolve(null);
      model = null; baseCap = null; baseIds = null; ablCap = null; ablKey = null; baseCont = null;
    } else if (m.type === 'run') {
      if (!model) throw new Error('no model loaded');
      if (hasAbl(m)) {
        // ablated run: reuse the base story, return a FULL capture (attention too) so
        // every instrument can switch to the damaged model's internals consistently.
        const cap = await ablOf(m);
        const payload = packRun(cap, baseIds, false);
        payload.type = 'run-ablated';
        payload.ablate = m.ablate || [];
        payload.steer = m.steer || [];
        self.postMessage(payload, [payload.attention]);
      } else {
        let ids = tok.encode(m.text, true, false);
        let truncated = false;
        if (ids.length > config.seqLen) { ids = ids.slice(0, config.seqLen); truncated = true; }
        const cap = await captureFast(model, ids, {
          topK: 12, pool: await poolP,
          onProgress: (done, total) => self.postMessage({ type: 'progress', pos: done, n: total, pieces: ids.length }),
        });
        baseCap = cap; baseIds = ids; ablCap = null; ablKey = null; baseCont = null; // a new base story invalidates the ablation cache
        const payload = packRun(cap, ids, false);
        payload.type = 'run-done';
        payload.truncated = truncated;
        payload.ablate = null;
        self.postMessage(payload, [payload.attention]);
      }
    } else if (m.type === 'lens') {
      if (!baseCap) throw new Error('run a story first');
      // ablation-aware lens: when heads are muted, project the ABLATED residual stream
      // (the whole point — the lens shows what the ablation did to the forming
      // prediction). No ablation -> the original un-ablated baseCap, unchanged.
      const ablate = hasAbl(m) ? (m.ablate || []) : null;
      const cap = ablate ? await ablOf(m) : baseCap;
      const rows = [];
      const nextTok = m.pos + 1 < baseIds.length ? baseIds[m.pos + 1] : null;
      // every lens row in one pass through the classifier (same numbers as logitLens / jLensLogits)
      const lens = await lensLogits(model, cap, m.pos, jl, await poolP);
      for (let l = 0; l < config.nLayers; l++) {
        for (const phase of ['attn', 'ffn']) {
          const logits = lens.row(2 * l + (phase === 'ffn' ? 1 : 0));
          let max = -Infinity;
          for (const v of logits) if (v > max) max = v;
          let sum = 0;
          for (const v of logits) sum += Math.exp(v - max);
          const logZ = max + Math.log(sum);
          const idx = topIndices(logits, 3);
          rows.push({
            layer: l, phase,
            top: idx.map(i => ({ piece: tok.decodePiece(baseIds[m.pos], i), prob: Math.exp(logits[i] - logZ) })),
            nextProb: nextTok !== null ? Math.exp(logits[nextTok] - logZ) : null,
          });
        }
      }
      // thinking ahead: the J-lens at the layers it was fitted for (word pieces only, ranked)
      const ahead = [];
      const pick = (order) => {
        const words = [];
        for (const i of order) { const p = tok.decodePiece(baseIds[m.pos], i); if (/[A-Za-z]/.test(p)) words.push(p); if (words.length >= 6) break; }
        return words;
      };
      lens.jLayers.forEach((l, i) => {
        const lg = lens.row(2 * config.nLayers + i);
        let words = pick(topIndices(lg, 200));   // the head of the full ranking, without sorting it all
        if (words.length < 6) words = pick([...lg.keys()].sort((a, b) => lg[b] - lg[a]));
        ahead.push({ layer: l, words });
      });
      self.postMessage({ type: 'lens-done', pos: m.pos, rows, ahead, ablate: ablate || null, steer: m.steer || [],
        nextPiece: nextTok !== null ? tok.decodePiece(baseIds[m.pos], nextTok) : null });
    } else if (m.type === 'neurons') {
      if (!baseCap) throw new Error('run a story first');
      const cap = hasAbl(m) ? await ablOf(m) : baseCap;  // which-neurons-light-up, in the ablated model when muted
      const top = neuronContrast(cap, m.spanStart, m.spanEnd, 12);
      self.postMessage({ type: 'neurons-done', top, spanStart: m.spanStart, spanEnd: m.spanEnd, ablate: m.ablate || null, steer: m.steer || [] });
    } else if (m.type === 'neuron-acts') {
      if (!baseCap) throw new Error('run a story first');
      const cap = hasAbl(m) ? await ablOf(m) : baseCap;
      const n = cap.nPos, HD = config.hiddenDim;
      const acts = new Float32Array(n);
      let hi = 0, lo = 0;   // signed extremes over the story (the card's steer fallback)
      for (let p = 0; p < n; p++) {
        const v = cap.ffn[(m.layer * n + p) * HD + m.neuron];
        acts[p] = Math.abs(v);
        if (p > 0 && v > hi) hi = v;
        if (p > 0 && v < lo) lo = v;
      }
      self.postMessage({ type: 'neuron-acts-done', layer: m.layer, neuron: m.neuron, acts: acts.buffer, hi, lo, ablate: m.ablate || null, steer: m.steer || [] }, [acts.buffer]);
    } else if (m.type === 'continue') {
      // "how does the story go on?" — the full model's greedy continuation (cached per
      // story) beside the continuation under the current mutes + steers
      if (!baseCap) throw new Error('run a story first');
      const piecesOf = (gen) => gen.map((t, i) => tok.decodePiece(i ? gen[i - 1] : baseIds[baseIds.length - 1], t));
      const steps = Math.min(CONT_STEPS, config.seqLen - baseIds.length);
      if (steps < 1) throw new Error('the story fills the whole model window — shorten it to see how it goes on');
      const changed = hasAbl(m);
      const total = changed ? 2 * steps : steps;
      const pool = await genPool();
      if (!baseCont) baseCont = await continueFast(model, baseIds, steps, { pool,
        onToken: (i) => self.postMessage({ type: 'continue-progress', done: i, total }) });
      const gen = changed ? await continueFast(model, baseIds, steps, { heads: m.ablate || [], steer: m.steer || [], pool,
        onToken: (i) => self.postMessage({ type: 'continue-progress', done: steps + i, total }) }) : baseCont;
      self.postMessage({ type: 'continue-done', base: piecesOf(baseCont), steered: piecesOf(gen),
        changed: !!changed, ablate: m.ablate || [], steer: m.steer || [] });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
}
