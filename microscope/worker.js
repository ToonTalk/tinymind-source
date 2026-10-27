// microscope/worker.js — Session A: all model computation for the Microscope runs here,
// off the main thread. Holds the loaded model, tokenizer, and the baseline capture;
// answers slice queries (lens, neurons) without re-running the model.
// NOTE for the artifact edition: this file + the src imports get inlined into the
// single-file HTML and the worker is built from a Blob (no imports); the dev edition
// keeps real modules so jsdom tests exercise identical code.
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun, logitLens, neuronContrast } from '../src/capture.js';
import { captureSteered, continueSteered, steerKey } from '../src/steer.js';
import { atlasFingerprint } from '../src/neuron-atlas.js';

let model = null, tok = null, config = null;
let baseCap = null, baseIds = null;
// One ablated capture serves every ablation-aware instrument (lens, neurons, neuron-acts,
// and the ablated run itself), keyed by the sorted muted-head set over the SAME story, so
// switching instruments under a fixed ablation recomputes nothing. Steered neurons
// (m.steer: [{layer, neuron, value}], value 0 = muted) join the same key: the whole
// microscope then shows the steered model. No steer -> the original head-only path.
let ablCap = null, ablKey = null;
function getAblCap(list, steer = []) {
  if (!baseCap) throw new Error('run the full model first');
  const key = steerKey(list, steer);
  if (key !== ablKey) {
    ablCap = steer.length ? captureSteered(model, baseIds, { topK: 12, heads: list, steer })
                          : captureRun(model, baseIds, { topK: 12, ablate: new Set(list) });
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

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      const ck = readCheckpoint(m.ckptBuf);
      config = ck.config;
      tok = new Tokenizer(m.tokBuf, config.vocabSize);
      model = createModel(ck);
      baseCap = null; baseIds = null;
      self.postMessage({ type: 'loaded', config: {
        dim: config.dim, nLayers: config.nLayers, nHeads: config.nHeads,
        nKvHeads: config.nKvHeads, hiddenDim: config.hiddenDim,
        vocabSize: config.vocabSize, seqLen: config.seqLen,
      }, fingerprint: atlasFingerprint(config, ck.weights) });
    } else if (m.type === 'run') {
      if (!model) throw new Error('no model loaded');
      if (hasAbl(m)) {
        // ablated run: reuse the base story, return a FULL capture (attention too) so
        // every instrument can switch to the damaged model's internals consistently.
        const cap = ablOf(m);
        const payload = packRun(cap, baseIds, false);
        payload.type = 'run-ablated';
        payload.ablate = m.ablate || [];
        payload.steer = m.steer || [];
        self.postMessage(payload, [payload.attention]);
      } else {
        let ids = tok.encode(m.text, true, false);
        let truncated = false;
        if (ids.length > config.seqLen) { ids = ids.slice(0, config.seqLen); truncated = true; }
        const cap = captureRun(model, ids, {
          topK: 12,
          onProgress: (pos, n) => { if (pos % 8 === 0 || pos === n) self.postMessage({ type: 'progress', pos, n }); },
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
      const cap = ablate ? ablOf(m) : baseCap;
      const rows = [];
      const nextTok = m.pos + 1 < baseIds.length ? baseIds[m.pos + 1] : null;
      for (let l = 0; l < config.nLayers; l++) {
        for (const phase of ['attn', 'ffn']) {
          const logits = logitLens(model, cap, l, m.pos, phase);
          let max = -Infinity;
          for (const v of logits) if (v > max) max = v;
          let sum = 0;
          for (const v of logits) sum += Math.exp(v - max);
          const logZ = max + Math.log(sum);
          const idx = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, 3);
          rows.push({
            layer: l, phase,
            top: idx.map(i => ({ piece: tok.decodePiece(baseIds[m.pos], i), prob: Math.exp(logits[i] - logZ) })),
            nextProb: nextTok !== null ? Math.exp(logits[nextTok] - logZ) : null,
          });
        }
      }
      self.postMessage({ type: 'lens-done', pos: m.pos, rows, ablate: ablate || null, steer: m.steer || [],
        nextPiece: nextTok !== null ? tok.decodePiece(baseIds[m.pos], nextTok) : null });
    } else if (m.type === 'neurons') {
      if (!baseCap) throw new Error('run a story first');
      const cap = hasAbl(m) ? ablOf(m) : baseCap;  // which-neurons-light-up, in the ablated model when muted
      const top = neuronContrast(cap, m.spanStart, m.spanEnd, 12);
      self.postMessage({ type: 'neurons-done', top, spanStart: m.spanStart, spanEnd: m.spanEnd, ablate: m.ablate || null, steer: m.steer || [] });
    } else if (m.type === 'neuron-acts') {
      if (!baseCap) throw new Error('run a story first');
      const cap = hasAbl(m) ? ablOf(m) : baseCap;
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
      if (!baseCont) baseCont = continueSteered(model, baseIds, steps, {
        onToken: (i) => self.postMessage({ type: 'continue-progress', done: i, total }) });
      const gen = changed ? continueSteered(model, baseIds, steps, { heads: m.ablate || [], steer: m.steer || [],
        onToken: (i) => self.postMessage({ type: 'continue-progress', done: steps + i, total }) }) : baseCont;
      self.postMessage({ type: 'continue-done', base: piecesOf(baseCont), steered: piecesOf(gen),
        changed: !!changed, ablate: m.ablate || [], steer: m.steer || [] });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
