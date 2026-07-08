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

let model = null, tok = null, config = null;
let baseCap = null, baseIds = null;
// One ablated capture serves every ablation-aware instrument (lens, neurons, neuron-acts,
// and the ablated run itself), keyed by the sorted muted-head set over the SAME story, so
// switching instruments under a fixed ablation recomputes nothing.
let ablCap = null, ablKey = null;
function getAblCap(list) {
  if (!baseCap) throw new Error('run the full model first');
  const key = list.slice().sort().join(',');
  if (key !== ablKey) { ablCap = captureRun(model, baseIds, { topK: 12, ablate: new Set(list) }); ablKey = key; }
  return ablCap;
}
const hasAbl = m => m.ablate && m.ablate.length;

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
      }});
    } else if (m.type === 'run') {
      if (!model) throw new Error('no model loaded');
      if (hasAbl(m)) {
        // ablated run: reuse the base story, return a FULL capture (attention too) so
        // every instrument can switch to the damaged model's internals consistently.
        const cap = getAblCap(m.ablate);
        const payload = packRun(cap, baseIds, false);
        payload.type = 'run-ablated';
        payload.ablate = m.ablate;
        self.postMessage(payload, [payload.attention]);
      } else {
        let ids = tok.encode(m.text, true, false);
        let truncated = false;
        if (ids.length > config.seqLen) { ids = ids.slice(0, config.seqLen); truncated = true; }
        const cap = captureRun(model, ids, {
          topK: 12,
          onProgress: (pos, n) => { if (pos % 8 === 0 || pos === n) self.postMessage({ type: 'progress', pos, n }); },
        });
        baseCap = cap; baseIds = ids; ablCap = null; ablKey = null; // a new base story invalidates the ablation cache
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
      const ablate = hasAbl(m) ? m.ablate : null;
      const cap = ablate ? getAblCap(ablate) : baseCap;
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
      self.postMessage({ type: 'lens-done', pos: m.pos, rows, ablate: ablate || null,
        nextPiece: nextTok !== null ? tok.decodePiece(baseIds[m.pos], nextTok) : null });
    } else if (m.type === 'neurons') {
      if (!baseCap) throw new Error('run a story first');
      const cap = hasAbl(m) ? getAblCap(m.ablate) : baseCap;  // which-neurons-light-up, in the ablated model when muted
      const top = neuronContrast(cap, m.spanStart, m.spanEnd, 12);
      self.postMessage({ type: 'neurons-done', top, spanStart: m.spanStart, spanEnd: m.spanEnd, ablate: m.ablate || null });
    } else if (m.type === 'neuron-acts') {
      if (!baseCap) throw new Error('run a story first');
      const cap = hasAbl(m) ? getAblCap(m.ablate) : baseCap;
      const n = cap.nPos, HD = config.hiddenDim;
      const acts = new Float32Array(n);
      for (let p = 0; p < n; p++) acts[p] = Math.abs(cap.ffn[(m.layer * n + p) * HD + m.neuron]);
      self.postMessage({ type: 'neuron-acts-done', layer: m.layer, neuron: m.neuron, acts: acts.buffer, ablate: m.ablate || null }, [acts.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
