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
      let ids = tok.encode(m.text, true, false);
      let truncated = false;
      if (ids.length > config.seqLen) { ids = ids.slice(0, config.seqLen); truncated = true; }
      const ablate = m.ablate && m.ablate.length ? new Set(m.ablate) : null;
      const cap = captureRun(model, ids, {
        topK: 12, ablate,
        onProgress: (pos, n) => { if (pos % 8 === 0 || pos === n) self.postMessage({ type: 'progress', pos, n }); },
      });
      const light = !!ablate; // ablated runs: bars/probs only, no attention transfer
      if (!ablate) { baseCap = cap; baseIds = ids; }
      const payload = packRun(cap, ids, light);
      payload.type = ablate ? 'run-ablated' : 'run-done';
      payload.truncated = truncated;
      payload.ablate = m.ablate ?? null;
      self.postMessage(payload, payload.attention ? [payload.attention] : []);
    } else if (m.type === 'lens') {
      if (!baseCap) throw new Error('run a story first');
      const rows = [];
      const nextTok = m.pos + 1 < baseIds.length ? baseIds[m.pos + 1] : null;
      for (let l = 0; l < config.nLayers; l++) {
        for (const phase of ['attn', 'ffn']) {
          const logits = logitLens(model, baseCap, l, m.pos, phase);
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
      self.postMessage({ type: 'lens-done', pos: m.pos, rows,
        nextPiece: nextTok !== null ? tok.decodePiece(baseIds[m.pos], nextTok) : null });
    } else if (m.type === 'neurons') {
      if (!baseCap) throw new Error('run a story first');
      const top = neuronContrast(baseCap, m.spanStart, m.spanEnd, 12);
      self.postMessage({ type: 'neurons-done', top, spanStart: m.spanStart, spanEnd: m.spanEnd });
    } else if (m.type === 'neuron-acts') {
      if (!baseCap) throw new Error('run a story first');
      const n = baseCap.nPos, HD = config.hiddenDim;
      const acts = new Float32Array(n);
      for (let p = 0; p < n; p++) acts[p] = Math.abs(baseCap.ffn[(m.layer * n + p) * HD + m.neuron]);
      self.postMessage({ type: 'neuron-acts-done', layer: m.layer, neuron: m.neuron, acts: acts.buffer }, [acts.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
