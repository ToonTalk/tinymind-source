// hunt/worker.js — Session C: the treasure-hunt engine. Runs the probe sentence,
// locates the character-name piece spans, gathers the correlational evidence the
// interpreter reasons from (head attention-to-name at the answer boundary, neuron
// contrasts), and runs ablation tests. The metric everything hangs on is the
// CONTINUATION PROBABILITY: product of the model's probabilities for the expected
// continuation pieces, teacher-forced. Ablation verdicts compare it before/after.
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun, neuronContrast } from '../src/capture.js';

let model = null, tok = null, config = null;
let state = null; // { ids, boundary, baseline: {contProb, top}, nameSpans }

function findNameSpans(ids, name) {
  // rebuild the decoded text with per-piece char ranges, then map name matches to spans
  const pieces = ids.map((id, i) => tok.decodePiece(i > 0 ? ids[i - 1] : 1, id));
  let text = '';
  const ranges = [];
  for (let i = 0; i < ids.length; i++) {
    const p = ids[i] === 1 || ids[i] === 2 ? '' : pieces[i];
    ranges.push([text.length, text.length + p.length]);
    text += p;
  }
  const spans = [];
  const needle = name.toLowerCase();
  const hay = text.toLowerCase();
  let at = 0;
  while ((at = hay.indexOf(needle, at)) !== -1) {
    const end = at + needle.length;
    let first = -1, last = -1;
    for (let i = 0; i < ids.length; i++) {
      if (ranges[i][1] > at && ranges[i][0] < end) { if (first === -1) first = i; last = i; }
    }
    if (first !== -1) spans.push([first, last]);
    at = end;
  }
  return { spans, pieces };
}

function contProbOf(cap, boundary, n) {
  let p = 1;
  for (let i = boundary; i < n - 1; i++) p *= cap.nextProb[i];
  return p;
}

function packTop(cap, ids, pos) {
  return cap.topK[pos].slice(0, 8).map(({ token, logit }) => ({
    piece: tok.decodePiece(ids[pos], token),
    prob: Math.exp(logit - cap.logZ[pos]),
  }));
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      const ck = readCheckpoint(m.ckptBuf);
      config = ck.config;
      tok = new Tokenizer(m.tokBuf, config.vocabSize);
      model = createModel(ck);
      state = null;
      self.postMessage({ type: 'loaded', config: { dim: config.dim, nLayers: config.nLayers,
        nHeads: config.nHeads, hiddenDim: config.hiddenDim, vocabSize: config.vocabSize } });
    } else if (m.type === 'probe') {
      if (!model) throw new Error('no model loaded');
      const probeIds = tok.encode(m.probe, true, false);
      const ids = tok.encode(m.probe + m.continuation, true, false);
      // the continuation must extend the probe's tokenization for teacher-forcing to
      // line up; BPE can merge across the seam — check and report if so
      for (let i = 0; i < probeIds.length - 1; i++) {
        if (ids[i] !== probeIds[i]) throw new Error('tokenizer merged across the probe/continuation seam — add a space or change the words');
      }
      const boundary = probeIds.length - 1;
      const cap = captureRun(model, ids, { topK: 12 });
      const { spans, pieces } = findNameSpans(ids, m.name);
      const n = ids.length;

      // head evidence: attention from the boundary position to name pieces
      const L = config.nLayers, H = config.nHeads;
      const headScores = [];
      for (let l = 0; l < L; l++) {
        for (let h = 0; h < H; h++) {
          const row = cap.attnAt(l, h, boundary);
          let s = 0;
          for (const [a, b] of spans) for (let i = a; i <= Math.min(b, boundary); i++) s += row[i];
          headScores.push({ layer: l, head: h, score: s });
        }
      }
      headScores.sort((a, b) => b.score - a.score);

      // neuron evidence: contrast over the first name span
      const neuronTop = spans.length
        ? neuronContrast(cap, spans[0][0], spans[0][1], 12)
        : [];

      state = { ids, boundary,
        baseline: { contProb: contProbOf(cap, boundary, n), top: packTop(cap, ids, boundary) },
        nameSpans: spans };
      self.postMessage({
        type: 'probe-done',
        pieces, ids: [...ids], boundary, nameSpans: spans,
        contProb: state.baseline.contProb,
        top: state.baseline.top,
        headScores: headScores.slice(0, 10),
        allHeadScores: headScores,
        neuronTop,
        // steer: neurons firing at the PREDICTION position (carry local/bigram signals,
        // which name-focused evidence misses), + the word that triggers the answer
        boundaryNeurons: neuronContrast(cap, boundary, boundary, 12),
        triggerWord: tok.decodePiece(boundary > 0 ? ids[boundary - 1] : 1, ids[boundary]),
        contPieces: pieces.slice(boundary + 1),
        // per-token breakdown: prob of each continuation piece (product == contProb).
        // reveals whether a low answer-chance is death-by-many-tokens or a single collapse.
        contBreakdown: (() => { const out = []; for (let j = boundary + 1; j < n; j++) out.push({ piece: pieces[j], prob: cap.nextProb[j - 1] }); return out; })(),
      });
    } else if (m.type === 'namecontrol') {
      if (!model) throw new Error('no model loaded');
      const esc = m.name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      const rx = new RegExp('\\b' + esc + '\\b', 'gi');
      const nameInProbe = rx.test(m.probe); rx.lastIndex = 0;
      const results = [];
      for (const ctrl of m.controls) {
        try {
          const probe2 = m.probe.replace(rx, ctrl); rx.lastIndex = 0;
          const pIds = tok.encode(probe2, true, false);
          const ids2 = tok.encode(probe2 + m.continuation, true, false);
          let seamOk = pIds.length > 0;
          for (let i = 0; i < pIds.length - 1; i++) if (ids2[i] !== pIds[i]) { seamOk = false; break; }
          if (!seamOk) { results.push({ name: ctrl, error: 'tokenizer seam' }); continue; }
          const b2 = pIds.length - 1;
          const cap2 = captureRun(model, ids2, { topK: 1 });
          results.push({ name: ctrl, contProb: contProbOf(cap2, b2, ids2.length) });
        } catch (e) { results.push({ name: ctrl, error: String(e && e.message || e) }); }
      }
      self.postMessage({ type: 'namecontrol-done',
        baseline: state ? state.baseline.contProb : null, nameInProbe, results });
    } else if (m.type === 'test') {
      if (!state) throw new Error('probe first');
      const ablate = { heads: new Set(m.heads ?? []), neurons: new Set(m.neurons ?? []) };
      const cap = captureRun(model, state.ids, { topK: 12, ablate });
      self.postMessage({
        type: 'test-done',
        id: m.id,
        heads: m.heads ?? [], neurons: m.neurons ?? [],
        contProb: contProbOf(cap, state.boundary, state.ids.length),
        baseline: state.baseline.contProb,
        top: packTop(cap, state.ids, state.boundary),
      });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
