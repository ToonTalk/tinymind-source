// hunt/worker.js — Session C: the treasure-hunt engine. Runs the probe sentence,
// locates the character-name piece spans, gathers the correlational evidence the
// interpreter reasons from (head attention-to-name at the answer boundary, neuron
// contrasts), and runs ablation tests. The metric everything hangs on is the
// CONTINUATION SCORE: geometric-mean per-token probability for the expected
// continuation, teacher-forced. Raw products are retained for display, while
// comparisons stay fair when answers have different token lengths.
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun, neuronContrast } from '../src/capture.js';

let model = null, tok = null, config = null;
let state = null; // { ids, boundary, baseline: {contProb, top}, nameSpans }
const CONTROL_COMPLETIONS = [
  { id: 'ordinary-happy', probe: 'The little girl was very', continuation: ' happy' },
  { id: 'ordinary-play', probe: 'They went to the park to', continuation: ' play' },
  { id: 'ordinary-little', probe: 'Once upon a time there was a', continuation: ' little' },
];

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

function continuationStats(cap, boundary, n) {
  let logProb = 0;
  for (let i = boundary; i < n - 1; i++) logProb += Math.log(Math.max(Number.MIN_VALUE, cap.nextProb[i]));
  const tokenCount = Math.max(1, n - 1 - boundary);
  return {
    contProb: Math.exp(logProb),
    tokenCount,
    meanLogProb: logProb / tokenCount,
    geometricMeanProbability: Math.exp(logProb / tokenCount),
  };
}

function captureCompletion(probe, continuation, ablate = null, topK = 12) {
  const probeIds = tok.encode(probe, true, false);
  const ids = tok.encode(probe + continuation, true, false);
  for (let i = 0; i < probeIds.length - 1; i++) {
    if (ids[i] !== probeIds[i]) throw new Error('tokenizer merged across the probe/continuation seam — add a space or change the words');
  }
  const boundary = probeIds.length - 1;
  const cap = captureRun(model, ids, { topK, ...(ablate ? { ablate } : {}) });
  return { ids, boundary, cap, stats: continuationStats(cap, boundary, ids.length) };
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
      const run = captureCompletion(m.probe, m.continuation);
      const { ids, boundary, cap } = run;
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

      const controls = CONTROL_COMPLETIONS.map(control => {
        const controlRun = captureCompletion(control.probe, control.continuation, null, 1);
        return { ...control, ids: controlRun.ids, boundary: controlRun.boundary, baseline: controlRun.stats };
      });
      state = { ids, boundary,
        baseline: { ...run.stats, top: packTop(cap, ids, boundary) },
        nameSpans: spans, controls };
      self.postMessage({
        type: 'probe-done',
        pieces, ids: [...ids], boundary, nameSpans: spans,
        contProb: state.baseline.contProb,
        contTokenCount: state.baseline.tokenCount,
        contMeanLogProb: state.baseline.meanLogProb,
        contGeometricMeanProbability: state.baseline.geometricMeanProbability,
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
          const control = captureCompletion(probe2, m.continuation, null, 1);
          results.push({ name: ctrl, ...control.stats });
        } catch (e) { results.push({ name: ctrl, error: String(e && e.message || e) }); }
      }
      self.postMessage({ type: 'namecontrol-done',
        baseline: state ? state.baseline.contProb : null,
        baselineMeanLogProb: state ? state.baseline.meanLogProb : null,
        baselineGeometricMeanProbability: state ? state.baseline.geometricMeanProbability : null,
        nameInProbe, results });
    } else if (m.type === 'test') {
      if (!state) throw new Error('probe first');
      const ablate = { heads: new Set(m.heads ?? []), neurons: new Set(m.neurons ?? []) };
      const cap = captureRun(model, state.ids, { topK: 12, ablate });
      const target = continuationStats(cap, state.boundary, state.ids.length);
      const controls = state.controls.map(control => {
        const controlCap = captureRun(model, control.ids, { topK: 1, ablate });
        const after = continuationStats(controlCap, control.boundary, control.ids.length);
        return {
          id: control.id, probe: control.probe, continuation: control.continuation,
          baseline: control.baseline.geometricMeanProbability,
          ablated: after.geometricMeanProbability,
          ratio: after.geometricMeanProbability / Math.max(Number.MIN_VALUE, control.baseline.geometricMeanProbability),
        };
      });
      self.postMessage({
        type: 'test-done',
        id: m.id, purpose: m.purpose || 'single', label: m.label || '',
        heads: m.heads ?? [], neurons: m.neurons ?? [],
        contProb: target.contProb,
        meanLogProb: target.meanLogProb,
        geometricMeanProbability: target.geometricMeanProbability,
        baseline: state.baseline.contProb,
        baselineMeanLogProb: state.baseline.meanLogProb,
        baselineGeometricMeanProbability: state.baseline.geometricMeanProbability,
        controls,
        top: packTop(cap, state.ids, state.boundary),
      });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.message || err) });
  }
};
