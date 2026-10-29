// capture.js — E3: activation capture over a teacher-forced token sequence.
// This is the microscope's substrate (kickoff §6 E3): per-layer/per-head attention
// matrices, residual-stream snapshots, FFN activations, top-k next-token logits.
// Typed arrays internally; toInterpreterJSON() emits a selective, rounded JSON view
// for the resident interpreter (Session C). DOM-free; Node + browser.

import { forward, matmul } from './model.js';

export class Capture {
  constructor(config, tokens) {
    const c = config;
    const n = tokens.length;
    this.config = c;
    this.tokens = tokens.slice();
    this.nPos = n;
    // attention[l][h] is an n×n row-major matrix; row p holds post-softmax scores
    // over positions 0..p (upper triangle stays 0).
    this.attention = new Float32Array(c.nLayers * c.nHeads * n * n);
    // residual stream AFTER the attention-add and AFTER the ffn-add, per layer/pos.
    this.residAttn = new Float32Array(c.nLayers * n * c.dim);
    this.residFfn = new Float32Array(c.nLayers * n * c.dim);
    // SwiGLU-gated FFN hidden activations (input to w2), per layer/pos.
    this.ffn = new Float32Array(c.nLayers * n * c.hiddenDim);
    this.topK = [];        // per pos: [{token, logit}] descending
    // Session A additions: full-softmax log partition per pos (exact probabilities
    // from stored logits: p = exp(logit - logZ)), and the model's probability for
    // the ACTUAL next token (teacher-forced), null at the last position.
    this.logZ = new Float32Array(n);
    this.nextProb = new Float32Array(n);
  }

  attnAt(l, h, pos = null) {
    const n = this.nPos;
    const m = this.attention.subarray((l * this.config.nHeads + h) * n * n,
                                      (l * this.config.nHeads + h + 1) * n * n);
    return pos === null ? m : m.subarray(pos * n, pos * n + pos + 1);
  }
  residualAt(l, pos, phase = 'ffn') {
    const c = this.config;
    const base = phase === 'attn' ? this.residAttn : this.residFfn;
    return base.subarray((l * this.nPos + pos) * c.dim, (l * this.nPos + pos + 1) * c.dim);
  }
  ffnAt(l, pos) {
    const c = this.config;
    return this.ffn.subarray((l * this.nPos + pos) * c.hiddenDim,
                             (l * this.nPos + pos + 1) * c.hiddenDim);
  }

  bytes() {
    return (this.attention.length + this.residAttn.length + this.residFfn.length +
            this.ffn.length) * 4;
  }

  // Selective JSON for the resident interpreter. Full dumps are too big to ship to an
  // LLM; callers pick layers/heads and positions. Values rounded to `digits`.
  toInterpreterJSON({ layers = null, heads = null, positions = null, digits = 4 } = {}) {
    const c = this.config;
    const round = v => Number(v.toFixed(digits));
    const L = layers ?? [...Array(c.nLayers).keys()];
    const H = heads ?? [...Array(c.nHeads).keys()];
    const P = positions ?? [...Array(this.nPos).keys()];
    const out = {
      schema: 'tiny-mind-capture-v1',
      config: { dim: c.dim, nLayers: c.nLayers, nHeads: c.nHeads, nKvHeads: c.nKvHeads,
                hiddenDim: c.hiddenDim, vocabSize: c.vocabSize },
      tokens: this.tokens,
      positions: P,
      topK: P.map(p => this.topK[p].map(({ token, logit }) => ({ token, logit: round(logit) }))),
      attention: {},
    };
    for (const l of L) for (const h of H) {
      out.attention[`L${l}H${h}`] =
        P.map(p => Array.from(this.attnAt(l, h, p), round));
    }
    return out;
  }
}

// Run the model over `tokens` (teacher-forced) recording activations.
// topK = 0 disables logit capture. ablate: Set of "l:h" (see model.js).
// onProgress(pos, n) fires per position for UI progress bars.
export function captureRun(model, tokens, { topK = 16, ablate = null, onProgress = null } = {}) {
  const c = model.config;
  if (tokens.length > c.seqLen) throw new Error(`captureRun: ${tokens.length} tokens > seqLen ${c.seqLen}`);
  const cap = new Capture(c, tokens);
  const n = tokens.length;

  const tap = {
    attention(l, h, pos, row) {
      cap.attention.set(row, (l * c.nHeads + h) * n * n + pos * n);
    },
    residual(l, phase, pos, x) {
      (phase === 'attn' ? cap.residAttn : cap.residFfn)
        .set(x, (l * n + pos) * c.dim);
    },
    ffn(l, pos, hb) {
      cap.ffn.set(hb, (l * n + pos) * c.hiddenDim);
    },
  };

  for (let pos = 0; pos < n; pos++) {
    const logits = forward(model, tokens[pos], pos, tap, ablate);
    if (topK > 0) {
      const idx = [...logits.keys()];
      idx.sort((a, b) => logits[b] - logits[a]);
      cap.topK.push(idx.slice(0, topK).map(i => ({ token: i, logit: logits[i] })));
      // exact softmax normalizer + probability of the actual next token
      let max = -Infinity;
      for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
      let sum = 0;
      for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
      cap.logZ[pos] = max + Math.log(sum);
      cap.nextProb[pos] = pos + 1 < n ? Math.exp(logits[tokens[pos + 1]] - cap.logZ[pos]) : 0;
    } else {
      cap.topK.push([]);
    }
    if (onProgress) onProgress(pos + 1, n);
  }
  return cap;
}

// Neuron contrast scores for a token span (Session A "which neurons light up here?").
// For every (layer, hidden unit): mean |activation| inside [spanStart, spanEnd] minus
// mean |activation| outside, in units of the outside std. Returns the top `limit`
// as { layer, neuron, score, inMean, outMean }. Correlational by design — the
// microscope pairs it with ablation, which is the arbiter.
export function neuronContrast(cap, spanStart, spanEnd, limit = 20) {
  const c = cap.config, n = cap.nPos, HD = c.hiddenDim;
  const results = [];
  for (let l = 0; l < c.nLayers; l++) {
    for (let j = 0; j < HD; j++) {
      let inSum = 0, inN = 0, outSum = 0, outSq = 0, outN = 0;
      for (let p = 0; p < n; p++) {
        const v = Math.abs(cap.ffn[(l * n + p) * HD + j]);
        if (p >= spanStart && p <= spanEnd) { inSum += v; inN++; }
        else { outSum += v; outSq += v * v; outN++; }
      }
      if (!inN || !outN) continue;
      const inMean = inSum / inN, outMean = outSum / outN;
      const outStd = Math.sqrt(Math.max(1e-12, outSq / outN - outMean * outMean));
      results.push({ layer: l, neuron: j, score: (inMean - outMean) / (outStd + 1e-6), inMean, outMean });
    }
  }
  results.sort((a, b) => b.score - a.score);
  return results.slice(0, limit);
}

// Logit lens: project a residual-stream snapshot through the final rmsnorm + classifier.
// "What would the model predict if it stopped at layer l?"
export function logitLens(model, cap, l, pos, phase = 'ffn') {
  const c = model.config, w = model.weights;
  const xb = lensInput(model, cap.residualAt(l, pos, phase));
  const logits = new Float32Array(c.vocabSize);
  matmul(logits, xb, w.wcls, 0, c.dim, c.vocabSize);   // the engine's fast kernel
  return logits;
}

// the final rmsnorm of a residual snapshot: what the classifier reads (batch.js sends
// many of these through the classifier at once)
export function lensInput(model, x) {
  const c = model.config, w = model.weights;
  const f = Math.fround;
  let ss = 0;
  for (let j = 0; j < c.dim; j++) ss = f(ss + f(x[j] * x[j]));
  ss = f(1 / f(Math.sqrt(f(f(ss / c.dim) + 1e-5))));
  const xb = new Float32Array(c.dim);
  for (let j = 0; j < c.dim; j++) xb[j] = w.rmsFinal[j] * f(ss * x[j]);
  return xb;
}
