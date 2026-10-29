// model.js — CPU fp32 forward pass with KV cache, ported from vendor/llama2.c/run.c
// forward(). Math.fround is applied where C rounds to float, EXCEPT inside matmul: there
// each dot product accumulates in double (rounded once, on store) four rows at a time, the
// J-Lens Explorer's kernel. That is ~3x faster (110M: 540 -> 163 ms per piece) and, if
// anything, closer to true arithmetic than float32 accumulation; greedy output still
// matches run.c exactly on every fixture. The parity gate (E2/G1) is exact greedy token
// match against C fixtures, so any drift would fail it.
//
// Optional `tap` hooks expose per-layer internals for capture.js (E3):
//   tap.attention(l, h, pos, attRow)  — post-softmax scores, view valid 0..pos (COPY if kept)
//   tap.residual(l, phase, pos, x)    — phase 'attn' | 'ffn', residual stream after the add
//   tap.ffn(l, pos, hb)               — SwiGLU-gated hidden activations before w2
// Optional `ablate` (Sessions A/C): either a Set of "l:h" head keys (legacy form), or
// { heads: Set<"l:h">, neurons: Set<"l:j"> }. Ablated heads contribute zeros to the
// attention output (their pattern is still computed and tapped, so the microscope can
// show WHAT was removed); ablated neurons have their SwiGLU-gated activation zeroed
// before w2. With ablate unset the code path is identical to the G1-parity-tested one.
// DOM-free; Node + browser.

const f = Math.fround;

export function createState(config) {
  const c = config;
  return {
    x: new Float32Array(c.dim),
    xb: new Float32Array(c.dim),
    xb2: new Float32Array(c.dim),
    hb: new Float32Array(c.hiddenDim),
    hb2: new Float32Array(c.hiddenDim),
    q: new Float32Array(c.dim),
    att: new Float32Array(c.nHeads * c.seqLen),
    logits: new Float32Array(c.vocabSize),
    keyCache: new Float32Array(c.nLayers * c.seqLen * c.kvDim),
    valueCache: new Float32Array(c.nLayers * c.seqLen * c.kvDim),
  };
}

export function createModel(checkpoint) {
  return { config: checkpoint.config, weights: checkpoint.weights, state: createState(checkpoint.config) };
}

function rmsnorm(o, x, w, wOff, size) {
  let ss = 0;
  for (let j = 0; j < size; j++) ss = f(ss + f(x[j] * x[j]));
  ss = f(ss / size);
  ss = f(ss + 1e-5);
  ss = f(1 / f(Math.sqrt(ss)));
  for (let j = 0; j < size; j++) o[j] = w[wOff + j] * f(ss * x[j]);
}

function softmax(x, off, size) {
  let max = x[off];
  for (let i = 1; i < size; i++) if (x[off + i] > max) max = x[off + i];
  let sum = 0;
  for (let i = 0; i < size; i++) {
    x[off + i] = f(Math.exp(f(x[off + i] - max)));
    sum = f(sum + x[off + i]);
  }
  for (let i = 0; i < size; i++) x[off + i] = f(x[off + i] / sum);
}

// xout[d] = W(d,n) @ x(n) ; W row-major at wOff. Double accumulation, four output rows
// per pass so x is read once per four weight rows (the explorer's matmulFast).
export function matmul(xout, x, w, wOff, n, d) {
  const d4 = d - (d % 4);
  let i = 0;
  for (; i < d4; i += 4) {
    const r0 = wOff + i * n, r1 = r0 + n, r2 = r1 + n, r3 = r2 + n;
    let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (let j = 0; j < n; j++) {
      const xv = x[j];
      a0 += xv * w[r0 + j]; a1 += xv * w[r1 + j];
      a2 += xv * w[r2 + j]; a3 += xv * w[r3 + j];
    }
    xout[i] = a0; xout[i + 1] = a1; xout[i + 2] = a2; xout[i + 3] = a3;
  }
  for (; i < d; i++) {
    const row = wOff + i * n;
    let val = 0;
    for (let j = 0; j < n; j++) val += w[row + j] * x[j];
    xout[i] = val;
  }
}

export function forward(model, token, pos, tap = null, ablate = null) {
  const c = model.config, w = model.weights, s = model.state;
  const dim = c.dim, kvDim = c.kvDim, hiddenDim = c.hiddenDim, headSize = c.headSize;
  const kvMul = c.nHeads / c.nKvHeads;
  const x = s.x;

  let abH = null, abNL = null;
  if (ablate) {
    const abN = ablate instanceof Set ? null : ablate.neurons;
    abH = ablate instanceof Set ? ablate : (ablate.heads ?? null);
    if (abN && abN.size) {
      abNL = Array.from({ length: c.nLayers }, () => []);
      for (const k of abN) {
        const [l, j] = k.split(':').map(Number);
        if (abNL[l]) abNL[l].push(j);
      }
    }
  }

  x.set(w.tokenEmbedding.subarray(token * dim, (token + 1) * dim));

  for (let l = 0; l < c.nLayers; l++) {
    rmsnorm(s.xb, x, w.rmsAtt, l * dim, dim);

    const loff = l * c.seqLen * kvDim;
    const kOff = loff + pos * kvDim;

    matmul(s.q, s.xb, w.wq, l * dim * dim, dim, dim);
    matmul(s.keyCache.subarray(kOff, kOff + kvDim), s.xb, w.wk, l * dim * kvDim, dim, kvDim);
    matmul(s.valueCache.subarray(kOff, kOff + kvDim), s.xb, w.wv, l * dim * kvDim, dim, kvDim);

    // RoPE, matching C exactly (per-pair freq from head_dim = i % head_size)
    const k = s.keyCache;
    for (let i = 0; i < dim; i += 2) {
      const headDim = i % headSize;
      const freq = f(1 / f(Math.pow(10000, f(headDim / headSize))));
      const val = f(pos * freq);
      const fcr = f(Math.cos(val));
      const fci = f(Math.sin(val));
      const rotn = i < kvDim ? 2 : 1;
      for (let v = 0; v < rotn; v++) {
        if (v === 0) {
          const v0 = s.q[i], v1 = s.q[i + 1];
          s.q[i] = f(v0 * fcr) - f(v1 * fci);
          s.q[i + 1] = f(v0 * fci) + f(v1 * fcr);
        } else {
          const v0 = k[kOff + i], v1 = k[kOff + i + 1];
          k[kOff + i] = f(v0 * fcr) - f(v1 * fci);
          k[kOff + i + 1] = f(v0 * fci) + f(v1 * fcr);
        }
      }
    }

    // multihead attention over the KV cache
    const sqrtHs = f(Math.sqrt(headSize));
    for (let h = 0; h < c.nHeads; h++) {
      const qOff = h * headSize;
      const attOff = h * c.seqLen;
      const kvHeadOff = ((h / kvMul) | 0) * headSize;
      for (let t = 0; t <= pos; t++) {
        const kBase = loff + t * kvDim + kvHeadOff;
        let score = 0;
        for (let i = 0; i < headSize; i++) score = f(score + f(s.q[qOff + i] * s.keyCache[kBase + i]));
        s.att[attOff + t] = f(score / sqrtHs);
      }
      softmax(s.att, attOff, pos + 1);
      if (tap && tap.attention) tap.attention(l, h, pos, s.att.subarray(attOff, attOff + pos + 1));

      const xbOff = h * headSize;
      s.xb.fill(0, xbOff, xbOff + headSize);
      if (abH && abH.has(l + ':' + h)) continue; // head contributes zeros
      for (let t = 0; t <= pos; t++) {
        const vBase = loff + t * kvDim + kvHeadOff;
        const a = s.att[attOff + t];
        for (let i = 0; i < headSize; i++) {
          s.xb[xbOff + i] = s.xb[xbOff + i] + f(a * s.valueCache[vBase + i]);
        }
      }
    }

    matmul(s.xb2, s.xb, w.wo, l * dim * dim, dim, dim);
    for (let i = 0; i < dim; i++) x[i] = x[i] + s.xb2[i];
    if (tap && tap.residual) tap.residual(l, 'attn', pos, x);

    rmsnorm(s.xb, x, w.rmsFfn, l * dim, dim);
    matmul(s.hb, s.xb, w.w1, l * dim * hiddenDim, dim, hiddenDim);
    matmul(s.hb2, s.xb, w.w3, l * dim * hiddenDim, dim, hiddenDim);
    for (let i = 0; i < hiddenDim; i++) {
      let val = s.hb[i];
      val = f(val * f(1 / f(1 + f(Math.exp(-val)))));
      val = f(val * s.hb2[i]);
      s.hb[i] = val;
    }
    if (abNL) for (const j of abNL[l]) s.hb[j] = 0;
    if (tap && tap.ffn) tap.ffn(l, pos, s.hb);

    matmul(s.xb, s.hb, w.w2, l * hiddenDim * dim, hiddenDim, dim);
    for (let i = 0; i < dim; i++) x[i] = x[i] + s.xb[i];
    if (tap && tap.residual) tap.residual(l, 'ffn', pos, x);
  }

  rmsnorm(x, x, w.rmsFinal, 0, dim);
  matmul(s.logits, x, w.wcls, 0, dim, c.vocabSize);
  return s.logits;
}

export function argmax(arr, n = arr.length) {
  let maxI = 0, maxP = arr[0];
  for (let i = 1; i < n; i++) if (arr[i] > maxP) { maxI = i; maxP = arr[i]; }
  return maxI;
}

// Greedy generation mirroring run.c generate() with temperature 0, except it does NOT
// stop on BOS — fixtures record a fixed number of steps for strict comparison.
export function generateGreedy(model, tokenizer, prompt, steps, tap = null) {
  const promptTokens = tokenizer.encode(prompt, true, false);
  const n = Math.min(steps, model.config.seqLen);
  const tokens = [promptTokens[0]];
  let token = promptTokens[0];
  for (let pos = 0; pos < n - 1; pos++) {
    const logits = forward(model, token, pos, tap);
    const next = pos < promptTokens.length - 1 ? promptTokens[pos + 1] : argmax(logits);
    tokens.push(next);
    token = next;
  }
  return { tokens, text: tokenizer.decode(tokens) };
}
