// train.js — E4: hand-rolled batched forward/backward + AdamW for the 260K trainee
// (llm.c-style manual backprop; kickoff §6 E4 prefers this over tfjs: one
// implementation, zero deps, full activation access). DOM-free; Node + browser.
//
// Precision: Float32Array storage with f64 loop accumulation (training has no C-parity
// gate; model.js remains the fround-faithful inference reference). Pass { f64: true }
// for Float64Array storage — used by the finite-difference gradient check.
//
// Weight names/shapes match bin-reader.js, so trainer params drop straight into
// model.js (generation from the fine-tuned net) and writeCheckpoint (saving).
//
// A step's gradient is the sum of each sequence's own gradient, added in sequence order
// (step / evalLoss below). That lets TrainPool hand sequences to worker threads, one
// core each, and still produce exactly the same numbers as working through them here
// one by one — the same model, bit for bit, whatever the number of cores (E15).

// --- deterministic RNG (mulberry32) for reproducible batches/experiments.
// getState/setState expose the 32-bit internal state so long runs can checkpoint
// and resume the exact random sequence (arc generation).
export function rng(seed) {
  let a = seed >>> 0;
  const next = function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  next.getState = () => a >>> 0;
  next.setState = (v) => { a = v >>> 0; };
  return next;
}

// Pack tokenized stories into one training stream: BOS before each story (matches
// llama2.c TinyStories pretokenization).
export function packStories(storyTokenLists) {
  let n = 0;
  for (const s of storyTokenLists) n += s.length + 1;
  const out = new Int32Array(n);
  let p = 0;
  for (const s of storyTokenLists) { out[p++] = 1; out.set(s, p); p += s.length; }
  return out;
}

// Sample B random windows of T+1 tokens from a stream (inputs + shifted targets).
export function sampleBatch(stream, B, T, rand) {
  const batch = new Int32Array(B * (T + 1));
  for (let b = 0; b < B; b++) {
    const start = Math.floor(rand() * (stream.length - T - 1));
    batch.set(stream.subarray(start, start + T + 1), b * (T + 1));
  }
  return batch;
}

const PARAM_SHAPES = c => ({
  tokenEmbedding: c.vocabSize * c.dim,
  rmsAtt: c.nLayers * c.dim,
  wq: c.nLayers * c.dim * c.dim,
  wk: c.nLayers * c.dim * c.kvDim,
  wv: c.nLayers * c.dim * c.kvDim,
  wo: c.nLayers * c.dim * c.dim,
  rmsFfn: c.nLayers * c.dim,
  w1: c.nLayers * c.hiddenDim * c.dim,
  w2: c.nLayers * c.dim * c.hiddenDim,
  w3: c.nLayers * c.hiddenDim * c.dim,
  rmsFinal: c.dim,
});
const DECAYED = new Set(['tokenEmbedding', 'wq', 'wk', 'wv', 'wo', 'w1', 'w2', 'w3']);

export class Trainer {
  constructor(checkpoint, {
    B = 8, T = 256, lr = 1e-4, weightDecay = 0.01,
    beta1 = 0.9, beta2 = 0.999, eps = 1e-8, clip = 1.0, f64 = false,
  } = {}) {
    const c = checkpoint.config;
    if (!c.sharedWeights) throw new Error('Trainer assumes tied embedding/classifier');
    if (T > c.seqLen) throw new Error(`T ${T} > seqLen ${c.seqLen}`);
    this.c = c;
    this.opt = { lr, weightDecay, beta1, beta2, eps, clip };
    this.stepCount = 0;
    this.B = B; this.T = T;
    const AT = f64 ? Float64Array : Float32Array;
    this.AT = AT;

    this.params = {}; this.grads = {}; this.m = {}; this.v = {};
    for (const [name, size] of Object.entries(PARAM_SHAPES(c))) {
      this.params[name] = new AT(size);
      this.params[name].set(checkpoint.weights[name]);
      this.grads[name] = new AT(size);
      this.m[name] = new AT(size);
      this.v[name] = new AT(size);
    }

    // activations (per step); L layers of everything that backward needs
    const { dim: C, kvDim: K, hiddenDim: HD, nLayers: L, nHeads: H, vocabSize: V } = c;
    const N = B * T;
    const per = () => new AT(N * C);
    this.act = {
      x0: per(),
      ln1: [], q: [], k: [], v: [], att: [], attOut: [], res1: [],
      ln2: [], h1: [], h3: [], hGate: [], res2: [],
      lnf: new AT(N * C),
      logits: new AT(N * V), // overwritten with dLogits during loss
    };
    for (let l = 0; l < L; l++) {
      this.act.ln1.push(per()); this.act.q.push(per());
      this.act.k.push(new AT(N * K)); this.act.v.push(new AT(N * K));
      this.act.att.push(new AT(B * H * T * T));
      this.act.attOut.push(per()); this.act.res1.push(per());
      this.act.ln2.push(per());
      this.act.h1.push(new AT(N * HD)); this.act.h3.push(new AT(N * HD));
      this.act.hGate.push(new AT(N * HD));
      this.act.res2.push(per());
    }
    // gradient scratch
    this.g = {
      dX: per(), dLn: per(), dQ: per(),
      dK: new AT(N * K), dV: new AT(N * K),
      dAttOut: per(),
      dH1: new AT(N * HD), dH3: new AT(N * HD), dHGate: new AT(N * HD),
      dLnf: per(),
    };
    // contiguous [B][KH][T][HS] transposes of k/v — the attention fast path streams these
    this.kT = new AT(N * K); this.vT = new AT(N * K);
    this.dKT = new AT(N * K); this.dVT = new AT(N * K);
  }

  // [b*T+t]*K + kvh*HS + i  <->  ((b*KH + kvh)*T + t)*HS + i
  transposeKV(src, dst, invert = false) {
    const { B, T } = this, { kvDim: K, nKvHeads: KH, headSize: HS } = this.c;
    for (let b = 0; b < B; b++) {
      for (let kvh = 0; kvh < KH; kvh++) {
        const tb = (b * KH + kvh) * T * HS;
        for (let t = 0; t < T; t++) {
          const a = (b * T + t) * K + kvh * HS, d = tb + t * HS;
          if (invert) for (let i = 0; i < HS; i++) dst[a + i] = src[d + i];
          else for (let i = 0; i < HS; i++) dst[d + i] = src[a + i];
        }
      }
    }
  }

  // ---- building blocks -----------------------------------------------------
  // matmul / matmulBackward are register-blocked: four weight rows against three input
  // rows at a time, so each value loaded is used three or four times. Every output is
  // still the same sum in the same order as the plain loop, so results don't change.

  // out[N,O] = in[N,C] @ W[O,C]^T
  static matmul(out, inp, W, N, C, O) {
    const O4 = O - (O % 4), N3 = N - (N % 3);
    for (let n = 0; n < N3; n += 3) {   // three input rows at a time (each output's sum unchanged)
      const ia = n * C, ib = ia + C, ic = ib + C, oa = n * O, ob = oa + O, oc = ob + O;
      let o = 0;
      for (; o < O4; o += 4) {
        const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0, b2 = 0, b3 = 0, c0 = 0, c1 = 0, c2 = 0, c3 = 0;
        for (let i = 0; i < C; i++) {
          const u = inp[ia + i], v = inp[ib + i], x = inp[ic + i];
          const p0 = W[w0 + i], p1 = W[w1 + i], p2 = W[w2 + i], p3 = W[w3 + i];
          a0 += u * p0; a1 += u * p1; a2 += u * p2; a3 += u * p3;
          b0 += v * p0; b1 += v * p1; b2 += v * p2; b3 += v * p3;
          c0 += x * p0; c1 += x * p1; c2 += x * p2; c3 += x * p3;
        }
        out[oa + o] = a0; out[oa + o + 1] = a1; out[oa + o + 2] = a2; out[oa + o + 3] = a3;
        out[ob + o] = b0; out[ob + o + 1] = b1; out[ob + o + 2] = b2; out[ob + o + 3] = b3;
        out[oc + o] = c0; out[oc + o + 1] = c1; out[oc + o + 2] = c2; out[oc + o + 3] = c3;
      }
      for (; o < O; o++) {
        const wb = o * C;
        let a = 0, b = 0, c = 0;
        for (let i = 0; i < C; i++) { a += inp[ia + i] * W[wb + i]; b += inp[ib + i] * W[wb + i]; c += inp[ic + i] * W[wb + i]; }
        out[oa + o] = a; out[ob + o] = b; out[oc + o] = c;
      }
    }
    for (let n = N3; n < N; n++) {
      const ib = n * C, ob = n * O;
      let o = 0;
      for (; o < O4; o += 4) {
        const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        for (let i = 0; i < C; i++) {
          const xv = inp[ib + i];
          a0 += xv * W[w0 + i]; a1 += xv * W[w1 + i];
          a2 += xv * W[w2 + i]; a3 += xv * W[w3 + i];
        }
        out[ob + o] = a0; out[ob + o + 1] = a1; out[ob + o + 2] = a2; out[ob + o + 3] = a3;
      }
      for (; o < O; o++) {
        const wb = o * C;
        let acc = 0;
        for (let i = 0; i < C; i++) acc += inp[ib + i] * W[wb + i];
        out[ob + o] = acc;
      }
    }
  }

  // dIn[N,C] += dOut @ W ; dW[O,C] += dOut^T @ in  (two streaming passes)
  static matmulBackward(dOut, inp, W, dIn, dW, N, C, O) {
    const O4 = O - (O % 4), N3 = N - (N % 3);
    for (let n = 0; n < N3; n += 3) {   // dIn, three rows at a time (each row's updates unchanged)
      const ia = n * C, ib = ia + C, ic = ib + C, oa = n * O, ob = oa + O, oc = ob + O;
      let o = 0;
      for (; o < O4; o += 4) {
        const a0 = dOut[oa + o], a1 = dOut[oa + o + 1], a2 = dOut[oa + o + 2], a3 = dOut[oa + o + 3];
        const b0 = dOut[ob + o], b1 = dOut[ob + o + 1], b2 = dOut[ob + o + 2], b3 = dOut[ob + o + 3];
        const c0 = dOut[oc + o], c1 = dOut[oc + o + 1], c2 = dOut[oc + o + 2], c3 = dOut[oc + o + 3];
        const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        for (let i = 0; i < C; i++) {
          const p0 = W[w0 + i], p1 = W[w1 + i], p2 = W[w2 + i], p3 = W[w3 + i];
          dIn[ia + i] += a0 * p0 + a1 * p1 + a2 * p2 + a3 * p3;
          dIn[ib + i] += b0 * p0 + b1 * p1 + b2 * p2 + b3 * p3;
          dIn[ic + i] += c0 * p0 + c1 * p1 + c2 * p2 + c3 * p3;
        }
      }
      for (; o < O; o++) {
        const a = dOut[oa + o], b = dOut[ob + o], c = dOut[oc + o], wb = o * C;
        for (let i = 0; i < C; i++) { dIn[ia + i] += a * W[wb + i]; dIn[ib + i] += b * W[wb + i]; dIn[ic + i] += c * W[wb + i]; }
      }
    }
    for (let n = N3; n < N; n++) {
      const ib = n * C, ob = n * O;
      let o = 0;
      for (; o < O4; o += 4) {
        const g0 = dOut[ob + o], g1 = dOut[ob + o + 1], g2 = dOut[ob + o + 2], g3 = dOut[ob + o + 3];
        const w0 = o * C, w1 = w0 + C, w2 = w1 + C, w3 = w2 + C;
        for (let i = 0; i < C; i++) {
          dIn[ib + i] += g0 * W[w0 + i] + g1 * W[w1 + i] + g2 * W[w2 + i] + g3 * W[w3 + i];
        }
      }
      for (; o < O; o++) {
        const g = dOut[ob + o], wb = o * C;
        for (let i = 0; i < C; i++) dIn[ib + i] += g * W[wb + i];
      }
    }
    const N4 = N - (N % 4), O2 = O - (O % 2);
    let n = 0;
    for (; n < N4; n += 4) {
      const ib0 = n * C, ib1 = ib0 + C, ib2 = ib1 + C, ib3 = ib2 + C;
      const ob0 = n * O, ob1 = ob0 + O, ob2 = ob1 + O, ob3 = ob2 + O;
      let o = 0;
      for (; o < O2; o += 2) {
        const g0a = dOut[ob0 + o], g1a = dOut[ob1 + o], g2a = dOut[ob2 + o], g3a = dOut[ob3 + o];
        const g0b = dOut[ob0 + o + 1], g1b = dOut[ob1 + o + 1], g2b = dOut[ob2 + o + 1], g3b = dOut[ob3 + o + 1];
        const wa = o * C, wb = wa + C;
        for (let i = 0; i < C; i++) {
          const x0 = inp[ib0 + i], x1 = inp[ib1 + i], x2 = inp[ib2 + i], x3 = inp[ib3 + i];
          dW[wa + i] += g0a * x0 + g1a * x1 + g2a * x2 + g3a * x3;
          dW[wb + i] += g0b * x0 + g1b * x1 + g2b * x2 + g3b * x3;
        }
      }
      for (; o < O; o++) {
        const g0 = dOut[ob0 + o], g1 = dOut[ob1 + o], g2 = dOut[ob2 + o], g3 = dOut[ob3 + o];
        const wc = o * C;
        for (let i = 0; i < C; i++) {
          dW[wc + i] += g0 * inp[ib0 + i] + g1 * inp[ib1 + i] + g2 * inp[ib2 + i] + g3 * inp[ib3 + i];
        }
      }
    }
    for (; n < N; n++) {
      const ib = n * C, ob = n * O;
      for (let o = 0; o < O; o++) {
        const g = dOut[ob + o];
        const wb = o * C;
        for (let i = 0; i < C; i++) dW[wb + i] += g * inp[ib + i];
      }
    }
  }

  rmsnormRows(out, inp, w, wOff, N, C) {
    for (let n = 0; n < N; n++) {
      const b = n * C;
      let ss = 0;
      for (let i = 0; i < C; i++) ss += inp[b + i] * inp[b + i];
      const inv = 1 / Math.sqrt(ss / C + 1e-5);
      for (let i = 0; i < C; i++) out[b + i] = w[wOff + i] * inp[b + i] * inv;
    }
  }

  // accumulates dIn += backprop(dOut), dW += ...; recomputes inv from inp
  rmsnormRowsBackward(dOut, inp, w, wOff, dIn, dW, N, C) {
    for (let n = 0; n < N; n++) {
      const b = n * C;
      let ss = 0;
      for (let i = 0; i < C; i++) ss += inp[b + i] * inp[b + i];
      const inv = 1 / Math.sqrt(ss / C + 1e-5);
      let dot = 0; // sum_k dOut_k * w_k * x_k
      for (let i = 0; i < C; i++) dot += dOut[b + i] * w[wOff + i] * inp[b + i];
      const k = dot * inv * inv * inv / C;
      for (let i = 0; i < C; i++) {
        dW[wOff + i] += dOut[b + i] * inp[b + i] * inv;
        dIn[b + i] += dOut[b + i] * w[wOff + i] * inv - k * inp[b + i];
      }
    }
  }

  rope(vec, C, K, T, headSize, forQ, invert) {
    // rotate q rows (width C) or k rows (width K); invert=true applies the transpose
    if (!this.ropeCos) {
      const half = headSize / 2;
      this.ropeCos = new Float64Array(T * half);
      this.ropeSin = new Float64Array(T * half);
      for (let pos = 0; pos < T; pos++) {
        for (let j = 0; j < half; j++) {
          const freq = 1 / Math.pow(10000, (2 * j) / headSize);
          this.ropeCos[pos * half + j] = Math.cos(pos * freq);
          this.ropeSin[pos * half + j] = Math.sin(pos * freq);
        }
      }
    }
    const B = this.B, W = forQ ? C : K, half = headSize / 2;
    const sgn = invert ? -1 : 1;
    for (let bt = 0; bt < B * T; bt++) {
      const pos = bt % T, base = bt * W, tb = pos * half;
      for (let i = 0; i < W; i += 2) {
        const j = (i % headSize) >> 1;
        const fcr = this.ropeCos[tb + j], fci = sgn * this.ropeSin[tb + j];
        const v0 = vec[base + i], v1 = vec[base + i + 1];
        vec[base + i] = v0 * fcr - v1 * fci;
        vec[base + i + 1] = v0 * fci + v1 * fcr;
      }
    }
  }

  // ---- forward -------------------------------------------------------------

  // batch: Int32Array [B * (T+1)]; returns mean cross-entropy loss.
  // computeGrads=false -> eval only (activations still written, grads untouched).
  forwardLoss(batch, computeGrads = true) {
    const { c, B, T, act, params } = this;
    const { dim: C, kvDim: K, hiddenDim: HD, nLayers: L, nHeads: H, vocabSize: V, headSize: HS } = c;
    const kvMul = H / c.nKvHeads;
    const N = B * T;

    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const tok = batch[b * (T + 1) + t];
        act.x0.set(params.tokenEmbedding.subarray(tok * C, (tok + 1) * C), (b * T + t) * C);
      }
    }

    let x = act.x0;
    for (let l = 0; l < L; l++) {
      this.rmsnormRows(act.ln1[l], x, params.rmsAtt, l * C, N, C);
      Trainer.matmul(act.q[l], act.ln1[l], params.wq.subarray(l * C * C), N, C, C);
      Trainer.matmul(act.k[l], act.ln1[l], params.wk.subarray(l * C * K), N, C, K);
      Trainer.matmul(act.v[l], act.ln1[l], params.wv.subarray(l * C * K), N, C, K);
      this.rope(act.q[l], C, K, T, HS, true, false);
      this.rope(act.k[l], C, K, T, HS, false, false);

      // causal attention (fast path for HS=8 — the 260K trainee — streaming contiguous
      // kT/vT rows with fully unrolled head dots; generic path otherwise)
      const att = act.att[l], q = act.q[l], k = act.k[l], v = act.v[l], attOut = act.attOut[l];
      attOut.fill(0);
      const scale = 1 / Math.sqrt(HS);
      const KH = c.nKvHeads;
      if (HS === 8) {
        this.transposeKV(k, this.kT); this.transposeKV(v, this.vT);
        const kT = this.kT, vT = this.vT;
        for (let b = 0; b < B; b++) {
          for (let h = 0; h < H; h++) {
            const base = (b * KH + ((h / kvMul) | 0)) * T * HS;
            for (let t = 0; t < T; t++) {
              const qb = (b * T + t) * C + h * HS;
              const q0 = q[qb], q1 = q[qb + 1], q2 = q[qb + 2], q3 = q[qb + 3],
                    q4 = q[qb + 4], q5 = q[qb + 5], q6 = q[qb + 6], q7 = q[qb + 7];
              const rowB = ((b * H + h) * T + t) * T;
              let max = -Infinity;
              for (let t2 = 0; t2 <= t; t2++) {
                const kb = base + t2 * 8;
                const s = (q0 * kT[kb] + q1 * kT[kb + 1] + q2 * kT[kb + 2] + q3 * kT[kb + 3] +
                           q4 * kT[kb + 4] + q5 * kT[kb + 5] + q6 * kT[kb + 6] + q7 * kT[kb + 7]) * scale;
                att[rowB + t2] = s;
                if (s > max) max = s;
              }
              let sum = 0;
              for (let t2 = 0; t2 <= t; t2++) {
                const e = Math.exp(att[rowB + t2] - max);
                att[rowB + t2] = e; sum += e;
              }
              const inv = 1 / sum;
              let a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
              for (let t2 = 0; t2 <= t; t2++) {
                const p = att[rowB + t2] * inv;
                att[rowB + t2] = p;
                const vb = base + t2 * 8;
                a0 += p * vT[vb];     a1 += p * vT[vb + 1]; a2 += p * vT[vb + 2]; a3 += p * vT[vb + 3];
                a4 += p * vT[vb + 4]; a5 += p * vT[vb + 5]; a6 += p * vT[vb + 6]; a7 += p * vT[vb + 7];
              }
              const ob = (b * T + t) * C + h * HS;
              attOut[ob] = a0; attOut[ob + 1] = a1; attOut[ob + 2] = a2; attOut[ob + 3] = a3;
              attOut[ob + 4] = a4; attOut[ob + 5] = a5; attOut[ob + 6] = a6; attOut[ob + 7] = a7;
            }
          }
        }
      } else {
        for (let b = 0; b < B; b++) {
          for (let h = 0; h < H; h++) {
            const kvh = (h / kvMul) | 0;
            for (let t = 0; t < T; t++) {
              const qb = (b * T + t) * C + h * HS;
              const rowB = ((b * H + h) * T + t) * T;
              let max = -Infinity;
              for (let t2 = 0; t2 <= t; t2++) {
                const kb = (b * T + t2) * K + kvh * HS;
                let s = 0;
                for (let i = 0; i < HS; i++) s += q[qb + i] * k[kb + i];
                s *= scale;
                att[rowB + t2] = s;
                if (s > max) max = s;
              }
              let sum = 0;
              for (let t2 = 0; t2 <= t; t2++) {
                const e = Math.exp(att[rowB + t2] - max);
                att[rowB + t2] = e; sum += e;
              }
              const inv = 1 / sum;
              const ob = (b * T + t) * C + h * HS;
              for (let t2 = 0; t2 <= t; t2++) {
                const p = att[rowB + t2] * inv;
                att[rowB + t2] = p;
                const vb = (b * T + t2) * K + kvh * HS;
                for (let i = 0; i < HS; i++) attOut[ob + i] += p * v[vb + i];
              }
            }
          }
        }
      }

      // wo + residual
      Trainer.matmul(act.res1[l], attOut, params.wo.subarray(l * C * C), N, C, C);
      for (let i = 0; i < N * C; i++) act.res1[l][i] += x[i];

      // ffn
      this.rmsnormRows(act.ln2[l], act.res1[l], params.rmsFfn, l * C, N, C);
      Trainer.matmul(act.h1[l], act.ln2[l], params.w1.subarray(l * HD * C), N, C, HD);
      Trainer.matmul(act.h3[l], act.ln2[l], params.w3.subarray(l * HD * C), N, C, HD);
      const h1 = act.h1[l], h3 = act.h3[l], hg = act.hGate[l];
      for (let i = 0; i < N * HD; i++) {
        const a = h1[i];
        hg[i] = (a / (1 + Math.exp(-a))) * h3[i];
      }
      Trainer.matmul(act.res2[l], hg, params.w2.subarray(l * C * HD), N, HD, C);
      for (let i = 0; i < N * C; i++) act.res2[l][i] += act.res1[l][i];
      x = act.res2[l];
    }

    this.rmsnormRows(act.lnf, x, params.rmsFinal, 0, N, C);
    Trainer.matmul(act.logits, act.lnf, params.tokenEmbedding, N, C, V); // tied classifier

    // softmax CE; overwrite logits with dLogits if computeGrads
    let loss = 0;
    const invN = this.invN ?? 1 / N;   // a one-sequence helper scales by the whole batch's N
    for (let n = 0; n < N; n++) {
      const b = (n / T) | 0, t = n % T;
      const target = batch[b * (T + 1) + t + 1];
      const lb = n * V;
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (act.logits[lb + i] > max) max = act.logits[lb + i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(act.logits[lb + i] - max);
      const logZ = max + Math.log(sum);
      loss += logZ - act.logits[lb + target];
      if (computeGrads) {
        for (let i = 0; i < V; i++) {
          act.logits[lb + i] = Math.exp(act.logits[lb + i] - logZ) * invN;
        }
        act.logits[lb + target] -= invN;
      }
    }
    return loss * invN;
  }

  // ---- backward ------------------------------------------------------------

  backward(batch) {
    const { c, B, T, act, params, grads, g } = this;
    const { dim: C, kvDim: K, hiddenDim: HD, nLayers: L, nHeads: H, vocabSize: V, headSize: HS } = c;
    const kvMul = H / c.nKvHeads;
    const N = B * T;

    // classifier (tied): dLnf, dEmb
    g.dLnf.fill(0);
    Trainer.matmulBackward(act.logits, act.lnf, params.tokenEmbedding,
      g.dLnf, grads.tokenEmbedding, N, C, V);

    // final rmsnorm -> dX = grad wrt res2[L-1]
    g.dX.fill(0);
    this.rmsnormRowsBackward(g.dLnf, act.res2[L - 1], params.rmsFinal, 0,
      g.dX, grads.rmsFinal, N, C);

    const scale = 1 / Math.sqrt(HS);
    for (let l = L - 1; l >= 0; l--) {
      const xIn = l === 0 ? act.x0 : act.res2[l - 1];

      // ---- FFN block: res2 = res1 + w2(silu(w1 ln2) * w3 ln2)
      g.dHGate.fill(0);
      Trainer.matmulBackward(g.dX, act.hGate[l], params.w2.subarray(l * C * HD),
        g.dHGate, grads.w2.subarray(l * C * HD), N, HD, C);
      const h1 = act.h1[l], h3 = act.h3[l];
      for (let i = 0; i < N * HD; i++) {
        const a = h1[i], sig = 1 / (1 + Math.exp(-a));
        const silu = a * sig;
        g.dH1[i] = g.dHGate[i] * h3[i] * (sig + a * sig * (1 - sig));
        g.dH3[i] = g.dHGate[i] * silu;
      }
      g.dLn.fill(0);
      Trainer.matmulBackward(g.dH1, act.ln2[l], params.w1.subarray(l * HD * C),
        g.dLn, grads.w1.subarray(l * HD * C), N, C, HD);
      Trainer.matmulBackward(g.dH3, act.ln2[l], params.w3.subarray(l * HD * C),
        g.dLn, grads.w3.subarray(l * HD * C), N, C, HD);
      // rmsnorm(res1) backward adds into dX (identity path already there)
      this.rmsnormRowsBackward(g.dLn, act.res1[l], params.rmsFfn, l * C,
        g.dX, grads.rmsFfn, N, C);

      // ---- attention block: res1 = xIn + wo(attnOut)
      g.dAttOut.fill(0);
      Trainer.matmulBackward(g.dX, act.attOut[l], params.wo.subarray(l * C * C),
        g.dAttOut, grads.wo.subarray(l * C * C), N, C, C);

      g.dQ.fill(0); g.dK.fill(0); g.dV.fill(0);
      const att = act.att[l], q = act.q[l], k = act.k[l], v = act.v[l];
      const dPRow = new this.AT(T);
      const KH = c.nKvHeads;
      if (HS === 8) {
        this.transposeKV(k, this.kT); this.transposeKV(v, this.vT);
        this.dKT.fill(0); this.dVT.fill(0);
        const kT = this.kT, vT = this.vT, dKT = this.dKT, dVT = this.dVT, dAO = g.dAttOut;
        for (let b = 0; b < B; b++) {
          for (let h = 0; h < H; h++) {
            const base = (b * KH + ((h / kvMul) | 0)) * T * HS;
            for (let t = 0; t < T; t++) {
              const rowB = ((b * H + h) * T + t) * T;
              const ob = (b * T + t) * C + h * HS;
              const d0 = dAO[ob], d1 = dAO[ob + 1], d2 = dAO[ob + 2], d3 = dAO[ob + 3],
                    d4 = dAO[ob + 4], d5 = dAO[ob + 5], d6 = dAO[ob + 6], d7 = dAO[ob + 7];
              let pdSum = 0;
              for (let t2 = 0; t2 <= t; t2++) {
                const vb = base + t2 * 8;
                const p = att[rowB + t2];
                const dp = d0 * vT[vb] + d1 * vT[vb + 1] + d2 * vT[vb + 2] + d3 * vT[vb + 3] +
                           d4 * vT[vb + 4] + d5 * vT[vb + 5] + d6 * vT[vb + 6] + d7 * vT[vb + 7];
                dVT[vb] += p * d0; dVT[vb + 1] += p * d1; dVT[vb + 2] += p * d2; dVT[vb + 3] += p * d3;
                dVT[vb + 4] += p * d4; dVT[vb + 5] += p * d5; dVT[vb + 6] += p * d6; dVT[vb + 7] += p * d7;
                dPRow[t2] = dp;
                pdSum += p * dp;
              }
              const q0 = q[ob], q1 = q[ob + 1], q2 = q[ob + 2], q3 = q[ob + 3],
                    q4 = q[ob + 4], q5 = q[ob + 5], q6 = q[ob + 6], q7 = q[ob + 7];
              let dq0 = 0, dq1 = 0, dq2 = 0, dq3 = 0, dq4 = 0, dq5 = 0, dq6 = 0, dq7 = 0;
              for (let t2 = 0; t2 <= t; t2++) {
                const dS = att[rowB + t2] * (dPRow[t2] - pdSum) * scale;
                const kb = base + t2 * 8;
                dq0 += dS * kT[kb];     dq1 += dS * kT[kb + 1]; dq2 += dS * kT[kb + 2]; dq3 += dS * kT[kb + 3];
                dq4 += dS * kT[kb + 4]; dq5 += dS * kT[kb + 5]; dq6 += dS * kT[kb + 6]; dq7 += dS * kT[kb + 7];
                dKT[kb] += dS * q0;     dKT[kb + 1] += dS * q1; dKT[kb + 2] += dS * q2; dKT[kb + 3] += dS * q3;
                dKT[kb + 4] += dS * q4; dKT[kb + 5] += dS * q5; dKT[kb + 6] += dS * q6; dKT[kb + 7] += dS * q7;
              }
              g.dQ[ob] = dq0; g.dQ[ob + 1] = dq1; g.dQ[ob + 2] = dq2; g.dQ[ob + 3] = dq3;
              g.dQ[ob + 4] = dq4; g.dQ[ob + 5] = dq5; g.dQ[ob + 6] = dq6; g.dQ[ob + 7] = dq7;
            }
          }
        }
        this.transposeKV(dKT, g.dK, true);  // dKT -> dK layout
        this.transposeKV(dVT, g.dV, true);
      } else {
        for (let b = 0; b < B; b++) {
          for (let h = 0; h < H; h++) {
            const kvh = (h / kvMul) | 0;
            for (let t = 0; t < T; t++) {
              const rowB = ((b * H + h) * T + t) * T;
              const ob = (b * T + t) * C + h * HS;
              const qb = ob; // q and attOut share head slot layout
              let pdSum = 0;
              for (let t2 = 0; t2 <= t; t2++) {
                const vb = (b * T + t2) * K + kvh * HS;
                let dp = 0;
                for (let i = 0; i < HS; i++) {
                  dp += g.dAttOut[ob + i] * v[vb + i];
                  g.dV[vb + i] += att[rowB + t2] * g.dAttOut[ob + i];
                }
                dPRow[t2] = dp;
                pdSum += att[rowB + t2] * dp;
              }
              for (let t2 = 0; t2 <= t; t2++) {
                const dS = att[rowB + t2] * (dPRow[t2] - pdSum) * scale;
                const kb = (b * T + t2) * K + kvh * HS;
                for (let i = 0; i < HS; i++) {
                  g.dQ[qb + i] += dS * k[kb + i];
                  g.dK[kb + i] += dS * q[qb + i];
                }
              }
            }
          }
        }
      }
      this.rope(g.dQ, C, K, T, HS, true, true);   // inverse rotation
      this.rope(g.dK, C, K, T, HS, false, true);

      g.dLn.fill(0);
      Trainer.matmulBackward(g.dQ, act.ln1[l], params.wq.subarray(l * C * C),
        g.dLn, grads.wq.subarray(l * C * C), N, C, C);
      Trainer.matmulBackward(g.dK, act.ln1[l], params.wk.subarray(l * C * K),
        g.dLn, grads.wk.subarray(l * C * K), N, C, K);
      Trainer.matmulBackward(g.dV, act.ln1[l], params.wv.subarray(l * C * K),
        g.dLn, grads.wv.subarray(l * C * K), N, C, K);
      // rmsnorm(xIn) backward adds into dX, which already carries the identity path
      this.rmsnormRowsBackward(g.dLn, xIn, params.rmsAtt, l * C,
        g.dX, grads.rmsAtt, N, C);
    }

    // embedding backward
    for (let b = 0; b < B; b++) {
      for (let t = 0; t < T; t++) {
        const tok = batch[b * (T + 1) + t];
        const src = (b * T + t) * C, dst = tok * C;
        for (let i = 0; i < C; i++) grads.tokenEmbedding[dst + i] += g.dX[src + i];
      }
    }
  }

  // ---- optimizer -----------------------------------------------------------

  zeroGrads() { for (const k of Object.keys(this.grads)) this.grads[k].fill(0); }

  gradNorm() {
    let sq = 0;
    for (const k of Object.keys(this.grads)) {
      const gr = this.grads[k];
      for (let i = 0; i < gr.length; i++) sq += gr[i] * gr[i];
    }
    return Math.sqrt(sq);
  }

  adamw() {
    const { lr, weightDecay, beta1, beta2, eps, clip } = this.opt;
    this.stepCount++;
    const norm = this.gradNorm();
    const gscale = clip > 0 && norm > clip ? clip / norm : 1;
    const bc1 = 1 - Math.pow(beta1, this.stepCount);
    const bc2 = 1 - Math.pow(beta2, this.stepCount);
    for (const key of Object.keys(this.params)) {
      const p = this.params[key], gr = this.grads[key], m = this.m[key], v = this.v[key];
      const wd = DECAYED.has(key) ? weightDecay : 0;
      for (let i = 0; i < p.length; i++) {
        const gi = gr[i] * gscale;
        m[i] = beta1 * m[i] + (1 - beta1) * gi;
        v[i] = beta2 * v[i] + (1 - beta2) * gi * gi;
        p[i] -= lr * ((m[i] / bc1) / (Math.sqrt(v[i] / bc2) + eps) + wd * p[i]);
      }
    }
    return norm;
  }

  // a B=1 helper sharing this trainer's weights: it works out one sequence at a time
  seqTrainer() {
    if (!this._seq) {
      this._seq = new Trainer({ config: this.c, weights: this.params }, { B: 1, T: this.T, f64: this.AT === Float64Array });
      this._seq.params = this.params;   // shared, so it always sees the current weights
    }
    return this._seq;
  }

  // sequence b of batch: its gradient into tr.grads; returns its share of the mean loss
  static seqGrad(tr, batch, b, Bn) {
    const T = tr.T, seq = batch.subarray(b * (T + 1), (b + 1) * (T + 1));
    tr.invN = 1 / (Bn * T);
    tr.zeroGrads();
    const loss = tr.forwardLoss(seq, true);
    tr.backward(seq);
    return loss;
  }

  // grads += one sequence's gradient: a trainer's grads, or a flat array in PARAM order
  addGrads(src) {
    let off = 0;
    for (const key of Object.keys(this.grads)) {
      const g = this.grads[key], s = src instanceof Float32Array || src instanceof Float64Array
        ? src.subarray(off, off + g.length) : src[key];
      for (let i = 0; i < g.length; i++) g[i] += s[i];
      off += g.length;
    }
  }

  // one full training step; returns { loss, gradNorm, ms }
  step(batch) {
    const t0 = performance.now();
    this.zeroGrads();
    const seq = this.seqTrainer(), Bn = batch.length / (this.T + 1);
    let loss = 0;
    for (let b = 0; b < Bn; b++) { loss += Trainer.seqGrad(seq, batch, b, Bn); this.addGrads(seq.grads); }
    const norm = this.adamw();
    return { loss, gradNorm: norm, ms: performance.now() - t0 };
  }

  // the same step with the sequences worked on by a TrainPool; identical numbers
  async stepWith(pool, batch) {
    const t0 = performance.now();
    this.zeroGrads();
    const parts = await pool.grads(this, batch);   // per sequence, in sequence order
    let loss = 0;
    for (const p of parts) { loss += p.loss; this.addGrads(p.grads); }
    const norm = this.adamw();
    return { loss, gradNorm: norm, ms: performance.now() - t0 };
  }

  // eval-only mean CE loss on a batch (no grads, no optimizer state touched): the sum of
  // each sequence's share, in order. A one-sequence trainer runs it on itself, so its
  // activations (act.logits) hold the evaluated sequence afterwards.
  evalLoss(batch) {
    const Bn = batch.length / (this.T + 1);
    const tr = this.B === 1 && Bn === 1 ? this : this.seqTrainer();
    let loss = 0;
    for (let b = 0; b < Bn; b++) {
      tr.invN = 1 / (Bn * this.T);
      loss += tr.forwardLoss(batch.subarray(b * (this.T + 1), (b + 1) * (this.T + 1)), false);
    }
    if (tr === this) this.invN = undefined;
    return loss;
  }

  async evalLossWith(pool, batch) {
    let loss = 0;
    for (const p of await pool.losses(this, batch)) loss += p.loss;
    return loss;
  }

  // weights dict compatible with model.js createModel / bin-reader writeCheckpoint
  modelWeights() {
    const w = {};
    for (const k of Object.keys(this.params)) {
      w[k] = this.params[k] instanceof Float32Array
        ? this.params[k] : Float32Array.from(this.params[k]);
    }
    w.wcls = w.tokenEmbedding;
    return w;
  }
}

// ---- TrainPool: sequences of a batch worked on by worker threads ---------------------
// Each worker holds a one-sequence Trainer (built from this file's own code via
// Function.prototype.toString, so no separate worker file). Per step it gets the current
// weights and some sequence numbers, and sends back each sequence's loss share and
// gradient; the main trainer adds them up in sequence order (stepWith), exactly as step()
// does alone. Sequence b goes to worker b % K.
function trainPoolMain(self) {
  let tr = null;
  const flat = (o) => { let n = 0; for (const k in o) n += o[k].length; const out = new Float32Array(n); let p = 0; for (const k in o) { out.set(o[k], p); p += o[k].length; } return out; };
  const setFlat = (o, a) => { let p = 0; for (const k in o) { o[k].set(a.subarray(p, p + o[k].length)); p += o[k].length; } };
  self.onmessage = (e) => {
    const q = e.data;
    try {
      if (q.op === 'init') {
        const shapes = PARAM_SHAPES(q.config), weights = {};
        let p = 0;
        for (const k of Object.keys(shapes)) { weights[k] = q.params.subarray(p, p + shapes[k]); p += shapes[k]; }
        tr = new Trainer({ config: q.config, weights }, { B: 1, T: q.T });
        self.postMessage({ id: q.id });
        return;
      }
      setFlat(tr.params, q.params);
      const out = [], transfers = [];
      for (const b of q.seqs) {
        if (q.op === 'grads') {
          const loss = Trainer.seqGrad(tr, q.batch, b, q.Bn);
          const grads = flat(tr.grads);
          out.push({ b, loss, grads }); transfers.push(grads.buffer);
        } else {
          tr.invN = 1 / (q.Bn * q.T);
          out.push({ b, loss: tr.forwardLoss(q.batch.subarray(b * (q.T + 1), (b + 1) * (q.T + 1)), false) });
        }
      }
      self.postMessage({ id: q.id, out }, transfers);
    } catch (err) {
      self.postMessage({ id: q.id, error: String((err && err.stack) || err) });
    }
  };
}

export function trainPoolSource() {
  return 'const PARAM_SHAPES = ' + PARAM_SHAPES.toString() + ';\n' +
    'const DECAYED = new Set(' + JSON.stringify([...DECAYED]) + ');\n' +
    Trainer.toString() + '\n(' + trainPoolMain.toString() + ')(self);\n';
}

export class TrainPool {
  // spawn() -> a Worker-like object running trainPoolSource(); by default a blob worker.
  // Resolves to null wherever workers can't start (training then runs on one core).
  static async create(trainer, { K = null, spawn = null, timeoutMs = 8000 } = {}) {
    if (!K) {
      // the fewest workers that finish a batch as soon as the most we could use would:
      // 8 sequences on 7 spare cores take two rounds either way, so 4 workers do it
      const hc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
      const most = Math.min(trainer.B, Math.max(2, hc - 1)), rounds = Math.ceil(trainer.B / most);
      K = Math.ceil(trainer.B / rounds);
    }
    let make = spawn;
    if (!make) {
      if (typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined' || !URL.createObjectURL) return null;
      let url;
      try { url = URL.createObjectURL(new Blob([trainPoolSource()], { type: 'text/javascript' })); } catch (e) { return null; }
      make = () => new Worker(url);
    }
    const pool = new TrainPool(trainer, K);
    try {
      for (let k = 0; k < K; k++) {
        const wk = make();
        wk.onmessage = (e) => pool._reply(e.data);
        wk.onerror = (e) => pool._failAll(new Error('train worker failed: ' + ((e && e.message) || 'error')));
        pool.workers.push(wk);
      }
      const params = pool._params(trainer);
      const init = Promise.all(pool.workers.map(wk => pool._call(wk, { op: 'init', config: trainer.c, T: trainer.T, params })));
      let timer;
      const late = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('train pool: workers did not start')), timeoutMs); });
      try { await Promise.race([init, late]); } finally { clearTimeout(timer); }
      return pool;
    } catch (e) {
      pool.dispose();
      return null;
    }
  }

  constructor(trainer, K) {
    this.K = K; this.T = trainer.T; this.workers = []; this.pending = new Map(); this._id = 0;
  }
  _params(trainer) {
    let n = 0;
    for (const k in trainer.params) n += trainer.params[k].length;
    const out = new Float32Array(n);
    let p = 0;
    for (const k in trainer.params) { out.set(trainer.params[k], p); p += trainer.params[k].length; }
    return out;
  }
  _call(wk, msg) {
    return new Promise((res, rej) => {
      const id = ++this._id;
      this.pending.set(id, { res, rej });
      msg.id = id;
      wk.postMessage(msg);
    });
  }
  _reply(d) {
    const p = d && this.pending.get(d.id);
    if (!p) return;
    this.pending.delete(d.id);
    d.error ? p.rej(new Error(d.error)) : p.res(d);
  }
  _failAll(err) {
    for (const p of this.pending.values()) p.rej(err);
    this.pending.clear();
    this.broken = true;
  }
  dispose() {
    for (const wk of this.workers) { try { wk.terminate(); } catch (e) { /* gone already */ } }
    this.workers = []; this._failAll(new Error('train pool disposed'));
  }
  async _run(op, trainer, batch) {
    if (this.broken) throw new Error('train pool is broken');
    if (trainer.T !== this.T) throw new Error('train pool: sequence length changed');
    const Bn = batch.length / (this.T + 1), params = this._params(trainer);
    const replies = await Promise.all(this.workers.map((wk, k) => {
      const seqs = [];
      for (let b = k; b < Bn; b += this.K) seqs.push(b);
      return seqs.length ? this._call(wk, { op, params, batch, Bn, T: this.T, seqs }) : { out: [] };
    }));
    const all = [];
    for (const r of replies) for (const o of r.out) all[o.b] = o;
    return all;
  }
  grads(trainer, batch) { return this._run('grads', trainer, batch); }
  losses(trainer, batch) { return this._run('loss', trainer, batch); }
}
