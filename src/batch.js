// batch.js — the Microscope's fast engine. model.js's forward() takes one piece through
// every layer, so it reads all the weights (440 MB for the Giant) once per piece. Here
// the same pass runs layer by layer over a whole run of pieces at once, so each layer's
// weights are read once per run; and where the browser allows, the big matrices are
// split by rows across a pool of worker threads (the J-Lens Explorer's pool), each one
// computing its slice of every product on its own core. Every number is made by the
// same operations in the same order as forward(), so captures, continuations and lens
// rows are bit-identical to the one-piece-at-a-time path for any number of workers
// (E14 checks this). DOM-free; Node + browser.

import { matmulBatch, rmsnorm, softmax, argmax, expertSwitches, routeExperts } from './model.js';
import { Capture, lensInput } from './capture.js';
import { jLensInput } from './jlens.js';
import { plan } from './steer.js';

// ---- the pool --------------------------------------------------------------------
// A pool worker holds row slices of every layer matrix and of the classifier, and
// answers one product at a time. Its source is built from this function plus model.js's
// matmulBatch (Function.prototype.toString), so it needs no file of its own and runs
// the very same kernel. (Its `f` is defined in that source, not here.)
function poolMain(self) {
  let S = null;
  self.onmessage = (e) => {
    const q = e.data;
    try {
      if (q.op === 'init') { S = q; self.postMessage({ id: q.id }); return; }
      const c = S.config, N = q.N;
      if (q.op === 'qkv') {
        const L = S.layers[q.l], dn = S.d1 - S.d0, kvn = S.kv1 - S.kv0;
        const qo = new Float32Array(N * dn), ko = new Float32Array(N * kvn), vo = new Float32Array(N * kvn);
        matmulBatch(qo, q.X, L.wq, 0, N, c.dim, dn);
        matmulBatch(ko, q.X, L.wk, 0, N, c.dim, kvn);
        matmulBatch(vo, q.X, L.wv, 0, N, c.dim, kvn);
        self.postMessage({ id: q.id, q: qo, k: ko, v: vo }, [qo.buffer, ko.buffer, vo.buffer]);
      } else if (q.op === 'wo' || q.op === 'w2') {
        const dn = S.d1 - S.d0, out = new Float32Array(N * dn);
        matmulBatch(out, q.X, S.layers[q.l][q.op], 0, N, q.op === 'wo' ? c.dim : c.hiddenDim, dn);
        self.postMessage({ id: q.id, out }, [out.buffer]);
      } else if (q.op === 'ffn') {
        const L = S.layers[q.l], hn = S.h1 - S.h0;
        const hb = new Float32Array(N * hn), hb2 = new Float32Array(N * hn);
        matmulBatch(hb, q.X, L.w1, 0, N, c.dim, hn);
        matmulBatch(hb2, q.X, L.w3, 0, N, c.dim, hn);
        for (let i = 0; i < N * hn; i++) {   // SwiGLU, exactly as model.js
          let val = hb[i];
          val = f(val * f(1 / f(1 + f(Math.exp(-val)))));
          val = f(val * hb2[i]);
          hb[i] = val;
        }
        self.postMessage({ id: q.id, out: hb }, [hb.buffer]);
      } else if (q.op === 'cls') {
        const vn = S.v1 - S.v0, out = new Float32Array(N * vn);
        matmulBatch(out, q.X, S.wcls, 0, N, c.dim, vn);
        self.postMessage({ id: q.id, out }, [out.buffer]);
      } else if (q.op === 'set-mat') {   // an extra matrix's row slice (Pool.addMatrix)
        (S.mats || (S.mats = {}))[q.name] = { m: q.m, rows: q.rows, cols: q.cols };
        self.postMessage({ id: q.id });
      } else if (q.op === 'mat') {
        const M = S.mats[q.name], out = new Float32Array(N * M.rows);
        matmulBatch(out, q.X, M.m, 0, N, M.cols, M.rows);
        self.postMessage({ id: q.id, out }, [out.buffer]);
      } else throw new Error('pool: unknown op ' + q.op);
    } catch (err) {
      self.postMessage({ id: q.id, error: String((err && err.stack) || err) });
    }
  };
}

export function poolWorkerSource() {
  return 'const f = Math.fround;\n' + matmulBatch.toString() + '\n(' + poolMain.toString() + ')(self);\n';
}

// how many workers: leave a core for the page and one for the Microscope's own worker
export function poolSize() {
  const hc = (typeof navigator !== 'undefined' && navigator.hardwareConcurrency) || 4;
  return Math.min(6, Math.max(2, hc - 2));
}

const splitRange = (R, K) => { const b = [0]; for (let k = 1; k <= K; k++) b.push(Math.round(R * k / K)); return b; };

// copy row slices [a, b) of a worker's [N, b - a] answer into dst [N, width]
function scatter(dst, src, N, width, a, b) {
  const n = b - a;
  for (let p = 0; p < N; p++) dst.set(src.subarray(p * n, (p + 1) * n), p * width + a);
}

export class Pool {
  // spawn() -> a Worker-like object running poolWorkerSource(); by default a blob worker.
  // Resolves to null wherever workers can't start (the caller then computes alone).
  static async create(model, { K = poolSize(), spawn = null, timeoutMs = 8000 } = {}) {
    let make = spawn;
    if (!make) {
      if (typeof Worker === 'undefined' || typeof Blob === 'undefined' || typeof URL === 'undefined' || !URL.createObjectURL) return null;
      let url;
      try { url = URL.createObjectURL(new Blob([poolWorkerSource()], { type: 'text/javascript' })); } catch (e) { return null; }
      make = () => new Worker(url);
    }
    const pool = new Pool(model.config, K);
    try {
      for (let k = 0; k < K; k++) {
        const wk = make();
        wk.onmessage = (e) => pool._reply(e.data);
        wk.onerror = (e) => pool._failAll(new Error('pool worker failed: ' + ((e && e.message) || 'error')));
        pool.workers.push(wk);
      }
      const init = Promise.all(pool.workers.map((wk, k) => {
        const { msg, transfers } = pool._slices(model.weights, k);
        return pool._call(wk, msg, transfers);
      }));
      let timer;
      const late = new Promise((_, rej) => { timer = setTimeout(() => rej(new Error('pool: workers did not start')), timeoutMs); });
      try { await Promise.race([init, late]); } finally { clearTimeout(timer); }
      return pool;
    } catch (e) {
      pool.dispose();
      return null;
    }
  }

  constructor(config, K) {
    const c = config;
    this.config = c; this.K = K; this.workers = []; this.pending = new Map(); this._id = 0;
    this.bD = splitRange(c.dim, K); this.bKV = splitRange(c.kvDim, K);
    this.bH = splitRange(c.hiddenDim, K); this.bV = splitRange(c.vocabSize, K);
  }

  _slices(w, k) {
    const c = this.config, dim = c.dim, kv = c.kvDim, hd = c.hiddenDim, transfers = [];
    const cut = (a, from, to) => { const s = a.slice(from, to); transfers.push(s.buffer); return s; };
    const [d0, d1, kv0, kv1, h0, h1, v0, v1] = [this.bD[k], this.bD[k + 1], this.bKV[k], this.bKV[k + 1],
      this.bH[k], this.bH[k + 1], this.bV[k], this.bV[k + 1]];
    const layers = [];
    for (let l = 0; l < c.nLayers; l++) {
      layers.push({
        wq: cut(w.wq, l * dim * dim + d0 * dim, l * dim * dim + d1 * dim),
        wk: cut(w.wk, l * dim * kv + kv0 * dim, l * dim * kv + kv1 * dim),
        wv: cut(w.wv, l * dim * kv + kv0 * dim, l * dim * kv + kv1 * dim),
        wo: cut(w.wo, l * dim * dim + d0 * dim, l * dim * dim + d1 * dim),
        w1: cut(w.w1, l * dim * hd + h0 * dim, l * dim * hd + h1 * dim),
        w3: cut(w.w3, l * dim * hd + h0 * dim, l * dim * hd + h1 * dim),
        w2: cut(w.w2, l * hd * dim + d0 * hd, l * hd * dim + d1 * hd),
      });
    }
    const config = { dim, kvDim: kv, hiddenDim: hd, vocabSize: c.vocabSize };
    return { msg: { op: 'init', config, d0, d1, kv0, kv1, h0, h1, v0, v1, layers, wcls: cut(w.wcls, v0 * dim, v1 * dim) }, transfers };
  }

  _call(wk, msg, transfers = []) {
    return new Promise((res, rej) => {
      const id = ++this._id;
      this.pending.set(id, { res, rej });
      msg.id = id;
      wk.postMessage(msg, transfers);
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
  _all(msg) {
    if (this.broken) return Promise.reject(new Error('pool is broken'));
    return Promise.all(this.workers.map(wk => this._call(wk, { ...msg })));
  }
  dispose() {
    for (const wk of this.workers) { try { wk.terminate(); } catch (e) { /* gone already */ } }
    this.workers = []; this._failAll(new Error('pool disposed'));
  }

  async qkv(l, X, N, Q, K, V) {
    const c = this.config, parts = await this._all({ op: 'qkv', l, X, N });
    for (let k = 0; k < this.K; k++) {
      scatter(Q, parts[k].q, N, c.dim, this.bD[k], this.bD[k + 1]);
      scatter(K, parts[k].k, N, c.kvDim, this.bKV[k], this.bKV[k + 1]);
      scatter(V, parts[k].v, N, c.kvDim, this.bKV[k], this.bKV[k + 1]);
    }
  }
  async mm(op, l, X, N, out) {   // op 'wo' | 'w2': the residual-stream outputs
    const parts = await this._all({ op, l, X, N });
    for (let k = 0; k < this.K; k++) scatter(out, parts[k].out, N, this.config.dim, this.bD[k], this.bD[k + 1]);
  }
  async ffn(l, X, N, out) {
    const parts = await this._all({ op: 'ffn', l, X, N });
    for (let k = 0; k < this.K; k++) scatter(out, parts[k].out, N, this.config.hiddenDim, this.bH[k], this.bH[k + 1]);
  }
  async cls(X, N, out) {
    const parts = await this._all({ op: 'cls', X, N });
    for (let k = 0; k < this.K; k++) scatter(out, parts[k].out, N, this.config.vocabSize, this.bV[k], this.bV[k + 1]);
  }
  // an extra matrix M [rows, cols], split by rows across the workers like the weights
  // (the Thought bubble's story-vocabulary readout); mat() then multiplies by it
  async addMatrix(name, M, rows, cols) {
    const b = splitRange(rows, this.K);
    (this.mats || (this.mats = {}))[name] = { rows, b };
    await Promise.all(this.workers.map((wk, k) => {
      const m = M.slice(b[k] * cols, b[k + 1] * cols);
      return this._call(wk, { op: 'set-mat', name, m, rows: b[k + 1] - b[k], cols }, [m.buffer]);
    }));
  }
  async mat(name, X, N, out) {
    const { rows, b } = this.mats[name], parts = await this._all({ op: 'mat', name, X, N });
    for (let k = 0; k < this.K; k++) scatter(out, parts[k].out, N, rows, b[k], b[k + 1]);
  }
}

// ---- the batched forward pass ------------------------------------------------------
// Positions start .. start+ids.length-1 through every layer, filling model.state's KV
// cache (so a later call carries on from there, as generation does). tap / ablate are
// forward()'s. Returns the final-normed residuals [N, dim]: what the classifier reads.
// edit: { layers: Set, fn(l, pos, x) } changes the residual stream x in place after a
// layer's ffn add, before anything reads it (the Thought bubble's thought swap).
export async function forwardBatch(model, ids, start, { tap = null, ablate = null, pool = null, onLayer = null, edit = null } = {}) {
  const c = model.config, w = model.weights, s = model.state, f = Math.fround;
  const N = ids.length, dim = c.dim, kvDim = c.kvDim, hd = c.hiddenDim, headSize = c.headSize;
  const kvMul = c.nHeads / c.nKvHeads;
  if (start + N > c.seqLen) throw new Error(`forwardBatch: ${start + N} positions > seqLen ${c.seqLen}`);

  const sw = expertSwitches(c, ablate);
  let abH = null, abNL = null;   // same reading of `ablate` as forward()
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

  const X = new Float32Array(N * dim), XB = new Float32Array(N * dim), Q = new Float32Array(N * dim);
  const KB = new Float32Array(N * kvDim), VB = new Float32Array(N * kvDim);
  const HB = new Float32Array(N * hd), O = new Float32Array(N * dim);
  const HB2 = pool ? null : new Float32Array(N * hd);
  const att = new Float32Array(c.seqLen);
  const row = (A, p, n) => A.subarray(p * n, (p + 1) * n);
  for (let p = 0; p < N; p++) X.set(w.tokenEmbedding.subarray(ids[p] * dim, (ids[p] + 1) * dim), p * dim);
  const sqrtHs = f(Math.sqrt(headSize));

  for (let l = 0; l < c.nLayers; l++) {
    for (let p = 0; p < N; p++) rmsnorm(row(XB, p, dim), row(X, p, dim), w.rmsAtt, l * dim, dim);
    if (pool) await pool.qkv(l, XB, N, Q, KB, VB);
    else {
      matmulBatch(Q, XB, w.wq, l * dim * dim, N, dim, dim);
      matmulBatch(KB, XB, w.wk, l * dim * kvDim, N, dim, kvDim);
      matmulBatch(VB, XB, w.wv, l * dim * kvDim, N, dim, kvDim);
    }

    // RoPE as forward() does it, then into the KV cache
    const loff = l * c.seqLen * kvDim;
    for (let p = 0; p < N; p++) {
      const pos = start + p, qo = p * dim, ko = p * kvDim;
      for (let i = 0; i < dim; i += 2) {
        const headDim = i % headSize;
        const freq = f(1 / f(Math.pow(10000, f(headDim / headSize))));
        const val = f(pos * freq);
        const fcr = f(Math.cos(val));
        const fci = f(Math.sin(val));
        const v0 = Q[qo + i], v1 = Q[qo + i + 1];
        Q[qo + i] = f(v0 * fcr) - f(v1 * fci);
        Q[qo + i + 1] = f(v0 * fci) + f(v1 * fcr);
        if (i < kvDim) {
          const k0 = KB[ko + i], k1 = KB[ko + i + 1];
          KB[ko + i] = f(k0 * fcr) - f(k1 * fci);
          KB[ko + i + 1] = f(k0 * fci) + f(k1 * fcr);
        }
      }
      s.keyCache.set(row(KB, p, kvDim), loff + pos * kvDim);
      s.valueCache.set(row(VB, p, kvDim), loff + pos * kvDim);
    }

    // attention over the cache; the head outputs replace this position's row of XB
    for (let p = 0; p < N; p++) {
      const pos = start + p, qo = p * dim;
      for (let h = 0; h < c.nHeads; h++) {
        const qOff = qo + h * headSize;
        const kvHeadOff = ((h / kvMul) | 0) * headSize;
        for (let t = 0; t <= pos; t++) {
          const kBase = loff + t * kvDim + kvHeadOff;
          let score = 0;
          for (let i = 0; i < headSize; i++) score = f(score + f(Q[qOff + i] * s.keyCache[kBase + i]));
          att[t] = f(score / sqrtHs);
        }
        softmax(att, 0, pos + 1);
        if (tap && tap.attention) tap.attention(l, h, pos, att.subarray(0, pos + 1));
        XB.fill(0, qOff, qOff + headSize);
        if (abH && abH.has(l + ':' + h)) continue;   // head contributes zeros
        for (let t = 0; t <= pos; t++) {
          const vBase = loff + t * kvDim + kvHeadOff;
          const a = att[t];
          for (let i = 0; i < headSize; i++) XB[qOff + i] = XB[qOff + i] + f(a * s.valueCache[vBase + i]);
        }
      }
    }

    if (pool) await pool.mm('wo', l, XB, N, O);
    else matmulBatch(O, XB, w.wo, l * dim * dim, N, dim, dim);
    for (let i = 0; i < N * dim; i++) X[i] = X[i] + O[i];
    if (tap && tap.residual) for (let p = 0; p < N; p++) tap.residual(l, 'attn', start + p, row(X, p, dim));

    for (let p = 0; p < N; p++) rmsnorm(row(XB, p, dim), row(X, p, dim), w.rmsFfn, l * dim, dim);
    if (c.experts) expertLayer(model, l, XB, N, start, sw, tap, O);   // the experts run here, not in the pool
    else {
      if (pool) await pool.ffn(l, XB, N, HB);
      else {
        matmulBatch(HB, XB, w.w1, l * dim * hd, N, dim, hd);
        matmulBatch(HB2, XB, w.w3, l * dim * hd, N, dim, hd);
        for (let i = 0; i < N * hd; i++) {
          let val = HB[i];
          val = f(val * f(1 / f(1 + f(Math.exp(-val)))));
          val = f(val * HB2[i]);
          HB[i] = val;
        }
      }
      for (let p = 0; p < N; p++) {
        const hb = row(HB, p, hd);
        if (abNL) for (const j of abNL[l]) hb[j] = 0;
        if (tap && tap.ffn) tap.ffn(l, start + p, hb);   // a clamp written here is what w2 reads
      }

      if (pool) await pool.mm('w2', l, HB, N, O);
      else matmulBatch(O, HB, w.w2, l * hd * dim, N, hd, dim);
    }
    for (let i = 0; i < N * dim; i++) X[i] = X[i] + O[i];
    if (edit && edit.layers.has(l)) for (let p = 0; p < N; p++) edit.fn(l, start + p, row(X, p, dim));
    if (tap && tap.residual) for (let p = 0; p < N; p++) tap.residual(l, 'ffn', start + p, row(X, p, dim));
    if (onLayer) onLayer(l + 1, c.nLayers);
  }

  for (let p = 0; p < N; p++) rmsnorm(row(X, p, dim), row(X, p, dim), w.rmsFinal, 0, dim);
  return X;
}

// A layer of experts over N rows of normed inputs XB (a mind made of experts; see model.js):
// the router for every row in one product, then each expert once, over just the rows sent to
// it. out [N, dim] gets each row's weighted sum, added best choice first as forward() adds it,
// so every number matches the one-piece path. The experts are small (15M-sized minds), so they
// run on this thread; the pool still does the attention and the classifier.
function expertLayer(model, l, XB, N, start, sw, tap, out) {
  const c = model.config, w = model.weights, ex = c.experts, f = Math.fround;
  const n = ex.n, k = ex.topK, eh = ex.hidden, dim = c.dim;
  const RL = new Float32Array(N * n);
  matmulBatch(RL, XB, w.router, l * n * dim, N, dim, n);
  const top = new Int32Array(N * k).fill(-1), gate = new Float32Array(N * k);
  const probs = new Float32Array(n), t1 = new Int32Array(k), g1 = new Float32Array(k);
  const offL = sw && sw.off[l], muteL = sw && sw.mute[l];
  for (let p = 0; p < N; p++) {
    const kk = routeExperts(RL.subarray(p * n, (p + 1) * n), probs, n, k, ex.renorm, offL, t1, g1);
    if (tap && tap.router) tap.router(l, start + p, probs, t1, g1, kk);
    for (let j = 0; j < kk; j++) { top[p * k + j] = t1[j]; gate[p * k + j] = g1[j]; }
  }
  const C = new Float32Array(k * N * dim), has = new Uint8Array(k * N);   // C[j][p]: row p's j-th choice
  for (let e = 0; e < n; e++) {
    if (muteL && muteL.has(e)) continue;   // muted: chosen, but adds nothing
    const rows = [];
    for (let p = 0; p < N; p++) for (let j = 0; j < k; j++) if (top[p * k + j] === e) rows.push(p, j);
    const R = rows.length / 2;
    if (!R) continue;
    const Xe = new Float32Array(R * dim);
    for (let r = 0; r < R; r++) Xe.set(XB.subarray(rows[2 * r] * dim, (rows[2 * r] + 1) * dim), r * dim);
    const H = new Float32Array(R * eh), H2 = new Float32Array(R * eh), Y = new Float32Array(R * dim), le = l * n + e;
    matmulBatch(H, Xe, w.ew1, le * eh * dim, R, dim, eh);
    matmulBatch(H2, Xe, w.ew3, le * eh * dim, R, dim, eh);
    for (let i = 0; i < R * eh; i++) {
      let val = H[i];
      val = f(val * f(1 / f(1 + f(Math.exp(-val)))));
      val = f(val * H2[i]);
      H[i] = val;
    }
    matmulBatch(Y, H, w.ew2, le * dim * eh, R, eh, dim);
    for (let r = 0; r < R; r++) {
      const p = rows[2 * r], j = rows[2 * r + 1], g = gate[p * k + j], o = (j * N + p) * dim;
      for (let i = 0; i < dim; i++) C[o + i] = f(g * Y[r * dim + i]);
      has[j * N + p] = 1;
    }
  }
  for (let p = 0; p < N; p++) {
    for (let i = 0; i < dim; i++) {
      let acc = 0;
      for (let j = 0; j < k; j++) if (has[j * N + p]) acc = f(acc + C[(j * N + p) * dim + i]);
      out[p * dim + i] = acc;
    }
  }
}

// logits [r1 - r0, vocab] for rows r0..r1 of classifier inputs Xn [*, dim]
export async function classify(model, Xn, r0, r1, pool = null) {
  const c = model.config, R = r1 - r0;
  const X = Xn.slice(r0 * c.dim, r1 * c.dim);   // a copy: a view would post its whole buffer
  const out = new Float32Array(R * c.vocabSize);
  if (pool) await pool.cls(X, R, out);
  else matmulBatch(out, X, model.weights.wcls, 0, R, c.dim, c.vocabSize);
  return out;
}

// the k largest entries, largest first — the same list as
// [...a.keys()].sort((i, j) => a[j] - a[i]).slice(0, k) (ties keep index order, as the
// stable sort does) without sorting 32,000 numbers
export function topIndices(a, k) {
  const idx = [], val = [];
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (idx.length === k && !(v > val[k - 1])) continue;
    let at = idx.length === k ? k - 1 : idx.length;
    while (at > 0 && v > val[at - 1]) { idx[at] = idx[at - 1]; val[at] = val[at - 1]; at--; }
    idx[at] = i; val[at] = v;
  }
  return idx;
}

// ---- what the Microscope asks for ----------------------------------------------------
const CLS_CHUNK = 64;

// captureRun / captureSteered, batched: same Capture, bit for bit. heads: ["l:h"];
// steer: [{layer, neuron, value}] (value 0 = muted). onProgress(done, total).
export async function captureFast(model, tokens, { topK = 16, heads = [], steer = [], pool = null, onProgress = null } = {}) {
  const c = model.config, n = tokens.length, V = c.vocabSize;
  if (n > c.seqLen) throw new Error(`captureFast: ${n} tokens > seqLen ${c.seqLen}`);
  const { ablate, clamps, anyClamp } = plan(c, heads, steer);
  const cap = new Capture(c, tokens);
  const tap = {
    attention(l, h, pos, row) { cap.attention.set(row, (l * c.nHeads + h) * n * n + pos * n); },
    residual(l, phase, pos, x) { (phase === 'attn' ? cap.residAttn : cap.residFfn).set(x, (l * n + pos) * c.dim); },
    ffn(l, pos, hb) {
      if (anyClamp) for (const [j, v] of clamps[l]) hb[j] = v;
      cap.ffn.set(hb, (l * n + pos) * c.hiddenDim);
    },
  };
  const chunks = topK > 0 ? Math.ceil(n / CLS_CHUNK) : 0, total = c.nLayers + chunks;
  const Xn = await forwardBatch(model, tokens, 0, { tap, ablate: (ablate.heads.size || ablate.neurons.size) ? ablate : null, pool,
    onLayer: onProgress && ((l) => onProgress(l, total)) });
  for (let r0 = 0, ch = 0; r0 < n; r0 += CLS_CHUNK, ch++) {
    const r1 = Math.min(n, r0 + CLS_CHUNK);
    const LG = topK > 0 ? await classify(model, Xn, r0, r1, pool) : null;
    for (let pos = r0; pos < r1; pos++) {
      if (!LG) { cap.topK.push([]); continue; }
      const logits = LG.subarray((pos - r0) * V, (pos - r0 + 1) * V);
      cap.topK.push(topIndices(logits, topK).map(i => ({ token: i, logit: logits[i] })));
      let max = -Infinity;
      for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
      let sum = 0;
      for (let i = 0; i < V; i++) sum += Math.exp(logits[i] - max);
      cap.logZ[pos] = max + Math.log(sum);
      cap.nextProb[pos] = pos + 1 < n ? Math.exp(logits[tokens[pos + 1]] - cap.logZ[pos]) : 0;
    }
    if (onProgress) onProgress(c.nLayers + ch + 1, total);
  }
  return cap;
}

// continueSteered, batched: the prompt in one pass, then a piece at a time. Same tokens.
export async function continueFast(model, tokens, steps, { heads = [], steer = [], pool = null, onToken = null } = {}) {
  const c = model.config;
  const anything = (heads && heads.length) || (steer && steer.length);
  const { ablate, clamps, anyClamp } = plan(c, heads, steer);
  const tap = anyClamp ? { ffn(l, pos, hb) { for (const [j, v] of clamps[l]) hb[j] = v; } } : null;
  const opts = { tap, ablate: anything ? ablate : null, pool };
  const out = [];
  const limit = Math.min(c.seqLen, tokens.length + steps);
  if (tokens.length >= limit) return out;
  let Xn = await forwardBatch(model, tokens, 0, opts), r = tokens.length - 1;
  for (let pos = tokens.length - 1; ;) {
    const token = argmax(await classify(model, Xn, r, r + 1, pool));
    if (token === 1 || token === 2) break;
    out.push(token);
    if (onToken) onToken(out.length, steps);
    if (++pos >= limit - 1) break;
    Xn = await forwardBatch(model, [token], pos, opts); r = 0;
  }
  return out;
}

// every lens row for one position through the classifier in one pass: the logit lens
// after each layer's attention and ffn adds (rows 2l, 2l+1), then the J-lens at each of
// jLayers (rows 2*nLayers + i). Same logits as logitLens / jLensLogits.
export async function lensLogits(model, cap, pos, jl = null, pool = null) {
  const c = model.config, d = c.dim;
  const inputs = [];
  for (let l = 0; l < c.nLayers; l++) for (const phase of ['attn', 'ffn']) inputs.push(lensInput(model, cap.residualAt(l, pos, phase)));
  const jLayers = jl ? jl.layers.filter(l => l < c.nLayers) : [];
  for (const l of jLayers) inputs.push(jLensInput(model, jl.J[l], cap.residualAt(l, pos, 'ffn')));
  const X = new Float32Array(inputs.length * d);
  inputs.forEach((x, i) => X.set(x, i * d));
  const LG = await classify(model, X, 0, inputs.length, pool);
  return { row: (i) => LG.subarray(i * c.vocabSize, (i + 1) * c.vocabSize), jLayers };
}
