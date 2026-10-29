// thought.js — the Thought bubble's engine, ported from the J-Lens Explorer onto Tiny
// Mind's batched engine (batch.js). Reading a story keeps, for every piece and every
// layer the mind's J-lens was fitted for, the score of every word in the story
// vocabulary (vocab.json: ~7,900 TinyStories pieces) under "thinking ahead" (the
// J-lens) and "layer by layer" (the logit lens), plus the mind's own next-word scores.
// The readout, the heat strip and DIFF all come from those tables. The scores are the
// very logits logitLens / jLensLogits give, restricted to the story vocabulary (E16).
//
// The thought swap is the explorer's readout-direction edit: at each chosen layer, from
// the chosen piece on, the residual's component along the "take out" word's direction
// is moved onto the "put in" word's direction:
//   beta = h · d̂_out ;  h += strength * beta * (d̂_in - d̂_out),
//   d_(l,t) = (wcls[t] ⊙ rmsFinal) @ J_l   (what that word looks like to the J-lens at l)
// DOM-free; Node + browser.

import { matmulBatch } from './model.js';
import { lensInput } from './capture.js';
import { forwardBatch, topIndices } from './batch.js';

const START_PIECE = 1, END_PIECE = 2;   // a TinyStories mind starts each story with 1 and may end with 2
const CHUNK = 16;   // pieces scored per pass through the story vocabulary

// xorshift32, seeded, so a rewrite with and without the swap can share one random stream
export function makeRng(seed) {
  let s = seed >>> 0 || 88675123;
  return () => { s ^= s << 13; s >>>= 0; s ^= s >> 17; s ^= s << 5; s >>>= 0; return s / 4294967296; };
}

export function sample(logits, temperature, rng) {
  const V = logits.length;
  if (temperature <= 0) {
    let best = 0;
    for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
    return best;
  }
  let max = -Infinity;
  for (let i = 0; i < V; i++) if (logits[i] > max) max = logits[i];
  const probs = new Float64Array(V);
  let sum = 0;
  for (let i = 0; i < V; i++) { probs[i] = Math.exp((logits[i] - max) / temperature); sum += probs[i]; }
  let r = rng() * sum;
  for (let i = 0; i < V; i++) { r -= probs[i]; if (r <= 0) return i; }
  return V - 1;
}

export class Thought {
  // model: createModel(...); jl: readJlens(...); vocab: vocab.json; pool: batch.js Pool or null
  constructor(model, jl, vocab, pool = null) {
    const c = model.config, d = c.dim, w = model.weights;
    this.model = model; this.jl = jl; this.pool = pool;
    this.band = jl.layers.filter(l => l < c.nLayers);
    this.ids = Int32Array.from(vocab.prunedIds.filter(id => id < c.vocabSize));
    this.P = this.ids.length;
    this.index = new Int32Array(c.vocabSize).fill(-1);
    this.ids.forEach((id, r) => { this.index[id] = r; });
    this.G = new Float32Array(this.P * d);   // the story-vocabulary rows of the classifier
    for (let r = 0; r < this.P; r++) this.G.set(w.wcls.subarray(this.ids[r] * d, (this.ids[r] + 1) * d), r * d);
    this.cap = null;
  }

  // ship the readout matrices to the pool, if there is one
  async ready() {
    if (!this.pool) return;
    const d = this.model.config.dim;
    await this.pool.addMatrix('story-vocab', this.G, this.P, d);
    for (const l of this.band) await this.pool.addMatrix('J' + l, this.jl.J[l], d, d);
  }
  async _mm(name, M, X, R, rows) {
    const d = this.model.config.dim, out = new Float32Array(R * rows);
    if (this.pool) await this.pool.mat(name, X, R, out);
    else matmulBatch(out, X, M, 0, R, d, rows);
    return out;
  }

  // read ids (with an optional edit, so a swapped story's thoughts are its own) and keep
  // every score; onProgress(done, total)
  async analyze(ids, { edit = null, onProgress = null } = {}) {
    const c = this.model.config, d = c.dim, L = this.band.length, n = ids.length, P = this.P;
    const bandAt = new Map(this.band.map((l, i) => [l, i]));
    const resid = new Float32Array(n * L * d);
    const tap = { residual(l, phase, pos, x) { if (phase === 'ffn' && bandAt.has(l)) resid.set(x, (pos * L + bandAt.get(l)) * d); } };
    const chunks = Math.ceil(n / CHUNK), total = c.nLayers + chunks;
    const Xn = await forwardBatch(this.model, ids, 0, { tap, edit, pool: this.pool,
      onLayer: onProgress && (l => onProgress(l, total)) });
    const jScores = new Float32Array(n * L * P), lScores = new Float32Array(n * (L + 1) * P);
    for (let p0 = 0, ch = 0; p0 < n; p0 += CHUNK, ch++) {
      const p1 = Math.min(n, p0 + CHUNK), cnt = p1 - p0, R = 2 * L + 1;
      const X = new Float32Array(cnt * R * d);
      for (let li = 0; li < L; li++) {
        const H = new Float32Array(cnt * d);
        for (let p = p0; p < p1; p++) H.set(resid.subarray((p * L + li) * d, (p * L + li + 1) * d), (p - p0) * d);
        const U = await this._mm('J' + this.band[li], this.jl.J[this.band[li]], H, cnt, d);   // J @ h for every piece
        for (let p = p0; p < p1; p++) {
          const r = (p - p0) * R;
          X.set(lensInput(this.model, U.subarray((p - p0) * d, (p - p0 + 1) * d)), (r + li) * d);        // thinking ahead
          X.set(lensInput(this.model, H.subarray((p - p0) * d, (p - p0 + 1) * d)), (r + L + li) * d);    // layer by layer
        }
      }
      for (let p = p0; p < p1; p++) X.set(Xn.subarray(p * d, (p + 1) * d), ((p - p0) * R + 2 * L) * d);   // its own next word
      const S = await this._mm('story-vocab', this.G, X, cnt * R, P);
      for (let p = p0; p < p1; p++) {
        const r = (p - p0) * R;
        for (let li = 0; li < L; li++) {
          jScores.set(S.subarray((r + li) * P, (r + li + 1) * P), (p * L + li) * P);
          lScores.set(S.subarray((r + L + li) * P, (r + L + li + 1) * P), (p * (L + 1) + li) * P);
        }
        lScores.set(S.subarray((r + 2 * L) * P, (r + 2 * L + 1) * P), (p * (L + 1) + L) * P);
      }
      if (onProgress) onProgress(c.nLayers + ch + 1, total);
    }
    this.cap = { ids: Array.from(ids), n, jScores, lScores };
    return this.cap;
  }

  // the top k story words per band layer under both lenses, and the mind's own next word
  readout(pos, k = 40) {
    const { jScores, lScores } = this.cap, L = this.band.length, P = this.P;
    const top = (arr, off) => { const row = arr.subarray(off, off + P);
      return topIndices(row, k).map(r => ({ id: this.ids[r], score: row[r] })); };
    return {
      pos,
      layers: this.band.map((l, li) => ({ layer: l, jlens: top(jScores, (pos * L + li) * P), logit: top(lScores, (pos * (L + 1) + li) * P) })),
      model: top(lScores, (pos * (L + 1) + L) * P).slice(0, 5),
    };
  }

  // for one word: its best thinking-ahead rank over the band layers at every piece
  // (0 = top); null if the word is not in the story vocabulary
  heat(tokenId) {
    const r = tokenId >= 0 && tokenId < this.index.length ? this.index[tokenId] : -1;
    if (r < 0 || !this.cap) return null;
    const { jScores, n } = this.cap, L = this.band.length, P = this.P;
    const ranks = new Int32Array(n).fill(P);
    for (let p = 0; p < n; p++) {
      for (let li = 0; li < L; li++) {
        const off = (p * L + li) * P, mine = jScores[off + r];
        let rank = 0;
        for (let q = 0; q < P; q++) if (jScores[off + q] > mine) rank++;
        if (rank < ranks[p]) ranks[p] = rank;
      }
    }
    return ranks;
  }

  // the swap: { layers, fn } for forwardBatch's edit, or null when nothing would change
  makeEdit(outId, inId, l0, l1, fromPos, strength) {
    const c = this.model.config, w = this.model.weights, d = c.dim;
    const layers = new Set(this.band.filter(l => l >= l0 && l <= l1));
    if (!layers.size || !strength) return null;
    const dirs = {};
    const direction = (l, t) => {
      const J = this.jl.J[l], g = new Float64Array(d), out = new Float32Array(d);
      for (let k = 0; k < d; k++) g[k] = w.wcls[t * d + k] * w.rmsFinal[k];
      for (let k = 0; k < d; k++) { const gk = g[k], row = k * d; if (gk) for (let j = 0; j < d; j++) out[j] += gk * J[row + j]; }
      let nn = 0;
      for (let j = 0; j < d; j++) nn += out[j] * out[j];
      nn = 1 / Math.sqrt(nn || 1e-30);
      for (let j = 0; j < d; j++) out[j] *= nn;
      return out;
    };
    for (const l of layers) dirs[l] = { s: direction(l, outId), t: direction(l, inId) };
    return {
      layers,
      fn(l, pos, x) {
        if (pos < fromPos) return;
        const { s, t } = dirs[l];
        let beta = 0;
        for (let j = 0; j < d; j++) beta += x[j] * s[j];
        const g = strength * beta;
        for (let j = 0; j < d; j++) x[j] += g * (t[j] - s[j]);
      },
    };
  }

  // the mind writes on from promptIds: the prompt in one pass, then a piece at a time,
  // sampled at `temperature` from a seeded stream. onToken(id, pos) may return false to
  // stop; it is awaited, so a caller can yield to its message loop there.
  async generate(promptIds, { maxNew = 120, temperature = 0.8, seed = 1, edit = null, onToken = null } = {}) {
    const c = this.model.config, d = c.dim, rng = makeRng(seed), w = this.model.weights;
    const limit = Math.min(c.seqLen, promptIds.length + maxNew);
    const out = promptIds.slice();
    if (promptIds.length >= limit) return out;
    let Xn = await forwardBatch(this.model, promptIds, 0, { edit, pool: this.pool }), r = promptIds.length - 1;
    const logits = new Float32Array(c.vocabSize);
    for (let pos = promptIds.length - 1; ;) {
      const x = Xn.slice(r * d, (r + 1) * d);
      if (this.pool) await this.pool.cls(x, 1, logits);
      else matmulBatch(logits, x, w.wcls, 0, 1, d, c.vocabSize);
      const next = sample(logits, temperature, rng);
      if (next === START_PIECE || next === END_PIECE) break;
      out.push(next);
      if (onToken && (await onToken(next, pos + 1)) === false) break;
      if (++pos >= limit - 1) break;
      Xn = await forwardBatch(this.model, [next], pos, { edit, pool: this.pool }); r = 0;
    }
    return out;
  }
}
