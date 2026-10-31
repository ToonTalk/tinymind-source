// experts.js — the Experts room's engine, on batch.js. A mind made of experts (bin-reader's
// config.experts; tools/moe) reads a story and keeps, for every piece and every layer, the
// router's votes for each expert and the expert(s) it chose; and it writes on, with experts
// switched off: muted (the router still chooses it, but its output is zero) or rerouted (the
// router may not choose it, so the word goes to its next choice). Switches are "l:e" keys
// (layer and expert, both from 0). Reading and writing go through forwardBatch, so a story
// and its routes are the very ones model.js forward() gives, with or without a pool (E18).
// DOM-free; Node + browser.

import { matmulBatch } from './model.js';
import { forwardBatch } from './batch.js';
import { makeRng, sample } from './thought.js';

const STORY_START = 1, STORY_END = 2;   // a TinyStories mind starts each story with 1 and may end with 2 (thought.js has the same, under its own names)

// forwardBatch's ablate for a set of switches, or null when nothing is switched off
export function expertAblate(mute = [], reroute = []) {
  return (mute && mute.length) || (reroute && reroute.length) ? { mute: new Set(mute || []), reroute: new Set(reroute || []) } : null;
}

export class Experts {
  // model: createModel(...) of a mind made of experts; pool: batch.js Pool or null
  constructor(model, pool = null) {
    const c = model.config;
    if (!c.experts) throw new Error('this mind is not made of experts');
    this.model = model; this.pool = pool;
    this.n = c.experts.n; this.k = c.experts.topK; this.L = c.nLayers;
  }

  // read ids with the given switches: for every piece and layer, the router's votes (probs,
  // after any reroute) and its choice(s), best first. onProgress(done, total) per layer.
  async read(ids, { mute = [], reroute = [], onProgress = null } = {}) {
    const { n, k, L } = this, N = ids.length;
    const top = new Int8Array(N * L * k).fill(-1), gate = new Float32Array(N * L * k), probs = new Float32Array(N * L * n);
    const tap = {
      router(l, pos, pr, t, g, kk) {
        probs.set(pr, (pos * L + l) * n);
        for (let j = 0; j < kk; j++) { top[(pos * L + l) * k + j] = t[j]; gate[(pos * L + l) * k + j] = g[j]; }
      },
    };
    await forwardBatch(this.model, ids, 0, { tap, ablate: expertAblate(mute, reroute), pool: this.pool,
      onLayer: onProgress && (l => onProgress(l, L)) });
    return reading({ ids: Array.from(ids), n: N, L, E: n, k, top, gate, probs });
  }

  // write on from promptIds (the prompt in one pass, then a piece at a time) with the given
  // switches; temperature 0 = always its most likely word, so a rewrite with an expert
  // switched off differs from one without ONLY by the switch. Returns prompt + new pieces.
  // onToken(id, pos) may return false to stop; it is awaited.
  async write(promptIds, { maxNew = 120, temperature = 0, seed = 1, mute = [], reroute = [], onToken = null } = {}) {
    const c = this.model.config, d = c.dim, w = this.model.weights, rng = makeRng(seed);
    const ablate = expertAblate(mute, reroute);
    const limit = Math.min(c.seqLen, promptIds.length + maxNew);
    const out = promptIds.slice();
    if (promptIds.length >= limit) return out;
    let Xn = await forwardBatch(this.model, promptIds, 0, { ablate, pool: this.pool }), r = promptIds.length - 1;
    const logits = new Float32Array(c.vocabSize);
    for (let pos = promptIds.length - 1; ;) {
      const x = Xn.slice(r * d, (r + 1) * d);
      if (this.pool) await this.pool.cls(x, 1, logits);
      else matmulBatch(logits, x, w.wcls, 0, 1, d, c.vocabSize);
      const next = sample(logits, temperature, rng);
      if (next === STORY_START || next === STORY_END) break;
      out.push(next);
      if (onToken && (await onToken(next, pos + 1)) === false) break;
      if (++pos >= limit - 1) break;
      Xn = await forwardBatch(this.model, [next], pos, { ablate, pool: this.pool }); r = 0;
    }
    return out;
  }
}

// a reading's arrays (what the worker posts to the page), with lookups
export function reading(r) {
  return Object.assign(r, {
    // the expert piece pos went to in layer l (its first choice), or -1
    chosen: (pos, l) => r.top[(pos * r.L + l) * r.k],
    // the router's votes for all experts at piece pos, layer l
    votes: (pos, l) => Array.from(r.probs.subarray((pos * r.L + l) * r.E, (pos * r.L + l + 1) * r.E)),
  });
}
