// steer.js — neuron steering for the Microscope ("turn this neuron up / mute it").
// Steering = the head-mute ablation plus per-neuron settings: a neuron set to 0 is
// MUTED through model.js's own neuron ablation (ablate.neurons), and a neuron set to any
// other value is CLAMPED to it at every position. The clamp uses the ffn tap, which
// model.js calls with the live SwiGLU buffer just before w2 reads it, so a value
// written there is what the next matmul sees. This module is bundled only into the
// Microscope worker; model.js and capture.js stay untouched (and so do the Nursery and
// Hunt workers). DOM-free; Node + browser.
//
// steer: [{ layer, neuron, value }]. With no heads and no steer, every function here
// takes the exact unsteered code path (captureRun / plain forward), so outputs are
// bit-identical to the un-steered Microscope.

import { forward, argmax } from './model.js';
import { Capture, captureRun } from './capture.js';

export function steerKey(heads, steer) {
  const h = (heads || []).slice().sort().join(',');
  if (!steer || !steer.length) return h;
  return h + '|' + steer.map(s => `${s.layer}:${s.neuron}=${s.value}`).sort().join(',');
}

// the model.js ablate argument plus the per-layer clamp lists
export function plan(config, heads, steer) {
  const muted = new Set(), clamps = Array.from({ length: config.nLayers }, () => []);
  for (const s of steer || []) {
    if (s.value === 0) muted.add(s.layer + ':' + s.neuron);
    else if (clamps[s.layer]) clamps[s.layer].push([s.neuron, Math.fround(s.value)]);
  }
  return {
    ablate: { heads: new Set(heads || []), neurons: muted },
    clamps, anyClamp: clamps.some(c => c.length),
  };
}

// captureRun with steering. Nothing steered -> captureRun itself (legacy Set form for
// heads, exactly what the Microscope worker has always passed).
export function captureSteered(model, tokens, { topK = 16, heads = [], steer = [], onProgress = null } = {}) {
  if (!steer || !steer.length) return captureRun(model, tokens, { topK, ablate: new Set(heads || []), onProgress });
  const c = model.config, n = tokens.length;
  if (n > c.seqLen) throw new Error(`captureSteered: ${n} tokens > seqLen ${c.seqLen}`);
  const { ablate, clamps } = plan(c, heads, steer);
  const cap = new Capture(c, tokens);
  const tap = {
    attention(l, h, pos, row) { cap.attention.set(row, (l * c.nHeads + h) * n * n + pos * n); },
    residual(l, phase, pos, x) { (phase === 'attn' ? cap.residAttn : cap.residFfn).set(x, (l * n + pos) * c.dim); },
    ffn(l, pos, hb) {
      for (const [j, v] of clamps[l]) hb[j] = v;   // live buffer: w2 reads the clamped value
      cap.ffn.set(hb, (l * n + pos) * c.hiddenDim);
    },
  };
  for (let pos = 0; pos < n; pos++) {
    const logits = forward(model, tokens[pos], pos, tap, ablate);
    if (topK > 0) {   // same bookkeeping as captureRun
      const idx = [...logits.keys()];
      idx.sort((a, b) => logits[b] - logits[a]);
      cap.topK.push(idx.slice(0, topK).map(i => ({ token: i, logit: logits[i] })));
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

// Greedy continuation of `tokens` (teacher-forced through the prompt, then argmax),
// under the same heads + steer. Stops at BOS/EOS (a TinyStories model starts a new story
// with BOS) or after `steps` new tokens, and never past the model window.
export function continueSteered(model, tokens, steps, { heads = [], steer = [], onToken = null } = {}) {
  const c = model.config;
  const anything = (heads && heads.length) || (steer && steer.length);
  const { ablate, clamps, anyClamp } = plan(c, heads, steer);
  const tap = anyClamp ? { ffn(l, pos, hb) { for (const [j, v] of clamps[l]) hb[j] = v; } } : null;
  const ab = anything ? ablate : null;
  const out = [];
  let token = tokens[0];
  const limit = Math.min(c.seqLen, tokens.length + steps);
  for (let pos = 0; pos < limit - 1; pos++) {
    const logits = forward(model, token, pos, tap, ab);
    if (pos < tokens.length - 1) { token = tokens[pos + 1]; continue; }
    token = argmax(logits);
    if (token === 1 || token === 2) break;
    out.push(token);
    if (onToken) onToken(out.length, steps);
  }
  return out;
}
