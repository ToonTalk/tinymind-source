// jlens.js — the J-lens ("thinking ahead"), ported from the J-Lens Explorer
// (Documents\GitHub\jlens-tinystories\app). A logit lens asks "what would the model say
// if it stopped at layer l?"; a J-lens first maps the layer-l residual through J_l, a
// matrix fitted (offline, from 1000 TinyStories prompts) to stand for how that layer's
// state goes on to shape later predictions — so it reads out what the mind is leaning
// toward further ahead, not just the next word:
//   J-lens(h) = wcls @ ( rmsFinal ⊙ (J_l @ h) / rms(J_l @ h) )      h = block-l output
// .jlens format: 'JLNS' | u32 version (1) | u32 d_model | u32 n_layers | n × u32 layer
// index | per layer fp32 row-major [d, d], rows = output dims (y = J @ h).

import { matmul } from './model.js';

export const JLENS_FILES = { 'stories110M.bin': 'stories110M.jlens', 'stories15M.bin': 'stories15M.jlens' };

export function readJlens(buffer) {
  const u8 = buffer instanceof Uint8Array ? buffer : new Uint8Array(buffer);
  const buf = u8.byteOffset % 4 === 0 ? u8 : u8.slice();   // Float32Array views need 4-byte alignment
  const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'JLNS') throw new Error('not a .jlens file');
  if (dv.getUint32(4, true) !== 1) throw new Error('unsupported .jlens version ' + dv.getUint32(4, true));
  const d = dv.getUint32(8, true), n = dv.getUint32(12, true);
  const layers = [];
  let p = 16;
  for (let i = 0; i < n; i++) { layers.push(dv.getUint32(p, true)); p += 4; }
  const J = {};
  for (const l of layers) { J[l] = new Float32Array(buf.buffer, buf.byteOffset + p, d * d); p += d * d * 4; }
  if (p !== buf.byteLength) throw new Error(`.jlens: read ${p} of ${buf.byteLength} bytes`);
  return { dModel: d, layers, J };
}

// logits of the J-lens at one residual h (Float32Array of d_model)
export function jLensLogits(model, J, h) {
  const c = model.config, w = model.weights, d = c.dim, f = Math.fround;
  const u = new Float32Array(d);
  matmul(u, h, J, 0, d, d);
  let ss = 0;
  for (let j = 0; j < d; j++) ss = f(ss + f(u[j] * u[j]));
  ss = f(1 / f(Math.sqrt(f(f(ss / d) + 1e-5))));
  const xb = new Float32Array(d);
  for (let j = 0; j < d; j++) xb[j] = w.rmsFinal[j] * f(ss * u[j]);
  const logits = new Float32Array(c.vocabSize);
  matmul(logits, xb, w.wcls, 0, d, c.vocabSize);
  return logits;
}
