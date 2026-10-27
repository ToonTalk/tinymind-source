// neuron-atlas.js — the offline "what does this neuron do?" atlas for stories15M.
// tools/build-neuron-atlas.js runs the model over TinyStories validation stories and
// keeps, for every FFN hidden neuron, its strongest snippets at BOTH poles: the
// positive pole (the neuron's activation is high) and the negative pole (it is strongly
// negative — SwiGLU units go both ways, and the negative side often has its own job).
// This module is the shared format: the page reads entries with atlasEntry(), the tests
// check files with validateAtlas(), the builder writes with encodeAtlas(). DOM-free.
//
// File shape (schema tiny-mind-neuron-atlas-v1), kept small by a piece table and
// percent quantization:
//   { schema, model, config: {dim, nLayers, hiddenDim, vocabSize},
//     fingerprint,                 // atlasFingerprint() of the weights the atlas came from
//     corpus: {file, stories, tokens}, window: {before, after}, perPole,
//     pieces: [string],            // every distinct piece the snippets use
//     neurons: { "l:j": [posPeak, negPeak, posSnips, negSnips] },
//     labels?: { "l:j": [posLabel|null, negLabel|null, sig] } }
// A snippet is [q, target, story, pos, ...pieceIdx]: q = round(100·|act| / pole peak),
// target = index of the lit piece inside the snippet, story/pos = where it was found
// (story = the <|endoftext|>-separated record number in the corpus file, pos = token
// position with BOS at 0). Peaks keep 3 significant digits; negPeak is negative.
// A pole with no snippets has peak 0 and []. Neurons whose peak is negligible are left
// out entirely (atlasEntry returns null for them).

export const ATLAS_SCHEMA = 'tiny-mind-neuron-atlas-v1';
export const ATLAS_FILE = 'tiny-mind-neuron-atlas.json';

// Cheap identity check that an atlas belongs to the loaded weights: FNV-1a over the
// float bits of a fixed stride through w1 (the FFN input matrix). Same weights ->
// same string; a fine-tune or another model -> different.
export function atlasFingerprint(config, weights) {
  const w = weights.w1;
  const u32 = new Uint32Array(w.buffer, w.byteOffset, w.length);
  let h = 0x811c9dc5;
  const step = Math.max(1, Math.floor(u32.length / 4096));
  for (let i = 0; i < u32.length; i += step) { h ^= u32[i]; h = Math.imul(h, 0x01000193); }
  return `${config.dim}x${config.hiddenDim}x${config.nLayers}x${config.vocabSize}-${(h >>> 0).toString(36)}`;
}

export function isNeuronAtlas(value) {
  return !!value && typeof value === 'object' && value.schema === ATLAS_SCHEMA;
}
// sniff raw JSON text without parsing megabytes when it is plainly something else
export function looksLikeAtlasText(text) {
  return typeof text === 'string' && text.slice(0, 200).includes('"' + ATLAS_SCHEMA + '"');
}

const peak3 = v => Number(v.toPrecision(3));

// candidates: Map "l:j" -> { pos: [{value, story, pos, pieces, target}], neg: [...] }
// with each list sorted strongest first; pieces are the snippet's decoded strings.
// minPeak: a neuron whose strongest |activation| is below it is left out; minPoleRatio:
// a pole weaker than this fraction of the other pole is emptied (its "strongest"
// moments are just noise around zero).
export function encodeAtlas({ model, config, fingerprint, corpus, window, perPole,
                              candidates, minPeak, minPoleRatio = 0, labels = null }) {
  const pieces = [], pieceIx = new Map();
  const pieceId = s => {
    let i = pieceIx.get(s);
    if (i === undefined) { i = pieces.length; pieces.push(s); pieceIx.set(s, i); }
    return i;
  };
  const neurons = {};
  for (let l = 0; l < config.nLayers; l++) {
    for (let j = 0; j < config.hiddenDim; j++) {
      const key = l + ':' + j;
      const c = candidates.get(key);
      if (!c) continue;
      const pp0 = c.pos.length ? Math.abs(c.pos[0].value) : 0, np0 = c.neg.length ? Math.abs(c.neg[0].value) : 0;
      if (Math.max(pp0, np0) < minPeak) continue;   // negligible neuron
      const pole = (list, other) => {
        const top = list.slice(0, perPole);
        if (!top.length || Math.abs(top[0].value) < minPoleRatio * other) return [0, []];
        const peak = Math.abs(top[0].value);
        return [peak3(top[0].value), top.map(s => [
          Math.round(100 * Math.abs(s.value) / peak), s.target, s.story, s.pos,
          ...s.pieces.map(pieceId),
        ])];
      };
      const [pp, ps] = pole(c.pos, np0), [np, ns] = pole(c.neg, pp0);
      neurons[key] = [pp, np, ps, ns];
    }
  }
  const atlas = {
    schema: ATLAS_SCHEMA, model,
    config: { dim: config.dim, nLayers: config.nLayers, hiddenDim: config.hiddenDim, vocabSize: config.vocabSize },
    fingerprint, corpus, window, perPole, minPeak, minPoleRatio, pieces, neurons,
  };
  if (labels) {
    const kept = {};
    for (const [key, lab] of Object.entries(labels)) {
      if (neurons[key] && lab[2] === snippetSig(atlas, key)) kept[key] = lab;
    }
    if (Object.keys(kept).length) atlas.labels = kept;
  }
  return atlas;
}

// A label is only valid for the exact snippets it was written from.
export function snippetSig(atlas, key) {
  const e = atlas.neurons[key];
  if (!e) return null;
  const text = [e[2], e[3]].map(list => list.map(s => s.slice(4).map(i => atlas.pieces[i]).join('')).join('|')).join('||');
  let h = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) { h ^= text.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return (h >>> 0).toString(36);
}

// Decoded view of one neuron for the card. value = the snippet's activation, recovered
// from its percent of the pole peak (the atlas is quantized to whole percents).
export function atlasEntry(atlas, layer, neuron) {
  if (!isNeuronAtlas(atlas)) return null;
  const key = layer + ':' + neuron;
  const e = atlas.neurons[key];
  if (!e) return null;
  const pole = (peak, list) => list.map(([q, target, story, pos, ...ix]) => ({
    q, value: peak3(peak * q / 100), target, story, pos, pieces: ix.map(i => atlas.pieces[i]),
  }));
  const lab = atlas.labels && atlas.labels[key];
  return {
    layer, neuron, key,
    pos: { peak: e[0], snippets: pole(e[0], e[2]), label: (lab && lab[0]) || null },
    neg: { peak: e[1], snippets: pole(e[1], e[3]), label: (lab && lab[1]) || null },
  };
}

// Structural check used by tests and by the page before trusting a dropped file.
// Returns a list of problems (empty = valid).
export function validateAtlas(atlas) {
  const errs = [];
  if (!isNeuronAtlas(atlas)) return ['not a ' + ATLAS_SCHEMA + ' file'];
  const c = atlas.config || {};
  for (const k of ['dim', 'nLayers', 'hiddenDim', 'vocabSize']) {
    if (!Number.isInteger(c[k]) || c[k] <= 0) errs.push('config.' + k + ' missing');
  }
  if (!Array.isArray(atlas.pieces) || !atlas.pieces.every(p => typeof p === 'string')) errs.push('pieces must be strings');
  if (typeof atlas.fingerprint !== 'string') errs.push('fingerprint missing');
  if (!atlas.neurons || typeof atlas.neurons !== 'object') return errs.concat('neurons missing');
  const nP = (atlas.pieces || []).length, per = atlas.perPole;
  const width = atlas.window ? atlas.window.before + 1 + atlas.window.after : Infinity;
  for (const [key, e] of Object.entries(atlas.neurons)) {
    const m = /^(\d+):(\d+)$/.exec(key);
    if (!m || +m[1] >= c.nLayers || +m[2] >= c.hiddenDim) { errs.push(key + ': bad key'); continue; }
    if (!Array.isArray(e) || e.length !== 4) { errs.push(key + ': entry must be [posPeak, negPeak, pos, neg]'); continue; }
    const [pp, np, ps, ns] = e;
    if (!(pp >= 0) || !(np <= 0)) errs.push(key + ': peaks must be >=0 (positive pole) and <=0 (negative pole)');
    for (const [name, peak, list] of [['pos', pp, ps], ['neg', np, ns]]) {
      if (!Array.isArray(list) || list.length > per) { errs.push(key + ' ' + name + ': too many snippets'); continue; }
      if ((peak === 0) !== (list.length === 0)) errs.push(key + ' ' + name + ': peak/snippets disagree');
      let prev = 101;
      for (const s of list) {
        const [q, target, story, pos, ...ix] = s;
        if (!Number.isInteger(q) || q < 0 || q > 100 || q > prev) errs.push(key + ' ' + name + ': q out of order');
        prev = q;
        if (!ix.length || ix.length > width || !Number.isInteger(target) || target < 0 || target >= ix.length) errs.push(key + ' ' + name + ': bad target');
        if (!Number.isInteger(story) || !Number.isInteger(pos) || pos < 1) errs.push(key + ' ' + name + ': bad location');
        if (!ix.every(i => Number.isInteger(i) && i >= 0 && i < nP)) errs.push(key + ' ' + name + ': piece index out of range');
      }
      if (list.length && list[0][0] !== 100) errs.push(key + ' ' + name + ': strongest snippet must be 100%');
    }
    if (errs.length > 20) break;
  }
  return errs;
}
