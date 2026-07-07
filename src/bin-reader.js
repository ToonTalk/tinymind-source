// bin-reader.js — llama2.c legacy (v0) checkpoint reader/writer.
// Format verified against vendor/llama2.c/run.c (read_checkpoint + memory_map_weights):
//   header: 7 x int32 LE: dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len
//   vocab_size < 0 => final classifier wcls is a separate tensor ("unshared"); use abs()
//   then fp32 weights in the order listed in layout() below,
//   then TWO legacy blocks (seq_len*head_size/2 floats each — old RoPE freq_cis) to skip,
//   then wcls [vocab, dim] only if unshared.
// DOM-free: operates on ArrayBuffer/Uint8Array, runs in Node and browser.

const HEADER_INTS = 7;
export const HEADER_BYTES = HEADER_INTS * 4;

function toArrayBuffer(data) {
  if (data instanceof ArrayBuffer) return { buf: data, off: 0, len: data.byteLength };
  if (ArrayBuffer.isView(data)) return { buf: data.buffer, off: data.byteOffset, len: data.byteLength };
  throw new Error('bin-reader: expected ArrayBuffer or typed array');
}

// Ordered tensor layout. rows/cols follow run.c matmul convention: w is (d, n) row-major,
// i.e. output-dim rows of input-dim columns.
function layout(c) {
  const headSize = c.dim / c.nHeads;
  const kvDim = (c.dim * c.nKvHeads) / c.nHeads;
  return [
    { name: 'tokenEmbedding', size: c.vocabSize * c.dim },
    { name: 'rmsAtt',         size: c.nLayers * c.dim },
    { name: 'wq',             size: c.nLayers * c.dim * c.dim },
    { name: 'wk',             size: c.nLayers * c.dim * kvDim },
    { name: 'wv',             size: c.nLayers * c.dim * kvDim },
    { name: 'wo',             size: c.nLayers * c.dim * c.dim },
    { name: 'rmsFfn',         size: c.nLayers * c.dim },
    { name: 'w1',             size: c.nLayers * c.hiddenDim * c.dim },
    { name: 'w2',             size: c.nLayers * c.dim * c.hiddenDim },
    { name: 'w3',             size: c.nLayers * c.hiddenDim * c.dim },
    { name: 'rmsFinal',       size: c.dim },
    { name: 'legacyFreqCis',  size: c.seqLen * headSize },   // 2 blocks of seq*headSize/2 — skip
    { name: 'wcls',           size: c.sharedWeights ? 0 : c.vocabSize * c.dim },
  ];
}

export function readCheckpoint(data) {
  const { buf, off, len } = toArrayBuffer(data);
  const dv = new DataView(buf, off, len);
  const raw = [];
  for (let i = 0; i < HEADER_INTS; i++) raw.push(dv.getInt32(i * 4, true));
  const [dim, hiddenDim, nLayers, nHeads, nKvHeads, vocabRaw, seqLen] = raw;
  const config = {
    dim, hiddenDim, nLayers, nHeads, nKvHeads,
    vocabSize: Math.abs(vocabRaw),
    seqLen,
    sharedWeights: vocabRaw > 0,
    headSize: dim / nHeads,
    kvDim: (dim * nKvHeads) / nHeads,
  };
  if ((off + HEADER_BYTES) % 4 !== 0) throw new Error('bin-reader: weights not 4-byte aligned');
  const floats = new Float32Array(buf, off + HEADER_BYTES, (len - HEADER_BYTES) / 4);

  const weights = {};
  const offsets = [];   // { name, offsetFloats, size } — for the E1 offset-sum check
  let p = 0;
  for (const t of layout(config)) {
    offsets.push({ name: t.name, offsetFloats: p, size: t.size });
    if (t.name !== 'legacyFreqCis' && t.size > 0) weights[t.name] = floats.subarray(p, p + t.size);
    p += t.size;
  }
  if (config.sharedWeights) weights.wcls = weights.tokenEmbedding;

  const expectedBytes = HEADER_BYTES + p * 4;
  if (expectedBytes !== len) {
    throw new Error(`bin-reader: layout consumed ${expectedBytes} bytes but file is ${len}`);
  }
  return { config, weights, offsets, fileBytes: len };
}

// Serialize a checkpoint in the same legacy format (legacy freq_cis blocks written as zeros;
// run.c and this reader both skip them). `weights` uses the field names from readCheckpoint.
export function writeCheckpoint(config, weights) {
  const c = config;
  const tensors = layout(c);
  const totalFloats = tensors.reduce((a, t) => a + t.size, 0);
  const out = new ArrayBuffer(HEADER_BYTES + totalFloats * 4);
  const dv = new DataView(out);
  const vocabField = c.sharedWeights ? c.vocabSize : -c.vocabSize;
  [c.dim, c.hiddenDim, c.nLayers, c.nHeads, c.nKvHeads, vocabField, c.seqLen]
    .forEach((v, i) => dv.setInt32(i * 4, v, true));
  const floats = new Float32Array(out, HEADER_BYTES);
  let p = 0;
  for (const t of tensors) {
    if (t.name !== 'legacyFreqCis' && t.size > 0) {
      const src = weights[t.name];
      if (!src || src.length !== t.size) {
        throw new Error(`writeCheckpoint: ${t.name} expected ${t.size} floats, got ${src ? src.length : 'none'}`);
      }
      floats.set(src, p);
    }
    p += t.size;
  }
  return out;
}
