// tokenizer.js — sentencepiece-style BPE tokenizer, ported from vendor/llama2.c/run.c
// (build_tokenizer / encode / decode). Not improvised: same dummy-prefix, same UTF-8
// codepoint chunking, same byte-fallback (+3), same best-score merge loop.
//
// File format: int32 max_token_length, then vocab_size x (float32 score, int32 len, len raw bytes).
// vocab_size is NOT in the file — it comes from the checkpoint header.
//
// Pieces are handled as "byte strings": one JS char per raw byte (latin1), so lookup and
// concatenation behave exactly like C strcmp/sprintf on bytes. decode() converts back to
// real text via UTF-8 at the end. DOM-free; Node + browser.

export const BOS = 1;
export const EOS = 2;

const textEncoder = new TextEncoder();
const textDecoder = new TextDecoder('utf-8');

function bytesToByteString(bytes) {
  let s = '';
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return s;
}

export class Tokenizer {
  constructor(data, vocabSize) {
    let buf, off, len;
    if (data instanceof ArrayBuffer) { buf = data; off = 0; len = data.byteLength; }
    else if (ArrayBuffer.isView(data)) { buf = data.buffer; off = data.byteOffset; len = data.byteLength; }
    else throw new Error('Tokenizer: expected ArrayBuffer or typed array');
    const dv = new DataView(buf, off, len);
    const bytes = new Uint8Array(buf, off, len);

    this.vocabSize = vocabSize;
    this.maxTokenLength = dv.getInt32(0, true);
    this.vocab = new Array(vocabSize);       // byte strings
    this.scores = new Float32Array(vocabSize);
    let p = 4;
    for (let i = 0; i < vocabSize; i++) {
      this.scores[i] = dv.getFloat32(p, true); p += 4;
      const n = dv.getInt32(p, true); p += 4;
      this.vocab[i] = bytesToByteString(bytes.subarray(p, p + n)); p += n;
    }
    if (p !== len) throw new Error(`Tokenizer: consumed ${p} of ${len} bytes`);

    // piece -> id. C uses qsort+bsearch over unique strings; a Map is equivalent.
    // Keep the FIRST id for duplicate pieces (higher-scored merges dominate anyway).
    this.lookup = new Map();
    for (let i = 0; i < vocabSize; i++) {
      if (!this.lookup.has(this.vocab[i])) this.lookup.set(this.vocab[i], i);
    }
  }

  // encode(text, bos, eos) -> array of token ids. Mirrors run.c encode() exactly.
  encode(text, bos = false, eos = false) {
    const tokens = [];
    if (bos) tokens.push(BOS);
    if (text.length !== 0) {
      const dummyPrefix = this.lookup.get(' ');
      if (dummyPrefix === undefined) throw new Error('Tokenizer: no " " piece in vocab');
      tokens.push(dummyPrefix);
    }
    const utf8 = textEncoder.encode(text);

    // chunk into UTF-8 codepoints (reset on non-continuation byte), lookup or byte-fallback
    let chunk = '';
    for (let i = 0; i < utf8.length; i++) {
      const b = utf8[i];
      if ((b & 0xC0) !== 0x80) chunk = '';
      chunk += String.fromCharCode(b);
      const next = utf8[i + 1];
      if (next !== undefined && (next & 0xC0) === 0x80 && chunk.length < 4) continue;
      const id = this.lookup.get(chunk);
      if (id !== undefined) {
        tokens.push(id);
      } else {
        for (let j = 0; j < chunk.length; j++) tokens.push(chunk.charCodeAt(j) + 3);
      }
      chunk = '';
    }

    // greedy merge: repeatedly merge the adjacent pair whose merged piece has the best score
    while (true) {
      let bestScore = -1e10, bestId = -1, bestIdx = -1;
      for (let i = 0; i < tokens.length - 1; i++) {
        const merged = this.vocab[tokens[i]] + this.vocab[tokens[i + 1]];
        const id = this.lookup.get(merged);
        if (id !== undefined && this.scores[id] > bestScore) {
          bestScore = this.scores[id]; bestId = id; bestIdx = i;
        }
      }
      if (bestIdx === -1) break;
      tokens[bestIdx] = bestId;
      tokens.splice(bestIdx + 1, 1);
    }

    if (eos) tokens.push(EOS);
    return tokens;
  }

  // Piece for `token` given previous token, as a byte string. Mirrors run.c decode().
  decodePiece(prevToken, token) {
    let piece = this.vocab[token];
    if (prevToken === BOS && piece.charCodeAt(0) === 0x20) piece = piece.slice(1);
    const m = /^<0x([0-9A-Fa-f]{2})>$/.exec(piece);
    if (m) piece = String.fromCharCode(parseInt(m[1], 16));
    return piece;
  }

  // Full-sequence decode to a real JS string (UTF-8). BOS/EOS contribute no output but
  // BOS still triggers the leading-space strip on the following piece.
  decode(tokens) {
    let bytes = '';
    let prev = BOS;
    for (const tok of tokens) {
      if (tok === BOS || tok === EOS) { prev = tok; continue; }
      bytes += this.decodePiece(prev, tok);
      prev = tok;
    }
    const arr = new Uint8Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) arr[i] = bytes.charCodeAt(i);
    return textDecoder.decode(arr);
  }
}
