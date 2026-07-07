// train-worker.js — E6 Web Worker: runs the 260K fine-tune benchmark and the 15M
// forward/generation benchmark off the main thread. Module worker; imports the same
// DOM-free modules Node uses.
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy, forward } from '../src/model.js';
import { captureRun } from '../src/capture.js';
import { Trainer, rng, sampleBatch } from '../src/train.js';

self.onmessage = async (e) => {
  const msg = e.data;
  try {
    if (msg.type === 'bench-train') {
      const ck = readCheckpoint(msg.ckptBuf);
      const tr = new Trainer(ck, { B: msg.B, T: msg.T, lr: 1e-4 });
      const stream = new Int32Array(msg.stream);
      const rand = rng(1);
      const times = [];
      for (let i = 0; i < msg.steps; i++) {
        const batch = sampleBatch(stream, msg.B, msg.T, rand);
        const r = tr.step(batch);
        times.push(r.ms);
        self.postMessage({ type: 'progress', step: i + 1, of: msg.steps, ms: r.ms, loss: r.loss });
      }
      const steady = times.slice(1); // drop warmup
      self.postMessage({
        type: 'bench-train-done', B: msg.B, T: msg.T,
        msPerStep: steady.reduce((a, b) => a + b, 0) / steady.length,
        times,
      });
    } else if (msg.type === 'bench-forward') {
      const ck = readCheckpoint(msg.ckptBuf);
      const tok = new Tokenizer(msg.tokBuf, ck.config.vocabSize);
      const model = createModel(ck);
      const t0 = performance.now();
      const { tokens, text } = generateGreedy(model, tok, msg.prompt, msg.steps);
      const ms = performance.now() - t0;
      self.postMessage({
        type: 'bench-forward-done',
        nTokens: tokens.length, ms, tokPerSec: tokens.length / (ms / 1000),
        text,
      });
    } else if (msg.type === 'capture') {
      const ck = readCheckpoint(msg.ckptBuf);
      const tok = new Tokenizer(msg.tokBuf, ck.config.vocabSize);
      const model = createModel(ck);
      const ids = tok.encode(msg.prompt, true, false);
      const cap = captureRun(model, ids, { topK: 8 });
      // ship one attention matrix + top-k rows for the heatmap demo
      const l = msg.layer ?? ck.config.nLayers - 1, h = msg.head ?? 0;
      const n = ids.length;
      const att = new Float32Array(n * n);
      for (let p = 0; p < n; p++) att.set(cap.attnAt(l, h, p), p * n);
      self.postMessage({
        type: 'capture-done', nPos: n, layer: l, head: h,
        pieces: ids.map((id, i) => tok.decodePiece(i > 0 ? ids[i - 1] : 1, id)),
        att: att.buffer, bytes: cap.bytes(),
        topLast: cap.topK[n - 1].map(({ token, logit }) => ({ piece: tok.vocab[token], logit })),
      }, [att.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String(err && err.stack || err) });
  }
};
