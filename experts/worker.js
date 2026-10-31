// experts/worker.js — the Experts room's engine thread: holds the mind made of experts, writes
// stories (always its most likely word unless asked for surprise) with experts switched off
// or not, and reads them back to find which expert every piece went to in every layer
// (src/experts.js). Pool calls are asynchronous, so messages are handled one at a time, in
// order; only "stop" jumps the queue. NOTE for the artifact edition: this file + the src
// imports are inlined into the single-file HTML and the worker is built from a Blob.
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { Pool } from '../src/batch.js';
import { Experts } from '../src/experts.js';

let tok = null, config = null, ex = null, poolP = Promise.resolve(null);
let stopFlag = false;

const yieldNow = () => new Promise(r => setTimeout(r, 0));   // lets "stop" in between pieces

let queue = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  if (m && m.type === 'stop') { stopFlag = true; return; }
  queue = queue.then(() => handle(m));
};
self.whenIdle = () => queue;   // for tests that drive this worker in-process

// a story with every piece's expert in every layer and the router's votes
async function story(type, ids, m, extra) {
  const r = await ex.read(ids, { mute: m.mute || [], reroute: m.reroute || [],
    onProgress: (done, total) => self.postMessage({ type: 'progress', tag: m.tag || null, done, total }) });
  const pieces = ids.map((id, i) => (i === 0 ? '' : tok.decodePiece(ids[i - 1], id)));
  self.postMessage({ type, tag: m.tag || null, ids: Array.from(ids), pieces, top: r.top, gate: r.gate, probs: r.probs,
    L: r.L, E: r.E, k: r.k, mute: m.mute || [], reroute: m.reroute || [], ...extra }, [r.top.buffer, r.gate.buffer, r.probs.buffer]);
}

async function handle(m) {
  try {
    if (m.type === 'load') {
      poolP.then(p => p && p.dispose());
      ex = null;
      const ck = readCheckpoint(m.ckptBuf);
      config = ck.config;
      const info = { dim: config.dim, nLayers: config.nLayers, nHeads: config.nHeads, vocabSize: config.vocabSize, seqLen: config.seqLen,
        experts: config.experts || null };
      if (!config.experts) { self.postMessage({ type: 'loaded', config: info, notExperts: true }); return; }
      tok = new Tokenizer(m.tokBuf, config.vocabSize);
      const model = createModel(ck);
      poolP = Pool.create(model).catch(() => null);
      const pool = await poolP;
      ex = new Experts(model, pool);
      self.postMessage({ type: 'loaded', config: info, cores: pool ? pool.K : 1 });
    } else if (m.type === 'unload') {
      poolP.then(p => p && p.dispose()); poolP = Promise.resolve(null);
      ex = null;
      self.postMessage({ type: 'unloaded' });
    } else if (m.type === 'write') {
      if (!ex) throw new Error('no mind made of experts is loaded');
      const t0 = performance.now();
      stopFlag = false;
      const prompt = m.promptIds || tok.encode(m.prompt, true, false);
      const ids = await ex.write(prompt, { maxNew: m.maxNew || 120, temperature: m.temperature || 0, seed: m.seed || 1,
        mute: m.mute || [], reroute: m.reroute || [],
        onToken: async (id) => {
          self.postMessage({ type: 'gen-token', tag: m.tag || null, piece: tok.decodePiece(0, id) });
          await yieldNow();
          return !stopFlag;
        } });
      await story('written', ids, m, { promptLen: prompt.length, stopped: stopFlag, ms: performance.now() - t0 });
    } else if (m.type === 'read') {
      if (!ex) throw new Error('no mind made of experts is loaded');
      const t0 = performance.now();
      let ids = m.ids || tok.encode(m.text, true, false);
      const truncated = ids.length > config.seqLen;
      if (truncated) ids = ids.slice(0, config.seqLen);
      await story('read', ids, m, { promptLen: ids.length, truncated, ms: performance.now() - t0 });
    }
  } catch (err) {
    self.postMessage({ type: 'error', tag: (m && m.tag) || null, message: String((err && err.message) || err) });
  }
}
