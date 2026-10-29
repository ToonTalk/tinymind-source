// thought/worker.js — the Thought bubble's engine thread: holds the mind, its J-lens and
// the story vocabulary, reads stories, writes on, and answers readout / heat queries
// from the tables src/thought.js keeps. Pool calls are asynchronous, so messages are
// handled one at a time, in order; only "stop" jumps the queue (it ends a story being
// written). NOTE for the artifact edition: this file + the src imports are inlined
// into the single-file HTML and the worker is built from a Blob.
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { readJlens } from '../src/jlens.js';
import { Pool } from '../src/batch.js';
import { Thought } from '../src/thought.js';

let model = null, tok = null, config = null, th = null, poolP = Promise.resolve(null);
let stopFlag = false;

function pieceOf(ids, i) { return i === 0 ? '' : tok.decodePiece(ids[i - 1], ids[i]); }

// the story, its pieces, and how long the work took
function storyMsg(type, ids, extra) {
  return { type, ids: ids.slice(), pieces: ids.map((_, i) => pieceOf(ids, i)), ...extra };
}

const yieldNow = () => new Promise(r => setTimeout(r, 0));   // lets "stop" in between pieces

let queue = Promise.resolve();
self.onmessage = (e) => {
  const m = e.data;
  if (m && m.type === 'stop') { stopFlag = true; return; }
  queue = queue.then(() => handle(m));
};
self.whenIdle = () => queue;   // for tests that drive this worker in-process

async function handle(m) {
  try {
    if (m.type === 'load') {
      poolP.then(p => p && p.dispose());
      th = null; model = null;
      const ck = readCheckpoint(m.ckptBuf);
      config = ck.config;
      tok = new Tokenizer(m.tokBuf, config.vocabSize);
      let jl = null;
      if (m.lensBuf) { try { const L = readJlens(m.lensBuf); if (L.dModel === config.dim) jl = L; } catch (_) { jl = null; } }
      const info = { dim: config.dim, nLayers: config.nLayers, nHeads: config.nHeads, hiddenDim: config.hiddenDim, vocabSize: config.vocabSize, seqLen: config.seqLen };
      if (!jl || config.vocabSize !== 32000) {   // the story vocabulary is for the 32,000-piece alphabet
        self.postMessage({ type: 'loaded', config: info, lensLayers: [], noLens: true });
        return;
      }
      model = createModel(ck);
      poolP = Pool.create(model).catch(() => null);
      const pool = await poolP;
      th = new Thought(model, jl, m.vocab, pool);
      await th.ready();
      // the piece each story-vocabulary word prints as (after another piece, so with its space)
      const pairs = Array.from(th.ids, id => [id, tok.decodePiece(0, id)]);
      self.postMessage({ type: 'loaded', config: info, lensLayers: th.band, cores: pool ? pool.K : 1, pairs });
    } else if (m.type === 'unload') {
      poolP.then(p => p && p.dispose()); poolP = Promise.resolve(null);
      th = null; model = null;
      self.postMessage({ type: 'unloaded' });
    } else if (m.type === 'analyze') {
      if (!th) throw new Error('no mind with thinking ahead is loaded');
      const t0 = performance.now();
      let ids = m.ids || tok.encode(m.text, true, false), truncated = false;
      const max = Math.min(config.seqLen, 256);
      if (ids.length > max) { ids = ids.slice(0, max); truncated = true; }
      await th.analyze(ids, { edit: m.edit ? editOf(m.edit) : null,
        onProgress: (done, total) => self.postMessage({ type: 'progress', stage: 'read', done, total, tag: m.tag || null }) });
      self.postMessage(storyMsg('analyzed', ids, { ms: performance.now() - t0, truncated, tag: m.tag || null }));
    } else if (m.type === 'generate') {
      if (!th) throw new Error('no mind with thinking ahead is loaded');
      const t0 = performance.now();
      stopFlag = false;
      const prompt = m.promptIds || tok.encode(m.prompt, true, false);
      const edit = m.edit ? editOf(m.edit) : null;
      const ids = await th.generate(prompt, { maxNew: m.maxNew || 120, temperature: m.temperature ?? 0.8, seed: m.seed || 1, edit,
        onToken: async (id, pos, all) => {
          self.postMessage({ type: 'gen-token', tag: m.tag || null, piece: tok.decodePiece(0, id), pos });
          await yieldNow();
          return !stopFlag;
        } });
      const msWrite = performance.now() - t0;
      // read what was written (with the same swap on), so its thoughts can be looked at;
      // a rewrite that is only there to compare with skips this (noRead)
      const max = Math.min(config.seqLen, 256);
      const read = ids.length > max ? ids.slice(0, max) : ids;
      if (m.noRead) { self.postMessage(storyMsg('generated', read, { ms: msWrite, msWrite, promptLen: prompt.length, stopped: stopFlag, tag: m.tag || null })); return; }
      await th.analyze(read, { edit,
        onProgress: (done, total) => self.postMessage({ type: 'progress', stage: 'read', done, total, tag: m.tag || null }) });
      self.postMessage(storyMsg('generated', read, { ms: performance.now() - t0, msWrite, promptLen: prompt.length, stopped: stopFlag, tag: m.tag || null }));
    } else if (m.type === 'readout') {
      if (!th || !th.cap) return;
      self.postMessage({ type: 'readout', ...th.readout(m.pos, m.k || 40) });
    } else if (m.type === 'heat') {
      if (!th || !th.cap) return;
      const r = th.heat(m.tokenId);
      self.postMessage({ type: 'heat', tokenId: m.tokenId, ranks: r ? Array.from(r) : null });
    }
  } catch (err) {
    self.postMessage({ type: 'error', message: String((err && err.message) || err) });
  }
}

function editOf(e) {
  return th.makeEdit(e.outId, e.inId, e.l0, e.l1, e.fromPos, e.strength);
}
