// eye/worker.js — the Tiny Eye's engine thread (src/eye.js): looks at a drawing, with and
// without the neurons the visitor switched off, and sweeps slices of the 5,000-doodle sample
// set so the page can build neuron cards live. It can hold more than one eye (by key: the full
// eye and the squeezed one) and the training snapshots of the full eye (float16, made into an
// eye the first time the scrubber stops there). A sweep yields between small batches, so a
// drawing sent meanwhile is still answered quickly (and the same-thread fallback the artifact
// sandbox may need keeps the page alive). NOTE for the artifact edition: this file + src/eye.js
// are inlined into the single-file HTML and the worker is built from a Blob.
import { TinyEye, eyeSampleInput } from '../src/eye.js';

const eyes = {}, sweepTokens = {};
let growth = null;                                  // { bytes, bytesEach, eyes: [] } for the full eye's snapshots
const pause = () => new Promise(r => setTimeout(r, 0));
const bytesOf = b => ArrayBuffer.isView(b) ? new Uint8Array(b.buffer, b.byteOffset, b.byteLength) : new Uint8Array(b);

function eyeFor(m) {
  if (m.snap !== null && m.snap !== undefined) {
    if (!growth) throw new Error('the training snapshots are not here yet');
    return growth.eyes[m.snap] || (growth.eyes[m.snap] = new TinyEye(eyes.full.manifest,
      growth.bytes.subarray(m.snap * growth.bytesEach, (m.snap + 1) * growth.bytesEach), { float16: true }));
  }
  const e = eyes[m.key || 'full'];
  if (!e) throw new Error('no eye loaded yet');
  return e;
}

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      const key = m.key || 'full';
      eyes[key] = new TinyEye(typeof m.manifest === 'string' ? JSON.parse(m.manifest) : m.manifest, m.weights);
      sweepTokens[key] = (sweepTokens[key] || 0) + 1;
      if (key === 'full') growth = null;
      const eye = eyes[key];
      self.postMessage({ type: 'loaded', key, classes: eye.classes, widths: eye.widths, hidden: eye.hidden, sizes: eye.sizes,
        params: eye.manifest.params, accuracy: eye.manifest.held_out_acc });
    } else if (m.type === 'growth') {
      growth = { bytes: bytesOf(m.weights), bytesEach: m.bytesEach, eyes: [] };
      self.postMessage({ type: 'growth-ready', count: Math.floor(growth.bytes.length / m.bytesEach) });
    } else if (m.type === 'see') {
      const eye = eyeFor(m), t0 = performance.now();
      const anyOff = m.off && m.off.some(a => a && a.some(v => v));
      const r = eye.forward(m.x, { off: anyOff ? m.off : null, maps: !!m.maps });
      const base = anyOff ? eye.forward(m.x) : null;   // the guess with every neuron on, to compare
      self.postMessage({ type: 'seen', id: m.id, key: m.key || 'full', snap: m.snap ?? null, probs: r.probs, neurons: r.neurons, maps: r.maps || null,
        baseProbs: base ? base.probs : null, ms: performance.now() - t0 });
    } else if (m.type === 'sweep') {
      // every neuron's value for doodles [start, start + count) of the sample (in m.doodles)
      const key = m.key || 'full', eye = eyeFor({ key });
      const token = sweepTokens[key], total = eye.hidden.reduce((a, b) => a + b, 0);
      const u8 = bytesOf(m.doodles);
      const table = new Float32Array(m.count * total), guesses = new Uint8Array(m.count), x = new Float32Array(784);
      for (let i = 0; i < m.count; i++) {
        const r = eye.forward(eyeSampleInput(u8, i, x));
        let at = i * total;
        for (const n of r.neurons) { table.set(n, at); at += n.length; }
        let best = 0; for (let k = 1; k < r.probs.length; k++) if (r.probs[k] > r.probs[best]) best = k;
        guesses[i] = best;
        if (i % 25 === 24) {
          self.postMessage({ type: 'sweep-progress', id: m.id, key, done: i + 1 });
          await pause();
          if (token !== sweepTokens[key]) return;     // a new eye came in under this key: this sweep is stale
        }
      }
      self.postMessage({ type: 'swept', id: m.id, key, start: m.start, count: m.count, table, guesses }, [table.buffer, guesses.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m && m.id, message: String((err && err.message) || err) });
  }
};
