// eye/worker.js — the Tiny Eye's engine thread (src/eye.js): looks at a drawing, with and
// without the neurons the visitor switched off, and sweeps slices of the 5,000-doodle sample
// set so the page can build neuron cards live. A sweep yields between small batches, so a
// drawing sent meanwhile is still answered quickly (and the same-thread fallback the
// artifact sandbox may need keeps the page alive). NOTE for the artifact edition: this file
// + src/eye.js are inlined into the single-file HTML and the worker is built from a Blob.
import { TinyEye, eyeSampleInput } from '../src/eye.js';

let eye = null, sweepToken = 0;
const pause = () => new Promise(r => setTimeout(r, 0));

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === 'load') {
      eye = new TinyEye(typeof m.manifest === 'string' ? JSON.parse(m.manifest) : m.manifest, m.weights);
      sweepToken++;
      self.postMessage({ type: 'loaded', classes: eye.classes, widths: eye.widths, hidden: eye.hidden, sizes: eye.sizes,
        params: eye.manifest.params, accuracy: eye.manifest.held_out_acc });
    } else if (m.type === 'see') {
      if (!eye) throw new Error('no eye loaded yet');
      const t0 = performance.now();
      const anyOff = m.off && m.off.some(a => a && a.some(v => v));
      const r = eye.forward(m.x, { off: anyOff ? m.off : null, maps: !!m.maps });
      const base = anyOff ? eye.forward(m.x) : null;   // the guess with every neuron on, to compare
      const out = { type: 'seen', id: m.id, probs: r.probs, neurons: r.neurons, maps: r.maps || null,
        baseProbs: base ? base.probs : null, ms: performance.now() - t0 };
      self.postMessage(out);
    } else if (m.type === 'sweep') {
      // every neuron's value for doodles [start, start + count) of the sample (in m.doodles)
      if (!eye) throw new Error('no eye loaded yet');
      const token = sweepToken, total = eye.hidden.reduce((a, b) => a + b, 0);
      const u8 = ArrayBuffer.isView(m.doodles) ? new Uint8Array(m.doodles.buffer, m.doodles.byteOffset, m.doodles.byteLength) : new Uint8Array(m.doodles);
      const table = new Float32Array(m.count * total), guesses = new Uint8Array(m.count), x = new Float32Array(784);
      for (let i = 0; i < m.count; i++) {
        const r = eye.forward(eyeSampleInput(u8, i, x));
        let at = i * total;
        for (const n of r.neurons) { table.set(n, at); at += n.length; }
        let best = 0; for (let k = 1; k < r.probs.length; k++) if (r.probs[k] > r.probs[best]) best = k;
        guesses[i] = best;
        if (i % 25 === 24) {
          self.postMessage({ type: 'sweep-progress', id: m.id, done: i + 1 });
          await pause();
          if (token !== sweepToken) return;          // a new eye came in: this sweep is stale
        }
      }
      self.postMessage({ type: 'swept', id: m.id, start: m.start, count: m.count, table, guesses }, [table.buffer, guesses.buffer]);
    }
  } catch (err) {
    self.postMessage({ type: 'error', id: m && m.id, message: String((err && err.message) || err) });
  }
};
