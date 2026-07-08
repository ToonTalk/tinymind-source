// scrubber.js — pure core for the training-arc time machine (scrubber session).
// No DOM, no fetch: manifest validation, arc-pack recognition inside a dropped file
// batch, the multi-arc pool, and label/caption selection. The Nursery page and the
// shell both consume this; tests exercise it directly in Node.
//
// Arc-pack contract (arc kickoff): a manifest is any JSON with a string `arc` and an
// array `snapshots`; each snapshot has { step, file, samples } and may carry
// trainLoss/zogHeld/baseLoss. A manifest may be ANY PREFIX of its schedule — never
// assume a count, spacing, or final step. Multiple arcs coexist keyed by `arc`.

export function isArcManifest(data) {
  return !!(data && typeof data === 'object' && typeof data.arc === 'string' &&
    Array.isArray(data.snapshots));
}

// Returns { ok, error?, manifest? } — steps must be finite, ascending, files named.
export function validateArcManifest(data) {
  if (!isArcManifest(data)) return { ok: false, error: 'not an arc manifest (needs arc + snapshots[])' };
  if (!data.snapshots.length) return { ok: false, error: 'arc "' + data.arc + '" has no snapshots' };
  let prev = -Infinity;
  for (const s of data.snapshots) {
    if (!s || !Number.isFinite(s.step) || typeof s.file !== 'string')
      return { ok: false, error: 'arc "' + data.arc + '": snapshot needs numeric step + file name' };
    if (s.step <= prev) return { ok: false, error: 'arc "' + data.arc + '": steps must ascend (prefix-valid)' };
    prev = s.step;
  }
  return { ok: true, manifest: data };
}

// Split a batch of dropped records [{name, u8}] into arc packs and the rest.
// A pack = a valid manifest JSON + every .bin the manifest names that is present in
// the same batch. Bins claimed by a manifest do NOT flow on to the model pool.
// Returns { arcs: [{manifest, bins: Map<file, u8>}], rest: [records...] }.
export function splitArcPacks(recs) {
  const arcs = [];
  const byName = new Map(recs.map(r => [r.name, r]));
  const claimed = new Set();
  for (const r of recs) {
    if (!/\.json$/i.test(r.name)) continue;
    let data;
    try { data = JSON.parse(new TextDecoder().decode(r.u8)); } catch { continue; }
    const v = validateArcManifest(data);
    if (!v.ok) continue;                      // ordinary json — storybook/replay/wordlist
    claimed.add(r);
    const bins = new Map();
    for (const s of data.snapshots) {
      const hit = byName.get(s.file);
      if (hit) { bins.set(s.file, hit.u8); claimed.add(hit); }
    }
    arcs.push({ manifest: data, bins });
  }
  return { arcs, rest: recs.filter(r => !claimed.has(r)) };
}

// Little-kid labels, data-driven with known variants (arc kickoff §build 2).
// Unknown arcs fall back to the manifest's own note, then the arc id — so Arc A
// arriving later needs zero code changes, and so does Arc Q.
export const KID_LABELS = {
  arcB: 'the trainee learning Zog (and then just memorizing)',
  arcA: 'a brand-new mind growing up',
};
export function arcKidLabel(manifest) {
  return KID_LABELS[manifest.arc] || manifest.note || manifest.arc;
}

// Caption for a slider stop: the manifest's greedy sample. When a snapshot carries
// several prompts, prefer the requested one, else the LAST key (manifest order puts
// the arc-specific prompt after the generic one in the packs generated so far).
export function stopCaption(snapshot, preferPrompt) {
  const samples = snapshot && snapshot.samples;
  if (!samples || typeof samples !== 'object') return { prompt: null, text: '' };
  const keys = Object.keys(samples);
  if (!keys.length) return { prompt: null, text: '' };
  const prompt = (preferPrompt && keys.includes(preferPrompt)) ? preferPrompt : keys[keys.length - 1];
  return { prompt, text: String(samples[prompt]) };
}

// Multi-arc pool. Adding an arc with an existing key replaces it (idempotent re-drop);
// bins/urlBase merge so a manifest push (no bins) and a zip drop (bins) compose.
export function createArcPool() {
  const arcs = new Map();     // arc -> { manifest, bins: Map<file,u8>, urlBase: string|null }
  return {
    add(manifest, { bins = null, urlBase = null } = {}) {
      const v = validateArcManifest(manifest);
      if (!v.ok) throw new Error(v.error);
      const prev = arcs.get(manifest.arc);
      const entry = {
        manifest,
        bins: bins || (prev && prev.bins) || new Map(),
        urlBase: urlBase !== null ? urlBase : (prev ? prev.urlBase : null),
      };
      if (bins && prev && prev.bins) for (const [k, u8] of prev.bins) if (!entry.bins.has(k)) entry.bins.set(k, u8);
      arcs.set(manifest.arc, entry);
      return entry;
    },
    get: (arc) => arcs.get(arc) || null,
    list: () => [...arcs.values()].map(e => e.manifest),
    size: () => arcs.size,
    // Bytes for a snapshot: from the pack's bins, else via the caller-supplied lazy
    // fetcher (RPC to the shell, or fetch(urlBase + file) on a static host).
    async snapshotBytes(arc, file, lazyFetch) {
      const e = arcs.get(arc);
      if (!e) throw new Error('unknown arc ' + arc);
      if (e.bins.has(file)) return e.bins.get(file);
      if (!lazyFetch) throw new Error(file + ' is not in the pack and no loader was provided');
      const u8 = await lazyFetch(e, file);
      e.bins.set(file, u8);                 // cache — scrub back and forth is free after first load
      return u8;
    },
  };
}

// Slider-stop model: index <-> snapshot, nearest stop for an arbitrary step.
export function nearestStopIndex(snapshots, step) {
  let best = 0, bestD = Infinity;
  snapshots.forEach((s, i) => { const d = Math.abs(s.step - step); if (d < bestD) { bestD = d; best = i; } });
  return best;
}
