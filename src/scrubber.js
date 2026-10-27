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
  arcB: 'a mind learning your character (and then just memorizing)',
  arcA: 'a brand-new mind growing up',
  arcC: 'a mind learning which facts belong to which character',
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

// The stored sample includes the prompt because it is useful for replay and
// export. The teaching UI separates the two so children can see which words
// were supplied and which words the model actually added.
export function sampleContinuation(text, prompt) {
  const full = String(text || '');
  const start = String(prompt || '');
  if (!start || !full.startsWith(start)) return full.trim();
  return full.slice(start.length).trimStart();
}

// Four balanced candidate answers make random guessing a one-in-four event.
// Keep this pure so wording and threshold behavior stay covered by Node tests.
export function chanceLevelInterpretation(accuracy, choices = 4) {
  const chance = choices > 0 ? 1 / choices : 0;
  const nearChance = Number.isFinite(accuracy) && Math.abs(accuracy - chance) <= 0.05;
  return {
    chance,
    nearChance,
    label: nearChance ? 'approximately chance' : (accuracy > chance ? 'above chance' : 'below chance'),
    explanation: `With ${choices} possible answers, a blind guess is right about 1 in ${choices} times (${(100 * chance).toFixed(0)}%).`,
  };
}

// Binding arcs carry behavioural measurements alongside story losses. Keep the
// teaching stage derived from those measurements so new schedules do not need
// hard-coded step numbers.
export function bindingStage(snapshot, baseLoss0 = null) {
  const b = snapshot && snapshot.binding;
  if (!b) return null;
  const direct = b.directAccuracy ?? 0;
  const specificity = b.specificityAccuracy ?? 0;
  const baseLossDelta = Number.isFinite(baseLoss0) && Number.isFinite(snapshot.baseLoss)
    ? snapshot.baseLoss - baseLoss0 : null;
  let key, label, headline, summary;
  if (snapshot.step === 0) {
    key = 'before-training';
    label = 'before character training';
    headline = 'The mind is still guessing.';
    summary = 'It knows how simple stories sound, but it has not learned which facts belong to Zog, Mip, Bex and Nia.';
  } else if (specificity >= 0.95 && direct >= 0.9 && baseLossDelta !== null && baseLossDelta >= 0.4) {
    key = 'very-strong-rigid';
    label = 'very strong, but more rigid';
    headline = 'The character facts are very strong, but the mind is more stuck in its ways.';
    summary = 'It matches names and facts very well, while its ordinary storytelling has become noticeably worse.';
  } else if (specificity >= 0.95 && direct >= 0.8) {
    key = 'strong';
    label = 'strong character learning';
    headline = 'The mind has learned the character facts strongly.';
    summary = 'The names now point to the right facts reliably, although some old storytelling skill has faded.';
  } else if (specificity >= 0.8 && direct >= 0.6) {
    key = 'separating';
    label = 'character facts separating';
    headline = 'The character facts are separating.';
    summary = 'The mind usually uses each name to find that character\'s facts, but it is not fully reliable yet.';
  } else if (specificity >= 0.5) {
    key = 'names-matter';
    label = 'names beginning to matter';
    headline = 'The names are beginning to matter.';
    summary = 'Changing the name is starting to change which fact the mind expects.';
  } else if (direct > 0.25 || specificity >= 0.25) {
    key = 'mixing';
    label = 'still mixing character facts';
    headline = 'The mind is still mixing the characters.';
    summary = 'It is learning the new stories, but the names do not reliably choose the right facts yet.';
  } else {
    key = 'learning-world';
    label = 'learning the new story world';
    headline = 'The new story world is sounding more familiar.';
    summary = 'The mind is getting used to these stories, but it is still guessing which fact belongs to each character.';
  }
  return {
    key,
    label,
    headline,
    summary,
    baseLossDelta,
    ordinaryStoryDamageWarning: baseLossDelta !== null && baseLossDelta >= 0.25,
  };
}

// Qualitative first, exact values second. Kid mode consumes only the labels;
// standard mode combines the percentages with these interpretations.
export function bindingStatus(snapshot, baseLoss0 = null, bindingHeld0 = null) {
  const b = snapshot && snapshot.binding;
  if (!b) return null;
  const direct = b.directAccuracy ?? 0;
  const specificity = b.specificityAccuracy ?? 0;
  const leakage = b.meanLeakageRatio ?? 0;
  const overall = b.bindingScore ?? 0;
  const chance = chanceLevelInterpretation(direct, 4);
  const baseDelta = Number.isFinite(baseLoss0) && Number.isFinite(snapshot.baseLoss)
    ? snapshot.baseLoss - baseLoss0 : null;
  const baseRelative = snapshot.step === 0 || baseDelta === null || !baseLoss0 ? null : baseDelta / baseLoss0;
  const heldDelta = Number.isFinite(bindingHeld0) && Number.isFinite(snapshot.bindingHeld)
    ? snapshot.bindingHeld - bindingHeld0 : null;
  const heldRelative = snapshot.step === 0 || heldDelta === null || !bindingHeld0 ? null : heldDelta / bindingHeld0;
  return {
    direct: chance.nearChance ? 'just guessing' : direct < 0.5 ? 'weak' : direct < 0.8 ? 'improving' : 'strong',
    specificity: specificity < 0.4 ? 'weak' : specificity < 0.8 ? 'improving' : 'strong',
    mixing: leakage >= 0.95 ? 'often mixes them' : leakage >= 0.8 ? 'still mixes them' : leakage >= 0.65 ? 'improving' : 'rarely mixes them',
    ordinary: baseRelative === null || baseRelative <= 0.07 ? 'unchanged' : baseRelative < 0.25 ? 'somewhat worse' : 'much worse',
    overall: overall < 0.4 ? 'just guessing' : overall < 0.65 ? 'weak' : overall < 0.85 ? 'improving' : 'strong',
    newStory: heldRelative === null ? 'starting point' : heldRelative <= -0.2 ? 'much easier' : heldRelative < -0.05 ? 'easier' : heldRelative > 0.05 ? 'harder again' : 'about the same',
    chance,
    directValue: direct,
    specificityValue: specificity,
    leakageValue: leakage,
    overallValue: overall,
    baseDelta,
    baseRelative,
    heldDelta,
    heldRelative,
  };
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
