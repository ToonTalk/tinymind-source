// Pure, testable decision helpers for the Treasure Hunt. The browser page owns
// rendering; these functions own the evidence thresholds and manifest-derived data.

// A one-character Nursery mind carries its character sheet ({name, kind, attr1, attr2});
// each fact becomes a quest whose answer is its last word, the rule the Nursery's
// pre-flight uses ("lives in the clouds" -> "Sisi lives in the" / " clouds").
export function questPresetsFromSheet(sheet) {
  const name = String(sheet && sheet.name || '').trim();
  if (!name) return [];
  const quest = (slot, phrase, lead) => {
    const words = String(phrase || '').trim().split(/\s+/).filter(Boolean);
    if (words.length < 2) return null;
    const answer = words[words.length - 1].replace(/[^\w']/g, '');
    if (!answer) return null;
    const probe = `${name} ${lead}${words.slice(0, -1).join(' ')}`;
    return { name, slot, label: `${probe} …?`, probe, continuation: ` ${answer}` };   // the label is the sentence to finish
  };
  return [quest('kind', sheet.kind, 'is '), quest('attr1', sheet.attr1, ''), quest('attr2', sheet.attr2, '')].filter(Boolean);
}

export function questPresetsFromManifest(manifest) {
  if (manifest && manifest.sheet) return questPresetsFromSheet(manifest.sheet);
  const chars = manifest && manifest.characters;
  if (!chars || typeof chars !== 'object') return [];
  const slots = ['kind', 'food', 'home', 'food'];
  return Object.entries(chars).map(([name, facts], i) => {
    const slot = slots[i % slots.length];
    const value = String(facts && facts[slot] || '').trim();
    if (!value) return null;
    if (slot === 'kind') return { name, slot, label: `what IS ${name}?`, probe: `${name} is a small`, continuation: ` ${value}` };
    if (slot === 'food') return { name, slot, label: `what does ${name} eat?`, probe: `${name} likes to eat`, continuation: ` ${value}` };
    return { name, slot, label: `where does ${name} live?`, probe: `${name}'s home is by the`, continuation: ` ${value}` };
  }).filter(Boolean);
}

export function controlNamesForManifest(manifest, intendedName, limit = 3) {
  const trained = Object.keys(manifest && manifest.characters || {});
  const fallback = ['Max', 'Lily', 'Tom', 'Anna', 'Ben', 'Mia', 'Sam'];
  const source = trained.length > 1 ? trained : fallback;
  return source.filter(name => name.toLowerCase() !== String(intendedName).toLowerCase()).slice(0, limit);
}

export function nameSpecificityVerdict({ intendedMeanLogProb, alternatives = [], nameInProbe = true }) {
  if (!nameInProbe) return { kind: 'invalid', label: 'not name-specific yet', margin: null, leakageRatio: null };
  const valid = alternatives.filter(r => Number.isFinite(r.meanLogProb));
  if (!Number.isFinite(intendedMeanLogProb) || !valid.length)
    return { kind: 'unavailable', label: 'not name-specific yet', margin: null, leakageRatio: null };
  const bestOther = valid.reduce((a, b) => b.meanLogProb > a.meanLogProb ? b : a);
  const margin = intendedMeanLogProb - bestOther.meanLogProb;
  const leakageRatio = Math.exp(bestOther.meanLogProb - intendedMeanLogProb);
  const intendedRanksFirst = margin > 0;
  if (intendedRanksFirst && leakageRatio <= 0.8 && margin >= 0.15)
    return { kind: 'specific', label: 'name-specific', margin, leakageRatio, bestOther };
  if (intendedRanksFirst)
    return { kind: 'partial', label: 'partly name-specific', margin, leakageRatio, bestOther };
  return { kind: 'not-specific', label: 'not name-specific yet', margin, leakageRatio, bestOther };
}

export function median(values) {
  const xs = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!xs.length) return 0;
  const mid = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[mid] : (xs[mid - 1] + xs[mid]) / 2;
}

export function causalVerdict(targetRatio, controlRatios = []) {
  const safeTarget = Number.isFinite(targetRatio) ? targetRatio : 1;
  const targetDamage = Math.max(0, 1 - safeTarget);
  const medianControlDamage = median(controlRatios.map(r => Math.max(0, 1 - r)));
  const selectivity = targetDamage - medianControlDamage;
  if (safeTarget < 0.5 && targetDamage >= 2 * Math.max(0.01, medianControlDamage)) {
    return { kind: 'selective', label: 'strong causal contributor', targetDamage, medianControlDamage, selectivity };
  }
  if (safeTarget < 0.5) {
    return { kind: 'general', label: 'general-purpose contributor', targetDamage, medianControlDamage, selectivity };
  }
  if (safeTarget > 1.15) {
    return { kind: 'suppressor', label: 'possible suppressor', targetDamage, medianControlDamage, selectivity };
  }
  return { kind: 'little', label: 'little measured effect', targetDamage, medianControlDamage, selectivity };
}
