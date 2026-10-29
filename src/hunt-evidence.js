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

// ---- the detectives --------------------------------------------------------------
// The built-in detective turns the probe's evidence (the hunt worker's probe-done: head
// attention to the name, name-piece and answer-boundary neurons) into clues, ending
// with a whole-layer "team" clue, since knowledge is usually shared.
export function builtinHypotheses(m, { name, continuation, nHeads }) {
  const out = [];
  for (const h of m.headScores.slice(0, 2)) {
    if (h.score < 0.02) break;
    out.push({ kind: 'head', layer: h.layer, index: h.head,
      source: 'name attention',
      claim: `Attention head L${h.layer}·H${h.head} looks back at "${name}" right when it guesses the answer (${(100 * h.score).toFixed(0)}% of its attention).`,
      prediction: `If we switch it OFF, the model should get worse at saying "${continuation}".` });
  }
  for (const nr of m.neuronTop.slice(0, 1)) {
    if (nr.score < 1) break;
    out.push({ kind: 'neuron', layer: nr.layer, index: nr.neuron, source: 'name-piece contrast',
      claim: `Neuron L${nr.layer}·n${nr.neuron} lights up ${nr.score.toFixed(1)}× brighter on the "${name}" pieces than anywhere else.`,
      prediction: `If we switch it OFF, the model should get worse at saying "${continuation}".` });
  }
  for (const nr of (m.boundaryNeurons || []).slice(0, 2)) {
    if (out.some(h => h.kind === 'neuron' && h.layer === nr.layer && h.index === nr.neuron)) continue;
    out.push({ kind: 'neuron', layer: nr.layer, index: nr.neuron, source: 'answer boundary',
      claim: `Neuron L${nr.layer}·n${nr.neuron} is unusually active at the answer boundary after "${m.triggerWord || 'the last piece'}".`,
      prediction: `If that boundary activity contributes selectively, switching it OFF should hurt "${continuation}" more than unrelated completions.` });
  }
  // the team hypothesis: knowledge is usually SHARED — single parts are often
  // redundant, whole layers rarely are. (This is the ladder after busted clues.)
  const layerTotals = {};
  for (const h of m.allHeadScores ?? m.headScores) layerTotals[h.layer] = (layerTotals[h.layer] ?? 0) + h.score;
  const bestLayer = +Object.entries(layerTotals).sort((a, b) => b[1] - a[1])[0][0];
  out.push({ kind: 'team', layer: bestLayer,
    source: 'layer team',
    claim: `Maybe no single head dominates — the WHOLE TEAM of layer ${bestLayer}'s attention heads may contribute together.`,
    prediction: `Switch off ALL ${nHeads} heads in layer ${bestLayer} at once, then compare target damage with unrelated controls.` });
  return out;
}

// The language-model detective: the prompt it gets (the same evidence as JSON), and the
// clues kept from its reply (a JSON array of heads / neurons with integer places, or
// "team" clues: every head of one layer).
export function detectivePrompt({ probe, continuation, name, config, m }) {
  const evidence = {
    task: `The tiny language model completes "${probe}" with "${continuation}" (probability ${m.contProb}).`,
    character: name,
    modelShape: config,
    topGuesses: m.top,
    headAttentionToName_atAnswerPosition: m.headScores,
    // the whole-layer picture, for team hypotheses: every head's attention to the name, summed per layer
    attentionToName_perLayer: (() => {
      const t = {};
      for (const h of m.allHeadScores || m.headScores || []) t[h.layer] = (t[h.layer] || 0) + h.score;
      return Object.entries(t).map(([layer, total]) => ({ layer: +layer, total: +total.toFixed(3) })).sort((a, b) => b.total - a.total);
    })(),
    neuronContrast_onNamePieces: m.neuronTop,
    neuronsActive_atAnswerBoundary: m.boundaryNeurons,
  };
  return `You are a friendly detective helping a child (age 10) test which parts of a tiny transformer contribute when it predicts a learned character fact. Study the correlational evidence, then propose EXACTLY 3 falsifiable hypotheses the child can test by switching parts off (ablation). Do not claim that a component stores the fact merely because it contributes. In a mind this small, knowledge is often shared: a single head or neuron is frequently redundant while a whole layer is not, so at least one hypothesis should be a "team" (every attention head of one layer switched off together). For a team, pick the layer whose heads pay the most attention to the name in total (attentionToName_perLayer), not the layer you would expect to hold facts in a big model.

Evidence (JSON):
${JSON.stringify(evidence)}

Answer with ONLY a JSON array, no prose:
[{"kind":"head"|"neuron"|"team","layer":<int>,"index":<int, omitted for a team>,"claim":"<one short kid-friendly sentence naming the exact part and what you think it does>","prediction":"<what should happen to the model's chance of answering correctly if the child switches that part off>"}]`;
}
export function parseHypotheses(raw) {
  const jsonText = String(raw).match(/\[[\s\S]*\]/)?.[0];
  const arr = JSON.parse(jsonText);
  return arr.filter(h => Number.isInteger(h.layer) &&
    ((h.kind === 'head' || h.kind === 'neuron') && Number.isInteger(h.index) || h.kind === 'team')).slice(0, 3);
}
