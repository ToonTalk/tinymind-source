// Shared evaluation helpers for the balanced character-binding experiment.
// Uses teacher-forced continuation scores, matching Tiny Mind's Treasure Hunt metric,
// plus per-token normalization when comparing answers of different token lengths.
import { createModel, forward, generateGreedy } from '../src/model.js';

function replaceName(prompt, name, replacement) {
  const esc = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return prompt.replace(new RegExp(`\\b${esc}\\b`, 'g'), replacement);
}

function logProbForTarget(logits, target) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  return logits[target] - max - Math.log(sum);
}

export function scoreContinuation(model, tok, prompt, continuation) {
  const probeIds = tok.encode(prompt, true, false);
  const ids = tok.encode(prompt + continuation, true, false);
  if (probeIds.length < 1 || ids.length <= probeIds.length) throw new Error(`empty continuation for: ${prompt}`);
  for (let i = 0; i < probeIds.length - 1; i++) {
    if (probeIds[i] !== ids[i]) throw new Error(`tokenizer merged across seam: ${JSON.stringify(prompt)} + ${JSON.stringify(continuation)}`);
  }
  const boundary = probeIds.length - 1;
  let logProb = 0;
  const pieces = [];
  for (let pos = 0; pos < ids.length - 1; pos++) {
    const logits = forward(model, ids[pos], pos);
    if (pos >= boundary) {
      const target = ids[pos + 1];
      logProb += logProbForTarget(logits, target);
      pieces.push({ id: target, piece: tok.decodePiece(ids[pos], target) });
    }
  }
  const tokenCount = ids.length - 1 - boundary;
  return {
    logProb,
    probability: Math.exp(logProb),
    tokenCount,
    meanLogProb: logProb / tokenCount,
    geometricMeanProbability: Math.exp(logProb / tokenCount),
    pieces,
  };
}

function winnerForItem(model, tok, item, values) {
  const candidates = values.map(value => {
    const score = scoreContinuation(model, tok, item.prompt, ' ' + value);
    return { value, ...score };
  }).sort((a, b) => b.meanLogProb - a.meanLogProb);
  const expected = item.continuation.trim();
  return { correct: candidates[0]?.value === expected, expected, candidates };
}

function evaluateSet(model, tok, items, challenges) {
  const rows = items.map(item => ({
    id: item.id, name: item.name, slot: item.slot, prompt: item.prompt,
    ...winnerForItem(model, tok, item, challenges.slots[item.slot].values),
  }));
  return { accuracy: rows.filter(r => r.correct).length / Math.max(1, rows.length), rows };
}

function evaluateSpecificityFromDirect(direct, challenges) {
  const names = Object.keys(challenges.characters);
  const rows = challenges.direct.map(item => {
    const target = item.continuation.trim();
    const scores = names.map(name => {
      const sourceRow = direct.rows.find(r => r.name === name && r.slot === item.slot);
      const candidate = sourceRow?.candidates.find(c => c.value === target);
      if (!candidate) throw new Error(`missing direct score for ${name}/${item.slot}/${target}`);
      return { name, ...candidate };
    }).sort((a, b) => b.meanLogProb - a.meanLogProb);
    const correctScore = scores.find(x => x.name === item.name);
    const bestWrong = scores.find(x => x.name !== item.name);
    return {
      id: item.id, expectedName: item.name, target,
      correct: scores[0]?.name === item.name,
      marginNatsPerToken: correctScore.meanLogProb - bestWrong.meanLogProb,
      leakageRatio: Math.exp(bestWrong.meanLogProb - correctScore.meanLogProb),
      scores,
    };
  });
  return {
    accuracy: rows.filter(r => r.correct).length / Math.max(1, rows.length),
    meanMarginNatsPerToken: rows.reduce((a, r) => a + r.marginNatsPerToken, 0) / Math.max(1, rows.length),
    meanLeakageRatio: rows.reduce((a, r) => a + r.leakageRatio, 0) / Math.max(1, rows.length),
    rows,
  };
}

function evaluateOverrides(model, tok, challenges) {
  const rows = challenges.override.map(item => {
    const wanted = scoreContinuation(model, tok, item.prompt, item.continuation);
    const learned = scoreContinuation(model, tok, item.prompt, item.learnedContinuation);
    return {
      id: item.id, name: item.name, prompt: item.prompt,
      wanted: item.continuation.trim(), learned: item.learnedContinuation.trim(),
      correct: wanted.meanLogProb > learned.meanLogProb,
      marginNatsPerToken: wanted.meanLogProb - learned.meanLogProb,
      wantedScore: wanted, learnedScore: learned,
    };
  });
  return { accuracy: rows.filter(r => r.correct).length / Math.max(1, rows.length), rows };
}

function repetitionStats(text) {
  const words = text.toLowerCase().match(/[a-z]+(?:'[a-z]+)?/g) ?? [];
  const grams = new Map();
  for (let i = 0; i + 3 < words.length; i++) {
    const g = words.slice(i, i + 4).join(' ');
    grams.set(g, (grams.get(g) ?? 0) + 1);
  }
  const repeated = [...grams.values()].filter(n => n > 1).reduce((a, n) => a + n - 1, 0);
  return { words: words.length, repeatedFourGramOccurrences: repeated, repetitionRate: repeated / Math.max(1, words.length - 3) };
}

// Small, explicit heuristics that complement repetition rate. They catch the
// characteristic failures in this arc, but are not presented as a complete prose
// quality metric.
export function languageQualityStats(text, expectedName, characters) {
  const words = text.match(/[A-Za-z]+(?:'[A-Za-z]+)?/g) ?? [];
  const lower = words.map(w => w.toLowerCase());
  const names = Object.keys(characters || {});
  const expectedLower = String(expectedName).toLowerCase();
  const expectedNameMentions = lower.filter(w => w === expectedLower).length;
  const wrongNames = names.filter(n => n.toLowerCase() !== expectedLower).map(name => ({
    name, count: lower.filter(w => w === name.toLowerCase()).length,
  })).filter(x => x.count > 0);
  const own = new Set(Object.values(characters?.[expectedName] || {}).map(v => String(v).toLowerCase()));
  const wrongValues = new Map();
  for (const [name, facts] of Object.entries(characters || {})) {
    if (name === expectedName) continue;
    for (const value of Object.values(facts || {})) {
      const v = String(value).toLowerCase();
      if (own.has(v)) continue;
      const count = lower.filter(w => w === v).length;
      if (count) wrongValues.set(v, (wrongValues.get(v) || 0) + count);
    }
  }
  const malformed = [];
  for (const word of words) {
    const w = word.toLowerCase();
    const prefix = names.find(n => w.startsWith(n.toLowerCase()) && w !== n.toLowerCase());
    if (prefix && !malformed.some(x => x.form.toLowerCase() === w)) malformed.push({ form: word, prefix });
  }
  return {
    expectedNameMentions,
    expectedNameRetained: expectedNameMentions > 0,
    wrongCharacterNameIntrusionCount: wrongNames.reduce((sum, x) => sum + x.count, 0),
    wrongCharacterNames: wrongNames,
    wrongAttributeIntrusionCount: [...wrongValues.values()].reduce((a, b) => a + b, 0),
    wrongAttributes: [...wrongValues].map(([value, count]) => ({ value, count })),
    malformedNamePrefixCount: malformed.length,
    malformedNamePrefixes: malformed,
  };
}

export function evaluateBindingModel(checkpoint, tok, challenges, { mode = 'full', samples = true } = {}) {
  const model = createModel(checkpoint);
  const direct = evaluateSet(model, tok, challenges.direct, challenges);
  const specificity = evaluateSpecificityFromDirect(direct, challenges);
  const out = {
    direct,
    specificity,
    summary: {
      directAccuracy: direct.accuracy,
      specificityAccuracy: specificity.accuracy,
      meanSpecificityMargin: specificity.meanMarginNatsPerToken,
      meanLeakageRatio: specificity.meanLeakageRatio,
    },
  };
  if (mode === 'full') {
    out.paraphrase = evaluateSet(model, tok, challenges.paraphrase, challenges);
    out.distractor = evaluateSet(model, tok, challenges.distractor, challenges);
    out.override = evaluateOverrides(model, tok, challenges);
    Object.assign(out.summary, {
      paraphraseAccuracy: out.paraphrase.accuracy,
      distractorAccuracy: out.distractor.accuracy,
      overrideAccuracy: out.override.accuracy,
    });
  }
  const components = [out.summary.directAccuracy, out.summary.specificityAccuracy];
  if (mode === 'full') components.push(out.summary.paraphraseAccuracy, out.summary.distractorAccuracy);
  out.summary.bindingScore = components.reduce((a, b) => a + b, 0) / components.length;

  if (samples) {
    out.samples = {};
    for (const name of Object.keys(challenges.characters)) {
      const text = generateGreedy(createModel(checkpoint), tok, `Once upon a time, ${name}`, 140).text;
      out.samples[name] = { text, ...repetitionStats(text), ...languageQualityStats(text, name, challenges.characters) };
    }
  }
  return out;
}

export function compactBindingSummary(result) {
  const s = result.summary;
  return {
    directAccuracy: +s.directAccuracy.toFixed(4),
    specificityAccuracy: +s.specificityAccuracy.toFixed(4),
    meanSpecificityMargin: +s.meanSpecificityMargin.toFixed(4),
    meanLeakageRatio: +s.meanLeakageRatio.toFixed(4),
    ...(s.paraphraseAccuracy === undefined ? {} : {
      paraphraseAccuracy: +s.paraphraseAccuracy.toFixed(4),
      distractorAccuracy: +s.distractorAccuracy.toFixed(4),
      overrideAccuracy: +s.overrideAccuracy.toFixed(4),
    }),
    bindingScore: +s.bindingScore.toFixed(4),
  };
}
