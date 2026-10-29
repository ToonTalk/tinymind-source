// ghostwriter.js — the Nursery's storybook ghostwriter, minus the page: the prompt it
// sends a language model, how it splits the reply into stories, the mechanical checks a
// story must pass to go in the storybook (with plain-language reasons), and the repair
// prompt for stories whose only problem is a few unknown words. The Nursery page and
// tools/ai-check.mjs (testing against a real model) both use this module, so the test
// exercises exactly what visitors get. DOM-free; Node + browser.
//
// sheet: { name, kind, attr1, attr2 } (the character a child invented); wordlist: Set
// of words the tiny mind met often enough to learn (corpus/wordlist.json).

export const unsmart = t => t.replace(/[‘’]/g, "'").replace(/[“”]/g, '"');
export const words = t => [...unsmart(t).toLowerCase().matchAll(/[a-z]+(?:'[a-z]+)?/g)].map(m => m[0]);

// the character's own words are always allowed, in every form: possessive, and the
// family the fact check accepts ("sleeps in a teapot" -> sleep, slept, sleeping…), or a
// story telling the fact in the past tense would be turned away for using "slept"
export function whitelistFor(sh) {
  const ws = new Set();
  for (const f of Object.values(sh)) for (const w of words(f)) { ws.add(w + "'s"); for (const x of familyOf(w)) ws.add(x); }
  return ws;
}
// glue words never required by the attribute check (modals, articles, pronouns, degree
// words). Position words (behind, under, over…) are NOT here — they carry meaning.
const STOP = new Set(['a', 'an', 'the', 'and', 'or', 'of', 'to', 'is', 'it', 'its',
  'can', 'could', 'will', 'would', 'may', 'might', 'must', 'should', 'do', 'does', 'did',
  'has', 'have', 'had', 'be', 'been', 'was', 'were', 'are', 'am', 'he', 'she', 'they',
  'his', 'her', 'their', 'very', 'really', 'always', 'only', 'likes', 'loves', 'that',
  // dwelling verbs are linking words — "lives behind the moon" is about the moon
  // ("home is behind the moon" must pass; Phase 0 checked the place phrase only)
  'live', 'lives', 'lived', 'living']);
// irregular kid-verbs: attribute matching accepts any form of the word's family
const IRREG = {
  eat: ['ate', 'eaten'], fly: ['flew', 'flown', 'flies'], go: ['went', 'gone', 'goes'],
  run: ['ran'], swim: ['swam', 'swum'], sleep: ['slept', 'asleep'], sing: ['sang', 'sung'],
  see: ['saw', 'seen'], say: ['said'], give: ['gave', 'given'], get: ['got'],
  make: ['made'], ride: ['rode', 'ridden'], hide: ['hid', 'hidden'], sit: ['sat'],
  stand: ['stood'], drink: ['drank', 'drunk'], fall: ['fell', 'fallen'], feel: ['felt'],
  find: ['found'], keep: ['kept'], know: ['knew', 'known'], leave: ['left'],
  tell: ['told'], think: ['thought'], bring: ['brought'], catch: ['caught'],
  come: ['came'], draw: ['drew', 'drawn'], grow: ['grew', 'grown'], hold: ['held'],
  put: [], read: [], take: ['took', 'taken'], wear: ['wore', 'worn'], blow: ['blew', 'blown'],
};
function familyOf(w) {
  const bases = new Set([w]);
  if (w.endsWith('ies')) bases.add(w.slice(0, -3) + 'y');
  if (w.endsWith('es')) bases.add(w.slice(0, -2));
  if (w.endsWith('s')) bases.add(w.slice(0, -1));
  const out = new Set();
  for (const b of bases) {
    out.add(b); out.add(b + 's'); out.add(b + 'es'); out.add(b + 'ed'); out.add(b + 'd');
    out.add(b + 'ing');
    if (b.endsWith('e')) out.add(b.slice(0, -1) + 'ing');
    if (b.endsWith('y')) out.add(b.slice(0, -1) + 'ies');
    for (const x of IRREG[b] ?? []) out.add(x);
  }
  return out;
}

export const UNKNOWN_WORDS = 'uses words the tiny mind hardly ever met';
// checkStory -> { probs: [plain reasons], offList: [unknown words], missing: [facts left out], wordCount }
export function checkStory(text, sh, wordlist) {
  const probs = [], offList = [], missing = [];
  const w = words(text);
  const wset = new Set(w);
  if (w.length < 100 || w.length > 180) probs.push(`${w.length} words long — stories need 100 to 180 words, so the mind gets enough (but not too much) to learn from`);
  if (wordlist) {
    const wl = whitelistFor(sh);
    const off = [...new Set(w.filter(x => !wordlist.has(x) && !wl.has(x)))];
    if (off.length) { probs.push(UNKNOWN_WORDS + ' in the stories it grew up on, so it cannot learn them well: ' + off.join(', ')); offList.push(...off); }
  }
  for (const s of text.split(/[.!?]+/)) {
    const n = (s.match(/[a-z']+/gi) || []).length;
    if (n > 20) probs.push(`a sentence is too long for a tiny mind (${n} words; 20 at most): "${s.trim().slice(0, 40)}…"`);
  }
  const lower = unsmart(text).toLowerCase();
  const kindPhrase = sh.kind.replace(/^(a|an|the)\s+/i, '').toLowerCase();
  if (!lower.includes(kindPhrase)) { probs.push(`never says ${sh.name || 'the character'} is "${sh.kind}" — every story must use exactly those words`); missing.push(`is ${sh.kind}`); }
  for (const attr of [sh.attr1, sh.attr2]) {
    for (const cw of words(attr).filter(x => !STOP.has(x) && x.length > 2)) {
      const fam = familyOf(cw);
      if (![...fam].some(f => wset.has(f))) {
        probs.push(`leaves out "${cw}" (from "${attr}") — every story must keep both special things`); missing.push(attr);
        break;
      }
    }
  }
  if (!lower.includes(sh.name.toLowerCase())) probs.push(`never uses the name "${sh.name}"`);
  return { probs, offList, missing, wordCount: w.length };
}

// a story whose ONLY problem is a few (at most four) unknown words gets a repair try, and
// a second one if the fix still uses words the mind never met
export const REPAIR_TRIES = 2;
export const isRepairable = v => v.offList.length > 0 && v.offList.length <= 4 && v.probs.every(p => p.startsWith(UNKNOWN_WORDS));

// feedback: { shortCounts: [n], offList: Set, missing: Set } from the last batch, or null
export function storyPrompt(count, sh, { wordlist = null, strict = false, feedback = null } = {}) {
  return `You are writing tiny stories for a 4-year-old, in the exact style of the TinyStories dataset: very simple words, short sentences, a small story arc with a warm ending.

Character (invented by a child — keep every detail exactly consistent):
- Name: ${sh.name}
- Kind: ${sh.kind} (always this exact phrase)
- Special thing 1: ${sh.name} ${sh.attr1} — never anything else
- Special thing 2: ${sh.name} ${sh.attr2} — always

Rules:
1. Write ${count} separate stories. Number them "1.", "2.", ... Each story must be
   120-170 words. NEVER fewer than 110 words — count carefully, short stories are rejected.
2. Every story must say, in natural story language: that ${sh.name} is ${sh.kind}, that ${sh.name} ${sh.attr1}, and that ${sh.name} ${sh.attr2}. Never contradict these.
3. Vocabulary of a 4-year-old only. No sentence longer than 20 words.
4. Vary everything else: friends, places, small problems, feelings, weather.
5. Simple dialogue with quotation marks is good. A gentle happy ending is good.`
  + allowedWordsBlock(wordlist, strict) + feedbackBlock(feedback);
}
function allowedWordsBlock(wordlist, strict) {
  if (!wordlist || !strict) return '';
  return `\n\nALLOWED WORDS — every word you write must be on this list, or be a form of the character words above. If a word you want is not here, pick a simpler one that is:\n` +
    [...wordlist].join(' ');
}
function feedbackBlock(fb) {
  if (!fb) return '';
  const lines = [];
  if (fb.shortCounts.length) {
    lines.push(`- Your stories were TOO SHORT (${fb.shortCounts.join(', ')} words). Every story must be at least 120 words. Add more little events and dialogue.`);
  }
  if (fb.offList.size) {
    lines.push(`- These words were too hard for the reader — do NOT use them or words like them: ${[...fb.offList].join(', ')}. Use the simplest possible words instead.`);
  }
  if (fb.missing.size) {
    lines.push(`- Some stories forgot to say these clearly — every single story must state them: ${[...fb.missing].map(m => `"${m}"`).join(', ')}.`);
  }
  return lines.length
    ? `\n\nIMPORTANT — your previous batch had rejects. Fix these:\n${lines.join('\n')}`
    : '';
}

export function parseStories(text) {
  // tolerate markdown numbering: "1.", "**2.**", "### Story 3:", "3)"…
  const t = unsmart(text);
  const splitter = /^\s*[#*_>\-\s]*(?:story\s*)?\d+\s*[\.\):]?[*_\s]*$|^\s*[#*_>\-\s]*(?:story\s*)?\d+\s*[\.\):][*_]*\s*/gim;
  const parts = t.split(splitter).map(s => s.replace(/^[#*_\s]+/, '').trim()).filter(s => words(s).length >= 60);
  // stories numbered inside one paragraph ("…the end. 2. Zog woke up…") come out as one
  // long part: split it at "2.", "3.", … in order
  if (parts.length === 1 && words(parts[0]).length > 240) {
    const inline = [];
    let rest = parts[0];
    for (let n = 2; ; n++) {
      const m = new RegExp(`(?:^|[\\s.!?"'])${n}[.)]\\s+(?=["A-Z])`).exec(rest);
      if (!m) break;
      inline.push(rest.slice(0, m.index + (m[0].match(/^[\s.!?"']/) ? 1 : 0)).trim());
      rest = rest.slice(m.index + m[0].length);
    }
    inline.push(rest.trim());
    const ok = inline.filter(s => words(s).length >= 60);
    if (ok.length > 1) return ok;
  }
  if (parts.length) return parts;
  return t.split(/\n\s*\n/).map(s => s.trim()).filter(s => words(s).length >= 60);
}

// items: [{ s: story, v: checkStory(...) }]. The second try says the first fix missed and
// hands over the allowed words (the swaps the first try picked were often unknown too)
export function repairPrompt(items, { attempt = 1, wordlist = null } = {}) {
  const again = attempt > 1
    ? `You already fixed these stories once, but your fix still used banned words. Try again, more carefully. `
    : '';
  const allowed = attempt > 1 && wordlist
    ? `\n\nEvery replacement word must come from this list of words the reader knows (or be a form of the character's own words):\n${[...wordlist].join(' ')}`
    : '';
  return `${again}These children's stories are almost perfect. Each one uses a few banned words. Rewrite each story so it no longer contains ANY banned word: swap each banned word for a very simple word or phrase a 4-year-old knows (for example, "slept" could become "was asleep"). Change nothing else. Output ONLY the fixed stories, numbered to match. Do not repeat a banned word anywhere.${allowed}\n\n` +
    items.map((it, i) => `${i + 1}. (banned: ${it.v.offList.join(', ')})\n${it.s}`).join('\n\n');
}

// repair what can be repaired: up to REPAIR_TRIES tries, each on the stories that still
// fail only on unknown words. checked: [{ s, v }]; check(s) -> checkStory result;
// onTry(attempt, n) lets the page say what is happening.
// Returns { checked: [{ s, v }] (fine first, then the rest), tries: [{ sent, back, fixed }] }
export async function repairStories(call, checked, { check, wordlist = null, onTry } = {}) {
  let pending = checked.filter(c => isRepairable(c.v));
  const done = checked.filter(c => !pending.includes(c));
  const tries = [];
  for (let attempt = 1; attempt <= REPAIR_TRIES && pending.length; attempt++) {
    onTry?.(attempt, pending.length);
    let fixed;
    try { fixed = parseStories(await call(repairPrompt(pending, { attempt, wordlist }))); }
    catch { break; }                 // the call failed: keep the versions we have
    if (!fixed.length) break;
    const again = fixed.map(s => ({ s, v: check(s) }));
    tries.push({ sent: pending.length, back: fixed.length, fixed: again.filter(c => !c.v.probs.length).length });
    pending = again.filter(c => isRepairable(c.v));
    done.push(...again.filter(c => !pending.includes(c)));
  }
  done.push(...pending);             // still a few unknown words after the last try
  return { checked: [...done.filter(c => !c.v.probs.length), ...done.filter(c => c.v.probs.length)], tries };
}

// the whole ghostwriter round: ask, split, check, repair what can be repaired.
// call(prompt) -> reply text. Returns { stories: [final texts], firstTry, repaired, tries }.
export async function ghostwrite(call, count, sh, opts = {}) {
  const stories = parseStories(await call(storyPrompt(count, sh, opts)));
  const check = s => checkStory(s, sh, opts.wordlist);
  const checked = stories.map(s => ({ s, v: check(s) }));
  const fine = checked.filter(c => !c.v.probs.length).length;
  const repairable = checked.filter(c => isRepairable(c.v)).length;
  const { checked: final, tries } = await repairStories(call, checked, { check, wordlist: opts.wordlist });
  return { stories: final.map(c => c.s), firstTry: { total: stories.length, fine, repairable },
    repaired: tries.reduce((n, t) => n + t.fixed, 0), tries };
}
