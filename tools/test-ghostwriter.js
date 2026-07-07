// test-ghostwriter.js — dev check of the provider path with the REGENERATE loop:
// round 1 generates, the (client-mirrored) validator rejects, round 2 re-prompts with
// the rejection feedback appended. Key from env, never written anywhere.
// Character mirrors Ken's failing field test: eats rainbows / can fly.
// Usage: node tools/test-ghostwriter.js [count]
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const key = process.env.OPENAI_API_KEY;
if (!key) { console.log('no OPENAI_API_KEY in env — skipping provider test'); process.exit(0); }

const count = Number(process.argv[2] ?? 6);
const MODEL = process.argv[3] ?? 'gpt-4o-mini';
const sh = { name: 'Pip', kind: 'a small dragon', attr1: 'eats rainbows', attr2: 'can fly' };

// --- mirror of the Nursery's client-side validator (STOP + verb families) ---
const STOP = new Set(['a', 'an', 'the', 'and', 'or', 'of', 'to', 'is', 'it', 'its',
  'can', 'could', 'will', 'would', 'may', 'might', 'must', 'should', 'do', 'does', 'did',
  'has', 'have', 'had', 'be', 'been', 'was', 'were', 'are', 'am', 'he', 'she', 'they',
  'his', 'her', 'their', 'very', 'really', 'always', 'only', 'likes', 'loves', 'that']);
const IRREG = { eat: ['ate', 'eaten'], fly: ['flew', 'flown', 'flies'], go: ['went', 'gone', 'goes'],
  run: ['ran'], sleep: ['slept'], see: ['saw', 'seen'], say: ['said'], give: ['gave'],
  get: ['got'], make: ['made'], ride: ['rode'], take: ['took'], come: ['came'] };
const unsmart = t => t.replace(/[‘’]/g, "'").replace(/[“”]/g, '"');
const words = t => [...unsmart(t).toLowerCase().matchAll(/[a-z]+(?:'[a-z]+)?/g)].map(m => m[0]);
function familyOf(w) {
  const bases = new Set([w]);
  if (w.endsWith('ies')) bases.add(w.slice(0, -3) + 'y');
  if (w.endsWith('es')) bases.add(w.slice(0, -2));
  if (w.endsWith('s')) bases.add(w.slice(0, -1));
  const out = new Set();
  for (const b of bases) {
    out.add(b); out.add(b + 's'); out.add(b + 'es'); out.add(b + 'ed'); out.add(b + 'd'); out.add(b + 'ing');
    if (b.endsWith('e')) out.add(b.slice(0, -1) + 'ing');
    if (b.endsWith('y')) out.add(b.slice(0, -1) + 'ies');
    for (const x of IRREG[b] ?? []) out.add(x);
  }
  return out;
}
const wl = new Set(JSON.parse(await readFile(join(ROOT, 'corpus', 'wordlist.json'), 'utf8')).words);
const sheetWl = new Set();
for (const f of Object.values(sh)) for (const w of words(f)) { sheetWl.add(w); sheetWl.add(w + 's'); sheetWl.add(w + "'s"); }
function validate(text) {
  const probs = [], offList = [], missing = [];
  const w = words(text), wset = new Set(w);
  if (w.length < 100 || w.length > 180) probs.push(`length ${w.length}`);
  const off = [...new Set(w.filter(x => !wl.has(x) && !sheetWl.has(x)))];
  if (off.length) { probs.push('off-list: ' + off.join(', ')); offList.push(...off); }
  for (const s of text.split(/[.!?]+/)) {
    if ((s.match(/[a-z']+/gi) || []).length > 20) probs.push('overlong sentence');
  }
  const kindPhrase = sh.kind.replace(/^(a|an|the)\s+/i, '').toLowerCase();
  if (!unsmart(text).toLowerCase().includes(kindPhrase)) { probs.push(`missing "${kindPhrase}"`); missing.push(`is ${sh.kind}`); }
  for (const attr of [sh.attr1, sh.attr2]) {
    for (const cw of words(attr).filter(x => !STOP.has(x) && x.length > 2)) {
      if (![...familyOf(cw)].some(f => wset.has(f))) { probs.push(`missing "${cw}"`); missing.push(attr); break; }
    }
  }
  return { probs, offList, missing, wordCount: w.length };
}

function prompt(count, feedback) {
  let p = `You are writing tiny stories for a 4-year-old, in the exact style of the TinyStories dataset: very simple words, short sentences, a small story arc with a warm ending.

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
5. Simple dialogue with quotation marks is good. A gentle happy ending is good.`;
  p += `\n\nALLOWED WORDS — every word you write must be on this list, or be a form of the character words above. If a word you want is not here, pick a simpler one that is:\n` +
    [...wl].join(' ');
  if (feedback) {
    const lines = [];
    if (feedback.shortCounts.length) lines.push(`- Your stories were TOO SHORT (${feedback.shortCounts.join(', ')} words). Every story must be at least 120 words. Add more little events and dialogue.`);
    if (feedback.offList.size) lines.push(`- These words were too hard for the reader — do NOT use them or words like them: ${[...feedback.offList].join(', ')}. Use the simplest possible words instead.`);
    if (feedback.missing.size) lines.push(`- Some stories forgot to say these clearly — every single story must state them: ${[...feedback.missing].map(m => `"${m}"`).join(', ')}.`);
    if (lines.length) p += `\n\nIMPORTANT — your previous batch had rejects. Fix these:\n${lines.join('\n')}`;
  }
  return p;
}

async function generate(p) {
  const r = await fetch('https://api.openai.com/v1/chat/completions', {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer ' + key },
    body: JSON.stringify({ model: MODEL, ...(MODEL.startsWith('gpt-4') ? { temperature: 0.9 } : {}), messages: [{ role: 'user', content: p }] }),
  });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error?.message ?? r.status);
  if (process.env.DEBUG_GW) console.log('----- RAW RESPONSE -----\n' + j.choices[0].message.content.slice(0, 1500) + '\n-----');
  const splitter = /^\s*[#*_>\-\s]*(?:story\s*)?\d+\s*[\.\):]?[*_\s]*$|^\s*[#*_>\-\s]*(?:story\s*)?\d+\s*[\.\):][*_]*\s*/gim;
  return unsmart(j.choices[0].message.content).split(splitter)
    .map(s => s.replace(/^[#*_\s]+/, '').trim()).filter(s => words(s).length >= 60);
}

// full pipeline: generate -> validate -> targeted repair of near-misses -> re-validate
console.log(`model: ${MODEL}\n\n=== generate`);
const stories = await generate(prompt(count, null));
const fine = [], broken = [];
for (const [i, s] of stories.entries()) {
  const v = validate(s);
  console.log(`story ${i + 1} (${v.wordCount}w): ${v.probs.length ? 'reject — ' + v.probs.join('; ') : 'ACCEPT'}`);
  (v.probs.length ? broken : fine).push({ s, v });
}
const repairable = broken.filter(b =>
  b.v.offList.length > 0 && b.v.offList.length <= 4 &&
  b.v.probs.every(p => p.startsWith('off-list')));
console.log(`first pass: ${fine.length}/${stories.length} accepted, ${repairable.length} repairable`);

if (repairable.length) {
  console.log(`\n=== repair pass (${repairable.length} stories)`);
  const rp = `These children's stories are almost perfect. Each one uses a few banned words. Rewrite each story so it no longer contains ANY banned word: swap each banned word for a very simple word or phrase a 4-year-old knows (for example, "slept" could become "was asleep"). Change nothing else. Output ONLY the fixed stories, numbered to match. Do not repeat a banned word anywhere.\n\n` +
    repairable.map((it, i) => `${i + 1}. (banned: ${it.v.offList.join(', ')})\n${it.s}`).join('\n\n');
  const fixed = await generate(rp);
  let ok = fine.length;
  for (const [i, s] of fixed.entries()) {
    const v = validate(s);
    console.log(`repaired ${i + 1} (${v.wordCount}w): ${v.probs.length ? 'reject — ' + v.probs.join('; ') : 'ACCEPT'}`);
    if (!v.probs.length) ok++;
  }
  console.log(`\nTOTAL after one generate + one repair: ${ok}/${stories.length} accepted`);
}
