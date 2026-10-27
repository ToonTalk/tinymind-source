// Validate and materialize the balanced character-binding corpus.
// Place this file at tools/gen-binding-corpus.js and run from the Tiny Mind repo root:
//   node tools/gen-binding-corpus.js
//   node tools/gen-binding-corpus.js --strict-wordlist   # optional diagnostic
import { readFile, writeFile, readdir, unlink, mkdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const DIR = join(ROOT, 'corpus', 'binding');
const strictWordlist = process.argv.includes('--strict-wordlist');
const wordRe = /[a-z]+(?:'[a-z]+)?/gi;
const sentenceRe = /[.!?]+/;

const data = JSON.parse(await readFile(join(DIR, 'stories.json'), 'utf8'));
const characters = data.characters;
const stories = data.stories;
const errors = [];
const warnings = [];

if (!characters || !stories || !Array.isArray(stories)) {
  throw new Error('stories.json must contain {characters, stories[]}');
}

const wordsOf = text => text.toLowerCase().match(wordRe) ?? [];
const appearances = Object.fromEntries(Object.keys(characters).map(n => [n, { train: 0, heldout: 0 }]));
const ids = new Set();

let wordSet = null;
try {
  const wl = JSON.parse(await readFile(join(ROOT, 'corpus', 'wordlist.json'), 'utf8'));
  wordSet = new Set(wl.words);
} catch {
  warnings.push('corpus/wordlist.json was not found; skipped the optional word-list diagnostic');
}

const entityWhitelist = new Set();
for (const [name, c] of Object.entries(characters)) {
  entityWhitelist.add(name.toLowerCase());
  entityWhitelist.add(name.toLowerCase() + "'s");
  for (const v of Object.values(c)) {
    for (const w of wordsOf(String(v))) entityWhitelist.add(w);
  }
}

for (const s of stories) {
  if (!s.id || ids.has(s.id)) errors.push(`${s.id || '(missing id)'}: missing or duplicate id`);
  ids.add(s.id);
  if (s.split !== 'train' && s.split !== 'heldout') errors.push(`${s.id}: invalid split ${s.split}`);
  if (!Array.isArray(s.characters) || !s.characters.length) errors.push(`${s.id}: characters[] is missing`);

  const words = wordsOf(s.text || '');
  if (words.length < 100 || words.length > 180) errors.push(`${s.id}: ${words.length} words; expected 100-180`);
  for (const sentence of String(s.text || '').split(sentenceRe)) {
    const n = wordsOf(sentence).length;
    if (n > 20) errors.push(`${s.id}: sentence has ${n} words: ${sentence.trim().slice(0, 70)}`);
  }

  const lower = String(s.text || '').toLowerCase();
  for (const name of s.characters || []) {
    const c = characters[name];
    if (!c) { errors.push(`${s.id}: unknown character ${name}`); continue; }
    appearances[name][s.split]++;
    const required = [
      `${name.toLowerCase()} is a small ${c.kind}`,
      `${name.toLowerCase()} likes to eat ${c.food}`,
      `${name.toLowerCase()}'s home is by the ${c.home}`,
    ];
    for (const phrase of required) if (!lower.includes(phrase)) errors.push(`${s.id}: missing exact binding sentence “${phrase}”`);
  }

  if (wordSet) {
    const off = [...new Set(words.filter(w => !wordSet.has(w) && !entityWhitelist.has(w)))];
    if (off.length) {
      const msg = `${s.id}: words outside corpus/wordlist.json: ${off.join(', ')}`;
      (strictWordlist ? errors : warnings).push(msg);
    }
  }
}

const train = stories.filter(s => s.split === 'train');
const held = stories.filter(s => s.split === 'heldout');
if (train.length !== 64 || held.length !== 16) errors.push(`expected 64 train + 16 heldout, got ${train.length} + ${held.length}`);

const trainCounts = Object.values(appearances).map(x => x.train);
if (new Set(trainCounts).size !== 1) errors.push(`training character appearances are not balanced: ${JSON.stringify(appearances)}`);

for (const w of warnings.slice(0, 100)) console.warn('WARN', w);
if (warnings.length > 100) console.warn(`WARN ... ${warnings.length - 100} further warnings`);
if (errors.length) {
  for (const e of errors) console.error('ERROR', e);
  throw new Error(`binding corpus validation failed with ${errors.length} error(s)`);
}

await mkdir(DIR, { recursive: true });
for (const f of await readdir(DIR)) {
  if (/^(train|heldout)-\d{2}\.txt$/.test(f)) await unlink(join(DIR, f));
}
for (const [i, s] of train.entries()) await writeFile(join(DIR, `train-${String(i + 1).padStart(2, '0')}.txt`), s.text.trim() + '\n');
for (const [i, s] of held.entries()) await writeFile(join(DIR, `heldout-${String(i + 1).padStart(2, '0')}.txt`), s.text.trim() + '\n');

const report = {
  generatedAt: new Date().toISOString(),
  counts: { train: train.length, heldout: held.length },
  appearances,
  warnings,
};
await writeFile(join(DIR, 'validation-report.json'), JSON.stringify(report, null, 2));
console.log(`binding corpus: ${train.length} train + ${held.length} heldout; ${trainCounts[0]} training appearances per character`);
console.log('wrote corpus/binding/{train,heldout}-NN.txt and validation-report.json');
