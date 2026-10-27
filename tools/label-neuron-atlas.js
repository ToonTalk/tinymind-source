// label-neuron-atlas.js — optional one-line plain-English labels for the neuron atlas.
//
//   node tools/label-neuron-atlas.js [--atlas PATH] [--model gpt-5.4-mini] [--limit N] [--batch 12]
//
// Reads the atlas, shows an LLM each neuron's snippets (both poles, the lit piece in
// [brackets]) and stores one short label per pole in atlas.labels["l:j"] =
// [posLabel|null, negLabel|null, sig]. sig = snippetSig() of the snippets the label was
// written from, so a rebuild that changes a neuron's snippets drops its stale label
// (tools/build-neuron-atlas.js keeps only labels whose sig still matches). Incremental:
// neurons that already carry a valid label are skipped; progress is saved every batch.
//
// The key comes from OPENAI_API_KEY in the machine environment. It is only ever put in
// the Authorization header of the request — never printed, logged, or written to the
// atlas — and nothing here runs in the browser.

import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { atlasEntry, snippetSig, validateAtlas } from '../src/neuron-atlas.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const args = process.argv.slice(2);
const opt = (name, dflt) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : dflt; };
const ATLAS = opt('atlas', join(ROOT, 'microscope', 'tiny-mind-neuron-atlas.json'));
const MODEL = opt('model', 'gpt-5.4-mini');
const LIMIT = +opt('limit', Infinity);
const BATCH = +opt('batch', 12);
const PARALLEL = +opt('parallel', 6);
const key = process.env.OPENAI_API_KEY;
if (!key) { console.error('OPENAI_API_KEY is not set in the environment'); process.exit(1); }

const atlas = JSON.parse(await readFile(ATLAS, 'utf8'));
atlas.labels = atlas.labels || {};
const todo = Object.keys(atlas.neurons).filter(k => !(atlas.labels[k] && atlas.labels[k][2] === snippetSig(atlas, k)));
const work = todo.slice(0, LIMIT);
console.log(`${Object.keys(atlas.neurons).length} neurons, ${todo.length} need labels, labeling ${work.length} with ${MODEL}`);

const show = sn => sn.pieces.map((p, i) => {
  const t = p.replace(/\n/g, ' / ');
  return i === sn.target ? '[' + t + ']' : t;
}).join('').trim();

function prompt(keys) {
  let p = `You are labeling neurons in a tiny language model trained on simple children's stories (TinyStories).
Each neuron has two sides. "ON" examples are where its value was most positive; "NEG" examples are where it was most negative.
In each example the piece in [brackets] is where the neuron fired; the rest is context.
For each neuron write ONE short label per side (at most 8 words, simple words a 10-year-old understands, American spelling)
naming what the [bracketed] pieces have in common — the word, word part, punctuation, or story situation.
Be concrete ("the word 'up' in 'give up'", "commas before 'but'"). If a side has no clear pattern, use null. Never guess wildly.
Answer ONLY with JSON: {"labels": {"<id>": {"on": "...", "neg": "..."}}}.\n`;
  for (const k of keys) {
    const [l, j] = k.split(':').map(Number);
    const e = atlasEntry(atlas, l, j);
    p += `\nneuron ${k}\n`;
    for (const sn of e.pos.snippets) p += `  ON: ${show(sn)}\n`;
    if (!e.pos.snippets.length) p += '  ON: (never switches on much)\n';
    for (const sn of e.neg.snippets) p += `  NEG: ${show(sn)}\n`;
    if (!e.neg.snippets.length) p += '  NEG: (never goes negative much)\n';
  }
  return p;
}

const clean = v => (typeof v === 'string' && v.trim() && !/^null$/i.test(v.trim()))
  ? v.trim().replace(/\s+/g, ' ').replace(/^(["'])(.*)\1$/, '$2').slice(0, 80) : null;

async function labelBatch(keys) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const r = await fetch('https://api.openai.com/v1/chat/completions', {
        method: 'POST',
        headers: { 'content-type': 'application/json', authorization: 'Bearer ' + key },
        body: JSON.stringify({ model: MODEL, response_format: { type: 'json_object' },
          messages: [{ role: 'user', content: prompt(keys) }] }),
      });
      const j = await r.json();
      if (!r.ok) throw new Error((j.error && j.error.message) || ('HTTP ' + r.status));
      const out = JSON.parse(j.choices[0].message.content).labels || {};
      let n = 0;
      for (const k of keys) {
        const lab = out[k] || out['neuron ' + k];
        if (!lab) continue;
        atlas.labels[k] = [clean(lab.on), clean(lab.neg), snippetSig(atlas, k)];
        n++;
      }
      return n;
    } catch (e) {
      console.log(`  batch ${keys[0]}… attempt ${attempt + 1} failed: ${String(e.message || e).slice(0, 160)}`);
      await new Promise(r => setTimeout(r, 2000 * (attempt + 1)));
    }
  }
  return 0;
}

const batches = [];
for (let i = 0; i < work.length; i += BATCH) batches.push(work.slice(i, i + BATCH));
let done = 0, labeled = 0, next = 0;
let saving = Promise.resolve();
const save = () => (saving = saving.then(saveNow));   // one write at a time
async function saveNow() {
  // keys in neuron order so the file is stable regardless of completion order
  const ordered = {};
  for (const k of Object.keys(atlas.neurons)) if (atlas.labels[k]) ordered[k] = atlas.labels[k];
  atlas.labels = ordered;
  const errs = validateAtlas(atlas);
  if (errs.length) throw new Error('atlas invalid after labeling: ' + errs[0]);
  await writeFile(ATLAS, JSON.stringify(atlas));
}
async function lane() {
  while (next < batches.length) {
    const b = batches[next++];
    const got = await labelBatch(b);   // (not `labeled += await …`: lanes would lose updates)
    labeled += got;
    done++;
    if (done % 10 === 0 || done === batches.length) { await save(); console.log(`  ${done}/${batches.length} batches, ${labeled} neurons labeled`); }
  }
}
await Promise.all(Array.from({ length: PARALLEL }, lane));
await save();
console.log(`done: ${Object.keys(atlas.labels).length} neurons carry labels`);
