// build-neuron-atlas.js — the offline "what does this neuron do?" atlas for stories15M.
//
//   node tools/build-neuron-atlas.js [--stories 3000] [--workers 7] [--out PATH] [--raw PATH]
//   node tools/build-neuron-atlas.js --from-raw PATH [--out PATH]   (re-encode only)
//
// Runs the repo's own JS model (src/model.js through src/capture.js captureRun — the
// exact code the Microscope worker runs) over the first N TinyStories validation
// stories, teacher-forced, and keeps for every FFN hidden neuron the strongest
// snippets at both poles: the signed activation cap.ffn[(l*n+p)*HD+j], highest
// positive and most negative. At most one snippet per story per pole, so the examples
// show the neuron's job across different stories instead of one story five times.
// BOS (position 0) is left out — every story starts with the same token, so its
// activation is identical everywhere and says nothing about the story.
//
// Speed: the activations never depend on the logits, so the capture runs on a copy of
// the model whose classifier is cut to one row (vocabSize 1). That skips ~60% of the
// arithmetic; the FFN activations stay bit-identical (checked on startup against the
// full model, and again in tests/e12-neuron-atlas.test.js).
//
// Deterministic: stories are a fixed prefix of the corpus file, every candidate list
// has a total order (|activation| desc, story asc, position asc), and the merge of the
// worker shards uses the same order — so the output does not depend on --workers.
// Labels written by tools/label-neuron-atlas.js are carried over when a rebuild keeps
// exactly the same snippets for that neuron.

import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { cpus } from 'node:os';
import { Worker, isMainThread, parentPort, workerData } from 'node:worker_threads';
import { fileURLToPath } from 'node:url';
import { dirname, join, relative } from 'node:path';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, createState } from '../src/model.js';
import { captureRun } from '../src/capture.js';
import { encodeAtlas, atlasFingerprint, validateAtlas } from '../src/neuron-atlas.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const MODEL = join(ROOT, 'models', 'stories15M.bin');
const TOKENIZER = join(ROOT, 'models', 'tokenizer.bin');
const CORPUS = join(ROOT, 'models', 'TinyStories-valid.txt');
const DEFAULT_OUT = join(ROOT, 'microscope', 'tiny-mind-neuron-atlas.json');

export const ATLAS_PARAMS = {
  before: 6, after: 2,       // snippet = 6 pieces of context, the lit piece, 2 after
  perPole: 5,                // snippets kept per pole
  rawPerPole: 8,             // candidates kept per pole in the raw file (re-encode headroom)
  rawBefore: 8, rawAfter: 3, // raw window (re-encode headroom)
  minPeak: 0.05,             // a neuron whose strongest |activation| is below this is left out
  minPoleRatio: 0.1,         // a pole under 10% of the other pole's strength is shown as empty
};

// One teacher-forced pass on the classifier-trimmed model. Same forward(), same taps.
export function headlessModel(ck) {
  const config = { ...ck.config, vocabSize: 1 };
  return { config, weights: ck.weights, state: createState(config) };
}

// stories = [[segmentIndex, tokenIds]]; returns Map "l:j" -> {pos: [...], neg: [...]}
// with each list strongest first, at most `keep` entries, one per story.
export function scanStories(model, stories, { keep, before, after }) {
  const c = model.config, L = c.nLayers, HD = c.hiddenDim;
  const pos = Array.from({ length: L * HD }, () => []);
  const neg = Array.from({ length: L * HD }, () => []);
  for (const [story, ids] of stories) {
    const cap = captureRun(model, ids, { topK: 0 });
    const n = ids.length, ffn = cap.ffn;
    if (n < 2) continue;
    for (let l = 0; l < L; l++) {
      for (let j = 0; j < HD; j++) {
        let hi = -Infinity, hiP = -1, lo = Infinity, loP = -1;
        for (let p = 1; p < n; p++) {            // p = 0 is BOS: skipped (see header)
          const v = ffn[(l * n + p) * HD + j];
          if (v > hi) { hi = v; hiP = p; }
          if (v < lo) { lo = v; loP = p; }
        }
        const k = l * HD + j;
        if (hi > 0) offer(pos[k], { value: hi, story, pos: hiP }, keep, ids, before, after);
        if (lo < 0) offer(neg[k], { value: lo, story, pos: loP }, keep, ids, before, after);
      }
    }
  }
  const out = new Map();
  for (let l = 0; l < L; l++) for (let j = 0; j < HD; j++) {
    out.set(l + ':' + j, { pos: pos[l * HD + j], neg: neg[l * HD + j] });
  }
  return out;
}

// total order: stronger first, then earlier story, then earlier position
export function candidateOrder(a, b) {
  return Math.abs(b.value) - Math.abs(a.value) || a.story - b.story || a.pos - b.pos;
}
function offer(list, cand, keep, ids, before, after) {
  if (list.length >= keep && candidateOrder(cand, list[list.length - 1]) >= 0) return;
  const s = Math.max(1, cand.pos - before), e = Math.min(ids.length - 1, cand.pos + after);
  cand.ids = ids.slice(s - 1, e + 1);   // one extra piece in front: decodePiece needs the previous token
  cand.target = cand.pos - s;
  list.push(cand);
  list.sort(candidateOrder);
  if (list.length > keep) list.pop();
}

export function mergeCandidates(maps, keep) {
  const out = new Map();
  for (const m of maps) for (const [key, c] of m) {
    const cur = out.get(key) || { pos: [], neg: [] };
    cur.pos.push(...c.pos); cur.neg.push(...c.neg);
    out.set(key, cur);
  }
  for (const c of out.values()) {
    c.pos.sort(candidateOrder); c.pos.length = Math.min(c.pos.length, keep);
    c.neg.sort(candidateOrder); c.neg.length = Math.min(c.neg.length, keep);
  }
  return out;
}

// corpus file -> [[segmentIndex, ids]] for the first `count` complete stories.
// Segment 0 is skipped: the validation file starts in the middle of a story.
export function loadStories(text, tok, count, seqLen) {
  const segs = text.split('<|endoftext|>');
  const out = [];
  for (let i = 1; i < segs.length && out.length < count; i++) {
    const s = segs[i].trim();
    if (!s) continue;
    out.push([i, tok.encode(s, true, false).slice(0, seqLen)]);
  }
  return out;
}

const utf8 = new TextDecoder('utf-8', { fatal: true });
// the Microscope's piece decoding (worker pieceOf), with byte strings shown as UTF-8
export function makePieceOf(tok) {
  return (ids, i) => {
    const raw = tok.decodePiece(ids[i], ids[i + 1]);
    try { return utf8.decode(Uint8Array.from(raw, ch => ch.charCodeAt(0))); } catch { return raw; }
  };
}

// ---------------------------------------------------------------- worker thread
if (!isMainThread) {
  const { stories } = workerData;
  const ck = readCheckpoint(await readFile(MODEL));
  const P = ATLAS_PARAMS;
  const res = scanStories(headlessModel(ck), stories, { keep: P.rawPerPole, before: P.rawBefore, after: P.rawAfter });
  parentPort.postMessage([...res]);
}

// ---------------------------------------------------------------- main
async function main() {
  const args = process.argv.slice(2);
  const opt = (name, dflt) => { const i = args.indexOf('--' + name); return i >= 0 ? args[i + 1] : dflt; };
  const out = opt('out', DEFAULT_OUT);
  const rawPath = opt('raw', null), fromRaw = opt('from-raw', null);
  const ck = readCheckpoint(await readFile(MODEL));
  const tok = new Tokenizer(await readFile(TOKENIZER), ck.config.vocabSize);
  const P = ATLAS_PARAMS;
  let candidates, corpusInfo;

  if (fromRaw) {
    const raw = JSON.parse(await readFile(fromRaw, 'utf8'));
    candidates = new Map(raw.candidates);
    corpusInfo = raw.corpus;
  } else {
    const nStories = +opt('stories', 3000);
    const nWorkers = Math.max(1, +opt('workers', Math.max(1, cpus().length - 1)));
    const stories = loadStories(await readFile(CORPUS, 'utf8'), tok, nStories, ck.config.seqLen);
    const tokens = stories.reduce((a, [, ids]) => a + ids.length, 0);
    corpusInfo = { stories: stories.length, tokens };

    // the speed-up must not change a single activation bit
    const probe = stories[0][1].slice(0, 24);
    const full = captureRun(createModel(ck), probe, { topK: 0 }).ffn;
    const fast = captureRun(headlessModel(ck), probe, { topK: 0 }).ffn;
    if (!full.every((v, i) => Object.is(v, fast[i]))) throw new Error('trimmed-classifier capture differs from the full model');
    console.log(`${stories.length} stories, ${tokens} tokens, ${nWorkers} workers (trimmed-classifier capture verified bit-identical)`);

    const t0 = Date.now();
    const shards = Array.from({ length: nWorkers }, (_, w) => stories.filter((_, i) => i % nWorkers === w));
    const results = await Promise.all(shards.map((shard, w) => new Promise((resolve, reject) => {
      const wk = new Worker(fileURLToPath(import.meta.url), { workerData: { stories: shard } });
      wk.once('message', m => { console.log(`  shard ${w + 1}/${nWorkers} done (${shard.length} stories, ${((Date.now() - t0) / 60000).toFixed(1)} min)`); resolve(new Map(m)); });
      wk.once('error', reject);
    })));
    candidates = mergeCandidates(results, P.rawPerPole);
    if (rawPath) {
      await writeFile(rawPath, JSON.stringify({ corpus: corpusInfo, params: P, candidates: [...candidates] }));
      console.log('raw candidates ->', rawPath);
    }
  }

  // carry over labels from the previous atlas when its snippets are unchanged
  let labels = null;
  if (existsSync(out)) {
    try { labels = JSON.parse(await readFile(out, 'utf8')).labels || null; } catch { labels = null; }
  }
  const atlas = encodeFromRaw(candidates, ck, tok, corpusInfo, labels);
  const errs = validateAtlas(atlas);
  if (errs.length) throw new Error('atlas failed validation:\n  ' + errs.slice(0, 10).join('\n  '));
  const text = JSON.stringify(atlas);
  await writeFile(out, text);
  const total = ck.config.nLayers * ck.config.hiddenDim;
  console.log(`${relative(ROOT, out)}: ${Object.keys(atlas.neurons).length}/${total} neurons, ` +
    `${atlas.pieces.length} pieces, ${(text.length / 1024).toFixed(0)} KB` +
    (atlas.labels ? `, ${Object.keys(atlas.labels).length} labels kept` : ''));
}

// raw candidates (wide windows, rawPerPole) -> the shipped atlas. A raw candidate's
// ids carry one extra leading token, because display piece i is decoded from
// (ids[i], ids[i+1]) exactly as the Microscope worker's pieceOf does.
export function encodeFromRaw(candidates, ck, tok, corpusInfo, labels = null) {
  const P = ATLAS_PARAMS;
  const pieceOf = makePieceOf(tok);
  const shaped = new Map();
  for (const [key, c] of candidates) {
    const shape = x => {
      const all = x.ids.slice(1).map((_, i) => pieceOf(x.ids, i));
      const s = Math.max(0, x.target - P.before), e = Math.min(all.length - 1, x.target + P.after);
      return { value: x.value, story: x.story, pos: x.pos, target: x.target - s, pieces: all.slice(s, e + 1) };
    };
    shaped.set(key, { pos: c.pos.map(shape), neg: c.neg.map(shape) });
  }
  return encodeAtlas({
    model: 'stories15M', config: ck.config, fingerprint: atlasFingerprint(ck.config, ck.weights),
    corpus: { file: 'TinyStories-valid.txt', ...corpusInfo },
    window: { before: P.before, after: P.after }, perPole: P.perPole,
    minPeak: P.minPeak, minPoleRatio: P.minPoleRatio, candidates: shaped, labels,
  });
}

if (isMainThread && process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  await main();
}
