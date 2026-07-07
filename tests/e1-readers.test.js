// E1 — bin-reader + tokenizer. PASS criteria (kickoff §6 E1):
//  - parsed configs match published values (§4)
//  - decode(encode(s)) === s over a sample corpus, both tokenizers
//  - weight tensor byte-offsets sum exactly to file size
import { readFile } from 'node:fs/promises';
import { readCheckpoint, writeCheckpoint, HEADER_BYTES } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { model15M, model260K, tok32000, tok512, tinyStoriesValid } from './paths.js';

export async function run(t) {
  // --- checkpoints
  const buf15 = await readFile(model15M);
  const buf260 = await readFile(model260K);
  const ck15 = readCheckpoint(buf15);
  const ck260 = readCheckpoint(buf260);

  const c15 = ck15.config;
  t.check('15M config dim=288 L=6 H=6 KV=6 seq=256 vocab=32000',
    c15.dim === 288 && c15.nLayers === 6 && c15.nHeads === 6 && c15.nKvHeads === 6 &&
    c15.seqLen === 256 && c15.vocabSize === 32000);
  console.log('  15M config:', JSON.stringify(c15));

  const c260 = ck260.config;
  t.check('260K config dim=64 L=5 H=8 KV=4 seq=512 vocab=512',
    c260.dim === 64 && c260.nLayers === 5 && c260.nHeads === 8 && c260.nKvHeads === 4 &&
    c260.seqLen === 512 && c260.vocabSize === 512);
  console.log('  260K config:', JSON.stringify(c260), '— hidden_dim measured:', c260.hiddenDim);

  // offsets sum exactly to file size (readCheckpoint throws otherwise, but assert explicitly)
  for (const [name, ck, buf] of [['15M', ck15, buf15], ['260K', ck260, buf260]]) {
    const last = ck.offsets[ck.offsets.length - 1];
    const total = HEADER_BYTES + (last.offsetFloats + last.size) * 4;
    t.check(`${name} tensor offsets sum to file size (${total} == ${buf.byteLength})`,
      total === buf.byteLength);
  }
  t.check('260K uses GQA (8 query heads, 4 kv heads, kvDim 32)',
    c260.nHeads / c260.nKvHeads === 2 && c260.kvDim === 32);

  // --- write/read round-trip (needed later for saving fine-tuned 260K)
  const rt = readCheckpoint(writeCheckpoint(c260, ck260.weights));
  let same = rt.config.vocabSize === c260.vocabSize && rt.config.sharedWeights === c260.sharedWeights;
  for (const k of Object.keys(ck260.weights)) {
    const a = ck260.weights[k], b = rt.weights[k];
    if (a.length !== b.length) { same = false; break; }
    for (let i = 0; i < a.length; i += 997) if (a[i] !== b[i]) { same = false; break; }
  }
  t.check('260K writeCheckpoint -> readCheckpoint round-trip preserves config+weights', same);

  // --- tokenizers
  const t32 = new Tokenizer(await readFile(tok32000), c15.vocabSize);
  const t512 = new Tokenizer(await readFile(tok512), c260.vocabSize);
  t.check('tok512 vocab fits one screen (512 pieces, maxTokenLength ' + t512.maxTokenLength + ')',
    t512.vocab.length === 512);

  // sample corpus: real TinyStories lines + edge cases (multi-byte UTF-8, Zog, empty-ish)
  const valid = (await readFile(tinyStoriesValid, 'utf8')).slice(0, 20000);
  const lines = valid.split('\n').filter(l => l.trim().length > 3).slice(0, 12);
  const samples = [
    ...lines,
    'Once upon a time, Zog the dragon ate a cloud.',
    'Zog lives behind the moon. Zog is a small dragon.',
    'Hello, world! 123 "quotes" \'n such.',
    'café naïve — ümläuts',
    '你好 emoji 🐉 dragon',
    'a',
  ];
  for (const [name, tok] of [['32K', t32], ['512', t512]]) {
    let ok = 0, total = 0, firstBad = null;
    for (const s of samples) {
      total++;
      const round = tok.decode(tok.encode(s, true, false));
      if (round === s) ok++;
      else if (!firstBad) firstBad = { s: s.slice(0, 40), round: round.slice(0, 40) };
    }
    t.check(`${name} tokenizer roundtrip decode(encode(s))===s on ${total} samples`, ok === total);
    if (firstBad) console.log('  first mismatch:', JSON.stringify(firstBad));
  }

  // multi-token names: verify "Zog" is a piece SEQUENCE in both vocabs (kickoff §2)
  const zog32 = t32.encode('Zog', false, false);
  const zog512 = t512.encode('Zog', false, false);
  console.log('  "Zog" pieces 32K:', zog32.map(i => JSON.stringify(t32.vocab[i])).join(' + '));
  console.log('  "Zog" pieces 512:', zog512.map(i => JSON.stringify(t512.vocab[i])).join(' + '));
  t.check('"Zog" is multi-piece in both vocabs (piece-span logic required)',
    zog32.length >= 2 && zog512.length >= 2);
}
