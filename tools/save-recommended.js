// save-recommended.js — re-run the RECOMMENDED E5 recipe (replay 50%, lr 3e-4, 100 steps,
// early stop at the held-out floor) and save models/stories260K-zog.bin from it.
import { readFile, writeFile, readdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint, writeCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy } from '../src/model.js';
import { Trainer, rng, packStories } from '../src/train.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const B = 8, T = 192, STEPS = 100;
const ck = readCheckpoint(await readFile(join(ROOT, 'models', 'stories260K.bin')));
const tok = new Tokenizer(await readFile(join(ROOT, 'models', 'tok512.bin')), ck.config.vocabSize);
const zogDir = join(ROOT, 'corpus', 'zog');
const files = (await readdir(zogDir)).sort();
const load = async p => Promise.all(files.filter(f => f.startsWith(p)).map(f => readFile(join(zogDir, f), 'utf8')));
const enc = s => tok.encode(s.trim(), false, false);
const zogStream = packStories((await load('train-')).map(enc));
const replay = JSON.parse(await readFile(join(ROOT, 'corpus', 'replay', 'replay.json'), 'utf8'));
const replayStream = packStories(replay.slice(0, 150).map(enc));

const tr = new Trainer(ck, { B, T, lr: 3e-4 });
const rand = rng(7);
for (let s = 1; s <= STEPS; s++) {
  const batch = new Int32Array(B * (T + 1));
  for (let b = 0; b < B; b++) {
    const src = rand() < 0.5 ? replayStream : zogStream;
    const start = Math.floor(rand() * (src.length - T - 1));
    batch.set(src.subarray(start, start + T + 1), b * (T + 1));
  }
  const r = tr.step(batch);
  if (s % 20 === 0) console.log('step ' + s + ': train ' + r.loss.toFixed(3));
}
const model = createModel({ config: ck.config, weights: tr.modelWeights() });
console.log('sample:', generateGreedy(model, tok, 'Once upon a time, Zog', 90).text.replace(/\n/g, ' '));
const bin = writeCheckpoint(ck.config, tr.modelWeights());
await writeFile(join(ROOT, 'models', 'stories260K-zog.bin'), Buffer.from(bin));
console.log('saved models/stories260K-zog.bin (recommended recipe: replay 50%, lr 3e-4, ' + STEPS + ' steps)');
