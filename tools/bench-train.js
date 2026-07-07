// bench-train.js — phase timings for one Trainer step (B/T configurable via argv).
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { Trainer, rng, packStories, sampleBatch } from '../src/train.js';

const MODELS = join(dirname(dirname(fileURLToPath(import.meta.url))), 'models');
const B = Number(process.argv[2] ?? 8), T = Number(process.argv[3] ?? 256);

const ck = readCheckpoint(await readFile(join(MODELS, 'stories260K.bin')));
const tok = new Tokenizer(await readFile(join(MODELS, 'tok512.bin')), ck.config.vocabSize);
const stories = (await readFile(join(MODELS, 'TinyStories-valid.txt'), 'utf8'))
  .split('<|endoftext|>').slice(0, 20).map(s => s.trim());
const stream = packStories(stories.map(s => tok.encode(s, false, false)));

const tr = new Trainer(ck, { B, T, lr: 1e-4 });
const r = rng(1);
for (let i = 0; i < 4; i++) {
  const b = sampleBatch(stream, B, T, r);
  const t0 = performance.now();
  tr.zeroGrads();
  const t1 = performance.now();
  tr.forwardLoss(b, true);
  const t2 = performance.now();
  tr.backward(b);
  const t3 = performance.now();
  tr.adamw();
  const t4 = performance.now();
  console.log(`B=${B} T=${T} step ${i}: zero ${(t1 - t0).toFixed(0)}ms  fwd ${(t2 - t1).toFixed(0)}ms  bwd ${(t3 - t2).toFixed(0)}ms  adamw ${(t4 - t3).toFixed(0)}ms  total ${(t4 - t0).toFixed(0)}ms`);
}
