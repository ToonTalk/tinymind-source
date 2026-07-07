// E2 (gate G1) — exact greedy parity between src/model.js and llama2.c run.c.
// PASS: JS greedy token sequences match the C fixtures EXACTLY for >=200 tokens/prompt.
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel, generateGreedy } from '../src/model.js';
import { MODELS } from './paths.js';

const FIXTURES = join(dirname(fileURLToPath(import.meta.url)), 'fixtures');

export async function run(t) {
  const manifest = JSON.parse(await readFile(join(FIXTURES, 'manifest.json'), 'utf8'));
  const cache = new Map(); // model file -> { model, tokenizer }

  for (const fx of manifest.fixtures) {
    if (!cache.has(fx.model)) {
      const ck = readCheckpoint(await readFile(join(MODELS, fx.model)));
      const tok = new Tokenizer(await readFile(join(MODELS, fx.tokenizer)), ck.config.vocabSize);
      cache.set(fx.model, { model: createModel(ck), tokenizer: tok });
    }
    const { model, tokenizer } = cache.get(fx.model);
    const golden = (await readFile(join(FIXTURES, fx.file), 'utf8'))
      .trim().split('\n').map(Number);

    const t0 = performance.now();
    const { tokens, text } = generateGreedy(model, tokenizer, fx.prompt, fx.steps);
    const ms = performance.now() - t0;

    let firstDiff = -1;
    const n = Math.min(tokens.length, golden.length);
    for (let i = 0; i < n; i++) if (tokens[i] !== golden[i]) { firstDiff = i; break; }

    t.check(`${fx.file}: ${golden.length} golden tokens, JS produced ${tokens.length}`,
      tokens.length === golden.length);
    t.check(`${fx.file}: exact match ${firstDiff === -1 ? 'over all ' + n : 'FAILED at index ' + firstDiff} (gate needs >=200)`,
      firstDiff === -1 && n >= 200);
    console.log(`  ${fx.file}: ${(n / (ms / 1000)).toFixed(1)} tok/s in Node  |  "${text.slice(0, 70)}..."`);
  }
}
