// E13 — the J-lens ("thinking ahead") ported from the J-Lens Explorer: the .jlens reader,
// and exact agreement with the explorer's Python pipeline on stories110M (eval prompt 4,
// all positions, layers 4–8). The 110M checkpoint is not in git; without it the parity
// half is reported as skipped.
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { readCheckpoint } from '../src/bin-reader.js';
import { createModel } from '../src/model.js';
import { captureRun } from '../src/capture.js';
import { readJlens, jLensLogits } from '../src/jlens.js';
import { MODELS } from './paths.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export async function run(t) {
  const l15 = readJlens(await readFile(join(MODELS, 'stories15M.jlens')));
  t.check('stories15M.jlens reads: d_model 288, layers 2 and 3', l15.dModel === 288 && l15.layers.join(',') === '2,3' &&
    l15.J[2].length === 288 * 288);
  t.check('a .jlens file is refused when it is not one', (() => { try { readJlens(new Uint8Array(32)); return false; } catch { return true; } })());

  const ck110 = join(MODELS, 'stories110M.bin');
  if (!existsSync(ck110)) { t.check('110M parity skipped (models/stories110M.bin not present)', true); return; }
  const fx = JSON.parse(await readFile(join(ROOT, 'tests', 'fixtures', 'jlens-110M-p4.json'), 'utf8'));
  const model = createModel(readCheckpoint(new Uint8Array(await readFile(ck110)).buffer));
  const lens = readJlens(await readFile(join(MODELS, 'stories110M.jlens')));
  t.check('stories110M.jlens reads: d_model 768, layers 4–8', lens.dModel === 768 && lens.layers.join(',') === '4,5,6,7,8');
  const cap = captureRun(model, fx.token_ids);
  let cells = 0, sum = 0, worst = 1;
  fx.layers.forEach((l, li) => {
    for (let pos = 0; pos < fx.token_ids.length; pos++) {
      const lg = jLensLogits(model, lens.J[l], cap.residualAt(l, pos, 'ffn'));
      const top = [...lg.keys()].sort((a, b) => lg[b] - lg[a]).slice(0, 10);
      const want = new Set(fx.jlens_top10[li][pos]);
      const ov = top.filter(i => want.has(i)).length / 10;
      sum += ov; cells++; worst = Math.min(worst, ov);
    }
  });
  t.check(`J-lens matches the explorer's pipeline exactly (top-10 overlap ${(sum / cells).toFixed(4)}, worst ${worst}, ${cells} cells)`,
    sum / cells === 1 && worst === 1);
}
