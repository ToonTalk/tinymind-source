// E16 — the Thought bubble's engine (src/thought.js), on the Big Storyteller (15M) and
// its J-lens. Readout scores must be exactly the logits logitLens / jLensLogits give,
// restricted to the story vocabulary; the heat strip must rank like the readout; a
// swap must change the story while no swap leaves it alone; and generation must be
// the same with or without a worker pool.
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun, logitLens } from '../src/capture.js';
import { readJlens, jLensLogits } from '../src/jlens.js';
import { Pool, poolWorkerSource } from '../src/batch.js';
import { Thought } from '../src/thought.js';
import { ROOT, MODELS, model15M, tok32000 } from './paths.js';

function fakeWorker(src) {
  const inner = { onmessage: null, postMessage(d) { setTimeout(() => outer.onmessage && outer.onmessage({ data: structuredClone(d) }), 0); } };
  const outer = { onmessage: null, onerror: null, terminate() {},
    postMessage(d) { setTimeout(() => inner.onmessage({ data: structuredClone(d) }), 0); } };
  (new Function('self', src))(inner);
  return outer;
}

export async function run(t) {
  const ck = readCheckpoint(await readFile(model15M));
  const tok = new Tokenizer(await readFile(tok32000), ck.config.vocabSize);
  const jl = readJlens(await readFile(join(MODELS, 'stories15M.jlens')));
  const vocab = JSON.parse(await readFile(join(ROOT, 'thought', 'vocab.json'), 'utf8'));
  t.check('story vocabulary: ~7,900 pieces, word starts, name fragments, theme words',
    vocab.prunedIds.length === 7912 && vocab.wordStartIds.length === 5226 && Object.keys(vocab.alias).length === 182 && vocab.themeWords.length === 300);

  const ids = tok.encode('Lily and Max went to the beach. Lily built a castle. Max dug a', true, false);
  const m1 = createModel(ck), pool = await Pool.create(m1, { K: 3, spawn: () => fakeWorker(poolWorkerSource()) });
  const th = new Thought(m1, jl, vocab, pool);
  await th.ready();
  await th.analyze(ids);
  const alone = new Thought(createModel(ck), jl, vocab, null);
  await alone.analyze(ids);
  const same = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));
  t.check('readout tables identical with and without a pool', same(th.cap.jScores, alone.cap.jScores) && same(th.cap.lScores, alone.cap.lScores));

  const ref = captureRun(createModel(ck), ids, { topK: 0 });
  const pos = ids.length - 1, L = th.band.length, P = th.P;
  let ok = true;
  th.band.forEach((l, li) => {
    const jl0 = jLensLogits(m1, jl.J[l], ref.residualAt(l, pos, 'ffn')), ll0 = logitLens(m1, ref, l, pos, 'ffn');
    for (let r = 0; r < P; r += 97) {
      if (!Object.is(th.cap.jScores[(pos * L + li) * P + r], jl0[th.ids[r]])) ok = false;
      if (!Object.is(th.cap.lScores[(pos * (L + 1) + li) * P + r], ll0[th.ids[r]])) ok = false;
    }
  });
  t.check('thinking-ahead and layer-by-layer scores = jLensLogits / logitLens on the story vocabulary, bit for bit', ok);
  const ro = th.readout(pos, 8);
  t.check('readout: one row per lens layer (15M: layers 2 and 3), sorted, plus the next word',
    ro.layers.map(r => r.layer).join() === '2,3' && ro.layers.every(r => r.jlens.every((e, i) => !i || e.score <= r.jlens[i - 1].score)) && ro.model.length === 5);
  const topWord = ro.layers[0].jlens[0].id, heat = th.heat(topWord);
  t.check('heat: the top thinking-ahead word ranks first at that piece', heat && heat[pos] === 0);
  t.check('heat: a word outside the story vocabulary has none', th.heat(0) === null);

  const hole = tok.vocab.indexOf(' hole'), castle = tok.vocab.indexOf(' castle');
  t.check('no strength or no layers in range means no edit', th.makeEdit(hole, castle, 2, 3, pos, 0) === null && th.makeEdit(hole, castle, 7, 9, pos, 0.4) === null);
  const plain = await th.generate(ids, { maxNew: 30, temperature: 0, seed: 7 });
  const plain2 = await alone.generate(ids, { maxNew: 30, temperature: 0, seed: 7 });
  t.check('greedy writing is the same with and without the pool', same(plain, plain2));
  const warm = await th.generate(ids, { maxNew: 30, temperature: 0.8, seed: 11 });
  const warm2 = await alone.generate(ids, { maxNew: 30, temperature: 0.8, seed: 11 });
  t.check('warm writing with one seed is the same with and without the pool', same(warm, warm2) && !same(warm, plain));
  const edit = th.makeEdit(hole, castle, 2, 3, pos, 1.5);
  const swapped = await th.generate(ids, { maxNew: 30, temperature: 0, seed: 7, edit });
  t.check('a strong swap changes what the mind writes, from the anchor on',
    !same(swapped, plain) && same(swapped.slice(0, ids.length), ids));
  console.log('  15M no swap:   ' + tok.decode(plain.slice(ids.length - 1)).replace(/\n/g, ' ').slice(0, 90));
  console.log('  15M with swap: ' + tok.decode(swapped.slice(ids.length - 1)).replace(/\n/g, ' ').slice(0, 90));
  await th.analyze(swapped, { edit });
  t.check("a swapped story's thoughts are read with the swap still on", th.cap.n === swapped.length);
  pool.dispose();
}
