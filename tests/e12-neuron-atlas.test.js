// E12 — the "what does this neuron do?" card: the offline neuron atlas and neuron steering.
// Atlas: the committed microscope/tiny-mind-neuron-atlas.json is well-formed, belongs to
// stories15M, and its snippets are TRUE — re-running the real model over the cited story
// reproduces the activation and the text. The builder is deterministic (shard count does
// not matter) and its classifier-trimmed capture is bit-identical to the full model.
// Steering: strictly additive. Nothing muted or steered -> byte-identical outputs (both
// the src/steer.js functions and the Microscope worker's message protocol); muting goes
// through model.js's own neuron ablation; a clamp holds the neuron at its value and moves
// the output the way the w2 column says; the continuation path reproduces the C golden
// greedy tokens when nothing is changed.
import { readFile } from 'node:fs/promises';
import { join } from 'node:path';
import { readCheckpoint } from '../src/bin-reader.js';
import { Tokenizer } from '../src/tokenizer.js';
import { createModel } from '../src/model.js';
import { captureRun } from '../src/capture.js';
import { atlasEntry, atlasFingerprint, validateAtlas, looksLikeAtlasText, snippetSig, ATLAS_SCHEMA } from '../src/neuron-atlas.js';
import { captureSteered, continueSteered, steerKey } from '../src/steer.js';
import { headlessModel, scanStories, mergeCandidates, loadStories, encodeFromRaw, ATLAS_PARAMS } from '../tools/build-neuron-atlas.js';
import { ROOT, model15M, model260K, tok32000, tok512, tinyStoriesValid } from './paths.js';

const sameF32 = (a, b) => a.length === b.length && a.every((v, i) => Object.is(v, b[i]));

export async function run(t) {
  // ------------------------------------------------------------------ atlas format
  const text = await readFile(join(ROOT, 'microscope', 'tiny-mind-neuron-atlas.json'), 'utf8');
  const atlas = JSON.parse(text);
  const errs = validateAtlas(atlas);
  t.check(`atlas: valid ${ATLAS_SCHEMA} (${errs.length ? errs[0] : 'no problems'})`, errs.length === 0);
  t.check('atlas: sniffable without parsing (shell + drop routing)', looksLikeAtlasText(text) && !looksLikeAtlasText('{"stories": []}'));
  const ck15 = readCheckpoint(await readFile(model15M));
  const c15 = ck15.config;
  t.check('atlas: built for stories15M (config + weight fingerprint match)',
    atlas.model === 'stories15M' && atlas.config.hiddenDim === c15.hiddenDim && atlas.config.nLayers === c15.nLayers &&
    atlas.config.dim === c15.dim && atlas.config.vocabSize === c15.vocabSize &&
    atlas.fingerprint === atlasFingerprint(c15, ck15.weights));
  const total = c15.nLayers * c15.hiddenDim, kept = Object.keys(atlas.neurons).length;
  t.check(`atlas: covers ${kept}/${total} neurons from ${atlas.corpus.stories} stories (5 per pole, 6 before + 2 after)`,
    kept >= 0.95 * total && atlas.corpus.stories >= 2000 && atlas.perPole === 5 &&
    atlas.window.before === 6 && atlas.window.after === 2);
  const kb = text.length / 1024;
  t.check(`atlas: small enough to ship as a same-folder pack (${kb.toFixed(0)} KB < 4 MB)`, kb < 4096);
  let bothPoles = 0;
  for (const e of Object.values(atlas.neurons)) if (e[2].length && e[3].length) bothPoles++;
  t.check(`atlas: most neurons have BOTH poles (${bothPoles}/${kept})`, bothPoles >= 0.9 * kept);
  const one = atlasEntry(atlas, 0, 5);
  t.check('atlasEntry decodes a neuron: peaks signed, values from percent, target in range',
    one && one.pos.peak > 0 && one.neg.peak < 0 && one.pos.snippets[0].q === 100 &&
    one.pos.snippets[0].value === one.pos.peak && one.neg.snippets.every(s => s.value <= 0) &&
    one.pos.snippets.every(s => s.target >= 0 && s.target < s.pieces.length));
  t.check('atlasEntry: unknown neuron -> null, labels only when signed for these snippets',
    atlasEntry(atlas, 99, 0) === null &&
    Object.entries(atlas.labels || {}).every(([k, lab]) => lab[2] === snippetSig(atlas, k)));

  // ground truth: the cited story, re-run through the real model, gives the same
  // activation at the cited piece and the same text around it
  const tok15 = new Tokenizer(await readFile(tok32000), c15.vocabSize);
  const segs = (await readFile(tinyStoriesValid, 'utf8')).split('<|endoftext|>');
  const fast = headlessModel(ck15);
  let truth = 0, tried = 0;
  for (const [l, j] of [[0, 5], [3, 100], [5, 700]]) {
    const e = atlasEntry(atlas, l, j);
    if (!e) continue;
    for (const side of [e.pos, e.neg]) {
      const sn = side.snippets[0];
      if (!sn) continue;
      tried++;
      const ids = tok15.encode(segs[sn.story].trim(), true, false).slice(0, c15.seqLen);
      const prefix = ids.slice(0, sn.pos + ATLAS_PARAMS.after + 1);   // causal: later pieces cannot change it
      const cap = captureRun(fast, prefix.slice(0, sn.pos + 1), { topK: 0 });
      const act = cap.ffn[(l * cap.nPos + sn.pos) * c15.hiddenDim + j];
      const pieces = [];
      for (let i = sn.pos - sn.target; i < sn.pos - sn.target + sn.pieces.length; i++) pieces.push(tok15.decodePiece(prefix[i - 1], prefix[i]));
      if (Math.abs(act - side.peak) <= 0.005 * Math.abs(side.peak) + 1e-6 && pieces.join('') === sn.pieces.join('')) truth++;
    }
  }
  t.check(`atlas snippets are true: ${truth}/${tried} re-run activations + texts match the cited story`, tried >= 4 && truth === tried);

  // builder: trimmed-classifier capture is bit-identical, merge is shard-independent,
  // encoding is deterministic
  const stories = loadStories(segs.slice(0, 40).join('<|endoftext|>'), tok15, 3, c15.seqLen).map(([i, ids]) => [i, ids.slice(0, 20)]);
  const opts = { keep: 3, before: ATLAS_PARAMS.rawBefore, after: ATLAS_PARAMS.rawAfter };
  t.check('builder: classifier-trimmed capture has bit-identical FFN activations',
    sameF32(captureRun(createModel(ck15), stories[0][1], { topK: 0 }).ffn, captureRun(fast, stories[0][1], { topK: 0 }).ffn));
  const whole = mergeCandidates([scanStories(fast, stories, opts)], 3);
  const sharded = mergeCandidates([scanStories(fast, [stories[2], stories[0]], opts), scanStories(fast, [stories[1]], opts)], 3);
  const strip = m => JSON.stringify([...m]);
  t.check('builder: merged candidates do not depend on how stories were sharded', strip(whole) === strip(sharded));
  const enc1 = JSON.stringify(encodeFromRaw(whole, ck15, tok15, { stories: 3, tokens: 60 }));
  const enc2 = JSON.stringify(encodeFromRaw(sharded, ck15, tok15, { stories: 3, tokens: 60 }));
  t.check('builder: encoding is deterministic and validates', enc1 === enc2 && validateAtlas(JSON.parse(enc1)).length === 0);
  const firstPos = [...whole.values()][0].pos;
  t.check('builder: BOS (position 0) is never a snippet; at most one snippet per story per pole',
    [...whole.values()].every(c => [...c.pos, ...c.neg].every(s => s.pos >= 1)) &&
    [...whole.values()].every(c => new Set(c.pos.map(s => s.story)).size === c.pos.length) && firstPos.length > 0);

  // ------------------------------------------------------------------ steering
  const ck = readCheckpoint(await readFile(model260K));
  const c = ck.config;
  const tok = new Tokenizer(await readFile(tok512), c.vocabSize);
  const ids = tok.encode('Zog is a small dragon. Zog eats clouds and lives behind the moon.', true, false);
  const base = captureRun(createModel(ck), ids, { topK: 8, ablate: new Set() });
  const same = (a, b) => sameF32(a.ffn, b.ffn) && sameF32(a.attention, b.attention) && sameF32(a.residFfn, b.residFfn) &&
    sameF32(a.logZ, b.logZ) && sameF32(a.nextProb, b.nextProb) && JSON.stringify(a.topK) === JSON.stringify(b.topK);
  t.check('steer: nothing steered -> bit-identical to captureRun', same(captureSteered(createModel(ck), ids, { topK: 8 }), base));
  t.check('steer: heads only -> bit-identical to the head-mute capture',
    same(captureSteered(createModel(ck), ids, { topK: 8, heads: ['2:3'] }), captureRun(createModel(ck), ids, { topK: 8, ablate: new Set(['2:3']) })));
  t.check('steerKey: no steer keeps the legacy head key; steer joins it order-free',
    steerKey(['1:2', '0:1'], []) === '0:1,1:2' &&
    steerKey([], [{ layer: 1, neuron: 2, value: 0 }, { layer: 0, neuron: 3, value: 1.5 }]) ===
    steerKey([], [{ layer: 0, neuron: 3, value: 1.5 }, { layer: 1, neuron: 2, value: 0 }]));
  const mutedNeurons = captureSteered(createModel(ck), ids, { topK: 8, heads: ['2:3'], steer: [{ layer: 2, neuron: 10, value: 0 }] });
  t.check('steer: "mute" is exactly model.js neuron ablation (with a head muted too)',
    same(mutedNeurons, captureRun(createModel(ck), ids, { topK: 8, ablate: { heads: new Set(['2:3']), neurons: new Set(['2:10']) } })));
  const L = c.nLayers - 1, J = 7, V = 3.5;
  const up = captureSteered(createModel(ck), ids, { topK: 8, steer: [{ layer: L, neuron: J, value: V }] });
  let held = true;
  for (let p = 0; p < ids.length; p++) if (up.ffnAt(L, p)[J] !== Math.fround(V)) held = false;
  t.check('steer: a clamped neuron reads its value at every position', held);
  // last layer: the residual moves by exactly w2[:, J] * (V - original activation)
  let maxErr = 0, moved = 0;
  const p = ids.length - 1, w2 = ck.weights.w2, HD = c.hiddenDim;
  const a0 = base.ffnAt(L, p)[J];
  for (let d = 0; d < c.dim; d++) {
    const want = w2[(L * c.dim + d) * HD + J] * (V - a0);
    const got = up.residualAt(L, p)[d] - base.residualAt(L, p)[d];
    maxErr = Math.max(maxErr, Math.abs(got - want)); moved = Math.max(moved, Math.abs(got));
  }
  t.check(`steer: clamp moves the residual along the neuron's w2 column (max err ${maxErr.toExponential(1)}, move ${moved.toFixed(3)})`,
    maxErr < 1e-4 && moved > 1e-2);

  // continuation: unchanged path = the C golden greedy tokens (up to the first BOS/EOS)
  const manifest = JSON.parse(await readFile(join(ROOT, 'tests', 'fixtures', 'manifest.json'), 'utf8'));
  for (const fx of manifest.fixtures.filter(f => f.model === 'stories260K.bin')) {
    const golden = (await readFile(join(ROOT, 'tests', 'fixtures', fx.file), 'utf8')).trim().split('\n').map(Number);
    const prompt = tok.encode(fx.prompt, true, false);
    const gen = continueSteered(createModel(ck), prompt, fx.steps - prompt.length);
    let want = golden.slice(prompt.length);
    const stop = want.findIndex(x => x === 1 || x === 2);
    if (stop >= 0) want = want.slice(0, stop);
    t.check(`continue: nothing steered reproduces ${fx.file} (${gen.length} tokens)`, gen.join() === want.join() && gen.length > 20);
  }
  const plainCont = continueSteered(createModel(ck), ids, 20);
  const pushed = continueSteered(createModel(ck), ids, 20, { steer: [{ layer: L, neuron: J, value: 40 }] });
  t.check('continue: a strong steer changes what the model writes', plainCont.join() !== pushed.join());

  // the Microscope worker's message protocol, run in Node: no steer field, steer: [],
  // and direct captureRun all agree byte for byte
  const posted = [];
  globalThis.self = { postMessage: (m) => posted.push(m) };
  await import('../microscope/worker.js');
  // the worker answers asynchronously (pool calls), one message at a time: wait for it to go idle
  const send = async (data) => { posted.length = 0; self.onmessage({ data }); await self.whenIdle(); return posted.slice(); };
  const bytes = (await readFile(model260K)), tbytes = await readFile(tok512);
  const loaded = (await send({ type: 'load', ckptBuf: bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
    tokBuf: tbytes.buffer.slice(tbytes.byteOffset, tbytes.byteOffset + tbytes.byteLength) })).find(m => m.type === 'loaded');
  t.check('worker: loaded reports the weight fingerprint the atlas is matched against',
    loaded && loaded.fingerprint === atlasFingerprint(c, ck.weights));
  const story = 'Zog is a small dragon. Zog eats clouds and lives behind the moon.';
  await send({ type: 'run', text: story });
  const ser = (msgs) => JSON.stringify(msgs.map(m => ({ ...m, attention: m.attention ? Array.from(new Float32Array(m.attention)) : undefined,
    acts: m.acts ? Array.from(new Float32Array(m.acts)) : undefined })), (k, v) => k === 'steer' ? undefined : v);
  let allSame = true;
  for (const q of [{ type: 'lens', pos: 5, ablate: [] }, { type: 'neurons', spanStart: 1, spanEnd: 3, ablate: [] },
                   { type: 'neuron-acts', layer: 2, neuron: 10, ablate: [] },
                   { type: 'lens', pos: 5, ablate: ['2:3'] }, { type: 'run', text: story, ablate: ['2:3'] }]) {
    if (ser(await send(q)) !== ser(await send({ ...q, steer: [] }))) allSame = false;
  }
  t.check('worker: steer: [] answers every query byte-identically to the un-steered message', allSame);
  const legacyAbl = captureRun(createModel(ck), tok.encode(story, true, false), { topK: 12, ablate: new Set(['2:3']) });
  const runAbl = (await send({ type: 'run', text: story, ablate: ['2:3'], steer: [] })).find(m => m.type === 'run-ablated');
  t.check('worker: head-mute run still equals the direct head-ablated capture',
    runAbl && runAbl.nextProb.every((v, i) => v === legacyAbl.nextProb[i]));
  const muteMsg = (await send({ type: 'run', text: story, ablate: [], steer: [{ layer: 2, neuron: 10, value: 0 }] })).find(m => m.type === 'run-ablated');
  const muteRef = captureRun(createModel(ck), tok.encode(story, true, false), { topK: 12, ablate: { heads: new Set(), neurons: new Set(['2:10']) } });
  t.check('worker: a muted neuron drives the whole microscope (run-ablated = neuron-ablated capture)',
    muteMsg && muteMsg.steer.length === 1 && muteMsg.nextProb.every((v, i) => v === muteRef.nextProb[i]));
  const lensSteer = (await send({ type: 'lens', pos: 5, ablate: [], steer: [{ layer: L, neuron: J, value: 40 }] })).find(m => m.type === 'lens-done');
  const lensPlain = (await send({ type: 'lens', pos: 5, ablate: [] })).find(m => m.type === 'lens-done');
  t.check('worker: the lens reflects the steered model', JSON.stringify(lensSteer.rows) !== JSON.stringify(lensPlain.rows));
  const contPlain = (await send({ type: 'continue' })).find(m => m.type === 'continue-done');
  const contUp = (await send({ type: 'continue', ablate: [], steer: [{ layer: L, neuron: J, value: 40 }] })).find(m => m.type === 'continue-done');
  t.check('worker: continue — unchanged model reports changed=false and one story; a steer changes it',
    contPlain && !contPlain.changed && contPlain.base.join() === contPlain.steered.join() &&
    contUp && contUp.changed && contUp.base.join() === contPlain.base.join() && contUp.steered.join() !== contUp.base.join());
  delete globalThis.self;
}
