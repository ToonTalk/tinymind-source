// E18 — a mind made of experts (plans/experts-room.md Phase 1; tools/moe). The Experts room's
// mind, models/stories-moe-top1.bin (tools/moe/export.py: llama2.c weights with hidden_dim 0,
// then a "MoE1" block of router and experts; top-1, each chosen expert scaled by its router
// probability), must read back exactly; the one-piece engine (model.js forward) must reproduce
// PyTorch's greedy stories, top logits and every layer's routing, plain and with one expert
// muted or rerouted (tests/fixtures/moe-top1-parity.json, tools/moe/parity.py, CPU float32);
// the batched engine (batch.js) must be bit-identical to it with no pool and with pools of
// different sizes; muting an expert must act exactly like zeroing its output weights; and the
// room's engine (src/experts.js) must read and write the same stories.
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';
import { readCheckpoint } from '../src/bin-reader.js';
import { createModel, forward, argmax } from '../src/model.js';
import { Pool, poolWorkerSource, forwardBatch, classify, continueFast, topIndices } from '../src/batch.js';
import { Experts } from '../src/experts.js';
import { MODELS, ROOT } from './paths.js';

function fakeWorker(src) {
  const inner = { onmessage: null, postMessage(d) { setTimeout(() => outer.onmessage && outer.onmessage({ data: structuredClone(d) }), 0); } };
  const outer = { onmessage: null, onerror: null, terminate() {},
    postMessage(d) { setTimeout(() => inner.onmessage({ data: structuredClone(d) }), 0); } };
  (new Function('self', src))(inner);
  return outer;
}
const fakePool = (model, K) => Pool.create(model, { K, spawn: () => fakeWorker(poolWorkerSource()) });
const sameArr = (a, b) => { if (a.length !== b.length) return false; for (let i = 0; i < a.length; i++) if (!Object.is(a[i], b[i])) return false; return true; };
const switchOf = s => !s ? null : s.kind === 'mute' ? { mute: new Set([`${s.layer}:${s.expert}`]) } : { reroute: new Set([`${s.layer}:${s.expert}`]) };

// one case of the fixture through forward(): every step's top 5, routes and gates
function oneByOne(ck, cs) {
  const m = createModel(ck), c = ck.config, ab = switchOf(cs.switch);
  const toks = [...cs.prompt_tokens], steps = [];
  let routes = [];
  const tap = { router(l, pos, probs, top, gate, k) { routes[l] = { top: Array.from(top.subarray(0, k)), gate: Array.from(gate.subarray(0, k)) }; } };
  for (let pos = 0; pos < cs.tokens.length - 1; pos++) {
    routes = [];
    const logits = forward(m, toks[pos], pos, tap, ab);
    if (pos >= cs.prompt_tokens.length - 1) {
      const order = topIndices(logits, 5);
      steps.push({ top5: order.map(i => [i, logits[i]]), routes: routes.map(r => r.top), gates: routes.map(r => r.gate) });
      if (toks.length < cs.tokens.length) toks.push(argmax(logits));
    }
  }
  return { toks, steps, nLayers: c.nLayers };
}

export async function run(t) {
  const file = join(MODELS, 'stories-moe-top1.bin');
  const u8 = await readFile(file);
  const ck = readCheckpoint(u8), c = ck.config;
  t.check('top-1 mind: llama2.c header with hidden_dim 0, then the MoE1 block (8 experts of 192, top-1, not renormalized)',
    c.dim === 288 && c.nLayers === 6 && c.hiddenDim === 0 && c.vocabSize === 32000 && c.sharedWeights &&
    c.experts && c.experts.n === 8 && c.experts.topK === 1 && c.experts.hidden === 192 && c.experts.renorm === false);
  // 19,186,848 weights, plus run.c's unused RoPE table (256 x 48) and the block's 8 header words
  t.check('top-1 mind: router and experts fill the file to the last byte (19.19M weights)',
    ck.weights.router.length === 6 * 8 * 288 && ck.weights.ew1.length === 6 * 8 * 192 * 288 && ck.weights.ew2.length === 6 * 8 * 288 * 192 &&
    ck.fileBytes === u8.length && (u8.length - 28) / 4 === 19186848 + 256 * 48 + 8);

  // parity with PyTorch, one piece at a time
  const fx = JSON.parse(await readFile(join(ROOT, 'tests', 'fixtures', 'moe-top1-parity.json'), 'utf8'));
  let tokOk = true, routeOk = true, maxD = 0, gateD = 0;
  const label = s => !s ? 'plain' : `${s.kind} L${s.layer + 1}·E${s.expert}`;
  const seen = new Set();
  for (const cs of fx.cases) {
    const r = oneByOne(ck, cs);
    if (!sameArr(r.toks, cs.tokens)) { tokOk = false; console.log('    token mismatch:', label(cs.switch), cs.prompt); }
    cs.steps.forEach((s, i) => {
      const mine = r.steps[i];
      s.top5.forEach(([id, v], j) => { if (mine.top5[j][0] !== id) routeOk = routeOk && j > 2; maxD = Math.max(maxD, Math.abs(mine.top5[j][1] - v)); });
      s.routes.forEach((ro, l) => { if (ro.join() !== mine.routes[l].join()) routeOk = false; gateD = Math.max(gateD, Math.abs(ro[0] === undefined ? 0 : s.gates[l][0] - mine.gates[l][0])); });
    });
    seen.add(label(cs.switch));
  }
  t.check(`greedy stories = PyTorch's, token for token (${fx.cases.length} cases: ${[...seen].join(', ')})`, tokOk);
  t.check('every layer chooses the same expert as PyTorch at every step, and the top 3 next words agree', routeOk);
  t.check(`top-5 logits within 2e-3 of PyTorch (max |diff| ${maxD.toExponential(2)}), gates within 1e-5 (${gateD.toExponential(2)})`, maxD < 2e-3 && gateD < 1e-5);
  const story = sw => fx.cases.find(cs => label(cs.switch) === sw && cs.prompt.startsWith('Once')).text;
  t.check('muting L2·E1 makes the mind write "to to to" (and the plain story does not)', /\bto to to\b/.test(story('mute L2·E1')) && !/\bto to\b/.test(story('plain')));

  // the batched engine, bit for bit: no pool, and pools of 2 and 3
  const ids = fx.cases[0].tokens.slice(0, 50);
  const ref = { logits: [], routes: [] };
  for (const ab of [null, { mute: new Set(['1:1', '4:3']) }, { reroute: new Set(['2:7', '0:6']) }]) {
    const m = createModel(ck), rows = [], rt = [];
    for (let pos = 0; pos < ids.length; pos++) rows.push(...forward(m, ids[pos], pos, { router(l, p, probs, top, gate, k) { rt.push(l, p, ...probs, ...top.subarray(0, k), ...gate.subarray(0, k)); } }, ab));
    ref.logits.push(Float32Array.from(rows)); ref.routes.push(rt);
  }
  async function batched(pool) {
    const outs = [];
    for (const [i, ab] of [null, { mute: new Set(['1:1', '4:3']) }, { reroute: new Set(['2:7', '0:6']) }].entries()) {
      const m = createModel(ck), rt = [];
      const tap = { router(l, p, probs, top, gate, k) { rt.push([p, l, ...probs, ...top.subarray(0, k), ...gate.subarray(0, k)]); } };
      // the prompt in two runs, as generation goes on from a filled cache
      const X1 = await forwardBatch(m, ids.slice(0, 30), 0, { tap, ablate: ab, pool });
      const X2 = await forwardBatch(m, ids.slice(30), 30, { tap, ablate: ab, pool });
      const L1 = await classify(m, X1, 0, 30, pool), L2 = await classify(m, X2, 0, ids.length - 30, pool);
      const all = new Float32Array(L1.length + L2.length); all.set(L1); all.set(L2, L1.length);
      rt.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      outs.push(sameArr(all, ref.logits[i]) && sameArr(rt.flatMap(r => [r[1], r[0], ...r.slice(2)]), ref.routes[i]));
    }
    return outs.every(Boolean);
  }
  t.check('batched, no pool: logits and every route = forward(), bit for bit (plain, muted, rerouted)', await batched(null));
  for (const K of [2, 3]) {
    const pool = await fakePool(createModel(ck), K);
    t.check(`batched, pool of ${K}: the same, bit for bit`, !!pool && await batched(pool));
    if (pool) pool.dispose();
  }

  // muting = zeroing that expert's output weights (ew2), bit for bit
  const zeroed = readCheckpoint(new Uint8Array(u8));
  const n = c.experts.n, eh = c.experts.hidden;
  zeroed.weights.ew2.fill(0, (1 * n + 1) * c.dim * eh, (1 * n + 2) * c.dim * eh);
  const muted = createModel(ck), zm = createModel(zeroed);
  let muteOk = true;
  for (let pos = 0; pos < 40; pos++) if (!sameArr(forward(muted, ids[pos], pos, null, { mute: new Set(['1:1']) }), forward(zm, ids[pos], pos))) muteOk = false;
  t.check('muting L2·E1 = zeroing its output weights, bit for bit', muteOk);
  // rerouting: the switched-off expert is never chosen, and the word goes to its next choice
  let rerouteOk = true, moved = 0;
  const plainR = [], offR = [];
  const m1 = createModel(ck), m2 = createModel(ck);
  for (let pos = 0; pos < 40; pos++) {
    forward(m1, ids[pos], pos, { router(l, p, probs, top) { if (l === 1) plainR.push([...probs]); } });
    forward(m2, ids[pos], pos, { router(l, p, probs, top) { if (l === 1) offR.push(top[0]); } }, { reroute: new Set(['1:1']) });
  }
  plainR.forEach((pr, p) => {
    if (offR[p] === 1) rerouteOk = false;
    const best = pr.indexOf(Math.max(...pr));
    if (best === 1) moved++;
  });
  t.check(`rerouting L2·E1: never chosen (its ${moved} words of 40 went elsewhere)`, rerouteOk && moved > 0);

  // continueFast, the shared greedy writer, honors the switches
  const p = fx.cases.find(cs => label(cs.switch) === 'mute L2·E1');
  const pool = await fakePool(createModel(ck), 3);
  const cont = await continueFast(createModel(ck), p.prompt_tokens, p.tokens.length - p.prompt_tokens.length, { heads: [], steer: [], pool });
  t.check('continueFast (plain) writes PyTorch\'s plain story', sameArr([...fx.cases.find(cs => !cs.switch && cs.prompt === p.prompt).prompt_tokens, ...cont],
    fx.cases.find(cs => !cs.switch && cs.prompt === p.prompt).tokens.slice(0, p.prompt_tokens.length + cont.length)));

  // the room's engine
  const ex = new Experts(createModel(ck), pool);
  const wrote = await ex.write(p.prompt_tokens, { maxNew: p.tokens.length - p.prompt_tokens.length, mute: ['1:1'] });
  t.check('Experts.write, L2·E1 muted, greedy = PyTorch\'s muted story', sameArr(wrote, p.tokens));
  const rd = await ex.read(p.tokens, { mute: ['1:1'] });
  const fxRoutes = p.steps.map(s => s.routes.map(r => r[0]));
  let readOk = rd.n === p.tokens.length;
  fxRoutes.forEach((rs, i) => rs.forEach((e, l) => { if (rd.chosen(p.prompt_tokens.length - 1 + i, l) !== e) readOk = false; }));
  t.check('Experts.read: each piece\'s expert in every layer = PyTorch\'s routing', readOk);
  const probs = rd.votes(10, 2);
  t.check('Experts.read: the router\'s eight votes at a piece add up to 1, and the chosen one is the biggest',
    probs.length === 8 && Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-5 && probs.indexOf(Math.max(...probs)) === rd.chosen(10, 2));
  pool.dispose();

  // the top-2 mind, if it is here: the same engine, renormalized top-2 routing
  const top2 = join(MODELS, 'stories-moe.bin'), fx2 = join(ROOT, 'tests', 'fixtures', 'moe-top2.fixture.json');
  if (existsSync(top2) && existsSync(fx2)) {
    const ck2 = readCheckpoint(await readFile(top2)), f2 = JSON.parse(await readFile(fx2, 'utf8'));
    let ok2 = ck2.config.experts.topK === 2 && ck2.config.experts.renorm;
    for (const fxc of f2.fixtures) {
      const m = createModel(ck2), toks = [...fxc.prompt_tokens];
      for (let pos = 0; pos < fxc.tokens.length - 1; pos++) {
        const lg = forward(m, toks[pos], pos);
        if (pos >= fxc.prompt_tokens.length - 1 && toks.length < fxc.tokens.length) toks.push(argmax(lg));
      }
      if (!sameArr(toks, fxc.tokens)) ok2 = false;
    }
    t.check('top-2 mind (renormalized): greedy stories = PyTorch\'s', ok2);
  } else console.log('    (the top-2 mind, models/stories-moe.bin, is not here: its check is skipped; it is not in the room)');
}
