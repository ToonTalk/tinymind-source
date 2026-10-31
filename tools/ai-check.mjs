// ai-check.mjs — test Tiny Mind's language-model features against a real model (OpenAI).
//   node tools/ai-check.mjs                 all three parts
//   node tools/ai-check.mjs ghost detective guide   any of them  (--compare: guides on gpt-5.4-mini, the old default, too;
//                                                    --room=eye: only the rooms whose name contains "eye")
// Needs OPENAI_API_KEY in the environment. The key is read here and sent only to
// api.openai.com: it is never written to a file, printed, or put into a web page (the
// guide part drives the real pages in headless Chrome, captures the prompts the guides
// WOULD send, and makes the calls from Node).
//
// 1. ghost — the Nursery's ghostwriter (src/ghostwriter.js, the code visitors run):
//    three characters, eight stories each, the app's checks and repair, then a second
//    round with the rejection feedback, as when a visitor presses ✍️ again.
// 2. detective — the Hunt's language-model detective vs the built-in one, on the Four
//    Friends mind: each clue is tested by switching that part off (the Hunt's own worker,
//    in-process) and judged by the Hunt's causal verdict against unrelated controls.
// 3. guide — every room's guide, in real states (Big Storyteller loaded, a story read,
//    the lens open…): typical visitor questions, answered by the guide's model, checked
//    mechanically (DO line, plain speech, page names) and graded by a stronger model
//    against what was on screen and what a right answer must contain.
// Report: reports/ai-check/<date>.md and .json.

import { readFileSync, writeFileSync, mkdirSync, mkdtempSync, rmSync, existsSync, statSync } from 'node:fs';
import { join, dirname, extname, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { createServer } from 'node:http';
import { spawn } from 'node:child_process';
import os from 'node:os';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const KEY = process.env.OPENAI_API_KEY;
if (!KEY) { console.error('OPENAI_API_KEY is not set in the environment'); process.exit(1); }
const imp = p => import(pathToFileURL(join(ROOT, p)).href);
const { PROVIDERS, callLLM } = await imp('src/llm.js');
const parts = process.argv.slice(2).filter(a => !a.startsWith('--'));
const want = p => !parts.length || parts.includes(p);
const ROOM = ((process.argv.find(a => a.startsWith('--room=')) || '').slice(7)).toLowerCase();   // only these guide rooms
const SITE = process.env.TM_SITE || 'C:/Users/toont/Documents/GitHub/tinymind';
const APP_MODEL = PROVIDERS.openai.defaultModel;    // what the ghostwriter and detective use
const GUIDE_MODELS = process.argv.includes('--compare') ? [APP_MODEL, 'gpt-5.4-mini'] : [APP_MODEL];   // the guides' default (tm-prelude), and the old one to compare
const JUDGE = 'gpt-5.4';
const report = { date: new Date().toISOString(), models: { app: APP_MODEL, guide: GUIDE_MODELS, judge: JUDGE }, usage: {} };

// ---- OpenAI, from Node only -----------------------------------------------------
async function chat(model, messages, { json = false } = {}) {
  for (let attempt = 0; ; attempt++) {
    const r = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST', headers: { 'content-type': 'application/json', authorization: 'Bearer ' + KEY },
      body: JSON.stringify({ model, messages, ...(json ? { response_format: { type: 'json_object' } } : {}) }),
    });
    const j = await r.json().catch(() => ({}));
    if (r.ok) {
      const u = report.usage[model] || (report.usage[model] = { in: 0, out: 0, calls: 0 });
      u.in += j.usage?.prompt_tokens || 0; u.out += j.usage?.completion_tokens || 0; u.calls++;
      return j.choices[0].message.content;
    }
    if (attempt < 3 && (r.status === 429 || r.status >= 500)) { await new Promise(res => setTimeout(res, 2000 * (attempt + 1))); continue; }
    throw new Error(`OpenAI ${model}: ${j.error?.message || 'HTTP ' + r.status}`);
  }
}
// the app's own client (src/llm.js) for the ghostwriter and detective, counted too
async function appCall(prompt, maxTokens) {
  const out = await callLLM({ provider: 'openai', key: KEY, model: APP_MODEL, prompt, maxTokens });
  const u = report.usage[APP_MODEL] || (report.usage[APP_MODEL] = { in: 0, out: 0, calls: 0 });
  u.calls++; u.in += Math.round(prompt.length / 4); u.out += Math.round(out.length / 4);   // estimated: callLLM returns text only
  return out;
}
const md = [];
const say = s => { console.log(s); md.push(s); };

// ---- 1. ghostwriter ----------------------------------------------------------------
async function ghostPart() {
  const { ghostwrite, checkStory } = await imp('src/ghostwriter.js');
  const { EMBEDDED_WORDLIST } = await imp('src/embed-data.js');
  const wordlist = new Set(EMBEDDED_WORDLIST);
  const CHARACTERS = [
    { name: 'Zog', kind: 'a small dragon', attr1: 'eats clouds', attr2: 'lives behind the moon' },
    { name: 'Sisi', kind: 'a little cloud', attr1: 'lives in the sky', attr2: 'likes to sing songs' },
    { name: 'Pip', kind: 'a tiny robot', attr1: 'loves red apples', attr2: 'sleeps in a teapot' },
  ];
  say(`\n## 1 · Ghostwriter (${APP_MODEL}, the Nursery's own prompt, checks and repair)\n`);
  say('| character | round | written | first try ok | repaired | accepted | turned away | why turned away |');
  say('|---|---|---|---|---|---|---|---|');
  const out = [];
  for (const sh of CHARACTERS) {
    let feedback = null;
    for (const round of [1, 2]) {
      const t0 = Date.now();
      let res;
      try { res = await ghostwrite(p => appCall(p, 6000), 8, sh, { wordlist, strict: true, feedback }); }
      catch (e) { say(`| ${sh.name} | ${round} | error: ${e.message} |||||| `); continue; }
      const checked = res.stories.map(s => ({ s, v: checkStory(s, sh, wordlist) }));
      const ok = checked.filter(c => !c.v.probs.length), bad = checked.filter(c => c.v.probs.length);
      const why = {};
      for (const b of bad) for (const p of b.v.probs) { const k = p.replace(/:.*$/, '').replace(/^\d+ words long.*/, 'too short or too long').replace(/^(a sentence is too long).*/, '$1')
        .replace(/^(leaves out "[^"]*").*/, '$1').replace(/^(never says|never uses).*/, '$1').trim(); why[k] = (why[k] || 0) + 1; }
      const unknown = [...new Set(bad.flatMap(b => b.v.offList))];
      say(`| ${sh.name} | ${round} | ${res.firstTry.total} | ${res.firstTry.fine} | ${res.repaired} of ${res.firstTry.repairable}${res.tries.length > 1 ? ` (tries: ${res.tries.map(t => `${t.fixed}/${t.sent}`).join(' + ')})` : ''} | **${ok.length}** | ${bad.length} | ${Object.entries(why).map(([k, n]) => `${k} ×${n}`).join('; ')}${unknown.length ? ` (words: ${unknown.slice(0, 12).join(', ')})` : ''} |`);
      out.push({ sheet: sh, round, ...res.firstTry, repaired: res.repaired, tries: res.tries, accepted: ok.length, rejected: bad.length, why, unknown,
        secs: (Date.now() - t0) / 1000, sample: ok[0] ? ok[0].s : (bad[0] ? bad[0].s : '') });
      // what the Nursery folds into the next prompt (addStories' feedback)
      feedback = { offList: new Set(unknown), missing: new Set(bad.flatMap(b => b.v.missing)), shortCounts: bad.filter(b => b.v.wordCount < 100).map(b => b.v.wordCount) };
      if (!bad.length) break;
    }
  }
  const sample = out.find(o => o.accepted);
  if (sample) say(`\nAn accepted story (${sample.sheet.name}):\n\n> ${sample.sample.replace(/\n+/g, ' ')}\n`);
  report.ghost = out;
}

// ---- 2. detective ------------------------------------------------------------------
async function detectivePart() {
  const { questPresetsFromManifest, builtinHypotheses, detectivePrompt, parseHypotheses, causalVerdict } = await imp('src/hunt-evidence.js');
  const posted = [];
  globalThis.self = { postMessage: m => posted.push(m) };
  await imp('hunt/worker.js');
  const send = m => { posted.length = 0; self.onmessage({ data: m }); const e = posted.find(x => x.type === 'error'); if (e) throw new Error(e.message); return posted.slice(); };
  const ck = readFileSync(join(ROOT, 'models', 'stories260K-binding.bin')), tk = readFileSync(join(ROOT, 'models', 'tok512.bin'));
  const loaded = send({ type: 'load', ckptBuf: ck.buffer.slice(ck.byteOffset, ck.byteOffset + ck.byteLength), tokBuf: tk.buffer.slice(tk.byteOffset, tk.byteOffset + tk.byteLength) }).find(m => m.type === 'loaded');
  const config = loaded.config;
  const corpus = JSON.parse(readFileSync(join(ROOT, 'corpus', 'binding', 'stories.json'), 'utf8'));
  const quests = questPresetsFromManifest({ characters: corpus.characters });
  const ablationFor = h => h.kind === 'head' ? { heads: [`${h.layer}:${h.index}`] } : h.kind === 'neuron' ? { neurons: [`${h.layer}:${h.index}`] }
    : { heads: Array.from({ length: config.nHeads }, (_, i) => `${h.layer}:${i}`) };
  const label = h => h.kind === 'head' ? `head L${h.layer}·H${h.index}` : h.kind === 'neuron' ? `neuron L${h.layer}·n${h.index}` : `all heads in layer ${h.layer}`;
  const test = h => {
    const m = send({ type: 'test', id: 't', ...ablationFor(h) }).find(x => x.type === 'test-done');
    const ratio = m.geometricMeanProbability / Math.max(Number.MIN_VALUE, m.baselineGeometricMeanProbability);
    return { ratio, verdict: causalVerdict(ratio, m.controls.map(c => c.ratio)) };
  };
  say(`\n## 2 · Detective (Four Friends mind; every clue tested by switching it off, judged against unrelated controls)\n`);
  say('Verdicts: **selective** = the answer fell much more than unrelated completions (the real find); general = everything fell; little = no real effect.\n');
  say('| quest | answer chance | detective | clues | selective | general | little / other | clues it gave |');
  say('|---|---|---|---|---|---|---|---|');
  const out = [];
  for (const q of quests) {
    const m = send({ type: 'probe', probe: q.probe, continuation: q.continuation, name: q.name }).find(x => x.type === 'probe-done');
    for (const who of ['built-in', 'language model']) {
      let hyps = [], err = '';
      try {
        hyps = who === 'built-in' ? builtinHypotheses(m, { name: q.name, continuation: q.continuation.trim(), nHeads: config.nHeads })
          : parseHypotheses(await appCall(detectivePrompt({ probe: q.probe, continuation: q.continuation, name: q.name, config, m }), 1500));
      } catch (e) { err = e.message; }
      const inRange = h => h.layer >= 0 && h.layer < config.nLayers && h.index >= 0 && h.index < (h.kind === 'head' ? config.nHeads : config.hiddenDim);
      const tested = hyps.filter(h => h.kind === 'team' || inRange(h)).map(h => ({ ...h, ...test(h) }));
      const count = k => tested.filter(t => t.verdict.kind === k).length;
      say(`| ${q.label} | ${(100 * m.contProb).toFixed(1)}% | ${who} | ${err ? 'failed: ' + err.slice(0, 60) : tested.length} | ${count('selective')} | ${count('general')} | ${tested.length - count('selective') - count('general')} | ${tested.map(t => `${label(t)}: ${t.verdict.kind} (${t.ratio.toFixed(2)}×)`).join('; ')} |`);
      out.push({ quest: q.label, who, error: err, clues: tested.map(t => ({ part: label(t), claim: t.claim, verdict: t.verdict.kind, ratio: t.ratio })),
        outOfRange: hyps.length - tested.length });
    }
  }
  const tally = who => { const r = out.filter(o => o.who === who); return { clues: r.reduce((a, o) => a + o.clues.length, 0), selective: r.reduce((a, o) => a + o.clues.filter(c => c.verdict === 'selective').length, 0), questsFound: r.filter(o => o.clues.some(c => c.verdict === 'selective')).length, failures: r.filter(o => o.error).length }; };
  for (const who of ['built-in', 'language model']) { const t = tally(who); say(`\n- **${who}**: ${t.selective} of ${t.clues} clues were selective finds; found one in ${t.questsFound} of ${quests.length} quests${t.failures ? `; ${t.failures} replies could not be read` : ''}.`); }
  report.detective = out;
  delete globalThis.self;
}

// ---- 3. guides -----------------------------------------------------------------------
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.bin': 'application/octet-stream', '.jlens': 'application/octet-stream', '.zip': 'application/zip' };
function serve(dir) {
  dir = resolve(dir);
  return new Promise(res => {
    const srv = createServer((req, rsp) => {
      let p = decodeURIComponent(new URL(req.url, 'http://x').pathname);
      if (p.endsWith('/')) p += 'index.html';
      const f = resolve(join(dir, p));
      if (!(f === dir || f.startsWith(dir + sep)) || !existsSync(f) || !statSync(f).isFile()) { rsp.writeHead(404); rsp.end(); return; }
      rsp.writeHead(200, { 'content-type': TYPES[extname(f)] || 'application/octet-stream' });
      rsp.end(readFileSync(f));
    }).listen(0, '127.0.0.1', () => res(srv));
  });
}
async function chrome() {
  const profile = mkdtempSync(join(os.tmpdir(), 'tm-ai-check-'));
  const port = 9300 + Math.floor(Math.random() * 500);
  const proc = spawn('C:/Program Files/Google/Chrome/Application/chrome.exe',
    ['--headless=new', `--remote-debugging-port=${port}`, `--user-data-dir=${profile}`, '--no-first-run', '--mute-audio', 'about:blank'], { stdio: 'ignore' });
  let list;
  for (let i = 0; i < 50; i++) { try { list = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json(); if (list.length) break; } catch (e) {} await new Promise(r => setTimeout(r, 200)); }
  const ws = new WebSocket(list.find(t => t.type === 'page').webSocketDebuggerUrl);
  await new Promise(r => ws.addEventListener('open', r, { once: true }));
  let id = 0; const pending = new Map();
  ws.addEventListener('message', e => { const m = JSON.parse(e.data); if (m.id && pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); } });
  const call = (method, params = {}) => new Promise(r => { const i = ++id; pending.set(i, r); ws.send(JSON.stringify({ id: i, method, params })); });
  const evaluate = async (expression, timeoutMs = 120000) => {
    const r = await Promise.race([call('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true }),
      new Promise((_, rej) => setTimeout(() => rej(new Error('evaluate timed out')), timeoutMs))]);
    if (r.result?.exceptionDetails) throw new Error(r.result.exceptionDetails.exception?.description || r.result.exceptionDetails.text);
    return r.result?.result?.value;
  };
  return { call, evaluate, close: () => { try { ws.close(); } catch (e) {} proc.kill(); setTimeout(() => { try { rmSync(profile, { recursive: true, force: true }); } catch (e) {} }, 1500); } };
}
// page-side helpers: wait for a condition; capture what a guide would send for a question
const HELPERS = `window.__ai = {
  until: async (fn, ms = 60000) => { const t0 = Date.now(); while (Date.now() - t0 < ms) { try { const v = fn(); if (v) return v; } catch (e) {} await new Promise(r => setTimeout(r, 150)); } throw new Error('timed out waiting: ' + fn); },
  frame: app => { const f = document.querySelector('iframe[data-app=' + app + ']'); return f && f.contentWindow; },
  capture: async (win, question, kid) => {
    const d = win.document, prefs = win.__vgPrefs || {}, wasKid = prefs.kid, hadClaude = win.claude, real = win.__vgAsk;
    prefs.kid = !!kid; win.__cap = null;
    // so the guide believes it can ask (no real call happens); the front door asks through
    // window.claude with one combined prompt, the rooms through __vgAsk(sys, usr)
    win.claude = { complete: async p => { if (!win.__cap) win.__cap = { combined: p }; return 'Noted.'; } };
    win.__vgAsk = async (sys, usr) => { win.__cap = { sys, usr }; return 'Noted.'; };
    d.getElementById('vgIn').value = question; d.getElementById('vgGo').click();
    await window.__ai.until(() => win.__cap, 15000);
    win.__vgAsk = real; if (hadClaude) win.claude = hadClaude; else delete win.claude; prefs.kid = wasKid;
    return win.__cap;
  },
};`;
const ROOMS = [
  { room: 'front door', app: null,
    setup: `await __ai.until(() => document.querySelector('[data-act=think]'));`,
    qs: [
      { q: 'What can I do here?', expect: 'Briefly names the seven activities: teach/raise a mind, watch one grow up, look inside one, see what one is thinking (and swap a thought), meet a mind made of experts (switch experts off), a mind that sees your drawings (the Tiny Eye), hunt for a hidden fact. May point at a card.' },
      { q: 'Where can I see what a mind is thinking and change one of its thoughts?', expect: 'The "See what a mind is thinking" card (the Thought bubble); ideally ends with DO: SHOW think or offers TAB thought.' },
      { q: 'I want to make my own mind', kid: true, expect: 'Teach a tiny mind: invent a character, give it stories, watch it learn. Simple words for a young child. Points at the teach card or offers the Nursery.' },
    ] },
  { room: 'Nursery', app: 'nursery',
    setup: `__tmShell.selectTab('home'); await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=raise]').click();
      await __ai.until(() => { const w = __ai.frame('nursery'); return w && w.document.getElementById('genBtn'); });
      await new Promise(r => setTimeout(r, 1500));`,
    qs: [
      { q: 'How do I get stories for my character?', expect: 'Fill in the character (name, kind, two special things), then the ghostwriter writes stories (press write 8 stories / ✍️, needs a key or built-in Claude), or paste your own; aim for about 60 stories.' },
      { q: 'Why do some stories get turned away?', expect: 'A tiny mind learns only from short simple stories: 100 to 180 words, sentences of 20 words at most, words it met in the stories it grew up on, and every story must keep the character\'s facts word for word (name, kind, both special things).' },
    ] },
  { room: 'Microscope (Big Storyteller)', app: 'microscope',
    setup: `__tmShell.selectTab('home'); await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=look]').click();
      const pick = await __ai.until(() => document.querySelector('#loading button[data-pick=big]') || (__ai.frame('microscope') && __ai.frame('microscope').document.body.dataset.cores && 'loaded'));
      if (pick !== 'loaded') pick.click();
      const w = await __ai.until(() => { const w = __ai.frame('microscope'); return w && w.document.body.dataset.cores && !w.document.getElementById('runBtn').disabled && w; }, 120000);
      const d = w.document; d.getElementById('story').value = 'Lily and Max went to the beach. Lily built a castle. Max dug a big hole in the sand.';
      d.getElementById('runBtn').click(); await __ai.until(() => /^read \\d+/.test(d.getElementById('status').textContent), 60000);
      [...d.querySelectorAll('#strip span')].find(s => s.textContent.trim() === 'castle').click(); await new Promise(r => setTimeout(r, 200));
      d.getElementById('lensBtn').click(); await __ai.until(() => d.getElementById('lensOut').textContent.length > 200, 30000);`,
    qs: [
      { q: 'How about farther than the next word?', expect: 'Thinking ahead: the gold column in layer by layer shows words the mind is already leaning toward further on (this mind has it, at layers 2 and 3; the Giant shows it best). Should talk about what it shows since the view is open, using the words in the status if helpful.' },
      { q: 'How can I tell if one part of the mind really matters?', expect: 'Switch it off in the switch-off lab (tick heads or mute a neuron, re-run) and see if the guess changes; looking (attention, neuron contrast) only suggests, switching off decides. Points at the switch-off lab.' },
      { q: 'What is the guess-o-meter?', kid: true, expect: 'For a young child: bars showing how sure the mind is about each possible next word after the chosen piece; longer bar = more sure.' },
    ] },
  { room: 'Thought bubble (Big Storyteller)', app: 'thought',
    setup: `__tmShell.selectTab('home'); await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=think]').click();
      const pick = await __ai.until(() => document.querySelector('#loading button[data-pick=big]') || (__ai.frame('thought') && /ready/.test(__ai.frame('thought').document.getElementById('status').textContent) && 'loaded'));
      if (pick !== 'loaded') pick.click();
      const w = await __ai.until(() => { const w = __ai.frame('thought'); return w && /ready/.test(w.document.getElementById('status').textContent) && w; }, 120000);
      const d = w.document; d.getElementById('readBtn').click();
      await __ai.until(() => /^read \\d+/.test(d.getElementById('status').textContent) && d.querySelectorAll('#readout .lrow').length, 60000);`,
    qs: [
      { q: 'What are the gold words?', expect: 'Gold chips are thinking-ahead words that layer by layer does not show, and they appear ONLY in the third mode, "what only thinking ahead sees". Box four is in the first mode ("thinking ahead"), where nothing is gold, so the right answer explains what gold means, says none are gold in this mode, and suggests (or offers) switching to "what only thinking ahead sees". Naming specific gold words is wrong.' },
      { q: 'How do I make the mind write moat?', expect: 'Use swap what it\'s thinking: take out hole, put in castle, from the last piece ("a") of the beach story, layers as set, low strength (0.4), press swap and rewrite; the Giant Storyteller does this most clearly; this smaller Big Storyteller may need more strength or may not.' },
      { q: 'My swap did not change anything. Why?', expect: 'Possible reasons: strength too low, too few layers, the take-out word was not really on its mind at that piece, a small mind with fainter thoughts; suggest a bigger strength, more layers, another pair of words, or the Giant Storyteller. Must not claim a swap was run if the status shows none.' },
      { q: 'What does the glow mean?', kid: true, expect: 'For a young child: clicking a word makes the story glow where the mind was thinking about that word, sometimes before it wrote it. Nothing is glowing yet unless the status says so.' },
    ] },
  { room: 'Tiny Eye (a house drawn, a neuron card open)', app: 'eye',
    setup: `__tmShell.selectTab('home'); await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=see]').click();
      const w = await __ai.until(() => { const w = __ai.frame('eye'); return w && w.document.body.dataset.swept === '1' && w; }, 120000);
      const d = w.document, pad = d.getElementById('pad'), r = pad.getBoundingClientRect();
      // draw a house with the pad's own pointer handlers: walls, a roof, a door
      const at = (type, x, y) => pad.dispatchEvent(new w.PointerEvent(type, { clientX: r.left + x * r.width, clientY: r.top + y * r.height, pointerId: 1, button: 0, bubbles: true }));
      for (const s of [[[0.25, 0.45], [0.25, 0.85], [0.75, 0.85], [0.75, 0.45], [0.25, 0.45]], [[0.18, 0.48], [0.5, 0.15], [0.82, 0.48]], [[0.45, 0.85], [0.45, 0.65], [0.56, 0.65], [0.56, 0.85]]]) {
        at('pointerdown', ...s[0]); for (const p of s.slice(1)) at('pointermove', ...p); at('pointerup', ...s[s.length - 1]);
        await new Promise(r => setTimeout(r, 150));
      }
      await __ai.until(() => d.querySelector('#guesses .gbar.top'), 20000);
      await new Promise(r => setTimeout(r, 500));
      d.querySelector('#busy .nchip.up').click();
      await __ai.until(() => d.querySelector('#cardBody .poles'), 20000);`,
    qs: [
      { q: 'What does this neuron do?', expect: 'Talks about the open card in the status: what its "fires most" and "fires least" doodles and class bars show (both ends), the bigger AI\'s name and whether it passed the test if the status shows one; may say most neurons have one main job you can name (not all) and that switching it off would test it. Must not invent doodles or numbers not in the status.' },
      { q: 'Why did it guess that?', expect: 'Refers to the guess bars in the status; it only knows 24 kinds of things and always picks one; it sees the drawing shrunk to 28 by 28 dots; neurons light up and vote; to find which neurons matter, switch some off and watch the guess. Grounded in the status.' },
      { q: 'Do bigger eyes have cleaner neurons?', expect: 'No: that was tested (three sizes of eye, about equally clean), so bigger is not cleaner; most neurons have one main job you can name at every size. Must not claim bigger eyes are cleaner.' },
      { q: 'What happens if I switch off a neuron?', kid: true, expect: 'For a young child: that part of the eye goes quiet (its output becomes zero) and the guess might change, which shows whether it was helping; try the switch-off button on the card or in box four.' },
      { q: 'Why do the other kinds say under 1% instead of 0%?', expect: 'It was taught never to be completely sure (label smoothing: each right answer counted as about 90% right in training), so even a confident guess stops near 97% and every other kind keeps a sliver under 1%; the "other kinds together" row adds the slivers up. Plain words; the term label smoothing is optional.' },
    ] },
  { room: 'Tiny Eye (scrubbed back to step 1,206, neuron 117\'s growth story open)', app: 'eye',
    setup: `const d = __ai.frame('eye').document;
      [...d.querySelectorAll('#stories button')].find(b => /neuron 117/.test(b.textContent)).click();
      await __ai.until(() => d.querySelectorAll('#strip .col').length, 30000);
      [...d.querySelectorAll('#strip .col')].find(c => /1,206/.test(c.textContent)).click();
      await __ai.until(() => /at step 1,206/.test(d.getElementById('cardTitle').textContent) && d.querySelector('#guesses .gbar'), 20000);`,
    qs: [
      { q: 'Is the eye finished learning here?', expect: 'No: box five shows step 1,206 of 9,360, not the end; it is right about 77 times in 100 there and its neurons\' jobs are only about 20% settled; the jobs settle late, and "the end" button returns to the finished eye. Numbers must match the status.' },
      { q: 'How did this neuron grow?', expect: 'About neuron 117 (its card and growth story): it becomes a house or roof neuron early, and later umbrellas (also peaked tops) join it as an offshoot, just below the houses; the strip shows its top doodles at several moments. May add that neurons try jobs, settle late, sometimes switch, and a few grow offshoots.' },
    ] },
  { room: 'Tiny Eye (squeezed, squeezed neuron 7 open)', app: 'eye',
    setup: `const d = __ai.frame('eye').document;
      d.getElementById('growEnd').click(); await new Promise(r => setTimeout(r, 500));
      d.getElementById('sq7Btn').click();
      await __ai.until(() => /answer to/.test(d.getElementById('squeezeNote').textContent) && d.querySelector('#cardBody .poles') && /squeezed eye/.test(d.getElementById('cardTitle').textContent), 90000);`,
    qs: [
      { q: 'What changed when I squeezed it?', expect: 'The squeezed eye has only 8 last-stage neurons (and 8 channels); each has to cover more kinds of things: box six\'s numbers (about 4 kinds each against about 3 for the full eye); it is still right about 84 times in 100; its top doodles still look tidy. Must not call the neurons muddled or confused.' },
      { q: 'Why does this neuron like rainbows and umbrellas?', kid: true, expect: 'For a young child: the squeezed eye has only 8 neurons at the end, so each one has to do more jobs; rainbows, umbrellas and castles all have an arch or bump on top, so this neuron fires for all of them. Grounded in the card (rainbow, umbrella, castle bars).' },
    ] },
  { room: 'Experts (a story written, layer 3, its expert 4 switched off)', app: 'experts',
    setup: `__tmShell.selectTab('home'); await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=experts]').click();
      const w = await __ai.until(() => { const w = __ai.frame('experts'); return w && /ready/.test(w.document.getElementById('status').textContent) && w; }, 180000);
      const d = w.document;
      d.getElementById('story').value = 'Tom and his dog Max went to the park.';
      d.getElementById('writeBtn').click();
      await __ai.until(() => d.body.dataset.story === '1' && !d.getElementById('writeBtn').disabled, 120000);
      d.querySelector('.ecard[data-expert="2:4"] .offb').click();   // the stop sign expert: not one of the challenges
      await __ai.until(() => /^done:|came out the same/.test(d.getElementById('status').textContent), 120000);`,
    qs: [
      { q: 'What does expert 5 do?', expect: 'About expert 5 of the layer shown (layer 3), from its card in the status: its name (or "no clear job found"), the words that almost always go to it, how busy it is; may add that experts sort words by kind (grammar jobs), not topics. Must not invent words or numbers that are not in the status.' },
      { q: 'Why did my story change?', expect: 'Both stories start from the same beginning and always pick the most likely word, so the only difference is the switched-off expert (layer 3, expert 4, muted); the lit-up words are the changes, and once one word changes, what follows can change too. Grounded in the status.' },
      { q: 'Which expert handles names?', expect: 'Honest: the experts sort words, they are not topic experts, so there is no expert for a topic; some experts get naming words or he/she (point to a card in the status if one shows such words); suggest watching an expert (👁) to see its words glow. Must not invent an expert or words the status does not show.' },
      { q: 'Which expert makes it say to to to?', expect: 'Must NOT name the layer and expert: that challenge is open in the status. May give the page\'s hint (look in layer 2) and say how to search: switch off one expert at a time and read the lit-up words.' },
      { q: "What's an expert?", kid: true, expect: 'For a young child: one of eight helpers in each layer of this mind; a router picks one helper for every word, so each helper gets its own kind of words. Short, warm, simple.' },
    ] },
  { room: 'Hunt', app: 'hunt',
    setup: `__tmShell.selectTab('home'); await new Promise(r => setTimeout(r, 300)); document.querySelector('[data-act=hunt]').click();
      await __ai.until(() => { const w = __ai.frame('hunt'); return w && w.document.getElementById('cluesBtn'); }, 60000);
      await new Promise(r => setTimeout(r, 2000));`,
    qs: [
      { q: 'What should I do first?', expect: 'Pick a quest (a sentence the mind should finish about a character) and probe/ask the mind, then get the detective\'s clues and test them by switching parts off. Consistent with what the status shows as done.' },
      { q: 'Why switch parts off instead of just looking at the attention?', expect: 'Looking (attention, neurons lighting up) is only correlation, a clue; switching a part off and seeing the answer get worse (more than unrelated sentences) is the test that shows the part really contributes.' },
    ] },
];
function mechanical(answer, sys, q) {
  const issues = [];
  const lines = answer.trim().split('\n');
  const doLine = /DO:\s*([A-Z_]+)\s*([^\n]*)$/.exec(answer.trim());
  const targets = (/Targets: ([a-z ,]+)\./i.exec(sys) || /Show targets: ([a-z ,]+)\./i.exec(sys) || [])[1];
  const tabs = (/Tab names: ([a-z ,]+)\./i.exec(sys) || [])[1];
  if (doLine) {
    const verb = doLine[1], arg = doLine[2].trim().replace(/[.!?,;:]+$/, '').toLowerCase();   // as vg-core reads it
    const okSet = verb === 'SHOW' ? (targets || '').split(/,\s*/) : verb === 'TAB' ? (tabs || '').split(/,\s*/) : [];
    if (!okSet.map(s => s.trim().toLowerCase()).includes(arg)) issues.push(`DO line points at something that does not exist: "${doLine[0]}"`);
    if (lines[lines.length - 1].trim().indexOf('DO:') !== 0) issues.push('DO line is not on its own last line (the app still catches it)');
  }
  if (/\*\*|^#+ |^\s*[-*] |`/m.test(answer.replace(/DO:.*$/, ''))) issues.push('markdown in a reply that is read aloud');
  const prose = answer.replace(/DO:.*$/, '');
  const asked = q.toLowerCase();
  for (const [term, page] of [['logit lens', 'layer by layer'], ['j-lens', 'thinking ahead'], ['ablation', 'switch off']])
    if (prose.toLowerCase().includes(term) && !asked.includes(term)) issues.push(`uses "${term}" instead of the page's "${page}"`);
  const sentences = prose.split(/[.!?]+\s/).filter(s => s.trim()).length;
  if (sentences > 5) issues.push(`long for speech (${sentences} sentences)`);
  return issues;
}
async function judge(room, q, status, answer) {
  const prompt = `You grade one reply from the voice guide inside Tiny Mind, an educational web app where children (or, without "young child", visitors 12 and up) explore real small language models. The guide sees a STATUS of the screen and must answer accurately, consistently with that status, briefly (it is read aloud), using the names shown on the page.

Room: ${room}
Audience: ${q.kid ? 'a young child' : '12 or older'}
STATUS (what is on screen right now):
${status}

Visitor's question: ${q.q}
What a good answer must get right: ${q.expect}

Guide's reply:
${answer}

Return JSON: {"grade": 0-3 (3 = right, grounded and clear; 2 = right with a small slip; 1 = partly wrong or unhelpful; 0 = wrong or misleading), "problems": ["short notes, empty if none"]}`;
  try { return JSON.parse(await chat(JUDGE, [{ role: 'user', content: prompt }], { json: true })); }
  catch (e) { return { grade: null, problems: ['judge failed: ' + e.message] }; }
}
async function guidePart() {
  if (!existsSync(join(SITE, 'index.html'))) throw new Error('site folder not found: ' + SITE + ' (build it with tools/build-site.mjs, or set TM_SITE)');
  const srv = await serve(SITE), port = srv.address().port;
  const c = await chrome();
  say(`\n## 3 · Guides (real pages in headless Chrome; answers by ${GUIDE_MODELS.join(' and ')}; graded 0–3 by ${JUDGE})\n`);
  const out = [];
  try {
    await c.call('Page.enable');
    await c.call('Page.navigate', { url: `http://127.0.0.1:${port}/` });
    await new Promise(r => setTimeout(r, 3000));
    await c.evaluate(HELPERS);
    for (const R of ROOMS) {
      if (ROOM && !R.room.toLowerCase().includes(ROOM)) continue;
      try { await c.evaluate(`(async () => { ${R.setup} return true; })()`, 240000); }
      catch (e) { say(`\n**${R.room}**: could not set the room up (${e.message.slice(0, 120)}) — skipped.`); continue; }
      say(`\n### ${R.room}\n`);
      say(`| question | model | grade | checks | reply |`);
      say(`|---|---|---|---|---|`);
      for (const q of R.qs) {
        const cap = await c.evaluate(`(async () => __ai.capture(${R.app ? `__ai.frame('${R.app}')` : 'window'}, ${JSON.stringify(q.q)}, ${!!q.kid}))()`, 30000);
        if (cap.combined) {   // sys + '\n\n' + usr, where usr starts with the conversation or the status
          const at = cap.combined.search(/\n\n(Conversation so far between|STATUS RIGHT NOW)/);
          cap.sys = cap.combined.slice(0, at); cap.usr = cap.combined.slice(at + 2);
        }
        const usr = cap.usr.slice(cap.usr.indexOf('STATUS RIGHT NOW'));   // each question on its own (drop earlier captured turns)
        const status = usr.slice(usr.indexOf('\n') + 1, usr.lastIndexOf('\n\nThe visitor now asks'));
        for (const model of GUIDE_MODELS) {
          let answer;
          try { answer = await chat(model, [{ role: 'system', content: cap.sys }, { role: 'user', content: usr }]); }
          catch (e) { answer = 'ERROR: ' + e.message; }
          const issues = mechanical(answer, cap.sys, q.q);
          const g = await judge(R.room, q, status, answer);
          const cell = s => String(s).replace(/\|/g, '\\|').replace(/\n+/g, ' ⏎ ');
          say(`| ${cell(q.q)}${q.kid ? ' (kid)' : ''} | ${model} | ${g.grade ?? '?'} | ${cell([...issues, ...(g.problems || [])].join('; ') || 'ok')} | ${cell(answer)} |`);
          out.push({ room: R.room, question: q.q, kid: !!q.kid, model, answer, issues, grade: g.grade, problems: g.problems || [] });
        }
      }
    }
  } finally { c.close(); srv.close(); }
  for (const model of GUIDE_MODELS) {
    const r = out.filter(o => o.model === model && o.grade != null);
    if (!r.length) continue;
    const avg = r.reduce((a, o) => a + o.grade, 0) / r.length;
    say(`\n- **${model}**: average grade ${avg.toFixed(2)} of 3 over ${r.length} answers; ${r.filter(o => o.grade >= 2).length} good, ${r.filter(o => o.grade <= 1).length} weak; ${out.filter(o => o.model === model && o.issues.length).length} with mechanical issues.`);
  }
  report.guide = out;
}

// ---- run -------------------------------------------------------------------------------
say(`# Tiny Mind AI check — ${new Date().toISOString().slice(0, 16).replace('T', ' ')}`);
say(`\nApp model ${APP_MODEL}; guide models ${GUIDE_MODELS.join(', ')}; judge ${JUDGE}.`);
for (const [name, fn] of [['ghost', ghostPart], ['detective', detectivePart], ['guide', guidePart]]) {
  if (!want(name)) continue;
  try { await fn(); } catch (e) { say(`\n**${name} failed:** ${e.stack || e.message}`); }
}
say('\n## Usage\n');
for (const [m, u] of Object.entries(report.usage)) say(`- ${m}: ${u.calls} calls, ~${u.in.toLocaleString()} tokens in, ~${u.out.toLocaleString()} out`);
const dir = join(ROOT, 'reports', 'ai-check');
mkdirSync(dir, { recursive: true });
const stamp = new Date().toISOString().slice(0, 16).replace(/[:T]/g, '-');
writeFileSync(join(dir, `${stamp}.md`), md.join('\n') + '\n');
writeFileSync(join(dir, `${stamp}.json`), JSON.stringify(report, null, 2));
console.log(`\nreport: reports/ai-check/${stamp}.md`);
process.exit(0);
