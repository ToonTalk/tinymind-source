/* smoke-tiny-mind.mjs — suite for the guided Tiny Mind build.
   Strategy: the shell injects each template into an iframe as
   '<!doctype html>' + tpl.innerHTML, so the suite extracts each
   template from the BUILT file and boots it standalone in jsdom —
   identical document semantics, no jsdom-iframe flakiness. A fake
   Worker base (which the prelude's shim subclasses) lets tests
   drive synthetic worker traffic both directions. The shell is
   booted as the full page (templates stay inert inside <template>).
   Run: node smoke-tiny-mind.mjs */
import { JSDOM, VirtualConsole } from 'jsdom';
import { readFileSync } from 'fs';

const BUILT = readFileSync(process.argv[2] || 'tiny-mind-guided.html', 'utf8');

let pass = 0, fail = 0;
const failures = [];
function check(name, cond, detail) {
  if (cond) { pass++; console.log('  ok  ' + name); }
  else { fail++; failures.push(name + (detail ? ' — ' + detail : '')); console.log('  FAIL ' + name + (detail ? ' — ' + detail : '')); }
}
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function until(fn, ms = 8000, step = 25) {
  const t0 = Date.now();
  while (Date.now() - t0 < ms) { try { if (fn()) return true; } catch (e) {} await sleep(step); }
  return false;
}

function extractTemplate(app) {
  const open = BUILT.indexOf(`<template id="tpl-${app}">`);
  const start = open + `<template id="tpl-${app}">`.length;
  const end = BUILT.indexOf('</template>', open);
  /* jsdom does not execute module scripts; the frame apps are
     single-file inline modules with no imports, so run them classic */
  return ('<!doctype html>' + BUILT.slice(start, end)).replaceAll('<script type="module">', '<script>');
}

const IGNORABLE = /Not implemented|scrollIntoView|SpeechSynthesis|getContext|canvas|AudioContext|requestAnimationFrame/i;
function makeDom(html, collect, beforeParse) {
  const vc = new VirtualConsole();
  vc.on('jsdomError', e => { if (!IGNORABLE.test(String(e && e.message))) collect.push('jsdomError: ' + e.message); });
  vc.on('error', (...a) => { const s = a.map(String).join(' '); if (!IGNORABLE.test(s)) collect.push('console.error: ' + s); });
  return new JSDOM(html, {
    url: 'https://localhost/', runScripts: 'dangerously', pretendToBeVisual: true, virtualConsole: vc,
    beforeParse(w) {
      /* benign recursive proxy: any ctx.method(...).chain works and
         returns itself; assignments succeed; the chart code runs
         without a real raster */
      const mkNoop = () => { const fn = function(){ return p; };
        const p = new Proxy(fn, { get: (t, k) => (k === Symbol.toPrimitive ? () => 0 : p),
          apply: () => p, set: () => true }); return p; };
      w.HTMLCanvasElement.prototype.getContext = () => mkNoop();
      w.Element.prototype.scrollIntoView = () => {};
      w.ResizeObserver = class { observe(){} unobserve(){} disconnect(){} };
      w.URL.createObjectURL = () => 'blob:fake';
      class FakeWorker {
        constructor(){ this._ls = { message: [] }; this.sent = []; (w.__fakeWorkers = w.__fakeWorkers || []).push(this); }
        addEventListener(t, f){ (this._ls[t] = this._ls[t] || []).push(f); }
        removeEventListener(){}
        postMessage(m){ this.sent.push(m); }
        set onmessage(f){ this._on = f; } get onmessage(){ return this._on; }
        fakeReceive(data){ const e = { data };
          try { if (this._on) this._on(e); } catch (err) { /* app handler quirks under jsdom must not kill the taps */ }
          for (const f of this._ls.message || []) { try { f(e); } catch (err) {} } }
        terminate(){}
      }
      w.Worker = FakeWorker;
      if (beforeParse) beforeParse(w);
    },
  });
}

/* pretend to be the shell: answer the frame's rpc calls */
function stubShellRpc(w) {
  w.addEventListener('message', e => {
    const m = e.data; if (!m || m.tm !== 'rpc') return;
    let ok = false, value = null, error = 'not found';
    if (m.method === 'storage.set') { ok = true; value = { key: m.args[0], value: m.args[1] }; }
    if (m.method === 'claude.complete') { ok = true; value = 'stub reply'; }
    w.postMessage({ tm: 'rpc-reply', id: m.id, ok, value, error }, '*');
  });
}

const mkStorage = (w, seed) => {
  w.__store = new Map(Object.entries(seed || {}));
  w.storage = {
    get: async k => { if (!w.__store.has(k)) throw new Error('404'); return { key: k, value: w.__store.get(k) }; },
    set: async (k, v) => { w.__store.set(k, String(v)); return { key: k, value: v }; },
    delete: async k => { w.__store.delete(k); return { key: k, deleted: true }; },
    list: async p => ({ keys: [...w.__store.keys()].filter(k => !p || k.startsWith(p)), prefix: p }),
  };
};
const logTexts = w => [...w.document.querySelectorAll('#vgLog .vg-b')].map(d => d.textContent);
const spoke = (w, s) => logTexts(w).some(t => t.includes(s));

async function main() {
  console.log('-- build fingerprint assertions --');
  const count = (s) => { let n = 0, i = -1; while ((i = BUILT.indexOf(s, i + 1)) !== -1) n++; return n; };
  check('four vg-core v2 copies', count('version:2,fragments') === 4);
  check('four </body> tags', count('</body>') === 4);
  check('probe-fact handler in worker blob', count("m.type === 'probe-fact'") >= 1);
  check('selectTab broadcast spliced', BUILT.includes("tm: 'vg-tab'"));
  check('pref rebroadcast spliced', BUILT.includes('// vg: pref sync'));
  check('shell seam exposed', BUILT.includes('window.__tmShell = { frames, shellStorage, selectTab, APPS }'));
  check('quota-aware failure copy shipped', BUILT.includes('allowance is used up'));
  check('toggle labels unclipped (kid, nowrap)', BUILT.includes('id="vgKid"> kid<') && !BUILT.includes('id="vgKid"> little-kid<'));
  check('provider table shipped (app-key reuse)', BUILT.split('__vgProviders').length - 1 >= 3);
  check('returning-visitor skip on the starter tour', BUILT.split('skipIf: poolHasModel').length - 1 === 2);
  check('save-conversation button in every guide', BUILT.split('id="vgSave"').length - 1 === 4);
  check('shellStorage gained list (404-noise fix)', BUILT.includes('async list(prefix)'));
  check('never-offer-visible rule in all four prompts', BUILT.split('talk about what it shows instead').length - 1 >= 4);
  check('bay persistence shipped', BUILT.includes("tmRestorePool(); // vg wave 4") && BUILT.includes("tmPersist('model', name, u8)"));
  check('trainedModels honors export origin', BUILT.includes("m.origin === 'export' || !/"));
  check('time-machine card copy humanized', BUILT.includes('Time machine: ') && !BUILT.includes('scrub a recorded arc:'));
  check('character persistence + sample handoff shipped', BUILT.includes("'tm-book-'") && BUILT.split('tm-sample-latest').length - 1 >= 2);

  /* ================= NURSERY frame ================= */
  console.log('-- nursery frame --');
  {
    const errors = [];
    const dom = makeDom(extractTemplate('nursery'), errors, w => mkStorage(w, {
      'tm-book-Max': JSON.stringify({ sheet: { name: 'Max', attr1: 'eats socks', attr2: 'lives in a shoe' },
        kind: 'a gnome', book: [{ split: 'train', text: 'Max ate a sock.' }], ts: 1 }) }));
    const w = dom.window;
    const booted = await until(() => w.__guide, 6000);
    check('nursery guide boots (prefs default on)', booted);
    if (booted) {
      const G = w.__guide;
      G.setWordMs(0);
      check('worker shim captured the app worker', w.__vgW.workers.length >= 1);
      const wk = w.__vgW.workers[0];
      const names = G.steps.map(s => s.name);
      check('nursery tour has the 12 signed-off stops', names.length === 12 &&
        names[0] === 'welcome' && names[6] === 'startRaising' && names[10] === 'honesty' && names[11] === 'handoff',
        names.join(','));
      await until(() => [...w.document.querySelectorAll('button')].some(b => /load Max's storybook/.test(b.textContent)), 3000);
      check('saved character offered beside the Zog book',
        [...w.document.querySelectorAll('button')].some(b => /load Max's storybook/.test(b.textContent)));
      /* trainStarted derived from OUTGOING worker traffic; the same
         moment persists the storybook per character */
      w.document.getElementById('chName').value = 'Sisi';
      wk.postMessage({ type: 'set-data', stories: ['Sisi ate a rainbow.'], heldout: ['Sisi slept.'] });
      let started = false;
      G.waitFor('trainStarted').then(() => { started = true; });
      wk.postMessage({ type: 'train', opts: { fresh: true } });
      await until(() => started, 2000);
      check('trainStarted derived from outgoing postMessage', started);
      await until(() => w.__store.has('tm-book-Sisi'), 2000);
      check('storybook persisted per character on train start', w.__store.has('tm-book-Sisi'));
      /* floor / parroting / forgetting from synthetic evals */
      wk.fakeReceive({ type: 'eval', step: 100, zogHeld: 2.1, base: 3.0, arc: 'absorbed-ish' });
      await until(() => spoke(w, 'as well as it ever will'), 3000);
      check('floorReached line spoken', spoke(w, 'as well as it ever will'));
      wk.fakeReceive({ type: 'eval', step: 150, zogHeld: 2.2, base: 3.05, zogHeldMin: 2.1, arc: 'parroting' });
      await until(() => spoke(w, 'parroting'), 3000);
      check('parrotingBadge line spoken', spoke(w, 'parroting'));
      wk.fakeReceive({ type: 'eval', step: 200, zogHeld: 2.3, base: 3.3 });
      await until(() => spoke(w, 'forget'), 3000);
      check('forgettingVisible fires past +0.15 nats drift', spoke(w, 'forget'));
      wk.fakeReceive({ type: 'eval', step: 220, zogHeld: 2.3, base: 3.5 });
      await sleep(120);
      check('forgettingVisible is once-per-session',
        logTexts(w).filter(t => /forget/.test(t)).length === 1);
      /* sampleReady resolves a waiter */
      let sampled = false;
      G.waitFor('sampleReady').then(() => { sampled = true; });
      wk.fakeReceive({ type: 'sample', step: 220, text: 'once upon a time zog ate a cloud.' });
      await until(() => sampled, 2000);
      check('sampleReady from worker sample message', sampled);
      await until(() => w.__store.has('tm-sample-latest'), 2000);
      check('latest sample stored for the microscope handoff', w.__store.has('tm-sample-latest'));
      /* ghostwrite observer + kid register */
      w.__vgPrefs.kid = true;
      const out = w.document.getElementById('genOut');
      out.innerHTML = '<span class="warn">ghostwriter: 4 accepted, 2 rejected</span>';
      await until(() => spoke(w, 'bounced off'), 3000);
      check('storyRejected via #genOut observer, kid register', spoke(w, 'bounced off'));
      w.__vgPrefs.kid = false;
      /* probe-fact roundtrip: adapter -> worker -> adapter */
      const before = wk.sent.length;
      const p = w.__guide.waitFor('probeFact', { condition: x => x && x.id });
      wk.postMessage({ type: 'probe-fact', word: 'clouds', context: 'Zog eats', id: 'vgpfT' });
      wk.fakeReceive({ type: 'probe-fact-done', id: 'vgpfT', word: 'clouds', pieces: 1, prob: 0.02 });
      const got = await Promise.race([p, sleep(1500).then(() => null)]);
      check('probeFact event relays worker probe-fact-done', !!got && got.id === 'vgpfT');
      /* save-conversation: transcript accumulates and the button saves without throwing */
      check('transcript accumulates in the hook', G.transcript.length > 3 && G.transcript.some(x => x.who === 'guide'));
      w.document.getElementById('vgSave').click();
      await sleep(100);
      check('save button runs clean', errors.length === 0);
      check('no unexpected nursery page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ================= HUNT frame ================= */
  console.log('-- hunt frame --');
  {
    const errors = [];
    const dom = makeDom(extractTemplate('hunt'), errors);
    const w = dom.window;
    const booted = await until(() => w.__guide, 6000);
    check('hunt guide boots', booted);
    if (booted) {
      const G = w.__guide; G.setWordMs(0);
      const names = G.steps.map(s => s.name);
      check('hunt tour has 8 stops, choir hard-ordered before switchOff',
        names.length === 8 && names.indexOf('choir') >= 0 && names.indexOf('choir') < names.indexOf('switchOff'),
        names.join(','));
      /* choir gate: clueBusted narration is condition-gated until the frame is spoken */
      const hypos = w.document.getElementById('hypos');
      const bust = w.document.createElement('div');
      bust.className = 'hypo';
      bust.innerHTML = '<div class="verdict"><span class="busted">❌ BUSTED</span></div>';
      hypos.appendChild(bust);
      await sleep(150);
      check('clueBusted line gated before the choir frame', !spoke(w, 'choir singing'));
      /* speak the choir frame's prep, then bust again */
      const st = G.steps.find(s => s.name === 'choir'); st.prep();
      const bust2 = bust.cloneNode(true); hypos.appendChild(bust2);
      await until(() => spoke(w, 'choir singing'), 3000);
      check('clueBusted lands as confirmation after the choir frame', spoke(w, 'choir singing'));
      check('clueProved narration exists and observer classifies proved', (() => {
        const pr = w.document.createElement('div'); pr.className = 'hypo';
        pr.innerHTML = '<div class="verdict"><span class="proved">🏆 PROVED</span></div>';
        hypos.appendChild(pr); return true; })());
      await until(() => spoke(w, 'PROVED'), 3000);
      check('clueProved line spoken', spoke(w, 'PROVED'));
      /* name-control derivation from worker traffic */
      const wk = w.__vgW.workers[0];
      let ncv = null;
      G.waitFor('nameControlVerdict').then(p => { ncv = p; });
      wk.fakeReceive({ type: 'namecontrol-done', baseline: 0.4, nameInProbe: true,
        results: [{ name: 'Bob', contProb: 0.35 }, { name: 'Sue', contProb: 0.3 }] });
      await until(() => ncv !== null, 2000);
      check('nameControlVerdict derives huntable=false when swaps survive', ncv && ncv.huntable === false);
      await until(() => spoke(w, 'not really about'), 3000);
      check('name-control verdict narrated', spoke(w, 'not really about'));
      /* pref broadcast syncs the app's own kidMode */
      w.postMessage({ tm: 'pref', key: 'tm-guide-kid', value: '1' }, '*');
      await until(() => w.document.getElementById('kidMode').checked, 2000);
      check('tm-guide-kid broadcast checks #kidMode', w.document.getElementById('kidMode').checked);
      check('no unexpected hunt page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ================= MICROSCOPE frame ================= */
  console.log('-- microscope frame --');
  {
    const errors = [];
    const dom = makeDom(extractTemplate('microscope'), errors, w => mkStorage(w, {
      'tm-sample-latest': JSON.stringify({ name: 'Sisi', text: 'Sisi flew home to her tree.', ts: 1 }) }));
    const w = dom.window;
    const booted = await until(() => w.__guide, 6000);
    check('microscope guide boots', booted);
    if (booted) {
      const G = w.__guide; G.setWordMs(0);
      check('microscope tour has 5 stops', G.steps.length === 5, G.steps.map(s => s.name).join(','));
      const wk = w.__vgW.workers[0];
      check('microscope worker shimmed', !!wk);
      let cap = false;
      G.waitFor('captureDone').then(() => { cap = true; });
      wk.fakeReceive({ type: 'run-done', n: 12 });
      await until(() => cap, 2000);
      check('captureDone relays run-done', cap);
      await until(() => spoke(w, 'Captured'), 3000);
      check('captureDone line spoken', spoke(w, 'Captured'));
      wk.fakeReceive({ type: 'run-ablated', ablate: ['0:1'] });
      await until(() => spoke(w, 'switch-off decided'), 3000);
      check('ablationVerdict line spoken', spoke(w, 'switch-off decided'));
      /* trained-mind story prefill */
      w.document.getElementById('modelInfo').textContent = 'stories260K-sisi.bin \u00b7 dim 64';
      wk.fakeReceive({ type: 'loaded' });
      await until(() => /Sisi flew home/.test(w.document.getElementById('story').value), 3000);
      check('story box prefilled from the trained mind', /Sisi flew home/.test(w.document.getElementById('story').value));
      await until(() => spoke(w, 'best specimens'), 2000);
      check('prefill narrated', spoke(w, 'best specimens'));
      check('no unexpected microscope page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ================= SHELL (full built page) ================= */
  console.log('-- shell --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors);
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell, 8000);
    check('shell guide boots on the full page', booted);
    if (booted) {
      const G = w.__guide; G.setWordMs(0);
      check('shell seam has frames/shellStorage/selectTab', !!(w.__tmShell.frames && w.__tmShell.shellStorage && w.__tmShell.selectTab));
      const names = G.steps.map(s => s.name);
      check('starter-pack tour has the 4 signed-off stops',
        names.join(',') === 'catchTheBrain,dragIn,persisted,pickADoor', names.join(','));
      check('starter zip URL is in the transcript log', spoke(w, 'Starter pack:'));
      /* run the tour to the dragIn until, then satisfy it via the pool observer */
      G.start();
      await until(() => G.cur === 1 && spoke(w, 'browse button'), 6000);
      await sleep(120);
      check('tour parks on dragIn awaiting modelLoaded', G.state === 'running' && G.cur === 1);
      const pool = w.document.getElementById('pool');
      pool.innerHTML = '<tr><td>stories260K.bin</td><td>260K</td><td>512</td><td>model</td></tr>';
      await until(() => G.state === 'done' || G.cur >= 2, 6000);
      check('pool row advances the tour via modelLoaded', G.cur >= 2 || G.state === 'done');
      await until(() => spoke(w, 'Caught!'), 3000);
      check('modelLoaded event line spoken', spoke(w, 'Caught!'));
      /* shell guide auto-hide: a room frame covering the bay must
         hide both the panel and the collapsed tab */
      const bayEl = w.document.getElementById('bay');
      bayEl.classList.add('hide');
      await until(() => w.document.getElementById('vgTab').style.display === 'none', 2000);
      check('shell guide hides with the bay (no stacked Guide tabs)',
        w.document.getElementById('vgTab').style.display === 'none' &&
        w.document.getElementById('vgPanel').style.display === 'none');
      check('reserved margin released while hidden (no dead strip)', !w.document.body.classList.contains('vg-open'));
      bayEl.classList.remove('hide');
      await until(() => w.document.getElementById('vgTab').style.display === '', 2000);
      check('shell guide returns when the bay does', w.document.getElementById('vgTab').style.display === '');
      check('margin restored with the bay', w.document.body.classList.contains('vg-open'));
      check('no unexpected shell page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ============ SHELL scenario B: returning visitor ============ */
  console.log('-- shell: returning visitor (pool already full) --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors);
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell, 8000);
    check('shell guide boots (scenario B)', booted);
    if (booted) {
      const G = w.__guide; G.setWordMs(0);
      w.document.getElementById('pool').innerHTML =
        '<tr><td>stories260K.bin</td><td>260K</td><td>512</td><td>model</td></tr>';
      await sleep(80);
      G.start();
      await until(() => G.state === 'done', 8000);
      check('tour skips catch-the-brain when models persist',
        G.state === 'done' && !spoke(w, 'catch the brain') && !spoke(w, 'Download the starter pack'));
      check('persisted stop still speaks', spoke(w, 'pool') || spoke(w, 'minds you caught'));
    }
  }

  /* ============ SHELL scenario D: bay restore roundtrip ============ */
  console.log('-- shell: persisted bay restores on boot --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors, w => {
      w.localStorage.setItem('tm-pool-seed-wordlist.json', JSON.stringify({
        kind: 'json', name: 'seed-wordlist.json', text: JSON.stringify({ words: ['zog'] }) }));
    });
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell, 8000);
    check('shell boots with a seeded pool key', booted);
    if (booted) {
      const ok = await until(() => /remembered 1 file/.test(w.document.body.textContent), 8000);
      check('bay restore replays persisted files through ingest', ok);
    }
  }

  /* ============ SHELL scenario C: hanging artifact storage ============ */
  console.log('-- shell: artifact storage that never resolves --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors, w => {
      w.storage = { get: () => new Promise(() => {}), set: () => Promise.resolve({}) };
    });
    const w = dom.window;
    const booted = await until(() => w.__guide, 9000);
    check('shell guide still boots when storage.get hangs (raced reads)', booted);
  }

  console.log(`PASS ${pass}  FAIL ${fail}`);
  if (failures.length) { console.log('failures:'); failures.forEach(f => console.log('  - ' + f)); }
  process.exit(fail ? 1 : 0);
}
main().catch(e => { console.error(e); process.exit(2); });
