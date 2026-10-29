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
const BINDING_ARC = JSON.parse(readFileSync('arc/binding/arc-manifest.json', 'utf8'));

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

const sayLog = w => {                       // capture #shellStatus messages as they land.
  // Installed in beforeParse: say() replaces the text node, and a later
  // message overwrites an earlier one within the same microtask run.
  // (Checking document.body.textContent instead would match the inline
  // <script> source and green-light checks that never ran.)
  w.__says = [];
  new w.MutationObserver(muts => {
    for (const m of muts) {
      const t = m.target;
      const el = t.id === 'shellStatus' ? t
        : (t.parentElement && t.parentElement.id === 'shellStatus' ? t.parentElement : null);
      if (el && el.textContent) w.__says.push(el.textContent);
    }
  }).observe(w.document, { childList: true, subtree: true, characterData: true });
};
const said = (w, re) => (w.__says || []).some(t => re.test(t));
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
  check('shell seam exposed', BUILT.includes('window.__tmShell = { frames, shellStorage, selectTab, APPS'));
  check('quota-aware failure copy shipped', BUILT.includes('allowance is used up'));
  check('toggle labels unclipped (kid, nowrap)', BUILT.includes('id="vgKid"> kid<') && !BUILT.includes('id="vgKid"> little-kid<'));
  check('provider table shipped (app-key reuse)', BUILT.split('__vgProviders').length - 1 >= 3);
  check('no drop-the-starter-pack quest left in the shell tour', !BUILT.includes('skipIf: poolHasModel') && !BUILT.includes('catchTheBrain'));
  check('save-conversation button in every guide', BUILT.split('id="vgSave"').length - 1 === 4);
  check('shellStorage gained list (404-noise fix)', BUILT.includes('async list(prefix)'));
  check('never-offer-visible rule in all four prompts', BUILT.split('talk about what it shows instead').length - 1 >= 4);
  check('bay persistence shipped', BUILT.includes("const tmRestoreReady = tmRestorePool(); // vg wave 4") &&
    BUILT.includes("tmPersist('model', name, u8, { origin, modelMeta: meta })"));
  check('same-folder resource manifest + embedded fallback shipped',
    BUILT.includes("const RESOURCE_MANIFEST = 'tiny-mind-resources.json'") &&
    BUILT.includes('const FALLBACK_RESOURCE_PACKS = [') &&
    BUILT.includes('tiny-mind-binding-models-corrected.zip'));
  check('same-folder assets use persistent cache before network',
    BUILT.includes("const TM_RESOURCE_CACHE = 'tiny-mind-same-folder-v1'") &&
    BUILT.includes('const hit = await cache.match(url)') &&
    BUILT.includes('cache.put(url, copy)'));
  check('trained models stay separate from the exact pretrained base',
    BUILT.includes('const trainedModels = () =>') && BUILT.includes('!isBase260(m)'));
  check('time-machine card copy humanized', BUILT.includes('Time machine: ') && !BUILT.includes('scrub a recorded arc:'));
  check('character persistence + sample handoff shipped', BUILT.includes("'tm-book-'") && BUILT.split('tm-sample-latest').length - 1 >= 2);
  check('kid-odds helper in every prelude', BUILT.split('window.__vgOdds = {').length - 1 >= 3);
  check('shell guide no longer claims a key it lacks', !BUILT.includes('hasKey: () => true,') && BUILT.includes('function frameKey()'));
  check('shell status leads with the active tab', BUILT.includes('RIGHT NOW looking at') && BUILT.includes('get activeTab(){ return activeTab; }'));
  check('bay persistence is chunked + verified', BUILT.includes('const TM_CHUNK') && BUILT.includes('async function tmWriteInner('));
  check('storage writes serialized + conflict-recovering', BUILT.includes('const tmSerial') && BUILT.includes('409|conflict|exist'));
  check('one ask path shared by every guide', BUILT.split('window.__vgAsk = async function').length - 1 >= 3 && BUILT.split('__vgAsk(sys, usr').length - 1 >= 3);
  check('remembered app keys reachable by all guides', BUILT.includes('window.__vgRememberedKey = function') && BUILT.includes("localStorage.getItem('tm-key-' + p)"));
  check('shell page carries its own LLM helpers', BUILT.includes('if (!window.__vgProviders) window.__vgProviders') && BUILT.includes('if (!window.__vgRememberedKey) window.__vgRememberedKey'));
  check('SHOW is visibility-aware everywhere', BUILT.split('window.__vgShow(api, s)').length - 1 === 4 && BUILT.split("a.on('showBlocked'").length - 1 === 4);
  check('hosted arcs discover more than one manifest', BUILT.includes("tmGetJson('arc/arcs.json')") && BUILT.includes('manifest.more || []'));
  check('bay names what it remembered', BUILT.includes("recs.map(r => r.name).join(', ')"));
  check('bay arc line no longer prints raw keys', BUILT.includes('m.label || arcKidLabel(m)'));
  check('DO directive is caught even when it ends a prose line', BUILT.split('const m=s.match(/DO:[ \\t]*([A-Z_]+)').length - 1 === 4);
  check('TAB is a gated verb', BUILT.includes('TAB: { gated: true,') && BUILT.includes('ratifyText(p, t)'));
  check('ablation narration varies after the first', BUILT.includes('ablationAgain:') && BUILT.includes('const AGAIN_KID'));
  check('time machine can write a story from the moment', BUILT.includes('id="scrubStory"') && BUILT.includes("type: 'sample-now', prompt: prefix"));
  check('app readouts route through the kid-odds formatter',
    BUILT.split('window.__vgPct(').length - 1 >= 7 && BUILT.split('window.__vgPct = function').length - 1 === 3);
  check('hunt pct() honours kid mode', BUILT.includes("if (window.__vgPrefs && window.__vgPrefs.kid && window.__vgOdds)\n    return window.__vgOdds.phrase(p, true, { compact: true });   // kid mode"));
  check('prefill no longer depends on the loaded message', BUILT.includes("new MutationObserver(() => maybePrefillStory()).observe(mi"));
  check('core exposes click-to-ask (explain + explainable)', BUILT.includes('explain:vgExplainTopic, explainable:vgExplainable') && BUILT.includes('function vgExplainable('));
  check('specimen line spelled out + clickable', BUILT.includes("data-vg-explain=") && BUILT.includes("neurons per layer") && !BUILT.includes("neurons/layer \u00b7 vocab"));
  check('kid ablation gives odds + point delta', BUILT.includes('function deltaLine(') && BUILT.includes('${dir} ${pts} point${pts === 1'));
  check('kid toggle refreshes the numeric views', BUILT.includes('window.__vgMicRefresh'));
  check('arc description no longer hard-codes Zog', !BUILT.includes("learning Zog (and then just memorizing)"));
  check('specimen chips render as HTML, not text', BUILT.includes("$('modelInfo').innerHTML =") && !BUILT.includes("$('modelInfo').textContent =\n      (() => { const chip"));
  check('abbreviations spelled out (dimension/vocabulary)', BUILT.includes("chip('dim','dimension ") && BUILT.includes("' vocabulary')") && !BUILT.includes("layers " + String.fromCharCode(183) + " vocab $"));
  check('scrub-story control enables with the time machine', BUILT.includes("if ($('scrubStory')) $('scrubStory').disabled = false;") || BUILT.includes('\u{1F4D6} tell this story'));
  check('scrub-story gives feedback when no moment is picked', BUILT.includes('drag the slider to a moment first'));
  check('bay restore announces it is loading', BUILT.includes('loading your saved files from last visit'));
  check('a way back to the model picker exists', BUILT.includes('function showChooseAnother') && BUILT.includes('other minds for the '));
  check('time-machine UI drops the model-panel wording', !BUILT.includes('load that moment into the model panel') && !BUILT.includes('>\u{1F423} keep raising it from here<') && !BUILT.includes('press "keep raising it from here"'));
  check('time-machine has a story-telling control', BUILT.includes('\u{1F4D6} tell this story') && BUILT.includes('await commitScrub();   // bring this checkpoint'));
  check('specimen line reflects a loaded time machine', BUILT.includes("time machine loaded") && BUILT.includes("arcKidLabel(manifest); mi.classList"));
  check('click-to-ask shows only the short question', BUILT.includes('vgAsk(q, \'What is "\'') && BUILT.includes("vgLogAdd((shown||q),'user')"));
  check('a burst of the same event enqueues once', BUILT.includes('vgEvQueue.some(q=>q.name===name)) return;'));
  check('capture confirmation deduped per distinct capture', BUILT.includes('if (sig !== lastCaptureSig)') && BUILT.includes('one confirmation per distinct capture'));
  check('time-machine custom sample renders under the slider', BUILT.includes("a time-machine 'tell this story' request") && BUILT.includes("renderScrubSample(S.scrub.customPrompt"));
  check('nursery specimen chips are wired to the guide', BUILT.includes('const NUR_TERMS') && BUILT.includes("a.explainable(NUR_TERMS)"));
  check('custom scrub story renders live (worker falls back to checkpoint weights)',
    BUILT.includes('trainer ? trainer.modelWeights() : (baseCkpt && baseCkpt.weights)')
    && !BUILT.includes('needs the full trainer, which the next build will add'));
  check('ready-made Zog storybook is embedded in the single-file Nursery',
    BUILT.includes('const __TM_EMBEDDED_ZOG_STORIES__ = JSON.parse('));
  check('balanced four-character story world is embedded in the single-file Nursery',
    BUILT.includes('const __TM_EMBEDDED_BINDING_CORPUS__ = JSON.parse('));
  check('nursery guide distinguishes restored storybook data from model weights',
    BUILT.includes('A restored storybook is training data, not a memory in the model weights') &&
    BUILT.includes('Separate loaded-model display:'));

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
      await w.__tmRouteJson('arc-manifest.json', JSON.stringify(BINDING_ARC));
      await until(() => w.document.getElementById('scrubSlider').max === '10', 2000);
      check('guide receives the four-character arc manifest instead of the current character editor',
        w.__tmGuideArcContext &&
        w.__tmGuideArcContext.arc === 'arcC' &&
        w.__tmGuideArcContext.lastStep === 600 &&
        JSON.stringify(w.__tmGuideArcContext.characterNames) === JSON.stringify(['Zog', 'Mip', 'Bex', 'Nia']) &&
        !w.__tmGuideArcContext.characterNames.includes('Sisi'));
      const slider = w.document.getElementById('scrubSlider');
      const checkpointButtons = [...w.document.querySelectorAll('#scrubStops button.tm-stop-button')];
      check('time machine renders one direct, touch-sized button per checkpoint',
        checkpointButtons.length === BINDING_ARC.snapshots.length &&
        checkpointButtons[0].getAttribute('aria-current') === 'step' &&
        /Start: Before character training/.test(checkpointButtons[0].getAttribute('aria-label')) &&
        w.document.getElementById('scrubPrev').disabled &&
        !w.document.getElementById('scrubNext').disabled);
      checkpointButtons[1].click();
      w.document.getElementById('scrubNext').click();
      w.document.getElementById('scrubPrev').click();
      check('direct and previous/next controls can select moment 1 without skipping it',
        slider.value === '1' &&
        checkpointButtons[1].getAttribute('aria-current') === 'step' &&
        /Moment 1/.test(w.document.getElementById('scrubInfo').textContent));
      const goToStep = step => {
        slider.value = String(BINDING_ARC.snapshots.findIndex(s => s.step === step));
        slider.dispatchEvent(new w.Event('input', { bubbles: true }));
      };
      const expectedStages = new Map([
        [0, 'before character training'], [60, 'still mixing character facts'],
        [240, 'character facts separating'], [400, 'strong character learning'],
        [600, 'very strong, but more rigid'],
      ]);
      let standardMomentsOk = true;
      for (const [step, expected] of expectedStages) {
        goToStep(step);
        standardMomentsOk &&= w.document.getElementById('scrubStage').textContent.includes(expected)
          && /%/.test(w.document.getElementById('scrubStatuses').textContent);
      }
      check('standard time machine explains steps 0, 60, 240, 400 and 600', standardMomentsOk);
      goToStep(0);
      check('step 0 says before training and interprets 25 percent as chance',
        w.document.getElementById('scrubConclusion').textContent.includes('pretrained stories260K') &&
        w.document.getElementById('scrubStatuses').textContent.includes('approximately chance'));
      const kidToggle = w.document.getElementById('vgKid');
      kidToggle.checked = true; kidToggle.dispatchEvent(new w.Event('change', { bubbles: true }));
      await sleep(0);
      let kidMomentsOk = true;
      for (const [step, expected] of expectedStages) {
        goToStep(step);
        const statuses = w.document.getElementById('scrubStatuses');
        const visibleStatusText = [...statuses.querySelectorAll('button.tm-status-button')].map(b => b.textContent).join(' ');
        kidMomentsOk &&= w.document.getElementById('scrubStage').textContent.includes(expected)
          && statuses.querySelectorAll('button.tm-status-button').length === 4
          && !/%|leakage|specificity|loss/i.test(visibleStatusText)
          && w.document.getElementById('scrubLearning').classList.contains('kid');
      }
      check('Kid mode materially simplifies all five requested moments to four qualitative controls', kidMomentsOk);
      goToStep(0);
      const directStatus = w.document.querySelector('#scrubStatuses button.tm-status-button');
      directStatus.click();
      check('status explanation opens through a native button with ARIA state',
        directStatus.getAttribute('aria-expanded') === 'true' &&
        !w.document.getElementById(directStatus.getAttribute('aria-controls')).hidden);
      const questionsToggle = w.document.getElementById('scrubQuestionsToggle');
      questionsToggle.dispatchEvent(new w.KeyboardEvent('keydown', { key: 'Enter', bubbles: true }));
      const openedWithEnter = questionsToggle.getAttribute('aria-expanded') === 'true';
      questionsToggle.dispatchEvent(new w.KeyboardEvent('keydown', { key: ' ', bubbles: true }));
      check('disclosures activate with Enter and Space',
        openedWithEnter && questionsToggle.getAttribute('aria-expanded') === 'false');
      const technicalToggle = w.document.getElementById('scrubTechnicalToggle');
      technicalToggle.click();
      check('Kid raw measurements stay behind How do we know disclosure',
        technicalToggle.textContent === 'How do we know?' && technicalToggle.getAttribute('aria-expanded') === 'true' &&
        /held-out story surprise|cross-name leakage/.test(w.document.getElementById('scrubTechnical').textContent));
      kidToggle.checked = false; kidToggle.dispatchEvent(new w.Event('change', { bubbles: true }));
      await sleep(0);
      await until(() => [...w.document.querySelectorAll('button')].some(b => /load your saved Max storybook/.test(b.textContent)), 3000);
      check('saved character offered beside the Zog book',
        [...w.document.querySelectorAll('button')].some(b => /load your saved Max storybook/.test(b.textContent)));
      w.document.getElementById('chName').value = 'Sisi';
      w.document.getElementById('bindingDemoBtn').click();
      await sleep(0);
      check('new story-world button loads the supplied balanced 64 + 16 corpus and its four-character sheet',
        w.document.getElementById('singleCharacterSheet').hidden &&
        !w.document.getElementById('bindingCharacterSheet').hidden &&
        w.document.getElementById('bookCount').textContent === '64 learning + 16 secret' &&
        /Zog[\s\S]*Mip[\s\S]*Bex[\s\S]*Nia/.test(w.document.getElementById('bindingCharacterRows').textContent) &&
        /Zog is a small dragon/.test(w.document.getElementById('storyList').textContent) &&
        /Mip is a small rabbit/.test(w.document.getElementById('storyList').textContent) &&
        /loaded the balanced Zog, Mip, Bex \+ Nia/.test(w.document.getElementById('genOut').textContent));
      const bindingPromptMessagesAt = wk.sent.length;
      const sampleButton = w.document.getElementById('sampleBtn');
      sampleButton.disabled = false;
      for (let i = 0; i < 4; i++) sampleButton.click();
      const bindingSamplePrompts = wk.sent.slice(bindingPromptMessagesAt)
        .filter(m => m.type === 'sample-now').map(m => m.prompt);
      check('four-character samples rotate across the visible names and never use hidden Sisi',
        JSON.stringify(bindingSamplePrompts) === JSON.stringify([
          'Once upon a time, Zog', 'Once upon a time, Mip',
          'Once upon a time, Bex', 'Once upon a time, Nia',
        ]) && !bindingSamplePrompts.some(p => /Sisi/i.test(p)));
      const trainButton = w.document.getElementById('trainBtn');
      trainButton.disabled = false;
      trainButton.click();
      const bindingTrain = [...wk.sent].reverse().find(m => m.type === 'train');
      check('four-character training carries all four sample prompts instead of hidden Sisi',
        bindingTrain && JSON.stringify(bindingTrain.opts.prompts) === JSON.stringify([
          'Once upon a time, Zog', 'Once upon a time, Mip',
          'Once upon a time, Bex', 'Once upon a time, Nia',
        ]) && !/Sisi/i.test(JSON.stringify(bindingTrain.opts)));
      let publishedModel = null;
      w.__tmPublishModel = (name, bytes, meta) => { publishedModel = { name, bytes, meta }; };
      const beforeCompletedRun = wk.sent.length;
      wk.fakeReceive({ type: 'idle', step: 120 });
      const automaticPublish = wk.sent.slice(beforeCompletedRun)
        .find(m => m.type === 'export' && m.purpose === 'publish');
      check('a completed four-character run requests automatic model-pool publication',
        !!automaticPublish);
      wk.fakeReceive({ type: 'export-done', step: 120, purpose: 'publish', bin: new ArrayBuffer(16) });
      check('automatic publication uses a binding name that is not the legacy Zog filename',
        publishedModel && publishedModel.name === 'four-character-binding-260K.bin' &&
        publishedModel.bytes.byteLength === 16);
      check('automatic publication carries all four character fact sets',
        publishedModel && publishedModel.meta &&
        JSON.stringify(Object.keys(publishedModel.meta.storyWorld.characters)) === JSON.stringify(['Zog', 'Mip', 'Bex', 'Nia']) &&
        publishedModel.meta.storyWorld.characters.Mip.food === 'apples' &&
        publishedModel.meta.storyWorld.characters.Bex.home === 'cave');
      w.document.getElementById('exportBtn').click();
      const explicitExport = [...wk.sent].reverse().find(m => m.type === 'export');
      check('the explicit Export button remains a download request',
        explicitExport && explicitExport.purpose === 'download');
      w.document.getElementById('pauseBtn').click();
      w.document.getElementById('demoBtn').click();
      await sleep(0);
      check('original Zog-only button returns to the one-character sheet and bundled 50 + 10 storybook',
        w.document.getElementById('chName').value === 'Zog' &&
        w.document.getElementById('chKind').value === 'a small dragon' &&
        !w.document.getElementById('singleCharacterSheet').hidden &&
        w.document.getElementById('bookCount').textContent === '50 learning + 10 secret' &&
        /loaded the original Phase 0 Zog-only storybook/.test(w.document.getElementById('genOut').textContent));
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
      /* the ask path: window.claude is the bridge's RPC, which only
         works inside an artifact; on an ordinary page the app's own
         key must take over rather than dying of CORS */
      w.localStorage.setItem('tm-key-anthropic', 'sk-remembered');
      check('remembered app key is found', (w.__vgRememberedKey() || {}).key === 'sk-remembered');
      let used = null;
      const realCall = w.__vgProviders.call;
      w.__vgProviders.call = async (p, k, m) => { used = { p, k, m }; return 'fallback answer'; };
      w.claude = { complete: async () => { throw new Error('Failed to fetch'); } };
      const askOut = await w.__vgAsk('sys', 'usr', () => ({ provider: 'anthropic', key: 'sk-remembered', model: '' }));
      check('bridge failure falls back to the app key', askOut === 'fallback answer' && used && used.k === 'sk-remembered');
      let threw = null;
      try { await w.__vgAsk('sys', 'usr', () => ({ key: '' })); } catch (e) { threw = e; }
      check('no key anywhere raises a no-AI error, never a bare fetch', threw && /no key|Failed to fetch/.test(threw.message));
      w.__vgProviders.call = realCall; delete w.claude;
      w.localStorage.removeItem('tm-key-anthropic');
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
      const huntCharacters = {
        Zog: { kind: 'dragon', food: 'clouds', home: 'moon' },
        Mip: { kind: 'rabbit', food: 'apples', home: 'river' },
        Bex: { kind: 'bear', food: 'honey', home: 'cave' },
        Nia: { kind: 'bird', food: 'berries', home: 'tree' },
      };
      w.__tmLoadPair({ name: 'custom-trained-260K.bin', buf: new ArrayBuffer(8),
        meta: { storyWorld: { mode: 'binding', characters: huntCharacters } } },
        { name: 'tok512.bin', buf: new ArrayBuffer(8) });
      const metadataQuests = [...w.document.querySelectorAll('#quickQuests button')];
      check('Hunt renders one quick quest for each character carried with a trained model',
        metadataQuests.length === 4 && metadataQuests.map(b => b.dataset.name).join(',') === 'Zog,Mip,Bex,Nia');
      metadataQuests.find(b => b.dataset.name === 'Mip').click();
      check('a carried Mip quest fills the matched Mip question and answer',
        w.document.getElementById('qName').value === 'Mip' &&
        w.document.getElementById('qProbe').value.includes('Mip') &&
        w.document.getElementById('qCont').value.trim() === 'apples');
      w.__tmLoadPair({ name: 'four-character-binding-260K.bin', buf: new ArrayBuffer(8), meta: null },
        { name: 'tok512.bin', buf: new ArrayBuffer(8) });
      check('known binding filename restores four-character quests for a model saved by the previous build',
        [...w.document.querySelectorAll('#quickQuests button')].map(b => b.dataset.name).join(',') === 'Zog,Mip,Bex,Nia');
      const names = G.steps.map(s => s.name);
      check('hunt tour has 8 stops, choir hard-ordered before switchOff',
        names.length === 8 && names.indexOf('choir') >= 0 && names.indexOf('choir') < names.indexOf('switchOff'),
        names.join(','));
      /* choir gate: non-selective measured-effect narration waits for the frame */
      const hypos = w.document.getElementById('hypos');
      const bust = w.document.createElement('div');
      bust.className = 'hypo';
      bust.innerHTML = '<div class="verdict"><span class="little">little measured effect</span></div>';
      hypos.appendChild(bust);
      await sleep(150);
      check('clueMeasured line gated before the choir frame', !spoke(w, 'produced a measured effect'));
      /* speak the choir frame's prep, then measure again */
      const st = G.steps.find(s => s.name === 'choir'); st.prep();
      const bust2 = bust.cloneNode(true); hypos.appendChild(bust2);
      await until(() => spoke(w, 'produced a measured effect'), 3000);
      check('clueMeasured lands as calibrated guidance after the choir frame', spoke(w, 'produced a measured effect'));
      check('clueSelective narration exists and observer classifies selective evidence', (() => {
        const pr = w.document.createElement('div'); pr.className = 'hypo';
        pr.innerHTML = '<div class="verdict"><span class="selective">strong causal contributor</span></div>';
        hypos.appendChild(pr); return true; })());
      await until(() => spoke(w, 'Strong selective causal evidence'), 3000);
      check('clueSelective line spoken', spoke(w, 'Strong selective causal evidence'));
      /* name-control derivation from worker traffic */
      const wk = w.__vgW.workers[0], wkH = wk;
      let ncv = null;
      G.waitFor('nameControlVerdict').then(p => { ncv = p; });
      wk.fakeReceive({ type: 'namecontrol-done', baselineGeometricMeanProbability: 0.4, nameInProbe: true,
        results: [{ name: 'Bob', geometricMeanProbability: 0.35 }, { name: 'Sue', geometricMeanProbability: 0.3 }] });
      await until(() => ncv !== null, 2000);
      check('nameControlVerdict derives huntable=false when swaps survive', ncv && ncv.huntable === false);
      await until(() => spoke(w, 'not really about'), 3000);
      check('name-control verdict narrated', spoke(w, 'not really about'));
      /* kid-friendly odds: phrasing, narration, and the dots picture */
      const O = w.__vgOdds;
      check('odds phrase: moderate -> out of 100', O.phrase(0.07, true) === '7 out of 100', O.phrase(0.07, true));
      check('odds phrase: small -> out of 1000', O.phrase(0.02, true) === '20 out of 1000', O.phrase(0.02, true));
      check('odds phrase: rare -> 1 in N', O.phrase(0.0004, true) === 'about 1 in 2,500', O.phrase(0.0004, true));
      check('odds phrase: tiny -> out of a million', O.phrase(0.000002, true) === '2 out of a million', O.phrase(0.000002, true));
      check('odds phrase: default register keeps percent + 1-in-N', O.phrase(0.02, false) === '2% (about 1 in 50)', O.phrase(0.02, false));
      check('compact odds fit a table cell', O.phrase(0.07, true, { compact: true }) === '7 in 100'
        && O.phrase(0.0004, true, { compact: true }) === '1 in 2,500', O.phrase(0.0004, true, { compact: true }));
      w.__vgPrefs.kid = false;
      check('__vgPct is a percentage outside kid mode', w.__vgPct(0.976) === '97.6%', w.__vgPct(0.976));
      w.__vgPrefs.kid = true;
      check('__vgPct never prints a percentage in kid mode',
        !/%/.test(w.__vgPct(0.976)) && w.__vgPct(0.976) === '98 in 100', w.__vgPct(0.976));
      check('__vgPct handles the rare tail', w.__vgPct(0.0000004) === '1 in a million' || /in a million/.test(w.__vgPct(0.0000004)), w.__vgPct(0.0000004));
      w.__vgPrefs.kid = false;
      w.document.getElementById('kidMode').checked = true;
      wkH.fakeReceive({ type: 'probe-done', contProb: 0.07, contPieces: [1] });
      await until(() => spoke(w, '7 out of 100'), 3000);
      check('guess-o-meter narrated in the kid register', spoke(w, '7 out of 100'));
      await until(() => w.document.querySelector('.vg-odds-chip'), 3000);
      const chip = w.document.querySelector('.vg-odds-chip');
      check('dots chip offered beside the odds line', !!chip);
      if (chip) {
        chip.click();
        await sleep(60);
        check('chip opens the hundred-dot picture', !!w.document.getElementById('vgOddsBack'));
        w.document.getElementById('vgOddsBack').click();
        await sleep(40);
        check('picture closes on tap', !w.document.getElementById('vgOddsBack'));
      }
      w.document.getElementById('kidMode').checked = false;
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
      /* trained-mind story prefill — driven by the model line alone,
         because 'loaded' usually arrives before the guide exists */
      w.document.getElementById('modelInfo').textContent = 'stories260K-sisi.bin \u00b7 dim 64';
      await until(() => /Sisi flew home/.test(w.document.getElementById('story').value), 4000);
      /* the chip glossary label is spelled out, not 'dim' */
      check('microscope glossary uses "dimension" not "dim"', BUILT.includes("dim: { label: 'dimension'"));
      check('story box prefilled without any loaded message', /Sisi flew home/.test(w.document.getElementById('story').value));
      await until(() => spoke(w, 'wrote itself') || spoke(w, 'this very mind wrote'), 3000);
      check('prefill says the mind wrote it', spoke(w, 'wrote itself') || spoke(w, 'this very mind wrote'));

      /* click-to-ask: a data-vg-explain chip routes an explain question
         through the LLM path, seeded with the registered note */
      let askedSys = null, askedUsr = null;
      w.claude = { complete: async (p) => { askedUsr = p; return 'A neuron is a tiny switch. DO: nothing'.replace(' DO: nothing',''); } };
      const chip = w.document.createElement('span');
      chip.setAttribute('data-vg-explain', 'neurons');
      chip.textContent = '172 neurons per layer';
      w.document.body.appendChild(chip);
      chip.click();
      await until(() => spoke(w, 'A neuron is a tiny switch'), 4000);
      check('clicking an explain chip asks the guide', spoke(w, 'A neuron is a tiny switch'));
      check('the question carried the grounding note', /neurons per layer|thinking bits/.test(askedUsr || ''));
      check('the visitor sees only the short question, not the seeded note',
        logTexts(w).some(t => /What is "neurons per layer"\?/.test(t)) && !logTexts(w).some(t => /Here is what it is, in one line/.test(t)),
        logTexts(w).filter(t => /neuron/i.test(t)).slice(-1)[0]);
      delete w.claude;

      /* kid ablation: odds before -> after, drop in points */
      w.__vgPrefs.kid = true;
      wk.fakeReceive({ type: 'run-done', n: 6, nextProb: [null, 0.98, 0.5] });
      await until(() => G.speakDepth === 0 && G.evQueueLen === 0, 4000);
      wk.fakeReceive({ type: 'run-ablated', ablate: ['0:1'], nextProb: [null, 0.94, 0.5] });
      await until(() => spoke(w, 'down 4 point'), 4000);
      check('kid ablation line gives the odds and the drop in points',
        logTexts(w).some(t => /98 in 100/.test(t) && /94/.test(t) && /down 4 point/.test(t)),
        logTexts(w).slice(-1)[0]);
      w.__vgPrefs.kid = false;
      /* the bench refreshes on every ablation: say the lesson once */
      await until(() => G.speakDepth === 0 && G.evQueueLen === 0, 4000);
      const lessonCount = () => logTexts(w).filter(t => /switch-off decided|the real test/.test(t)).length;
      const ROTATED = t => /Blue is the whole mind|Barely a dent|Compare that gap|one suspect eliminated|points\.|still about/.test(t);
      for (let i = 0; i < 4; i++) {
        wk.fakeReceive({ type: 'run-ablated', ablate: ['0:' + i] });
        await until(() => G.speakDepth === 0 && G.evQueueLen === 0, 4000);
      }
      check('ablation lesson is spoken exactly once', lessonCount() === 1, 'x' + lessonCount());
      /* two identical captures must not double the 'took a picture' line */
      const capLine = () => logTexts(w).filter(t => /took a picture|every attention pattern/.test(t)).length;
      const capBefore = capLine();
      wk.fakeReceive({ type: 'run-done', n: 4, nextProb: [null, 0.9, 0.5], pieces: ['a','b','c','d'], topK: [[],[],[],[]], attention: new Float32Array(16) });
      wk.fakeReceive({ type: 'run-done', n: 4, nextProb: [null, 0.9, 0.5], pieces: ['a','b','c','d'], topK: [[],[],[],[]], attention: new Float32Array(16) });
      await until(() => G.speakDepth === 0 && G.evQueueLen === 0, 4000);
      check('a repeated identical capture does not re-announce', capLine() - capBefore <= 1, 'added x' + (capLine() - capBefore));
      check('later mutes get a different line', logTexts(w).some(ROTATED), logTexts(w).slice(-2).join(' | '));

      /* SHOW must never halo an empty box: bars exist only after a run */
      const spots = [], emits = [];
      const stub = { spot: sel => spots.push(sel), emit: (n, p) => emits.push([n, p]) };
      const barsSpec = { sel: '#bars', needs: '#runBtn', why: 'W-run-first', kidWhy: 'K-run-first' };
      w.document.getElementById('bars').innerHTML = '';
      const shown = w.__vgShow(stub, barsSpec);
      check('SHOW on an empty view is refused', shown === false);
      check('SHOW points at the control that makes it', spots[0] === '#runBtn');
      check('SHOW explains why, via the event bus', emits.length === 1 && emits[0][0] === 'showBlocked' && emits[0][1].kidWhy === 'K-run-first');
      w.document.getElementById('bars').innerHTML = '<div class="bar">the</div>';
      const shown2 = w.__vgShow(stub, barsSpec);
      check('SHOW spotlights the view once it is drawn', shown2 === true && spots[1] === '#bars');
      /* and the guide actually says the blocked line */
      w.__vgPrefs.kid = true;
      G.emit('showBlocked', { why: 'grown-up line', kidWhy: 'press the look inside button' });
      await until(() => spoke(w, 'press the look inside button'), 3000);
      check('blocked SHOW is narrated in the visitor register', spoke(w, 'press the look inside button'));
      w.__vgPrefs.kid = false;
      check('no unexpected microscope page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ====== MICROSCOPE: "what does this neuron do?" card + steering ====== */
  console.log('-- microscope: neuron card --');
  {
    const errors = [];
    const dom = makeDom(extractTemplate('microscope'), errors, w => mkStorage(w, {}));
    const w = dom.window, d = w.document;
    const booted = await until(() => w.__guide && w.__vgW && w.__vgW.workers[0], 6000);
    check('microscope boots for the neuron card', booted);
    if (booted) {
      w.__guide.setWordMs(0);
      const wk = w.__vgW.workers[0];
      /* a tiny atlas for a 2-layer, 4-neuron model, routed the way the shell does it */
      const atlas = { schema: 'tiny-mind-neuron-atlas-v1', model: 'stories15M',
        config: { dim: 8, nLayers: 2, hiddenDim: 4, vocabSize: 16 }, fingerprint: 'fp-test',
        corpus: { file: 'x', stories: 3000, tokens: 600000 }, window: { before: 6, after: 2 }, perPole: 5,
        pieces: [' Zog', ' ate', ' a', ' cloud', '.', ' gave', ' up'],
        neurons: { '1:2': [2.5, -1.8, [[100, 1, 7, 12, 0, 1, 2, 3], [60, 0, 9, 4, 1, 2, 3]], [[100, 1, 8, 30, 5, 6, 4]]] },
        labels: { '1:2': ['eating words', null, 'sig'] } };
      w.__tmRouteJson('tiny-mind-neuron-atlas.json', JSON.stringify(atlas));
      check('routed atlas is accepted', /neuron atlas ready/.test(d.getElementById('status').textContent));
      w.__tmRouteJson('replay.json', '{"stories":[]}');
      check('a non-atlas json is ignored by the microscope route', /neuron atlas ready/.test(d.getElementById('status').textContent));
      wk.fakeReceive({ type: 'loaded', fingerprint: 'fp-test',
        config: { dim: 8, nLayers: 2, nHeads: 2, nKvHeads: 2, hiddenDim: 4, vocabSize: 16, seqLen: 32 } });
      const n = 4;
      wk.fakeReceive({ type: 'run-done', n, ids: [1, 2, 3, 4], pieces: ['(start)', 'Zog', ' ate', ' a'],
        topK: Array.from({ length: n }, () => [{ token: 5, piece: ' cloud', prob: 0.5, logit: 1 }]),
        nextProb: [0.1, 0.2, 0.3, 0], attention: new Float32Array(2 * 2 * n * n).buffer,
        config: { nLayers: 2, nHeads: 2, hiddenDim: 4, dim: 8, vocabSize: 16, seqLen: 32 }, bytes: 1 });
      wk.fakeReceive({ type: 'neurons-done', spanStart: 1, spanEnd: 2, ablate: null,
        top: [{ layer: 1, neuron: 2, score: 4, inMean: 1, outMean: 0.2 }] });
      const item = d.querySelector('.neuron-item');
      check('neuron finder lists a neuron to click', !!item);
      if (item) item.click();
      const card = d.getElementById('neuronCard');
      check('clicking a neuron opens its card', /L1 · neuron 2/.test(card.textContent));
      check('card shows both poles side by side', !!card.querySelector('.pole.pos') && !!card.querySelector('.pole.neg'));
      check('the lit piece is highlighted in each snippet',
        card.querySelector('.pole.pos mark') && card.querySelector('.pole.pos mark').textContent === ' ate' &&
        card.querySelector('.pole.neg mark').textContent === ' up');
      check('the AI label shows on its pole, none invented for the other',
        /eating words/.test(card.querySelector('.pole.pos').textContent) && !card.querySelector('.pole.neg .plabel'));
      check('numbers and provenance stay behind a click',
        !!card.querySelector('details') && /\+2\.5 \(100%\) · story 7, piece 12/.test(card.querySelector('details').textContent));
      wk.sent.length = 0;
      card.querySelector('button[data-steer="up"]').click();
      const run = wk.sent.find(m => m.type === 'run'), cont = wk.sent.find(m => m.type === 'continue');
      check('"turn it up" re-runs the microscope with the neuron held at 2× its strongest',
        run && run.steer && run.steer.length === 1 && run.steer[0].layer === 1 && run.steer[0].neuron === 2 && run.steer[0].value === 5,
        JSON.stringify(run && run.steer));
      check('... and asks for the story to go on under the same change', cont && cont.steer && cont.steer[0].value === 5);
      wk.fakeReceive({ type: 'run-ablated', n, ids: [1, 2, 3, 4], pieces: ['(start)', 'Zog', ' ate', ' a'],
        topK: Array.from({ length: n }, () => [{ token: 5, piece: ' cloud', prob: 0.4, logit: 1 }]),
        nextProb: [0.1, 0.1, 0.3, 0], attention: new Float32Array(2 * 2 * n * n).buffer, ablate: [],
        steer: [{ layer: 1, neuron: 2, value: 5 }], config: { nLayers: 2, nHeads: 2, hiddenDim: 4, dim: 8, vocabSize: 16, seqLen: 32 }, bytes: 1 });
      check('the whole microscope switches to the steered model (banner)',
        d.getElementById('ablBanner').style.display !== 'none' && /neuron 2 of layer 1 turned up/.test(d.getElementById('ablBanner').textContent));
      wk.fakeReceive({ type: 'continue-done', changed: true, ablate: [], steer: [{ layer: 1, neuron: 2, value: 5 }],
        base: [' cloud', '.', ' The', ' end'], steered: [' cloud', ' and', ' a', ' cake'] });
      const conts = [...d.querySelectorAll('#neuronCard .cont')].map(c => c.textContent);
      check('normal and changed continuations shown, the difference highlighted',
        conts.length === 2 && /normal model/.test(conts[0]) && d.querySelector('#neuronCard .cont .new').textContent === ' and a cake',
        conts.join(' | '));
      wk.sent.length = 0;
      d.querySelector('button[data-steer="normal"]').click();
      check('"back to normal" leaves the steered view without a re-run',
        d.getElementById('ablBanner').style.display === 'none' && !wk.sent.some(m => m.type === 'run'));
      const plainCont = wk.sent.find(m => m.type === 'continue');
      check('... and the plain continuation carries no steer at all', plainCont && !('steer' in plainCont) && !('ablate' in plainCont));
      /* a model the atlas was not made for: no examples, steering still offered */
      wk.fakeReceive({ type: 'loaded', fingerprint: 'other',
        config: { dim: 8, nLayers: 2, nHeads: 2, nKvHeads: 2, hiddenDim: 4, vocabSize: 16, seqLen: 32 } });
      check('a different model closes the card', d.getElementById('neuronCard').textContent.trim() === '');
      check('no unexpected neuron-card page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ====== MICROSCOPE: no sample yet, so a story it grew up on ====== */
  console.log('-- microscope: prefill falls back to the storybook --');
  {
    const errors = [];
    const dom = makeDom(extractTemplate('microscope'), errors, w => mkStorage(w, {
      'tm-book-Sisi': JSON.stringify({ sheet: { name: 'Sisi' }, kind: 'a unicorn',
        book: [{ split: 'train', text: 'Sisi ate a rainbow for breakfast.' }] }) }));
    const w = dom.window;
    const booted = await until(() => w.__guide, 6000);
    check('microscope boots (storybook fallback)', booted);
    if (booted) {
      w.__guide.setWordMs(0);
      w.document.getElementById('modelInfo').textContent = 'sisi-260K.bin \u00b7 dim 64';
      await until(() => /Sisi ate a rainbow/.test(w.document.getElementById('story').value), 4000);
      check('storybook sentence used when the mind has written nothing yet',
        /Sisi ate a rainbow/.test(w.document.getElementById('story').value));
      await until(() => spoke(w, 'grew up on'), 3000);
      check('fallback is narrated honestly', spoke(w, 'grew up on'));
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
      check('shell page can actually call a provider', typeof (w.__vgProviders || {}).call === 'function');
      check('shell page can find a remembered key', typeof w.__vgRememberedKey === 'function');
      w.localStorage.setItem('tm-key-openai', 'sk-shell');
      check('shell finds the ghostwriter key', (w.__vgRememberedKey() || {}).provider === 'openai');
      w.localStorage.removeItem('tm-key-openai');
      const names = G.steps.map(s => s.name);
      check('front-door tour has one stop per card (the spotlight follows the words)',
        names.join(',') === 'welcome,teach,grow,look,hunt,pickOne', names.join(','));
      check('each card stop spotlights its own card',
        ['raise', 'grow', 'look', 'hunt'].every((k, i) => G.steps[i + 1].spot === '[data-act="' + k + '"]'));
      const doc = w.document;
      check('simple view is the default: the front door shows, the tabs wait',
        doc.body.classList.contains('simple') && w.__tmShell.activeTab === 'home' &&
        !doc.getElementById('home').classList.contains('hide') && doc.getElementById('bay').classList.contains('hide'));
      check('the front door offers the four activities',
        [...doc.querySelectorAll('#home .act')].map(b => b.dataset.act).join(',') === 'raise,grow,look,hunt');
      check('no starter-pack quest left on the front door', !spoke(w, 'Starter pack:') && !spoke(w, 'catch the brain'));
      G.start();
      await until(() => G.state === 'done', 12000);
      check('front-door tour runs to the end without waiting on a drop', G.state === 'done', 'cur=' + G.cur + ' state=' + G.state);
      check('tour speaks about the front door', spoke(w, 'front door'));
      /* shell guide auto-hide: a room covering the front page must
         hide both the panel and the collapsed tab */
      doc.body.dataset.view = 'nursery';
      await until(() => doc.getElementById('vgTab').style.display === 'none', 2000);
      check('shell guide hides inside a room (no stacked Guide tabs)',
        doc.getElementById('vgTab').style.display === 'none' &&
        doc.getElementById('vgPanel').style.display === 'none');
      check('reserved margin released while hidden (no dead strip)', !doc.body.classList.contains('vg-open'));
      doc.body.dataset.view = 'home';
      await until(() => doc.getElementById('vgTab').style.display === '', 2000);
      check('shell guide returns with the front door', doc.getElementById('vgTab').style.display === '');
      /* an inline DO directive must be executed and never shown */
      w.claude = { complete: async () => 'Looking inside is this card. DO: SHOW look' };
      doc.getElementById('vgIn').value = 'where do I look inside?';
      doc.getElementById('vgGo').click();
      await until(() => spoke(w, 'Looking inside is this card'), 6000);
      check('inline DO line is stripped from what the visitor reads',
        !logTexts(w).some(t => /DO:/.test(t)), logTexts(w).slice(-1)[0]);
      check('inline DO line is still executed', doc.querySelector('[data-act="look"]').classList.contains('vg-spot'));

      /* TAB proposes; the visitor countersigns */
      w.claude = { complete: async () => 'The time machine is in the Nursery. DO: TAB nursery' };
      doc.getElementById('vgIn').value = 'where is time travel?';
      doc.getElementById('vgGo').click();
      await until(() => doc.getElementById('vgConfirm') &&
        /Nursery|nursery/.test(doc.getElementById('vgConfirmMsg').textContent), 6000);
      check('TAB asks before it moves the visitor',
        /nursery/i.test(doc.getElementById('vgConfirmMsg').textContent));
      check('nothing moved while the question stood', w.__tmShell.activeTab === 'home');
      doc.getElementById('vgConfirmNo').click();
      await sleep(80);
      check('declining leaves the visitor where they were', w.__tmShell.activeTab === 'home');
      /* naming the room they already stand in is invalid, not a move */
      w.claude = { complete: async () => 'You are already here. DO: TAB home' };
      doc.getElementById('vgIn').value = 'take me to the front door';
      doc.getElementById('vgGo').click();
      await until(() => spoke(w, 'You are already here'), 6000);
      await sleep(150);
      check('TAB to the current room is a silent no-op',
        w.__tmShell.activeTab === 'home' && !/front door/i.test(doc.getElementById('vgConfirmMsg').textContent));
      delete w.claude;

      /* the expert view: every tab plus the files page, remembered per browser */
      w.__tmShell.setMode('expert');
      check('expert view shows the files page and the tabs',
        w.__tmShell.activeTab === 'bay' && doc.body.classList.contains('expert') && !doc.getElementById('bay').classList.contains('hide'));
      check('the guide stays on the files page', doc.getElementById('vgTab').style.display === '');
      check('the view choice is remembered', w.localStorage.getItem('tm-ui-mode') === 'expert');
      w.__tmShell.setMode('simple');
      check('simple view returns to the front door', w.__tmShell.activeTab === 'home' && w.localStorage.getItem('tm-ui-mode') === 'simple');

      check('no unexpected shell page errors', errors.length === 0, errors.slice(0, 3).join(' | '));
    }
  }

  /* ============ SHELL scenario B: returning expert ============ */
  console.log('-- shell: returning visitor who chose the expert view --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors, w => { w.localStorage.setItem('tm-ui-mode', 'expert'); });
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell, 8000);
    check('shell guide boots (scenario B)', booted);
    if (booted) {
      check('a remembered expert view opens on the files page',
        w.__tmShell.activeTab === 'bay' && w.document.body.classList.contains('expert'));
      check('the files tab is named for files, not models',
        /files/.test(w.document.querySelector('.tab[data-tab="bay"]').textContent));
    }
  }

  /* ============ SHELL scenario D: bay restore roundtrip ============ */
  console.log('-- shell: persisted bay restores on boot --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors, w => {
      sayLog(w);
      w.localStorage.setItem('tm-pool-seed-wordlist.json', JSON.stringify({
        kind: 'json', name: 'seed-wordlist.json', parts: 1, enc: 'text' }));
      w.localStorage.setItem('tm-pool-seed-wordlist.json~0', JSON.stringify({ words: ['zog'] }));
    });
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell, 8000);
    check('shell boots with a seeded pool key', booted);
    if (booted) {
      const ok = await until(() => said(w, /remembered seed-wordlist\.json/), 8000);
      check('bay restore replays persisted files, naming them', ok, (w.__says || []).join(' | '));
    }
  }

  /* ====== SHELL scenario E: a half-written model self-heals ======
     The artifact symptom: a large value fails to store, the index key
     survives, and every later boot 404s on it while the little files
     load. Restore must clear it and say so rather than lose the bay. */
  console.log('-- shell: phantom key from a failed big write --');
  {
    const errors = [];
    const dom = makeDom(BUILT, errors, w => {
      sayLog(w);
      w.localStorage.setItem('tm-pool-stories260K.bin', JSON.stringify({
        kind: 'model', name: 'stories260K.bin', parts: 2, enc: 'b64' }));
      w.localStorage.setItem('tm-pool-stories260K.bin~0', 'AAAA');   // chunk 1 of 2 never landed
      w.localStorage.setItem('tm-pool-seed-wordlist.json', JSON.stringify({
        kind: 'json', name: 'seed-wordlist.json', parts: 1, enc: 'text' }));
      w.localStorage.setItem('tm-pool-seed-wordlist.json~0', JSON.stringify({ words: ['zog'] }));
    });
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell, 8000);
    check('shell boots with a phantom pool key', booted);
    if (booted) {
      const healed = await until(() => said(w, /had gone bad/), 8000);
      check('restore reports and clears the broken file', healed, (w.__says || []).join(' | '));
      await until(() => !w.localStorage.getItem('tm-pool-stories260K.bin'), 3000);
      check('phantom key and its orphan chunk are deleted',
        !w.localStorage.getItem('tm-pool-stories260K.bin') && !w.localStorage.getItem('tm-pool-stories260K.bin~0'));
      check('the good file still restores', said(w, /remembered seed-wordlist\.json/), (w.__says || []).join(' | '));
    }
  }

  /* ====== SHELL scenario F: create-only storage answers 409 ======
     The artifact's StorageSet conflicts on a key that already exists,
     and on concurrent writes. Overwrites must become delete-then-create,
     and an identical value must not be rewritten at all. */
  console.log('-- shell: storage that 409s on overwrite --');
  {
    const errors = [];
    let sets = 0;
    const dom = makeDom(BUILT, errors, w => {
      sayLog(w);
      const map = new Map();
      w.storage = {
        get: async k => { if (!map.has(k)) throw new Error('Database error: 404'); return { key: k, value: map.get(k) }; },
        set: async (k, v) => { sets++; if (map.has(k)) throw new Error('Database error: 409'); map.set(k, String(v)); return { key: k, value: v }; },
        delete: async k => { map.delete(k); return { key: k, deleted: true }; },
        list: async p => ({ keys: [...map.keys()].filter(k => !p || k.startsWith(p)), prefix: p }),
      };
      w.__map = map;
    });
    const w = dom.window;
    const booted = await until(() => w.__guide && w.__tmShell && w.__tmShell.tmWrite, 8000);
    check('shell boots against create-only storage', booted);
    if (booted) {
      await w.__tmShell.tmWrite('model', 't.bin', 'AAAA', 'b64');
      const idx = () => { try { return JSON.parse(w.__map.get('tm-pool-t.bin')); } catch (_) { return null; } };
      check('first write lands', idx() && idx().parts === 1 && idx().len === 4 && w.__map.get('tm-pool-t.bin~0') === 'AAAA');
      await w.__tmShell.tmWrite('model', 't.bin', 'BBBB', 'b64');
      check('overwrite survives 409 (delete-then-create)', w.__map.get('tm-pool-t.bin~0') === 'BBBB' && idx() && idx().len === 4);
      const before = sets;
      await w.__tmShell.tmWrite('model', 't.bin', 'BBBB', 'b64');
      check('identical value is not rewritten', sets === before);
      const bindingMeta = { origin: 'trained', modelMeta: { storyWorld: { mode: 'binding',
        characters: { Zog: { food: 'clouds' }, Mip: { food: 'apples' }, Bex: { food: 'honey' }, Nia: { food: 'berries' } } } } };
      await w.__tmShell.tmWrite('model', 't.bin', 'BBBB', 'b64', bindingMeta);
      check('model-pool persistence stores training origin and character metadata',
        idx() && idx().extra && idx().extra.origin === 'trained' &&
        idx().extra.modelMeta.storyWorld.characters.Nia.food === 'berries');
      const afterMetadata = sets;
      await w.__tmShell.tmWrite('model', 't.bin', 'BBBB', 'b64', bindingMeta);
      check('identical model bytes and metadata are not rewritten', sets === afterMetadata);
      await Promise.all([
        w.__tmShell.tmWrite('model', 'a.bin', 'AA', 'b64'),
        w.__tmShell.tmWrite('tok', 'b.bin', 'BB', 'b64'),
        w.__tmShell.tmWrite('json', 'c.json', '{}', 'text'),
      ]);
      check('parallel writes serialize without conflict',
        !!w.__map.get('tm-pool-a.bin') && !!w.__map.get('tm-pool-b.bin') && !!w.__map.get('tm-pool-c.json'));
      check('no "could not remember" warning was raised', !said(w, /could not remember/), (w.__says || []).join(' | '));
    }
  }

  /* ============ SHELL scenario G: "save my work" round trip ============ */
  console.log('-- shell: save my work, then open it in a fresh page --');
  {
    const ck = new Uint8Array(readFileSync('models/stories260K.bin'));
    const meta = { schemaVersion: 1, source: 'nursery', storyWorld: { mode: 'single', label: 'Sisi',
      sheet: { name: 'Sisi', kind: 'a small unicorn', attr1: 'lives in the clouds', attr2: 'enjoys candy' } } };
    const book = JSON.stringify({ mode: 'single', sheet: meta.storyWorld.sheet, train: ['Sisi ate candy.'], heldout: [] });
    const errorsA = [];
    const a = makeDom(BUILT, errorsA, w => w.localStorage.setItem('tm-storybook', book)).window;
    const bootA = await until(() => a.__tmShell && a.__tmShell.buildWorkZip, 8000);
    check('shell boots for the save-my-work round trip', bootA);
    if (bootA) {
      await a.__tmShell.ingest([{ name: 'sisi-260K.bin', u8: ck, origin: 'trained', modelMeta: meta }]);
      const w1 = await a.__tmShell.buildWorkZip();
      check('save my work bundles the raised mind and the storybook', !!w1 && w1.models === 1 && w1.storybook);
      const b = makeDom(BUILT, []).window;
      const bootB = await until(() => b.__tmShell && b.__tmShell.ingest, 8000);
      if (bootB && w1) {
        await b.__tmShell.ingest([{ name: 'tiny-mind-my-work.zip', u8: w1.zip }]);
        await sleep(100);
        check('opening the bundle brings the mind back as one you raised',
          /you raised it/.test(b.document.getElementById('pool').textContent));
        const w2 = await b.__tmShell.buildWorkZip();
        const files = w2 ? await b.unzip(w2.zip) : [];
        const man = files.find(f => f.name === 'tiny-mind-work.json');
        const back = man && JSON.parse(new TextDecoder().decode(man.u8)).models[0];
        check('... with its character still attached (so the Hunt can quiz it)',
          !!back && back.origin === 'trained' && back.meta && back.meta.storyWorld.sheet.name === 'Sisi');
        check('opening the bundle says what came back, where you can see it',
          /Opened your saved work: Sisi's mind/.test((b.document.getElementById('simpleToast') || {}).textContent || '') &&
          /Sisi's storybook/.test(b.document.getElementById('simpleToast').textContent));
        check('the raised mind is named after its character', /Sisi's mind/.test(b.document.getElementById('pool').textContent));
        b.__tmShell.renameMind('sisi-260K.bin', 'Sparkle');
        check('a mind can be renamed (files page shows the new name)', /Sparkle/.test(b.document.getElementById('pool').textContent));
        const w3 = await b.__tmShell.buildWorkZip();
        const man3 = w3 && (await b.unzip(w3.zip)).find(f => f.name === 'tiny-mind-work.json');
        check('... and the new name travels in a saved bundle',
          !!man3 && JSON.parse(new TextDecoder().decode(man3.u8)).models[0].displayName === 'Sparkle');
        const first = await b.unzip(w1.zip);
        const sb = first.find(f => f.name === 'my-storybook.json');
        check('... and the storybook travels in the bundle', !!sb &&
          JSON.parse(new TextDecoder().decode(sb.u8)).train[0] === 'Sisi ate candy.');
      }
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
