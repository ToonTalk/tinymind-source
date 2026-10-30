/* ============================================================
   tm-shell-adapter.js — Tiny Mind shell guide (vg-core v2).
   Owns the front-door tour (the four activities on the home page)
   and the master guide/kid/voice toggles, which it persists via
   shellStorage and rebroadcasts to the frames. The build exposes
   window.__tmShell = {frames, shellStorage, selectTab, startActivity,
   APPS} for this adapter. The minds load by themselves now, so the
   old "catch the brain" starter-pack quest is gone.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
/* The prelude is spliced into the three frame templates, not into this
   page, so the shell guide has to carry the few helpers it needs. Keep
   in sync with tm-prelude.js -- better still, extract them into a
   shared vg-llm.js that both splices pull from. */
if (!window.__vgHidden) window.__vgHidden = function(el){
  if (!el) return true;
  if (el.hidden || /(^|\s)hide(\s|$)/.test(el.className || '')) return true;
  try { return el.ownerDocument.defaultView.getComputedStyle(el).display === 'none'; } catch (_) { return false; }
};
if (!window.__vgShow) window.__vgShow = function(api, spec){
  const t = typeof spec === 'string' ? { sel: spec } : (spec || {});
  const el = t.sel && document.querySelector(t.sel);
  const empty = !el || !((el.textContent || '').trim() || el.children.length || el.value);
  if (el && !window.__vgHidden(el) && !empty) { api.spot(t.sel); return true; }
  if (t.needs) api.spot(t.needs);
  api.emit('showBlocked', { why: t.why, kidWhy: t.kidWhy });
  return false;
};
if (!window.__vgShowBlockedEvent) window.__vgShowBlockedEvent = {
  priority: 'moment', cooldownMs: 1200,
  lines: p => (p && p.why) || 'That view has not been drawn yet \u2014 press the control I am pointing at first.',
  kidLines: p => (p && (p.kidWhy || p.why)) || 'That is not on the screen yet! Press the button I am pointing at first.',
};
if (!window.__vgRememberedKey) window.__vgRememberedKey = function(){
  const DEF = { anthropic: 'claude-sonnet-4-6', openai: 'gpt-6-luna', gemini: 'gemini-3.5-flash' };
  try {
    for (const p of ['anthropic', 'openai', 'gemini']) {
      const k = localStorage.getItem('tm-key-' + p);
      if (k && k.trim()) return { provider: p, key: k.trim(), model: DEF[p] };
    }
  } catch (_) {}
  return null;
};
if (!window.__vgProviders) window.__vgProviders = {
  async call(provider, key, model, sys, usr){
    if (!key) { const e = new Error('no key'); e.vgNoAI = true; throw e; }
    if (provider === 'openai') {
      const r = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
        body: JSON.stringify({ model: model || 'gpt-6-luna', messages: [{ role: 'system', content: sys }, { role: 'user', content: usr }] }) });
      if (!r.ok) throw new Error('OpenAI HTTP ' + r.status);
      const d = await r.json(); return d.choices[0].message.content;
    }
    if (provider === 'gemini' || provider === 'google') {
      const m = model || 'gemini-3.5-flash';
      const r = await fetch('https://generativelanguage.googleapis.com/v1beta/models/' + m + ':generateContent?key=' + encodeURIComponent(key), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ systemInstruction: { parts: [{ text: sys }] },
          contents: [{ role: 'user', parts: [{ text: usr }] }] }) });
      if (!r.ok) throw new Error('Gemini HTTP ' + r.status);
      const d = await r.json(); return d.candidates[0].content.parts.map(p => p.text || '').join('');
    }
    const r = await fetch('https://api.anthropic.com/v1/messages', { method: 'POST',
      headers: { 'Content-Type': 'application/json', 'x-api-key': key,
        'anthropic-version': '2023-06-01', 'anthropic-dangerous-direct-browser-access': 'true' },
      body: JSON.stringify({ model: model || 'claude-sonnet-4-6', max_tokens: 1000,
        system: sys, messages: [{ role: 'user', content: usr }] }) });
    if (!r.ok) throw new Error('Anthropic HTTP ' + r.status);
    const d = await r.json(); return (d.content || []).map(c => c.text || '').join('');
  }
};

const SH = () => window.__tmShell || {};
const prefs = { kid: false, voice: false, on: true };
let api = null;

/* The shell has no key box of its own. When it is not running inside
   a Claude artifact (where the keyless endpoint works), reuse the key
   the visitor already gave the Nursery's ghostwriter -- the frames are
   srcdoc, so same-origin and readable. Never fetch the keyless
   endpoint from an ordinary page: it has no key and no browser-access
   header, so it fails CORS and looks like a mysterious network error. */
const VG_LIVE = ['anthropic', 'openai', 'gemini'];
function frameKey(){
  try {
    const f = SH().frames && SH().frames.nursery;
    const d = f && f.contentWindow && f.contentWindow.document;
    if (!d) return null;
    const p = (d.getElementById('provider') || {}).value || '';
    const k = ((d.getElementById('apiKey') || {}).value || '').trim();
    if (!k || !VG_LIVE.includes(p)) return null;
    return { provider: p, key: k, model: ((d.getElementById('apiModel') || {}).value || '').trim() };
  } catch (_) { return null; }
}
const inClaude = () => !!(api && api.inClaude);
const shellKey = () => frameKey() || (window.__vgRememberedKey && window.__vgRememberedKey());
const activeTab = () => (SH().activeTab) || 'home';
const onFrontPage = () => ['home', 'bay'].includes(document.body.dataset.view || activeTab());   // the shell marks the view on <body data-view>

async function loadPrefs(){
  const st = SH().shellStorage; if (!st) return;
  const race = p => Promise.race([p, new Promise(r => setTimeout(() => r(null), 1200))]);
  let have = null;
  try { if (typeof st.list === 'function') {
    const r = await race(st.list('tm-guide-'));
    if (r && r.keys) have = new Set(r.keys);
  } } catch (_) {}
  const read = k => (have && !have.has(k)) ? Promise.resolve(null)
    : race(Promise.resolve().then(() => st.get(k)).then(r => r && r.value)).catch(() => null);
  const [on, kid, voice] = await Promise.all([read('tm-guide-on'), read('tm-guide-kid'), read('tm-guide-voice')]);
  if (on !== null) prefs.on = on !== '0';
  if (kid !== null) prefs.kid = kid === '1';
  if (voice !== null) prefs.voice = voice === '1';
}
function setPref(key, value){
  const st = SH().shellStorage;
  if (st) { try { st.set(key, value); } catch (_) {} }
  const frames = SH().frames || {};
  for (const a of Object.keys(frames)) {
    try { frames[a].contentWindow.postMessage({ tm: 'pref', key, value }, '*'); } catch (_) {}
  }
}
const kid = () => prefs.kid;

/* the pool table still lives on the files page (expert view); a change
   there is narrated only as a provenance remark, never as a quest */
const poolText = () => ($('pool') && $('pool').textContent) || '';
function wire(emit){}

const EVENTS = {};

function STEPS(){
  return [
  { name: 'welcome', spot: '#home h1', text: () => kid()
      ? 'Hi! This is the front door of Tiny Mind. Each big button is a different game with a real little computer mind.'
      : 'This is the front door of Tiny Mind. Each card is one thing you can do with a real, very small language model. The minds load by themselves; nothing leaves your computer.' },
  { name: 'teach', spot: '[data-act="raise"]', text: () => kid()
      ? 'Teach a tiny mind: make up a character, and watch the mind learn about it.'
      : 'Teach a tiny mind trains one live, on stories about a character you invent.' },
  { name: 'grow', spot: '[data-act="grow"]', text: () => kid()
      ? 'Watch a mind grow up: see a recording of a mind going from babble to stories.'
      : 'Watch a mind grow up replays recordings of that kind of training, from random babble to whole stories, with no waiting.' },
  { name: 'look', spot: '[data-act="look"]', text: () => kid()
      ? 'Look inside a mind shows what it is thinking, word by word.'
      : 'Look inside a mind shows the words a model is weighing up, layer by layer, and what single neurons respond to.' },
  { name: 'think', spot: '[data-act="think"]', text: () => kid()
      ? 'See what a mind is thinking lets you peek at its thoughts, and even swap one!'
      : 'See what a mind is thinking shows the words a mind is holding in mind as it reads or writes; then you swap one of its thoughts and watch the story change.' },
  { name: 'hunt', spot: '[data-act="hunt"]', text: () => kid()
      ? 'Hunt for a hidden fact is a treasure hunt inside a mind.'
      : 'Hunt for a hidden fact is a detective game: find where a mind keeps something, then prove it by switching parts off.' },
  { name: 'pickOne', spot: '#home .acts', allow: '.act,#modeBtn,#homeExpert', text: () => kid()
      ? 'Now pick one! A guide is waiting inside each.'
      : 'Pick one. A guide waits inside each. If you know your way around already, the button at the top right shows every tab and file at once.' },
  ];
}

function statusText(){
  const t = activeTab();
  const where = t === 'home' ? 'the front door, with its five activity cards \u2014 this very page'
    : t === 'bay' ? 'the files page (the expert view\'s "files" tab) \u2014 this very page'
    : 'the ' + t + ' room, whose own guide is beside them there';
  return `The visitor is RIGHT NOW looking at ${where}. Never tell them to go where they already are. ` +
    `Rooms booted so far: ${(SH().frames && Object.keys(SH().frames).join(',')) || 'none'}. ` +
    `Shell status: ${($('shellStatus') && $('shellStatus').textContent) || ''}. ` +
    `View: ${(SH().uiMode) || 'simple'} (simple = front door; expert = all tabs). ` +
    `Minds loaded (truncated): ${poolText().slice(0, 200) || 'none yet'}.`;
}
function systemPrompt(){
  return 'You are the guide on the front page of Tiny Mind, a four-room exhibit about how small language models work: a Nursery for raising one, a Microscope for looking inside one, a Thought bubble for seeing and swapping what one is thinking, and a Treasure Hunt for finding where a fact lives. This page is the front door: five activity cards (Teach a tiny mind, Watch a mind grow up, Look inside a mind, See what a mind is thinking, Hunt for a hidden fact); the minds load by themselves, and a button at the top right switches to an expert view with every tab and a files page. ' +
    'You are honestly an A I yourself — a much bigger mind of the same kind. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is ten or older: plain, concrete language. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target, or DO: TAB name to offer to take them to another room. Show targets: teach, grow, look, think, hunt, tabs. Tab names: home, nursery, microscope, thought, hunt, files. ' +
    ' A TAB directive only offers: the app asks the visitor to confirm before anything moves, so word it as an offer, and never claim you have already taken them. ' +
    ' The time machine lives in the Nursery, below the training controls. ' +
    ' Never offer to open or point at a view the status marks as already visible \u2014 talk about what it shows instead. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'switch to the Nursery', escalations: 'load or delete models', safeVerb: 'SHOW' });
}
const SHOW = { teach: '[data-act="raise"]', grow: '[data-act="grow"]', look: '[data-act="look"]', think: '[data-act="think"]', hunt: '[data-act="hunt"]', tabs: '#modeBtn' };
const TAB_ALIAS = { files: 'bay', home: 'home', nursery: 'nursery', microscope: 'microscope', thought: 'thought', hunt: 'hunt' };

function createGuide(){
  const A = {
    id: 'tm-shell',
    ui: { mascot: '\u{1F9E0}', greeting: () => kid()
        ? 'Hi! I am the guide. Press start tour and I will show you around.'
        : 'Welcome to Tiny Mind. Press start tour for a one-minute look at the five things you can do, or ask me anything.',
      voiceEnabled: () => prefs.voice, doneStatus: 'Pick a room — a guide waits in each.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 0 },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ pool: poolText() }),
      diffRemark: o => { if (!o) return '';
        if (o.pool !== poolText()) return 'More minds arrived while I was quiet.';
        return ''; } },
    prompts: { system: systemPrompt, status: statusText },
    llm: { hasKey: () => !!(window.claude && typeof window.claude.complete === 'function') || inClaude() || !!shellKey(),
      keyedCall: async (sys, usr) => {
        if (window.claude && typeof window.claude.complete === 'function') {
          try { return await window.claude.complete(sys + '\n\n' + usr); } catch (e) { if (!shellKey()) throw e; }
        } else if (inClaude()) {
          try { return await api.keylessCall(sys, usr); } catch (e) { if (!shellKey()) throw e; }
        }
        const c = shellKey();
        if (!c) { const e = new Error('no key'); e.vgNoAI = true; throw e; }
        return window.__vgProviders.call(c.provider, c.key, c.model, sys, usr);
      },
      noAI: { say: 'I cannot reach a live A I from this page on my own. Open the Nursery and paste a key into the ghostwriter panel, and I can talk live from any room. The tour works fine without one.' } },
    strings: { llmFail: e => { const m = e && e.message ? e.message : String(e);
      if (/failed to fetch|network|cors|no key/i.test(m) && !shellKey())
        return { log: 'Live AI call failed: ' + m,
          say: 'I could not reach a live A I from this page. Open the Nursery, pick a provider and paste your key into the ghostwriter panel — then I can answer from any room.' };
      return null; } },
    verbs: {
      SHOW: { exec(p){ const s = SHOW[(p[0] || '').toLowerCase()]; if (s) window.__vgShow(api, s); } },
      /* moving the visitor to another room is an escalation, not a
         pointing gesture: the guide proposes, the visitor countersigns.
         Naming the room they are already in is invalid, so the guide
         cannot "take" them where they stand. */
      TAB: { gated: true,
        validate(p){ const t = TAB_ALIAS[(p[0] || '').toLowerCase()];
          return t && t !== activeTab() && !!SH().selectTab ? t : false; },
        ratifyText(p, t){ const n = t === 'bay' ? 'files page' : t === 'home' ? 'front door' : t;
          return kid() ? `Shall I take you to the ${n}?` : `Take you to the ${n}?`; },
        exec(p, a, t){ SH().selectTab(t); } },
    },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    testHook: { app: () => ({ prefs }) },
    init(a){
      api = a; a.on('showBlocked', window.__vgShowBlockedEvent);
      wire((n, p) => a.emit(n, p));
      /* master toggles: guide on/off (for the frames), kid, voice */
      const head = document.getElementById('vgHead');
      if (head) {
        head.style.flexWrap = 'wrap';                 /* never clip: overflow wraps */
        const wrap = document.createElement('span');
        wrap.style.cssText = 'display:flex;gap:10px;margin-left:auto;font-size:11px;align-items:center';
        wrap.innerHTML =
          '<label style="display:flex;gap:3px;align-items:center;cursor:pointer;white-space:nowrap"><input type="checkbox" id="vgOnAll" checked> guides</label>' +
          '<label style="display:flex;gap:3px;align-items:center;cursor:pointer;white-space:nowrap"><input type="checkbox" id="vgKid"> kid</label>' +
          '<label style="display:flex;gap:3px;align-items:center;cursor:pointer;white-space:nowrap"><input type="checkbox" id="vgVoice"> voice</label>';
        head.appendChild(wrap);
        const on = wrap.querySelector('#vgOnAll'), k = wrap.querySelector('#vgKid'), v = wrap.querySelector('#vgVoice');
        on.checked = prefs.on; k.checked = prefs.kid; v.checked = prefs.voice;
        on.onchange = () => { prefs.on = on.checked; setPref('tm-guide-on', on.checked ? '1' : '0'); };
        k.onchange = () => { prefs.kid = k.checked; setPref('tm-guide-kid', k.checked ? '1' : '0'); };
        v.onchange = () => { prefs.voice = v.checked; setPref('tm-guide-voice', v.checked ? '1' : '0'); };
      }
      /* frame-originated pref changes (hunt's own kidMode) arrive via
         the rpc rebroadcast the build adds; mirror them here */
      window.addEventListener('message', e => { const m = e.data; if (!m || m.tm !== 'pref') return;
        if (m.key === 'tm-guide-kid') prefs.kid = m.value === '1';
        if (m.key === 'tm-guide-voice') prefs.voice = m.value === '1'; });
      /* the shell guide belongs to the front page (home or files): when a
         room's frame covers the shell, hide both the panel and the collapsed
         tab so it never stacks on the room guide's tab */
      { let wasOpen = false;
        const upd = () => { const off = !onFrontPage();
          for (const id of ['vgPanel', 'vgTab']) { const el = document.getElementById(id); if (el) el.style.display = off ? 'none' : ''; }
          if (off) { if (document.body.classList.contains('vg-open')) wasOpen = true;
            document.body.classList.remove('vg-open');   /* release the reserved margin: no dead strip beside the rooms */
          } else if (wasOpen) { document.body.classList.add('vg-open'); wasOpen = false; } };
        new MutationObserver(upd).observe(document.body, { attributes: true, attributeFilter: ['data-view'] }); upd(); }
    },
  };
  window.VoiceGuide.create(A);
}

(async function boot(){
  await loadPrefs();
  if (prefs.on !== false) createGuide(); else {
    /* guides off is a frame concern; the shell master guide still
       boots so the switch remains reachable */
    createGuide();
  }
})();
})();
