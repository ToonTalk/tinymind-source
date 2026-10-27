/* ============================================================
   tm-shell-adapter.js — Tiny Mind shell guide (vg-core v2).
   Owns the starter-pack "catch the brain" tour (artifact edition's
   front door) and the master guide/kid/voice toggles, which it
   persists via shellStorage and rebroadcasts to the frames. The
   build exposes window.__tmShell = {frames, shellStorage, selectTab,
   APPS} for this adapter. The starter zip URL is read from
   window.TM_STARTER_ZIP, defaulting to the confirmed host
   (toontalk.github.io/misc/).
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
  const DEF = { anthropic: 'claude-sonnet-4-6', openai: 'gpt-4o-mini', gemini: 'gemini-3.5-flash' };
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
        body: JSON.stringify({ model: model || 'gpt-4o-mini', messages: [{ role: 'system', content: sys }, { role: 'user', content: usr }] }) });
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
const STARTER = () => window.TM_STARTER_ZIP || 'https://toontalk.github.io/misc/tiny-mind-starter.zip';
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
const activeTab = () => (SH().activeTab) || 'bay';

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

/* modelLoaded without app edits: rows appear in the bay's #pool
   table when a bundle is ingested. Unlatched: any pool change while
   models are present re-emits, so an until-waiter armed AFTER the
   drop still resolves (narration stays single via once:true). */
const poolHasModel = () => { const p = $('pool');
  return !!p && p.querySelectorAll('tr').length > 0 && !/nothing yet/.test(p.textContent || ''); };
function wire(emit){
  const pool = $('pool');
  if (pool && pool.nodeType === 1) new MutationObserver(() => { if (poolHasModel()) emit('modelLoaded', {}); })
    .observe(pool, { childList: true, subtree: true });
}

const EVENTS = {
  modelLoaded: { priority: 'moment', once: true, spot: '#pool',
    lines: 'Caught! The brain is in the bay. It will still be here next visit — the bay remembers.',
    kidLines: 'You caught the brain! And it will still be here next time you come back.' },
};

function STEPS(){
  return [
  { name: 'catchTheBrain', skipIf: poolHasModel, spot: '#dropzone', text: () => kid()
      ? 'First quest: catch a brain! Ask a grown-up to download the starter pack from the address I put in my log, then drag the file into this drop zone.'
      : `First, catch the brain. Download the starter pack — I have put the address in my transcript log so you can copy it: ${STARTER()} — then drag the downloaded zip into this drop zone. Everything runs on your machine; nothing is uploaded anywhere.` },
  { name: 'dragIn', skipIf: poolHasModel, spot: '#dropzone', allow: '#dropzone,#browseBtn,#fileInput',
    until: 'modelLoaded',
    text: () => kid()
      ? 'Drag the file in, or press browse. I will wait right here.'
      : 'Drag it in, or use the browse button. I will wait — the bay will list every model and tokenizer it finds inside.' },
  { name: 'persisted', spot: '#pool', text: () => kid()
      ? 'See the list? Those are the minds you caught. They stay saved, so next time they will already be waiting.'
      : 'There is your pool: the trainee, its tokenizer, and any recordings. It persists — next time you open this page, the minds will already be here waiting.' },
  { name: 'pickADoor', spot: '#startNursery', allow: '#startNursery,.tab', text: () => kid()
      ? 'Now pick a door! The Nursery button starts you raising your very own tiny mind. See you inside!'
      : 'Now pick a door. The Nursery is the front one — press the start button, or use the tabs along the top: Nursery to raise a mind, Microscope to look inside one, Treasure Hunt to find where a fact lives. A guide is waiting in each room.' },
  ];
}

function statusText(){
  const t = activeTab();
  const where = t === 'bay' ? 'the model bay (the "models" tab) \u2014 this very page'
    : 'the ' + t + ' room, whose own guide is beside them there';
  return `The visitor is RIGHT NOW looking at ${where}. Never tell them to go where they already are. ` +
    `Rooms booted so far: ${(SH().frames && Object.keys(SH().frames).join(',')) || 'none'}. ` +
    `Shell status: ${($('shellStatus') && $('shellStatus').textContent) || ''}. ` +
    `Pool (truncated): ${($('pool') && $('pool').textContent.slice(0, 200)) || 'empty'}. ` +
    `Tokenizers: ${($('tokList') && $('tokList').textContent) || 'none'}.`;
}
function systemPrompt(){
  return 'You are the guide on the front page of Tiny Mind, a three-room exhibit about how small language models work: a Nursery for raising one, a Microscope for looking inside one, and a Treasure Hunt for finding where a fact lives. This page is the model bay, where bundles are dropped in. ' +
    'You are honestly an A I yourself — a much bigger mind of the same kind. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is ten or older: plain, concrete language. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target, or DO: TAB name to offer to take them to another room. Show targets: dropzone, pool, start. Tab names: bay, nursery, microscope, hunt. ' +
    ' A TAB directive only offers: the app asks the visitor to confirm before anything moves, so word it as an offer, and never claim you have already taken them. ' +
    ' The time machine lives in the Nursery, below the training controls. ' +
    ' Never offer to open or point at a view the status marks as already visible \u2014 talk about what it shows instead. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'switch to the Nursery', escalations: 'load or delete models', safeVerb: 'SHOW' });
}
const SHOW = { dropzone: '#dropzone', pool: '#pool', start: '#startNursery' };

function createGuide(){
  const A = {
    id: 'tm-shell',
    ui: { mascot: '\u{1F9E0}', greeting: () => kid()
        ? 'Hi! I am the guide. Press start tour and we will catch a brain together.'
        : 'Welcome to Tiny Mind. Press start tour for the two-minute front door, or ask me anything.',
      voiceEnabled: () => prefs.voice, doneStatus: 'Pick a room — a guide waits in each.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 0 },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ pool: ($('pool') && $('pool').textContent) || '' }),
      diffRemark: o => { if (!o) return '';
        const pool = ($('pool') && $('pool').textContent) || '';
        if (o.pool !== pool) return 'The bay changed while I was quiet — new minds in the pool.';
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
      noAI: { say: 'I cannot reach a live A I from this page on my own. Open the Nursery and paste a key into the ghostwriter panel, and I can talk live from any room. The tour and the bay work fine without one.' } },
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
        validate(p){ const t = (p[0] || '').toLowerCase();
          return ['bay', 'nursery', 'microscope', 'hunt'].includes(t) && t !== activeTab() && !!SH().selectTab ? t : false; },
        ratifyText(p, t){ return kid() ? `Shall I take you to the ${t}?` : `Take you to the ${t} tab?`; },
        exec(p, a, t){ SH().selectTab(t); } },
    },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    testHook: { app: () => ({ prefs, starter: STARTER() }) },
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
      /* the shell guide belongs to the bay: when a room's frame
         covers the shell, hide both the panel and the collapsed tab
         so it never stacks on the room guide's tab */
      const bay = document.getElementById('bay');
      if (bay && bay.nodeType === 1) { let wasOpen = false;
        const upd = () => { const off = bay.classList.contains('hide');
          for (const id of ['vgPanel', 'vgTab']) { const el = document.getElementById(id); if (el) el.style.display = off ? 'none' : ''; }
          if (off) { wasOpen = document.body.classList.contains('vg-open');
            document.body.classList.remove('vg-open');   /* release the reserved margin: no dead strip beside the rooms */
          } else if (wasOpen) { document.body.classList.add('vg-open'); } };
        new MutationObserver(upd).observe(bay, { attributes: true, attributeFilter: ['class'] }); upd(); }
      /* put the copyable starter URL straight into the transcript */
      a.logAdd('Starter pack: ' + STARTER());
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
