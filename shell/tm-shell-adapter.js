
/* ============================================================
   tm-shell-adapter.js — Tiny Mind shell guide (vg-core v2).
   Owns the starter-pack "catch the brain" tour (artifact edition's
   front door) and the master guide/kid/voice toggles, which it
   persists via shellStorage and rebroadcasts to the frames. The
   build exposes window.__tmShell = {frames, shellStorage, selectTab,
   APPS} for this adapter. The starter zip URL is read from
   window.TM_STARTER_ZIP with a placeholder default — CONFIRM URL.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const SH = () => window.__tmShell || {};
const STARTER = () => window.TM_STARTER_ZIP || 'https://toontalk.github.io/misc/tiny-mind-starter.zip';
const prefs = { kid: false, voice: false, on: true };
let api = null;

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
  if (pool) new MutationObserver(() => { if (poolHasModel()) emit('modelLoaded', {}); })
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
  return `Active tab: ${(SH().frames && Object.keys(SH().frames).join(',')) || 'none booted'}. ` +
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
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target, or DO: TAB name to switch rooms when the visitor asked to go there. Show targets: dropzone, pool, start. Tab names: bay, nursery, microscope, hunt. ' +
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
    llm: { hasKey: () => true,
      keyedCall: async (sys, usr) => {
        if (window.claude && typeof window.claude.complete === 'function') return window.claude.complete(sys + '\n\n' + usr);
        const r = await fetch('https://api.anthropic.com/v1/messages', { method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ model: 'claude-sonnet-4-6', max_tokens: 1000,
            messages: [{ role: 'user', content: sys + '\n\n' + usr }] }) });
        if (!r.ok) throw new Error('HTTP ' + r.status);
        const d = await r.json();
        return (d.content || []).map(c => c.text || '').join('');
      },
      noAI: { say: 'I cannot reach a live A I from here right now, but the tour and the bay work fine without one.' } },
    verbs: {
      SHOW: { exec(p){ const sel = SHOW[(p[0] || '').toLowerCase()]; if (sel) api.spot(sel); } },
      TAB: { exec(p){ const t = (p[0] || '').toLowerCase();
        if (['bay', 'nursery', 'microscope', 'hunt'].includes(t) && SH().selectTab) SH().selectTab(t); } },
    },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    testHook: { app: () => ({ prefs, starter: STARTER() }) },
    init(a){
      api = a; wire((n, p) => a.emit(n, p));
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
      if (bay) { let wasOpen = false;
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

