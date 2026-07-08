
/* ============================================================
   tm-prelude.js — Tiny Mind guide prelude. Spliced at the TOP of
   each frame template's <body> because the frames build their
   Worker at top level: the shim must exist before the app script
   runs. Inert until an adapter attaches. No app code is touched.
   ============================================================ */
(function(){
'use strict';

/* Worker shim: subclass records every instance and fans out both
   directions of traffic to taps. onmessage/onerror on the instance
   are untouched; addEventListener coexists with the app's handler.
   Outgoing traffic is seen via the postMessage override — that is
   how 'trainStarted' is derived with zero app edits. */
window.__vgW = { workers: [], taps: [] };
if (window.Worker) {
  const Base = window.Worker;
  window.Worker = class extends Base {
    constructor(...a){
      super(...a);
      window.__vgW.workers.push(this);
      this.addEventListener('message', e => {
        for (const t of window.__vgW.taps) { try { t('in', e.data, this); } catch (_) {} }
      });
    }
    postMessage(...a){
      for (const t of window.__vgW.taps) { try { t('out', a[0], this); } catch (_) {} }
      return super.postMessage(...a);
    }
  };
}

/* Emit buffer: app-side one-liners and observers can call
   window.vgEmit before (or without) a guide; emits queue until an
   adapter attaches, and are dropped if none ever does. */
const q = [];
window.vgEmit = function(n, p){ q.push([n, p]); if (q.length > 50) q.shift(); };
window.__vgAttach = function(fn){
  window.vgEmit = fn;
  for (const [n, p] of q.splice(0)) { try { fn(n, p); } catch (_) {} }
};

/* Pref cache + shell broadcasts. Keys live under tm- in the shell's
   storage (window.storage -> localStorage -> memory), relayed over
   the existing tm RPC; the shell rebroadcasts changes as
   {tm:'pref'} and tab switches as {tm:'vg-tab'}. */
window.__vgPrefs = { on: true, kid: false, voice: false, active: true };
window.addEventListener('message', e => {
  const m = e.data; if (!m) return;
  if (m.tm === 'pref' && /^tm-guide-/.test(m.key || '')) {
    if (m.key === 'tm-guide-on')    window.__vgPrefs.on    = m.value !== '0';
    if (m.key === 'tm-guide-kid')   window.__vgPrefs.kid   = m.value === '1';
    if (m.key === 'tm-guide-voice') window.__vgPrefs.voice = m.value === '1';
    window.vgEmit('prefChanged', { key: m.key, value: m.value });
  } else if (m.tm === 'vg-tab') {
    window.__vgPrefs.active = !!m.active;
    window.vgEmit('tabChanged', { active: !!m.active, tab: m.tab });
  }
});

/* Helpers the frame adapters share (called after the app script has
   run, so the frame's top-level `function rpc` is on window). */
window.__vgLoadPrefs = async function(){
  /* the frame bridge proxies window.storage to the shell's storage
     when framed; standalone there is no window.storage and defaults
     hold */
  const race = p => Promise.race([p, new Promise(r => setTimeout(() => r(null), 1200))]);
  /* list first: getting keys that do not exist logs loud 404s
     host-side in artifacts, so only get what list says is there */
  let have = null;
  try { if (window.storage && typeof window.storage.list === 'function') {
    const r = await race(window.storage.list('tm-guide-'));
    if (r && r.keys) have = new Set(r.keys);
  } } catch (_) {}
  const read = k => (have && !have.has(k)) ? Promise.resolve(null)
    : race((async () => { if (!window.storage) return null;
        const r = await window.storage.get(k); return r && r.value; })()).catch(() => null);
  const [on, kid, voice] = await Promise.all([read('tm-guide-on'), read('tm-guide-kid'), read('tm-guide-voice')]);
  if (on !== null)    window.__vgPrefs.on    = on !== '0';
  if (kid !== null)   window.__vgPrefs.kid   = kid === '1';
  if (voice !== null) window.__vgPrefs.voice = voice === '1';
  return window.__vgPrefs;
};
window.__vgSetPref = function(key, value){
  try { if (window.storage) window.storage.set(key, value); } catch (_) {}
};

/* Multi-provider chat call so a guide can reuse the key the
   visitor already gave the app (ghostwriter / detective) when it
   runs standalone. Mirrors the Piskel adapter's provider table. */
window.__vgProviders = {
  async call(provider, key, model, sys, usr){
    if (!key) { const e = new Error('no key'); e.vgNoAI = true; throw e; }
    if (provider === 'openai') {
      if (!model) throw new Error('fill the model box next to the key');
      const r = await fetch('https://api.openai.com/v1/chat/completions', { method: 'POST',
        headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + key },
        body: JSON.stringify({ model, messages: [{ role: 'system', content: sys }, { role: 'user', content: usr }] }) });
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

/* Kid-mode and voice toggles appended to the guide panel head.
   The hunt wires its own existing #kidMode/#voiceOn instead. */
window.__vgPanelToggles = function(api){
  const head = document.getElementById('vgHead');
  if (!head) return;
  const wrap = document.createElement('span');
  wrap.id = 'vgTmToggles';
  head.style.flexWrap = 'wrap';                       /* never clip: overflow wraps to a new row */
  wrap.style.cssText = 'display:flex;gap:10px;margin-left:auto;font-size:11px;align-items:center';
  wrap.innerHTML =
    '<label style="display:flex;gap:3px;align-items:center;cursor:pointer;white-space:nowrap">' +
    '<input type="checkbox" id="vgKid"> kid</label>' +
    '<label style="display:flex;gap:3px;align-items:center;cursor:pointer;white-space:nowrap">' +
    '<input type="checkbox" id="vgVoice"> voice</label>';
  head.appendChild(wrap);
  const kid = wrap.querySelector('#vgKid'), voice = wrap.querySelector('#vgVoice');
  kid.checked = window.__vgPrefs.kid; voice.checked = window.__vgPrefs.voice;
  kid.onchange = () => { window.__vgPrefs.kid = kid.checked; window.__vgSetPref('tm-guide-kid', kid.checked ? '1' : '0'); };
  voice.onchange = () => { window.__vgPrefs.voice = voice.checked; window.__vgSetPref('tm-guide-voice', voice.checked ? '1' : '0'); };
  window.addEventListener('message', e => {
    const m = e.data; if (!m || m.tm !== 'pref') return;
    if (m.key === 'tm-guide-kid') kid.checked = m.value === '1';
    if (m.key === 'tm-guide-voice') voice.checked = m.value === '1';
  });
};

/* Boot gate: create the guide only when tm-guide-on allows it, and
   boot live if the shell switches it on later. */
window.__vgDeferBoot = async function(create){
  /* the app itself is a module script (runs after classic scripts,
     before load); the bridge attaches window.storage/window.claude
     during parse. Waiting for load puts the guide after both. */
  if (document.readyState !== 'complete')
    await Promise.race([
      new Promise(r => window.addEventListener('load', r, { once: true })),
      new Promise(r => setTimeout(r, 4000)),   /* artifact hosts can stall load; the guide must still appear */
    ]);
  try { await window.__vgLoadPrefs(); } catch (_) {}
  if (window.__vgPrefs.on) { create(); return; }
  const once = e => {
    const m = e.data;
    if (m && m.tm === 'pref' && m.key === 'tm-guide-on' && m.value === '1') {
      window.removeEventListener('message', once); create();
    }
  };
  window.addEventListener('message', once);
};
})();

