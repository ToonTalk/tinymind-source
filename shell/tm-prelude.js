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

/* Kid-friendly odds (wave 5). Percentages are a poor first
   language for chance: "3 out of 100" and "about 1 in 2,000" are
   concrete, and a hundred dots you can look at is better still.
   phrase() picks the denominator by magnitude; grid() draws the
   dots; chip() offers the picture in the guide panel. */
window.__vgOdds = {
  phrase(p, kid, opts){
    const tight = !!(opts && opts.compact);          // compact: fits a table cell ("7 in 100")
    if (p == null || !isFinite(p) || p <= 0) return kid ? (tight ? 'never' : 'almost never') : 'about 0%';
    if (p >= 0.995) return kid ? (tight ? 'always' : 'almost always') : 'about 100%';
    const nice = n => (n >= 1000 ? Math.round(n / 100) * 100 : Math.round(n)).toLocaleString();
    if (kid && tight) {
      if (p >= 0.05)     return Math.round(p * 100) + ' in 100';
      if (p >= 0.005)    return Math.round(p * 1000) + ' in 1000';
      if (p >= 0.000005) return '1 in ' + nice(1 / p);
      return Math.max(1, Math.round(p * 1e6)) + ' in a million';
    }
    if (kid) {
      if (p >= 0.05)     return Math.round(p * 100) + ' out of 100';
      if (p >= 0.005)    return Math.round(p * 1000) + ' out of 1000';
      if (p >= 0.000005) return 'about 1 in ' + nice(1 / p);
      return Math.max(1, Math.round(p * 1e6)) + ' out of a million';
    }
    const pct = p >= 0.01 ? (p * 100).toFixed(0) + '%' : (p * 100).toPrecision(2) + '%';
    return p < 0.05 ? pct + ' (about 1 in ' + nice(1 / p) + ')' : pct;
  },
  base(p){ return p >= 0.01 ? 100 : p >= 0.001 ? 1000 : 10000; },
  grid(p, label){
    const base = this.base(p), lit = Math.max(1, Math.round(p * base));
    const cols = base === 100 ? 10 : base === 1000 ? 40 : 100;
    const rows = base / cols, cell = Math.min(28, Math.floor(300 / cols));
    const w = cols * cell, h = rows * cell;
    const back = document.createElement('div');
    back.id = 'vgOddsBack';
    back.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,.72);z-index:2147483000;' +
      'display:flex;align-items:center;justify-content:center;padding:16px';
    const card = document.createElement('div');
    card.style.cssText = 'background:#151a2e;color:#e8ecff;border-radius:12px;padding:16px 18px;' +
      'max-width:min(92vw,420px);font:14px/1.45 system-ui,sans-serif;text-align:center';
    const cv = document.createElement('canvas');
    cv.width = w; cv.height = h;
    cv.style.cssText = 'width:' + w + 'px;height:' + h + 'px;max-width:100%;image-rendering:pixelated';
    const ctx = cv.getContext('2d');
    if (ctx) for (let i = 0; i < base; i++) {
      const x = (i % cols) * cell, y = Math.floor(i / cols) * cell, r = Math.max(1, cell / 2 - 1);
      ctx.beginPath();
      ctx.arc(x + cell / 2, y + cell / 2, r, 0, 6.284);
      ctx.fillStyle = i < lit ? '#ff9ec2' : '#2b3252';
      ctx.fill();
    }
    const cap = document.createElement('div');
    cap.style.cssText = 'margin-top:10px';
    cap.textContent = lit + ' out of ' + base.toLocaleString() + ' \u2014 ' + (label || "that is the mind's guess") + '.';
    const hint = document.createElement('div');
    hint.style.cssText = 'margin-top:6px;opacity:.7;font-size:12px';
    hint.textContent = 'tap anywhere to close';
    card.appendChild(cv); card.appendChild(cap); card.appendChild(hint);
    back.appendChild(card);
    back.onclick = () => back.remove();
    document.body.appendChild(back);
  },
  chip(p, label){
    if (p == null || !isFinite(p) || p <= 0) return;
    const log = document.getElementById('vgLog');
    if (!log) return;
    const b = document.createElement('button');
    b.className = 'vg-odds-chip';
    b.style.cssText = 'display:block;margin:4px 0 6px;padding:3px 9px;border-radius:9px;cursor:pointer;' +
      'font:12px system-ui,sans-serif;background:#2b3252;color:#e8ecff;border:0';
    b.textContent = '\u{1F440} see it as dots';
    b.onclick = () => window.__vgOdds.grid(p, label);
    log.appendChild(b); log.scrollTop = log.scrollHeight;
  },
  chipAfterSpeak(p, label){ setTimeout(() => { try { this.chip(p, label); } catch (_) {} }, 350); },
};

/* Multi-provider chat call so a guide can reuse the key the
   visitor already gave the app (ghostwriter / detective) when it
   runs standalone. Mirrors the Piskel adapter's provider table. */
window.__vgProviders = {
  async call(provider, key, model, sys, usr){
    if (!key) { const e = new Error('no key'); e.vgNoAI = true; throw e; }
    if (provider === 'openai') {
      model = model || 'gpt-4o-mini';
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

/* The one formatter the app's own readouts call, so a percentage never
   survives kid mode. Probabilities only: a percentage-point *change*
   stays a percentage, because "3 in 100" would misdescribe it. */
window.__vgPct = function(p, dp){
  if (p == null || !isFinite(p)) return '\u2014';
  if (window.__vgPrefs && window.__vgPrefs.kid && window.__vgOdds)
    return window.__vgOdds.phrase(p, true, { compact: true });
  return (100 * p).toFixed(dp == null ? 1 : dp) + '%';
};

/* SHOW must not spotlight a view that has not been drawn yet: an
   empty div gets an invisible halo and the visitor sees nothing. If
   the target is missing, hidden, or empty, point at the control that
   produces it and say so. (offsetParent is unreliable under test
   harnesses, so visibility is judged by the app's own .hide
   convention plus computed display.) */
window.__vgHidden = function(el){
  if (!el) return true;
  if (el.hidden || /(^|\s)hide(\s|$)/.test(el.className || '')) return true;
  try { return el.ownerDocument.defaultView.getComputedStyle(el).display === 'none'; } catch (_) { return false; }
};
window.__vgShow = function(api, spec){
  const t = typeof spec === 'string' ? { sel: spec } : (spec || {});
  const el = t.sel && document.querySelector(t.sel);
  const empty = !el || !((el.textContent || '').trim() || el.children.length || el.value);
  if (el && !window.__vgHidden(el) && !empty) { api.spot(t.sel); return true; }
  if (t.needs) api.spot(t.needs);
  api.emit('showBlocked', { why: t.why, kidWhy: t.kidWhy });
  return false;
};
window.__vgShowBlockedEvent = {
  priority: 'moment', cooldownMs: 1200,
  lines: p => (p && p.why) || 'That view has not been drawn yet — press the control I am pointing at first.',
  kidLines: p => (p && (p.kidWhy || p.why)) || 'That is not on the screen yet! Press the button I am pointing at first.',
};

/* The ghostwriter and the detective remember keys in localStorage
   under tm-key-<provider>; srcdoc frames share the shell's origin, so
   any guide can find a key the visitor already gave the app. */
window.__vgRememberedKey = function(){
  const DEF = { anthropic: 'claude-sonnet-4-6', openai: 'gpt-4o-mini', gemini: 'gemini-3.5-flash' };
  try {
    for (const p of ['anthropic', 'openai', 'gemini']) {
      const k = localStorage.getItem('tm-key-' + p);
      if (k && k.trim()) return { provider: p, key: k.trim(), model: DEF[p] };
    }
  } catch (_) {}
  return null;
};

/* One ask path for every guide. window.claude is the bridge's RPC to
   the shell, which can only serve it inside a Claude artifact -- on an
   ordinary page that call reaches the keyless endpoint with no key and
   no browser-access header and dies of CORS. So: try it, and if it
   fails while the app holds a key of its own, use that instead. */
window.__vgAsk = async function(sys, usr, cfgFn){
  const cfg = (cfgFn && cfgFn()) || {};
  if (window.claude && typeof window.claude.complete === 'function') {
    try { return await window.claude.complete(sys + '\n\n' + usr); }
    catch (e) { if (!cfg.key) throw e; }
  }
  if (!cfg.key) { const e = new Error('no key'); e.vgNoAI = true; throw e; }
  return window.__vgProviders.call(cfg.provider, cfg.key, cfg.model, sys, usr);
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
