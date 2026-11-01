/* ============================================================
   tm-ai.js — which AI answers: one setting for the whole app.
   Spliced (by build-release) into the shell and into every room,
   wrapped in one function together with src/llm.js, so PROVIDERS,
   KEYED, callLLM, listModels, pickModel and keylessAvailable are
   in scope here. Exposes window.__tmAI, read by every guide (via
   tm-prelude's __vgRememberedKey / __vgProviders), the Nursery's
   ghostwriter and the Hunt's detective, and adds a 🔑 button to
   the guide panel that opens the settings.

   Storage (all tm-, every access guarded: an artifact or a private
   window may refuse it):
     tm-ai          localStorage  {provider, model:{p: id|''}}  '' = newest of the family
     tm-remember    localStorage  '1' when keys are remembered (the ghostwriter's flag)
     tm-key-<p>     localStorage when remembered, else sessionStorage (this visit only)
     tm-ai-models   localStorage  {p:{at, list:[{id,name}]}} what the key could use, when last asked
   Rooms are same-origin frames, so they all see the same storage;
   a 'storage' event tells the other frames something changed.
   ============================================================ */
if (!window.__tmAI) (function(){
const mem = {};
const store = (area) => ({
  get(k){ try { const v = window[area].getItem(k); return v === null ? (mem[area + k] ?? null) : v; } catch (_) { return mem[area + k] ?? null; } },
  set(k, v){ mem[area + k] = v; try { window[area].setItem(k, v); } catch (_) {} },
  del(k){ delete mem[area + k]; try { window[area].removeItem(k); } catch (_) {} },
});
const L = store('localStorage'), S = store('sessionStorage');
const json = (s, d) => { try { return s ? JSON.parse(s) : d; } catch (_) { return d; } };

/* inside a claude.ai artifact the page may not call other sites at
   all: keys cannot work there, and the guide uses built-in Claude */
const inArtifact = (() => { try {
  if (/claudeusercontent\.com$/i.test(location.hostname)) return true;
  return Array.from(location.ancestorOrigins || []).some(o => /claudeusercontent\.com|claude\.(ai|com)/i.test(o));
} catch (_) { return false; } })();

const remember = () => L.get('tm-remember') === '1';
const keyFor = p => ((remember() ? L.get('tm-key-' + p) : S.get('tm-key-' + p)) || L.get('tm-key-' + p) || S.get('tm-key-' + p) || '').trim();
function state(){
  const s = json(L.get('tm-ai'), null) || {};
  if (!KEYED.includes(s.provider)) s.provider = KEYED.find(p => keyFor(p)) || 'anthropic';   // an older visit's remembered key decides
  s.model = s.model || {};
  return s;
}
const cached = p => (json(L.get('tm-ai-models'), {})[p] || null);
const modelsFor = p => { const c = cached(p); return c && c.list && c.list.length ? c.list : PROVIDERS[p].models.map(id => ({ id, name: id })); };
const autoModel = p => pickModel(p, modelsFor(p).map(m => m.id));
const modelFor = p => state().model[p] || autoModel(p);

const listeners = [];
function changed(){ for (const f of listeners) { try { f(); } catch (_) {} } }
window.addEventListener('storage', e => { if (e.key && /^tm-(ai|key-|remember)/.test(e.key)) { changed(); render(); } });

function set(patch){
  const s = state();
  if (patch.provider && KEYED.includes(patch.provider)) s.provider = patch.provider;
  const p = patch.for || s.provider;
  if ('model' in patch) s.model[p] = patch.model || '';
  L.set('tm-ai', JSON.stringify({ provider: s.provider, model: s.model }));
  if ('remember' in patch) {
    L.set('tm-remember', patch.remember ? '1' : '');
    for (const q of KEYED) {   // move the keys to the other kind of storage
      const k = keyFor(q);
      L.del('tm-key-' + q); S.del('tm-key-' + q);
      if (k) (patch.remember ? L : S).set('tm-key-' + q, k);
    }
  }
  if ('key' in patch) {
    const k = String(patch.key || '').trim();
    L.del('tm-key-' + p); S.del('tm-key-' + p);
    if (k) (remember() ? L : S).set('tm-key-' + p, k);
  }
  changed();
}

/* ask the provider which models this key can use; cache the answer */
let asking = {};
async function refresh(p){
  p = p || state().provider;
  const key = keyFor(p);
  if (!key || inArtifact) return null;
  if (asking[p]) return asking[p];
  asking[p] = (async () => {
    try {
      const list = await listModels(p, key);
      if (!list.length) throw new Error('no chat models listed');
      const all = json(L.get('tm-ai-models'), {}); all[p] = { at: Date.now(), list: list.slice(0, 40) };
      L.set('tm-ai-models', JSON.stringify(all)); changed();
      return { ok: true, list };
    } catch (e) {
      // Anthropic's model list answers a wrong key without the header a browser needs, so
      // a bad Claude key looks like a blocked request rather than "invalid key"
      const blocked = /failed to fetch|network|load failed/i.test(String(e && e.message));
      return { ok: false, error: !blocked ? String(e && e.message || e)
        : p === 'anthropic' ? 'Claude did not accept that key, or the network is down' : 'the request was blocked or the network is down' };
    } finally { asking[p] = null; }
  })();
  return asking[p];
}
/* once a day at most, quietly, so "newest" keeps meaning newest */
setTimeout(() => { const p = state().provider, c = cached(p);
  if (keyFor(p) && (!c || Date.now() - c.at > 864e5)) refresh(p); }, 3000);

function config(){ const p = state().provider; return { provider: p, key: keyFor(p), model: modelFor(p) }; }
function callWith(provider, key, model, sys, usr){
  if (!key) { const e = new Error('no key'); e.vgNoAI = true; throw e; }
  return callLLM({ provider, key, model: model || modelFor(provider), system: sys, prompt: usr, maxTokens: 1500 });
}

/* ---------- the 🔑 panel inside the guide panel ---------- */
const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
let box = null, note = '';
const CSS =
  '#vgAI{flex:0 0 auto;padding:10px 12px 12px;border-bottom:1px solid var(--line,rgba(255,255,255,.14));background:rgba(0,0,0,.14);' +
  'font-size:12.5px;color:var(--ink,#eef4ff);display:flex;flex-direction:column;gap:8px;max-height:60vh;overflow-y:auto}' +
  '#vgAI b{font-size:13px}#vgAI .ai-m{color:var(--muted,#b7c5e8);line-height:1.45}' +
  '#vgAI .ai-seg{display:flex;gap:6px}#vgAI .ai-seg button{flex:1;padding:6px 4px;border-radius:9px;cursor:pointer;font:inherit;font-weight:700;' +
  'border:1px solid var(--line,rgba(255,255,255,.14));background:rgba(255,255,255,.06);color:var(--ink,#eef4ff)}' +
  '#vgAI .ai-seg button.on{background:linear-gradient(180deg,#aaf4ff,#65d4ea);color:#06111f;border-color:transparent}' +
  '#vgAI .ai-row{display:flex;gap:6px;align-items:center}#vgAI .ai-row>span{flex:0 0 42px;color:var(--muted,#b7c5e8)}' +
  '#vgAI input[type=password],#vgAI input[type=text],#vgAI select{flex:1;min-width:0;font:inherit;padding:6px 8px;border-radius:8px;' +
  'border:1px solid var(--line,rgba(255,255,255,.14));background:rgba(0,0,0,.25);color:var(--ink,#eef4ff)}' +
  '#vgAI select option{background:#111a33;color:#eef4ff}' +
  '#vgAI .ai-small{font:inherit;font-size:11.5px;padding:4px 8px;border-radius:8px;cursor:pointer;border:1px solid var(--line,rgba(255,255,255,.14));' +
  'background:rgba(255,255,255,.06);color:var(--accent,#7bdff2)}' +
  '#vgAI a{color:var(--accent,#7bdff2)}#vgAI label{display:flex;gap:6px;align-items:flex-start;cursor:pointer}';

function render(){
  if (!box || box.style.display === 'none') return;
  const s = state(), p = s.provider, P = PROVIDERS[p], key = keyFor(p);
  if (inArtifact) {
    box.innerHTML = '<b>Which AI answers?</b><div class="ai-m">Here, inside Claude, the guides, the ghostwriter and the detective use Claude through your claude.ai account, so no key is needed. ' +
      'Keys for Claude, ChatGPT or Gemini work when Tiny Mind runs as an ordinary web page, at <a href="https://toontalk.github.io/tinymind/" target="_blank" rel="noopener">toontalk.github.io/tinymind</a>.</div>' +
      '<div class="ai-row"><button class="ai-small" data-ai="done">done</button></div>';
    return;
  }
  const list = modelsFor(p), auto = autoModel(p), chosen = s.model[p] || '';
  const c = cached(p);
  const known = list.some(m => m.id === chosen);
  box.innerHTML =
    '<b>Which AI answers?</b>' +
    '<div class="ai-seg">' + KEYED.map(q => `<button data-ai-p="${q}" class="${q === p ? 'on' : ''}" title="${esc(PROVIDERS[q].company)}">${esc(PROVIDERS[q].label)}</button>`).join('') + '</div>' +
    `<div class="ai-row"><span>key</span><input type="password" id="vgAIKey" autocomplete="off" spellcheck="false" placeholder="paste your ${esc(P.label)} key" value="${esc(key)}">` +
    '<button class="ai-small" data-ai="show">show</button></div>' +
    `<label><input type="checkbox" id="vgAIRemember"${remember() ? ' checked' : ''}> <span>remember keys on this computer (untick on a shared computer: then a key lasts until this tab closes)</span></label>` +
    '<div class="ai-row"><span>model</span><select id="vgAIModel">' +
      `<option value="">newest ${esc(P.familyName)}: ${esc(auto)}</option>` +
      list.map(m => `<option value="${esc(m.id)}"${m.id === chosen ? ' selected' : ''}>${esc(m.id)}</option>`).join('') +
      (chosen && !known ? `<option value="${esc(chosen)}" selected>${esc(chosen)}</option>` : '') +
      '<option value="?">another model…</option>' +
    '</select></div>' +
    `<div class="ai-m" id="vgAINote">${esc(note || (c ? `Models checked with your key ${ago(c.at)}.` : key ? 'Checking which models your key can use…' : `Without a key the list is the one built in on 1 November 2026; with a key I ask ${esc(P.company)} for the newest.`))}</div>` +
    '<div class="ai-m">This choice is used by the guide in every room, the Nursery\'s ghostwriter and the Hunt\'s detective. The key goes only to ' + esc(P.company) + '.</div>' +
    `<div class="ai-row" style="flex-wrap:wrap"><a href="${esc(P.keyPage)}" target="_blank" rel="noopener">get a ${esc(P.label)} key</a>` +
    '<span style="flex:1"></span>' + (key ? '<button class="ai-small" data-ai="check">check again</button><button class="ai-small" data-ai="forget">forget key</button>' : '') +
    '<button class="ai-small" data-ai="done">done</button></div>';
}
const ago = t => { const m = Math.round((Date.now() - t) / 6e4); return m < 2 ? 'just now' : m < 120 ? m + ' minutes ago' : Math.round(m / 60) < 48 ? Math.round(m / 60) + ' hours ago' : Math.round(m / 1440) + ' days ago'; };

async function check(){
  const p = state().provider;
  note = `Asking ${PROVIDERS[p].company} which models your key can use…`; render();
  const r = await refresh(p);
  note = !r ? '' : r.ok ? `Your key can use ${r.list.length} chat models; the newest ${PROVIDERS[p].familyName} is ${autoModel(p)}.`
    : `I could not get the model list (${r.error}), so the menu shows the built-in names. If answers fail too, check the key.`;
  render();
}

function wire(){
  box.addEventListener('click', e => {
    const b = e.target.closest('button'); if (!b) return;
    if (b.dataset.aiP) { note = ''; set({ provider: b.dataset.aiP }); render(); if (keyFor(b.dataset.aiP) && !cached(b.dataset.aiP)) check(); return; }
    const a = b.dataset.ai;
    if (a === 'done') toggle(false);
    if (a === 'show') { const k = box.querySelector('#vgAIKey'); if (k) { k.type = k.type === 'password' ? 'text' : 'password'; b.textContent = k.type === 'password' ? 'show' : 'hide'; } }
    if (a === 'forget') { note = ''; set({ key: '' }); render(); }
    if (a === 'check') check();
  });
  box.addEventListener('change', e => {
    const t = e.target;
    if (t.id === 'vgAIKey') { note = ''; set({ key: t.value }); if (t.value.trim()) check(); else render(); }
    if (t.id === 'vgAIRemember') set({ remember: t.checked });
    if (t.id === 'vgAIModel') {
      if (t.value === '?') {
        const row = t.parentElement, inp = document.createElement('input');
        inp.type = 'text'; inp.placeholder = 'type the exact model name'; inp.spellcheck = false;
        inp.onchange = () => { if (inp.value.trim()) set({ model: inp.value.trim() }); render(); };
        t.replaceWith(inp); inp.focus(); return;
      }
      set({ model: t.value });
    }
  });
}
function toggle(open){
  if (!box) return;
  box.style.display = open ? 'flex' : 'none';
  const btn = document.getElementById('vgKey'); if (btn) btn.setAttribute('aria-expanded', open ? 'true' : 'false');
  if (open) { note = ''; render(); const p = state().provider, c = cached(p); if (keyFor(p) && (!c || Date.now() - c.at > 864e5)) check(); }
}
function mount(){
  const controls = document.getElementById('vgControls');
  if (!controls || document.getElementById('vgKey')) return !!controls;
  const st = document.createElement('style'); st.textContent = CSS; document.head.appendChild(st);
  const btn = document.createElement('button');
  btn.id = 'vgKey'; btn.className = 'vg-btn'; btn.textContent = '\u{1F511}';
  btn.title = 'Choose which AI answers (Claude, ChatGPT or Gemini) and give it your key';
  btn.setAttribute('aria-expanded', 'false');
  controls.appendChild(btn);
  box = document.createElement('div'); box.id = 'vgAI'; box.style.display = 'none';
  controls.insertAdjacentElement('afterend', box);
  wire();
  btn.onclick = () => toggle(box.style.display === 'none');
  return true;
}
/* the guide panel is built after this script runs: mount as soon as it appears */
if (!mount()) {
  const mo = new MutationObserver(() => { if (mount()) mo.disconnect(); });
  const start = () => mo.observe(document.body, { childList: true, subtree: true });
  if (document.body) start(); else document.addEventListener('DOMContentLoaded', () => { if (!mount()) start(); });
}

window.__tmAI = { PROVIDERS, KEYED, inArtifact, config, callWith, set, refresh, modelsFor, modelFor, autoModel, keyFor,
  remember, provider: () => state().provider, onChange: f => listeners.push(f), open: () => toggle(true) };
})();
