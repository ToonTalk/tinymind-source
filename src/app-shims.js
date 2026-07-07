// app-shims.js — environment shims shared by the three apps (merged back from the
// mega/artifact edition, 2026-07-06). Three problems these solve:
//   1. the claude.ai artifact sandbox refuses blob-URL Workers -> same-thread shim
//   2. persistence differs per host -> Store cascade (window.storage -> localStorage -> memory)
//   3. downloads are blocked in sandboxed/nested frames -> deliverFile ladder
// Browser-oriented but import-safe in Node (nothing runs at import time).

// Same-thread stand-in for a Worker, speaking the identical postMessage/onmessage
// protocol. The worker source is DOM-free and uses only globals present on the main
// thread, so it runs unchanged — just not off-thread. The training loop already yields
// via setTimeout(0) each step, so the page stays responsive between steps, but each
// individual step blocks paint for its duration.
export function makeInlineWorker(src) {
  const outbound = { message: [], error: [] };   // main-side listeners (worker -> main)
  const inbound = { message: [], error: [] };    // worker-side listeners (main -> worker)
  const later = (fn) => setTimeout(fn, 0);        // async delivery, like a real Worker
  const scope = {
    onmessage: null, onerror: null,
    postMessage(data) {                            // worker -> main; transfer list ignored (same realm)
      later(() => { const ev = { data }; if (api.onmessage) api.onmessage(ev); for (const l of outbound.message) l(ev); });
    },
    addEventListener(t, fn) { (inbound[t] || (inbound[t] = [])).push(fn); },
    removeEventListener(t, fn) { const a = inbound[t]; if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); } },
    importScripts() { throw new Error('importScripts is unavailable in the inline-worker fallback'); },
  };
  scope.self = scope;
  const api = {
    onmessage: null, onerror: null,
    postMessage(data) {                            // main -> worker
      later(() => { const ev = { data }; if (scope.onmessage) scope.onmessage(ev); for (const l of inbound.message) l(ev); });
    },
    addEventListener(t, fn) { (outbound[t] || (outbound[t] = [])).push(fn); },
    removeEventListener(t, fn) { const a = outbound[t]; if (a) { const i = a.indexOf(fn); if (i >= 0) a.splice(i, 1); } },
    terminate() {},
  };
  try {
    // eslint-disable-next-line no-new-func
    (new Function('self', src))(scope);            // bare globals (setTimeout, TextEncoder, …) resolve to window
  } catch (err) {
    later(() => { const ev = { message: String(err && err.message || err), error: err }; if (api.onerror) api.onerror(ev); for (const l of outbound.error) l(ev); });
  }
  return api;
}

// persistence — one build for every environment. Durable backend chosen by feature
// detection; all keys under "tm-". A synchronous in-memory cache is hydrated once at
// boot so sync read sites need no rewrite; writes fire-and-forget to the backend.
export function createStore() {
  const mem = new Map();
  const ws = (typeof window !== 'undefined' && window.storage && typeof window.storage.get === 'function')
    ? window.storage : null;
  const ls = (() => { try { localStorage.setItem('tm-probe', '1'); localStorage.removeItem('tm-probe'); return localStorage; } catch { return null; } })();
  async function persist(k, v) {
    if (ws) { try { v === null ? await ws.delete(k, false) : await ws.set(k, v, false); } catch { } }
    else if (ls) { try { v === null ? ls.removeItem(k) : ls.setItem(k, v); } catch { } }
  }
  return {
    backend: ws ? 'window.storage' : ls ? 'localStorage' : 'memory',
    get(k) { return mem.has(k) ? mem.get(k) : null; },              // sync read from cache
    set(k, v) { mem.set(k, v); persist(k, v); },                    // fire-and-forget
    del(k) { mem.delete(k); persist(k, null); },
    async hydrate() {                                               // fill the cache before boot restore
      if (ws) {
        try {
          const res = await ws.list('tm-', false);
          for (const k of (res && res.keys) || []) {
            try { const r = await ws.get(k, false); if (r && r.value != null) mem.set(k, r.value); } catch { }
          }
        } catch { }
      } else if (ls) {
        try { for (let i = 0; i < ls.length; i++) { const k = ls.key(i); if (k && k.startsWith('tm-')) mem.set(k, ls.getItem(k)); } } catch { }
      }
    },
  };
}

// File egress ladder: frame bridge -> real download (blob + download attr + _blank —
// the sandbox rewrites blob URLs cross-scheme, which disables the download attribute;
// a bare click then navigates the frame → "This content is blocked". _blank hands the
// navigation to a fresh top-level context, which downloads) -> clipboard for text ->
// storage backup for bytes when sandboxed.
export function makeDeliverFile({ status, store, isSandboxed }) {
  return function deliverFile(name, data, mime) {
    if (window.__tmBridge && window.__tmBubbleExport) {
      const u8 = typeof data === 'string' ? new TextEncoder().encode(data)
        : data instanceof Uint8Array ? data : new Uint8Array(data);
      window.__tmBubbleExport(name, u8);
      status(`saving ${name}…`);
      return true;
    }
    let clicked = false;
    try {
      const url = URL.createObjectURL(new Blob([data], { type: mime }));
      const a = document.createElement('a');
      a.href = url; a.download = name; a.target = '_blank'; a.rel = 'noopener';
      document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      clicked = true;
    } catch { }
    if (typeof data === 'string' && navigator.clipboard?.writeText) {
      navigator.clipboard.writeText(data).then(
        () => status(`saved ${name} — also copied to the clipboard as backup`),
        () => { if (!clicked) status(`could not save ${name} here — try the file:// or Pages edition`, 'warn'); });
    }
    if (isSandboxed() && typeof data !== 'string') {
      // belt: keep the bytes in browser storage too, in case the save dialog never appears
      try {
        const u8 = data instanceof Uint8Array ? data : new Uint8Array(data);
        let s = ''; const CH = 0x8000;
        for (let i = 0; i < u8.length; i += CH) s += String.fromCharCode.apply(null, u8.subarray(i, i + CH));
        store.set('tm-file-' + name, btoa(s));
      } catch { }
    }
    return clicked;
  };
}

// fetch the first path that answers — bridges the dev-server layout (../models/…),
// the flat Pages/mega layout, and dist-next-to-models
export async function fetchFirst(paths) {
  for (const p of paths) {
    try {
      const r = await fetch(p);
      if (r.ok) return r;
    } catch { }
  }
  throw new Error('none of ' + paths.join(', ') + ' reachable');
}
