/* tiny-mind frame bridge — inert when the app runs standalone.
   build-release.js injects this as a plain <script> ahead of each app's module script
   (dist singles AND the mega templates), so an app inside the shell's iframe gets:
   window.claude + window.storage proxied over postMessage RPC to the shell,
   load-pair / json pushes routed to the app's __tmLoadPair / __tmRouteJson hooks,
   and __tmBubbleExport to hand files up to the shell (nested frames cannot download). */
(function () {
  if (window.parent === window) return;
  window.__tmBridge = true;
  let seq = 0; const pending = new Map();
  function rpc(method, args) {
    return new Promise((resolve, reject) => {
      const id = ++seq; pending.set(id, { resolve, reject });
      parent.postMessage({ tm: 'rpc', id, method, args }, '*');
      setTimeout(() => { const p = pending.get(id); if (p) { pending.delete(id); p.reject(new Error(method + ' timed out')); } }, 60000);
    });
  }
  window.addEventListener('message', (e) => {
    const m = e.data; if (!m || !m.tm) return;
    if (m.tm === 'rpc-reply') {
      const p = pending.get(m.id); if (!p) return; pending.delete(m.id);
      if (m.ok) p.resolve(m.value); else p.reject(new Error(m.error || 'rpc failed'));
    } else if (m.tm === 'load-pair') {
      const tryLoad = (n) => {
        if (window.__tmLoadPair) window.__tmLoadPair({ name: m.ckName, buf: m.ck }, { name: m.tokName, buf: m.tok });
        else if (n > 0) setTimeout(() => tryLoad(n - 1), 150);
      };
      tryLoad(100);
    } else if (m.tm === 'json') {
      const tryRoute = (n) => {
        if (window.__tmRouteJson) window.__tmRouteJson(m.name, m.text);
        else if (n > 0) setTimeout(() => tryRoute(n - 1), 150);
      };
      tryRoute(100);
    }
  });
  if (!(window.claude && typeof window.claude.complete === 'function'))
    window.claude = { complete: (prompt) => rpc('claude.complete', [prompt]) };
  if (!window.storage)
    window.storage = {
      get: (k, s) => rpc('storage.get', [k, !!s]),
      set: (k, v, s) => rpc('storage.set', [k, v, !!s]),
      delete: (k, s) => rpc('storage.delete', [k, !!s]),
      list: (p, s) => rpc('storage.list', [p || '', !!s]),
    };
  window.__tmBubbleExport = (name, u8) => {
    const buf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    parent.postMessage({ tm: 'export', name, bytes: buf }, '*');
  };
  window.addEventListener('load', () => parent.postMessage({ tm: 'ready', app: document.title }, '*'));
})();
