/* tiny-mind frame bridge — inert when the app runs standalone.
   build-release.js injects this as a plain <script> ahead of each app's module script
   (dist singles AND the mega templates), so an app inside the shell's iframe gets:
   window.claude + window.storage proxied over postMessage RPC to the shell,
   load-pair / json / arc-selection pushes routed to the app's public hooks,
   __tmBubbleExport to hand downloads up to the shell, and __tmPublishModel to
   register a completed training run without forcing a download. */
(function () {
  if (window.parent === window) return;
  window.__tmBridge = true;
  // inside the combined app the shell's "mind" menu chooses and adds models, so each
  // room's own drop-a-file hints and load buttons (marked tm-file-ctl) step aside
  document.documentElement.classList.add('tm-embedded');
  { const st = document.createElement('style'); st.textContent = '.tm-embedded .tm-file-ctl{display:none !important}';
    (document.head || document.documentElement).appendChild(st); }
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
        if (window.__tmLoadPair) window.__tmLoadPair({ name: m.ckName, buf: m.ck, meta: m.modelMeta || null }, { name: m.tokName, buf: m.tok });
        else if (n > 0) setTimeout(() => tryLoad(n - 1), 150);
      };
      tryLoad(100);
    } else if (m.tm === 'json') {
      const tryRoute = (n) => {
        if (window.__tmRouteJson) window.__tmRouteJson(m.name, m.text);
        else if (n > 0) setTimeout(() => tryRoute(n - 1), 150);
      };
      tryRoute(100);
    } else if (m.tm === 'arc') {
      const tryArc = (n) => {
        if (window.__tmArc) window.__tmArc(m.manifest, m.source);
        else if (n > 0) setTimeout(() => tryArc(n - 1), 150);
      };
      tryArc(100);
    } else if (m.tm === 'select-arc') {
      const trySelectArc = (n) => {
        if (window.__tmSelectArc) window.__tmSelectArc(m.arc);
        else if (n > 0) setTimeout(() => trySelectArc(n - 1), 150);
      };
      trySelectArc(100);
    }
  });
  window.__tmRPC = rpc;   // generic escape hatch (arc.snapshot etc.); inert standalone
  // relay Claude to the shell only when the shell itself has it (inside a claude.ai
  // artifact); on an ordinary web page "built-in Claude" must not be offered at all
  let shellHasClaude = false;
  try { shellHasClaude = !!parent.__tmKeyless; } catch (_) {}
  if (shellHasClaude && !(window.claude && typeof window.claude.complete === 'function'))
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
  window.__tmPublishModel = (name, bytes, meta) => {
    const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
    const buf = u8.buffer.slice(u8.byteOffset, u8.byteOffset + u8.byteLength);
    parent.postMessage({ tm: 'publish-model', name, bytes: buf, meta: meta || null }, '*');
  };
  window.addEventListener('load', () => parent.postMessage({ tm: 'ready', app: document.title }, '*'));
})();
