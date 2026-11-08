/* tm-report.js — ERROR REPORTS, ASKED FOR FIRST (Ken, 8 Nov 2026; the same design as Ani,
   ToonTalk 3D and tt-wasm). Spliced by build-release into the shell and into every room.

   When Tiny Mind runs into an error — a page script throws, a promise fails with nobody
   catching it, or a room's compute engine (its worker) reports a failure — a small notice asks
   whether to send Ken a report. Nothing is sent unless the visitor presses Send: Sentry's
   script is not even fetched before then. A report has the error, where in the code it
   happened, which room, the page's address and the browser. Removed first: every value this
   site keeps in the browser (stories, storybook, character and mind names, AI keys and
   settings), whatever is typed in any box in any room, and anything shaped like an AI key.

   One notice, in the shell: the rooms run in same-origin srcdoc frames and hand their errors
   up (window.parent.__tmReport). A page opened on its own (a dist single) shows its own.
   On only on toontalk.github.io with a DSN: never in the claude.ai artifact, on localhost or
   from a file. Checks: tests/e21-error-reports.test.js; window.__reportOn and
   window.__sentryStub are its seams. */
(function () {
  'use strict';
  var SENTRY_DSN = 'https://1a44467fda170918840e8716c2408c4f@o61153.ingest.us.sentry.io/4512394709041152';   // Ken's tinymind project in Sentry (public; it only accepts reports)
  var SENTRY_SDK = { src: 'https://browser.sentry-cdn.com/10.50.0/bundle.min.js',
                     integrity: 'sha384-9cVKNfUSoIRWDL3QQHe0fgJboGyb51IyiAv8x+KeuylA5GArkSaqyGFFm14hqajd' };
  var REPORT_KEY = 'tm-report';
  var SITE = 'toontalk.github.io';

  // which room this page is (a srcdoc frame knows from its <iframe data-app>)
  function roomName() {
    try { var f = window.frameElement; if (f && f.dataset && f.dataset.app) return f.dataset.app; } catch (e) {}
    return window === window.top ? 'front door' : 'a room';
  }
  function hostOf(w) { try { return w.location.hostname; } catch (e) { return null; } }   // null: another site's frame

  // a room's compute engine (its worker) only sends back a message. Ones that mean the visitor
  // gave it a file that is not a mind, or asked before something was loaded, are not faults
  var NOT_A_FAULT = /^(bin-reader|Tokenizer:|no zip end|bad central directory|zip method|not a \.jlens|unsupported \.jlens|\.jlens:|not a Tiny Eye manifest|the weights are shorter|this mind is not made of experts|no mind made of experts is loaded|no eye loaded yet|the training snapshots are not here yet)/;
  window.tmReportWorker = function (message) {
    if (!NOT_A_FAULT.test(String(message || '')) && window.tmReportFault) window.tmReportFault('the compute engine', String(message || 'unknown error'));
  };

  // A ROOM IN THE SHELL: hand every error up, and let the shell ask
  var up = null;
  try { if (window !== window.parent && window.parent.__tmReport) up = window.parent.__tmReport; } catch (e) { up = null; }
  if (up) {
    var room = roomName();
    var pass = function (err, where) { try { up.keep(err, where || ('in the ' + room + ' room')); } catch (e) {} };
    window.addEventListener('error', function (ev) { pass(ev.error || ev.message); });
    window.addEventListener('unhandledrejection', function (ev) { pass(ev.reason); });
    window.tmReportFault = function (where, e) { pass(e, where + ' (' + room + ' room)'); };
    return;
  }

  var report = { sdk: null, consent: new Set(), held: [], open: false, sent: 0 };
  // the site's own page only: the top window, or a frame inside a page of the same site
  function reportOn() {
    if (window.__reportOn) return true;
    if (!SENTRY_DSN || location.hostname !== SITE) return false;
    return window === window.top || hostOf(window.top) === SITE;
  }
  function reportNever() { try { return localStorage.getItem(REPORT_KEY) === 'never'; } catch (e) { return false; } }
  function reportSentry() {
    if (window.__sentryStub) return Promise.resolve(window.__sentryStub);   // a check's stand-in
    if (report.sdk) return report.sdk;
    report.sdk = new Promise(function (resolve) {
      var sc = document.createElement('script');
      sc.src = SENTRY_SDK.src; sc.integrity = SENTRY_SDK.integrity; sc.crossOrigin = 'anonymous';
      sc.onload = function () {
        var S = window.Sentry;
        try {
          S.init({
            dsn: SENTRY_DSN, sendDefaultPii: false, defaultIntegrations: false, sendClientReports: false,
            maxBreadcrumbs: 0, integrations: [S.linkedErrorsIntegration()],   // no handlers of its own
            beforeBreadcrumb: function () { return null; },
            beforeSend: reportScrubbed,
            initialScope: { tags: { app: 'tinymind' } }
          });
          resolve(S);
        } catch (e) { resolve(null); }
      };
      sc.onerror = function () { resolve(null); };
      document.head.appendChild(sc);
    });
    return report.sdk;
  }
  // an Error from a room's frame is not an instanceof this window's Error: go by its shape
  function asError(err) {
    if (err && typeof err === 'object' && typeof err.message === 'string') return err;
    var e = new Error(typeof err === 'string' ? err : (function () { try { return JSON.stringify(err); } catch (e2) { return String(err); } })());
    e.__noStack = true;   // made here, so its stack is this file's, not the fault's
    return e;
  }
  // keep an error, and ask about it
  function reportKeep(err, where) {
    if (!reportOn() || reportNever() || err == null) return;
    var e = asError(err);
    // a browser's own noise, a script of somebody else's with nothing to show, and the network
    // being away (an AI provider out of reach is not a bug in Tiny Mind)
    if (/ResizeObserver loop|^Script error\.?$|^Failed to fetch$|NetworkError when attempting|^Load failed$/.test(e.message || '')) return;
    if (where) { try { e.__where = where; } catch (x) {} }
    // the same error twice is one report, and a page in trouble asks about at most ten
    if (report.held.some(function (x) { return x.message === e.message && x.stack === e.stack && x.__where === e.__where; }) || report.held.length >= 10) return;
    report.held.push(e);
    reportAsk();
  }
  function reportFault(where, e) { try { reportKeep(e, where); } catch (e2) {} }
  window.addEventListener('error', function (ev) { reportKeep(ev.error || ev.message); });
  window.addEventListener('unhandledrejection', function (ev) { reportKeep(ev.reason); });

  // WHAT MUST NEVER LEAVE: every value kept for this site in the browser (and every string inside
  // one kept as JSON: the storybook, the minds' names, the AI settings and remembered keys), and
  // what is typed in any box of the shell or any room (a key not remembered, a story being
  // written, a character's name); longest first
  function reportSecrets() {
    var out = new Set();
    var add = function (v, least) { if (typeof v === 'string' && v.trim().length >= (least || 12)) out.add(v.trim()); };
    var leaves = function (x) { if (typeof x === 'string') add(x, 3); else if (x && typeof x === 'object') Object.keys(x).forEach(function (k) { leaves(x[k]); }); };
    [function () { return localStorage; }, function () { return sessionStorage; }].forEach(function (get) {
      try {
        var st = get();
        for (var k = 0; k < st.length; k++) {
          var v = st.getItem(st.key(k));
          add(v);
          try { var j = JSON.parse(v); if (j && typeof j === 'object') leaves(j); } catch (e) {}
        }
      } catch (e) {}
    });
    var docs = [document];
    try { document.querySelectorAll('iframe').forEach(function (f) { try { if (f.contentDocument) docs.push(f.contentDocument); } catch (e) {} }); } catch (e) {}
    docs.forEach(function (d) {
      try {
        d.querySelectorAll('textarea, input:not([type]), input[type=text], input[type=password], input[type=search], input[type=url], input[type=email]')
          .forEach(function (i) { add(i.value, /name/i.test(i.id || '') ? 2 : 4); });
        d.querySelectorAll('[contenteditable=""], [contenteditable=true]').forEach(function (n) { add(n.textContent, 4); });
      } catch (e) {}
    });
    return Array.from(out).sort(function (a, b) { return b.length - a.length; });
  }
  // and anything shaped like an AI key, wherever it came from
  var KEY_SHAPES = /\b(sk-ant-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,}|AIza[0-9A-Za-z_-]{20,})/g;
  var esc = function (s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); };
  function hideOne(t, h) {
    // a short secret (a character called Zog) goes only where it is a whole word, so it cannot
    // chew holes in the code's own words
    if (h.length < 8) return t.replace(new RegExp('(^|[^A-Za-z0-9_])' + esc(h) + '(?![A-Za-z0-9_])', 'g'), '$1[removed]');
    return t.split(h).join('[removed]');
  }
  function reportScrubOut(x, hide) {
    if (typeof x === 'string') return hide.reduce(hideOne, x).replace(KEY_SHAPES, '[removed]');
    if (Array.isArray(x)) return x.map(function (y) { return reportScrubOut(y, hide); });
    if (x && typeof x === 'object') { Object.keys(x).forEach(function (k) { x[k] = reportScrubOut(x[k], hide); }); return x; }
    return x;
  }
  // what a report of an error has in it, as it will be sent (and shown before it is)
  function reportOf(e) {
    var r = { error: (e.name || 'Error') + ': ' + e.message };
    if (e.__where) r.during = e.__where;
    if (!e.__noStack) r.where = String(e.stack || '').split('\n').slice(1, 9).map(function (l) { return l.trim(); }).filter(Boolean);
    r.page = location.origin + location.pathname;
    r.browser = navigator.userAgent;
    return reportScrubOut(r, reportSecrets());
  }
  // Sentry's beforeSend: only what the visitor sent (its consent noted when they pressed Send),
  // with no user, no breadcrumbs, the page's address without its query, and no kept value
  // anywhere in it. Anything else is dropped.
  function reportScrubbed(event) {
    var ok = event && event.tags && event.tags.kind === 'error-sent-by-visitor' && event.extra && report.consent.delete(event.extra.consent);
    if (!ok) return null;
    delete event.user;
    delete event.breadcrumbs;
    delete event.extra.consent;
    event.request = { url: location.origin + location.pathname, headers: { 'User-Agent': navigator.userAgent } };
    return reportScrubOut(event, reportSecrets());
  }
  // THE NOTICE: small, bottom left (the guide panel lives on the right), above everything,
  // gone once answered. Built from text nodes, never markup.
  function reportAsk() {
    if (!report.held.length || reportNever()) return;
    var box = document.getElementById('tmReportAsk');
    if (!box) {
      box = document.createElement('div');
      box.id = 'tmReportAsk';
      box.setAttribute('role', 'dialog');
      box.setAttribute('aria-label', 'Report an error to Ken');
      box.style.cssText = 'position:fixed;left:14px;bottom:40px;width:min(380px,calc(100vw - 28px));z-index:10002;box-sizing:border-box;'
        + 'padding:12px 14px;border-radius:12px;background:#1d1f24;border:1px solid #3a3f4a;color:#ddd;'
        + 'box-shadow:0 8px 28px rgba(0,0,0,.55);font:13px/1.45 system-ui,-apple-system,"Segoe UI",sans-serif;'
        + 'display:flex;flex-direction:column;gap:8px';
      (document.body || document.documentElement).appendChild(box);
    }
    var n = report.held.length;
    box.textContent = '';
    var p = document.createElement('div');
    p.textContent = (n > 1 ? 'Tiny Mind ran into ' + n + ' errors.' : 'Tiny Mind ran into an error.')
      + ' Send a report to Ken, without your stories, your name or any key?';
    var det = document.createElement('details');
    var sum = document.createElement('summary');
    sum.textContent = 'What would be sent';
    sum.style.cursor = 'pointer';
    var pre = document.createElement('pre');
    pre.style.cssText = 'max-height:180px;overflow:auto;white-space:pre-wrap;font:11px/1.4 ui-monospace,Menlo,Consolas,monospace;margin:6px 0 0;color:#bbb';
    pre.textContent = JSON.stringify(report.held.map(reportOf), null, 2);
    var note = document.createElement('div');
    note.style.cssText = 'font-size:11px;color:#888;margin-top:4px';
    note.textContent = 'Sent to Ken’s error reports (Sentry) only if you press Send: nothing about you, and nothing this page keeps in your browser.';
    det.append(sum, pre, note);
    var row = document.createElement('div');
    row.style.cssText = 'display:flex;gap:8px;flex-wrap:wrap;align-items:center';
    var btn = function (label, what) {
      var b = document.createElement('button');
      b.type = 'button'; b.textContent = label; b.setAttribute('data-report', what);
      b.style.cssText = 'font:inherit;cursor:pointer;border-radius:14px;padding:4px 12px;border:1px solid #555;'
        + (what === 'send' ? 'background:#20351f;color:#eafff0;border-color:#4a7' : 'background:none;color:inherit');
      b.onclick = function () { reportAnswer(what); };
      return b;
    };
    row.append(btn('Send', 'send'), btn('Don’t send', 'no'), btn('Don’t ask again', 'never'));
    box.append(p, det, row);
    box.style.display = 'flex';
    report.open = true;
  }
  // the way back after "Don't ask again": a line on the front door, only then
  function reportAgainLine() {
    var line = document.getElementById('homeReports');
    var show = reportOn() && reportNever();
    if (!line && show) {
      var wrap = document.querySelector('#home .wrap');
      if (!wrap) return;
      line = document.createElement('p');
      line.className = 'foot'; line.id = 'homeReports';
      var a = document.createElement('a');
      a.href = '#'; a.style.color = '#8fd3a7';
      a.textContent = 'Offer to report errors to Ken again';
      a.addEventListener('click', function (ev) {
        ev.preventDefault();
        try { localStorage.removeItem(REPORT_KEY); } catch (e) {}
        line.textContent = 'Error reports will be offered again, if Tiny Mind runs into one.';
      });
      line.append('You chose not to be asked about error reports. ', a, '.');
      wrap.appendChild(line);
    } else if (line && !show && line.querySelector('a')) line.remove();
  }
  function reportAnswer(what) {
    var box = document.getElementById('tmReportAsk');
    var held = report.held.splice(0);
    report.open = false;
    if (what === 'never') { try { localStorage.setItem(REPORT_KEY, 'never'); } catch (e) {} reportAgainLine(); }
    if (what !== 'send') { if (box) box.style.display = 'none'; return Promise.resolve(); }
    if (box) box.textContent = 'Sending…';
    return reportSentry().then(function (S) {
      if (!S) return false;
      held.forEach(function (err) {
        var consent = Date.now() + '-' + Math.random().toString(36).slice(2);
        report.consent.add(consent);
        S.withScope(function (scope) {
          scope.setTag('kind', 'error-sent-by-visitor');
          if (err.__where) scope.setTag('during', String(err.__where).slice(0, 190));
          scope.setExtras({ consent: consent });
          S.captureException(err);
        });
      });
      return S.flush(5000).catch(function () { return false; });
    }).then(function (sent) {
      if (sent) report.sent += held.length;
      // (unless another error came while it was sent, and is being asked about)
      if (box && !report.open) {
        box.textContent = sent ? 'Sent. Thank you — Ken will see it with Tiny Mind’s other reports.' : 'It couldn’t be sent from here just now.';
        setTimeout(function () { if (!report.open) box.style.display = 'none'; }, 4000);
      }
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', reportAgainLine); else reportAgainLine();
  window.tmReportFault = reportFault;
  // for the rooms (their frames hand errors up) and for checks
  window.__tmReport = {
    keep: reportKeep,
    state: function () { return { on: reportOn(), held: report.held.length, open: report.open, sent: report.sent }; },
    scrub: reportScrubbed, preview: function () { return report.held.map(reportOf); }
  };
})();
