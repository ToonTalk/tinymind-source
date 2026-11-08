// e21 — error reports, asked for first (shell/tm-report.js). Off on localhost and in the claude.ai
// artifact; on the site an error raises a notice and sends nothing by itself; what would be sent,
// and what Sentry is handed, have a planted key, a planted story sentence and a character's name
// taken out; a report nobody sent is dropped; Don't send sends nothing; Don't ask again is kept and
// the front door offers the way back; a compute-engine failure is offered, "that is not a mind" is not.
// Sentry itself is a stand-in (window.__sentryStub): no network.
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export async function run({ check }) {
  let JSDOM;
  try { ({ JSDOM } = await import('jsdom')); }
  catch (e) { check('e21 needs jsdom (npm install)', false); return; }
  const code = await readFile(join(ROOT, 'shell', 'tm-report.js'), 'utf8');
  const page = (url) => {
    const dom = new JSDOM('<!doctype html><html><head></head><body class="simple" data-view="home"><div id="home"><div class="wrap"><p class="foot">hi</p></div></div></body></html>',
      { url, runScripts: 'outside-only', pretendToBeVisual: true });
    dom.window.eval(code);
    return dom.window;
  };
  const wait = (ms) => new Promise(r => setTimeout(r, ms));
  const throwIn = (w, msg) => w.dispatchEvent(new w.ErrorEvent('error', { error: new w.Error(msg), message: msg }));

  // 1. off where it must be off
  const local = page('http://localhost:8123/');
  throwIn(local, 'on localhost');
  check('off on localhost: nothing held, no notice', local.__tmReport.state().on === false && local.__tmReport.state().held === 0 && !local.document.getElementById('tmReportAsk'));
  const art = page('https://abc123.claudeusercontent.com/');
  throwIn(art, 'in the artifact');
  check('off in the claude.ai artifact', art.__tmReport.state().on === false && !art.document.getElementById('tmReportAsk'));

  // 2. on the site
  const w = page('https://toontalk.github.io/tinymind/');
  check('on at toontalk.github.io (a DSN is set)', w.__tmReport.state().on === true);
  const KEY = 'sk-ant-api03-PLANTEDKEYPLANTEDKEY';
  const STORY = 'Zog the dragon hid the golden acorn under the purple bridge.';
  w.localStorage.setItem('tm-key-anthropic', KEY);
  w.localStorage.setItem('tm-storybook', JSON.stringify({ character: { name: 'Zog' }, stories: [STORY] }));
  w.sessionStorage.setItem('tm-key-openai', 'sk-proj-SESSIONKEYSESSIONKEY');
  const ta = w.document.createElement('textarea'); ta.value = 'A story still being typed about the moon and Zog.'; w.document.body.appendChild(ta);
  const nm = w.document.createElement('input'); nm.id = 'chName'; nm.value = 'Pim'; w.document.body.appendChild(nm);
  const events = [];
  w.__sentryStub = {
    withScope(fn) { this.s = { tags: {}, extra: {}, setTag(k, v) { this.tags[k] = v; }, setExtras(x) { Object.assign(this.extra, x); } }; fn(this.s); },
    captureException(e) {
      events.push(w.__tmReport.scrub({ tags: { ...this.s.tags }, extra: { ...this.s.extra }, user: { ip_address: '10.0.0.1' }, breadcrumbs: [{ message: 'x' }],
        exception: { values: [{ type: e.name, value: e.message, stacktrace: String(e.stack) }] }, request: { url: 'https://toontalk.github.io/tinymind/?q=1#x' } }));
    },
    flush() { return Promise.resolve(true); },
  };
  const box = () => w.document.getElementById('tmReportAsk');
  const press = (what) => box().querySelector('button[data-report="' + what + '"]').click();

  throwIn(w, 'carrying ' + KEY + ' and "' + STORY + '" and Pim and Zog and the moon and Zog.');
  const asked = w.__tmReport.state().held === 1 && box() && box().style.display !== 'none' && events.length === 0;
  check('an error raises the notice and sends nothing by itself', asked);
  check('the notice says what it is, in plain words', /Tiny Mind ran into an error\. Send a report to Ken, without your stories, your name or any key\?/.test(box().textContent));
  const preview = box().querySelector('pre').textContent;
  const clean = (t) => !t.includes(KEY) && !t.includes('PLANTEDKEY') && !t.includes('golden acorn') && !/\bPim\b/.test(t) && !/\bZog\b/.test(t) && !t.includes('SESSIONKEY');
  check('what would be sent: no key, no story, no names', clean(preview) && /\[removed\]/.test(preview));
  check('what would be sent: the error is still recognizable', /carrying \[removed\]/.test(preview) && /"page": "https:\/\/toontalk\.github\.io\/tinymind\/"/.test(preview));

  press('send');
  await wait(50);
  const ev = events[0];
  const all = JSON.stringify(events);
  check('Send hands Sentry one report, scrubbed the same way', events.length === 1 && clean(all));
  check('…with no user, no breadcrumbs, no consent token, no query in the address',
    ev && !ev.user && !ev.breadcrumbs && !('consent' in ev.extra) && ev.request.url === 'https://toontalk.github.io/tinymind/' && ev.tags.kind === 'error-sent-by-visitor');
  check('the notice thanks them', /Sent\. Thank you/.test(box().textContent));
  check('a report nobody sent (a forged consent) is dropped', w.__tmReport.scrub({ tags: { kind: 'error-sent-by-visitor' }, extra: { consent: 'forged' } }) === null
    && w.__tmReport.scrub({ tags: {}, extra: {} }) === null);

  // 3. Don't send
  throwIn(w, 'second');
  press('no');
  await wait(20);
  check("Don't send sends nothing", events.length === 1 && w.__tmReport.state().held === 0 && box().style.display === 'none');

  // 4. the compute engine: a real failure is offered, a visitor's wrong file is not
  w.tmReportWorker('bin-reader: layout consumed 10 bytes but file is 12');
  check('"that file is not a mind" is not offered as a fault', w.__tmReport.state().held === 0);
  w.tmReportWorker('train pool is broken');
  check('a compute-engine failure is offered, named as one', w.__tmReport.state().held === 1 && /the compute engine/.test(box().querySelector('pre').textContent));

  // 5. Don't ask again, and the way back
  press('never');
  check("Don't ask again is kept", w.localStorage.getItem('tm-report') === 'never');
  throwIn(w, 'third');
  check('then nothing more is asked', w.__tmReport.state().held === 0 && box().style.display === 'none');
  const back = w.document.getElementById('homeReports');
  check('the front door offers the way back', !!back && /Offer to report errors to Ken again/.test(back.textContent));
  back.querySelector('a').click();
  check('…and it works', w.localStorage.getItem('tm-report') === null);
  throwIn(w, 'fourth');
  check('asked again after that', w.__tmReport.state().held === 1);
}
