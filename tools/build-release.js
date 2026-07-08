// build-release.js — package each app (microscope, nursery, hunt) into ONE self-
// contained HTML file in dist/, and compose all three into the MEGA edition
// (dist/tiny-mind-mega.html): the shell from shell/index.html hosting the apps
// byte-intact in iframes, per the artifact session's architecture (merged back
// 2026-07-07). All src/ modules are inlined; workers are built from Blobs with a
// same-thread fallback for the artifact sandbox; the frame bridge (shell/bridge.js)
// is injected ahead of every app's module script and is inert standalone.
// Every generated script is syntax-checked at build time.
// Usage: node tools/build-release.js
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const APPS = ['microscope', 'nursery', 'hunt'];
const CORE = ['bin-reader.js', 'tokenizer.js', 'model.js', 'capture.js', 'train.js'];
const PAGE_EXTRA = ['llm.js', 'app-shims.js', 'ingest.js', 'embed-data.js'];
const WORKER_EXTRA = { nursery: ['embed-data.js'], microscope: [], hunt: [] };

// the boot that replaces the dev marker block: real blob Worker, or the same-thread
// shim when the sandbox refuses blob URLs (makeInlineWorker comes from app-shims)
const WORKER_BOOT = `let worker, workerMode = 'background worker';
try {
  worker = new Worker(URL.createObjectURL(new Blob([__WORKER_SRC__], { type: 'text/javascript' })));
} catch {
  worker = makeInlineWorker(__WORKER_SRC__);       // sandbox blocked the blob worker — run on this thread
  workerMode = 'inline (sandboxed page — compute shares the UI thread)';
}`;

// strip ESM syntax so modules concatenate into one shared scope; collect aliases
// (`import { a as b }` -> `const b = a;`) to emit after the inlined definitions
function stripModule(code, aliases) {
  code = code.replace(/^import\s+\{([\s\S]*?)\}\s*from\s*['"][^'"]+['"];?\s*$/gm, (_, names) => {
    for (const part of names.split(',')) {
      const m = part.trim().match(/^(\w+)\s+as\s+(\w+)$/);
      if (m) aliases.push(`const ${m[2]} = ${m[1]};`);
    }
    return '';
  });
  return code.replace(/^export\s+/gm, '');
}

// a JS string literal safe to embed inside an inline <script>
function embedString(s) {
  return JSON.stringify(s).replace(/<\//g, '<\\/');
}

async function bundleOf(names) {
  const parts = [];
  for (const f of names) {
    parts.push(`// ---- inlined src/${f} ----\n` + stripModule(await readFile(join(ROOT, 'src', f), 'utf8'), []));
  }
  return parts.join('\n');
}

async function checkScript(app, label, code) {
  const tmp = join(ROOT, 'dist', `_check-${app}-${label}.mjs`);
  await writeFile(tmp, code);
  try {
    execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' });
  } catch (err) {
    throw new Error(`${app} ${label} bundle failed syntax check:\n` + err.stderr);
  }
  await rm(tmp);
}

await mkdir(join(ROOT, 'dist'), { recursive: true });
const bridge = await readFile(join(ROOT, 'shell', 'bridge.js'), 'utf8');
const pageBundle = await bundleOf([...CORE, ...PAGE_EXTRA]);
const built = {};

for (const app of APPS) {
  const html = await readFile(join(ROOT, app, 'index.html'), 'utf8');
  const workerAliases = [];
  const workerCode = stripModule(await readFile(join(ROOT, app, 'worker.js'), 'utf8'), workerAliases);
  const workerBundle = await bundleOf([...CORE, ...WORKER_EXTRA[app]]);
  const workerSrc = `'use strict';\n${workerBundle}\n${workerAliases.join('\n')}\n// ---- ${app}/worker.js ----\n${workerCode}`;

  const scriptMatch = html.match(/<script type="module">([\s\S]*?)<\/script>/);
  if (!scriptMatch) throw new Error(app + ': no module script found');
  const pageAliases = [];
  let pageCode = stripModule(scriptMatch[1], pageAliases);
  const bootPattern = /\/\/ @tm-worker-boot[\s\S]*?\/\/ @tm-worker-boot-end/;
  if (!bootPattern.test(pageCode)) throw new Error(app + ': @tm-worker-boot markers not found');
  pageCode = pageCode.replace(bootPattern, WORKER_BOOT);

  const bundled =
    `\nconst __WORKER_SRC__ = ${embedString(workerSrc)};\n` +
    pageBundle + '\n' + pageAliases.join('\n') + '\n' + pageCode;

  await checkScript(app, 'page', bundled);
  await checkScript(app, 'worker', workerSrc);

  // function-form replacements throughout: the inserted code contains `$&`/`$'`
  // sequences (e.g. the hunt worker's regex-escape `'\\$&'`), which string-form
  // .replace() would expand into the matched text — silently corrupting the output
  // AFTER the syntax checks ran. Caught by the guided-build verification pass.
  let out = html.replace(scriptMatch[0], () =>
    `<script>/* injected by build-release: frame bridge (inert standalone) */\n${bridge}</script>\n` +
    `<script type="module">${bundled}</script>`);
  out = out.replace('<head>', `<head>\n<!-- Tiny Mind single-file release, built ${new Date().toISOString().slice(0, 10)} from the dev tree. Works from file://, any static host, or pasted into a claude.ai artifact (keyless Claude appears automatically). Models arrive by file-drop. -->`);
  await writeFile(join(ROOT, 'dist', `tiny-mind-${app}.html`), out);
  built[app] = out;
  console.log(`dist/tiny-mind-${app}.html  (${(out.length / 1024).toFixed(0)} KB) — bundles parse OK`);
}

// ---- mega edition: shell + the three apps as iframe templates -----------------
let shell = await readFile(join(ROOT, 'shell', 'index.html'), 'utf8');
const sharedAliases = [];
const shellShared = stripModule(await readFile(join(ROOT, 'src', 'ingest.js'), 'utf8'), sharedAliases);
shell = shell.replace('@@SHELL_SHARED@@', () => shellShared);
for (const app of APPS) {
  const doc = built[app].replace(/^<!doctype html>\s*/i, ''); // ensureFrame re-adds the doctype
  const marker = `@@APP_${app.toUpperCase()}@@`;
  if (!shell.includes(marker)) throw new Error('shell: missing ' + marker);
  shell = shell.replace(marker, () => doc);
}
// check the shell's own (plain) script
const shellScripts = [...shell.matchAll(/<script>([\s\S]*?)<\/script>/g)];
await checkScript('mega', 'shell', shellScripts[shellScripts.length - 1][1]);
await writeFile(join(ROOT, 'dist', 'tiny-mind-mega.html'), shell);
console.log(`dist/tiny-mind-mega.html  (${(shell.length / 1024).toFixed(0)} KB) — shell script parses OK`);
console.log('done');
