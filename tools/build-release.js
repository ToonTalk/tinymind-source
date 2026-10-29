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
const APPS = ['microscope', 'nursery', 'hunt', 'thought'];
const CORE = ['bin-reader.js', 'tokenizer.js', 'model.js', 'capture.js', 'train.js'];
const PAGE_EXTRA = ['llm.js', 'app-shims.js', 'scrubber.js', 'hunt-evidence.js', 'ingest.js', 'embed-data.js'];
const WORKER_EXTRA = { nursery: ['embed-data.js'], microscope: ['steer.js', 'neuron-atlas.js', 'jlens.js', 'batch.js'], hunt: [],
  thought: ['steer.js', 'jlens.js', 'batch.js', 'thought.js'] };
// page-only extras for one app (the others' pages stay byte-identical)
const PAGE_APP_EXTRA = { nursery: [], microscope: ['neuron-atlas.js'], hunt: [], thought: [] };

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
const prelude = await readFile(join(ROOT, 'shell', 'tm-prelude.js'), 'utf8');
const vgCore = await readFile(join(ROOT, 'shell', 'vg-core.js'), 'utf8');
const pageBundle = await bundleOf([...CORE, ...PAGE_EXTRA]);
const built = {};
const zogStories = JSON.parse(await readFile(join(ROOT, 'corpus', 'zog', 'stories.json'), 'utf8'));
if (zogStories.filter(s => s.split === 'train').length !== 50 || zogStories.filter(s => s.split === 'heldout').length !== 10) {
  throw new Error('nursery: ready-made Zog storybook must contain 50 training and 10 held-out stories');
}
const bindingCorpus = JSON.parse(await readFile(join(ROOT, 'corpus', 'binding', 'stories.json'), 'utf8'));
if (Object.keys(bindingCorpus.characters || {}).length !== 4 ||
    bindingCorpus.stories.filter(s => s.split === 'train').length !== 64 ||
    bindingCorpus.stories.filter(s => s.split === 'heldout').length !== 16) {
  throw new Error('nursery: binding story world must contain four characters, 64 training and 16 held-out stories');
}
// the Thought bubble's story vocabulary (the J-Lens Explorer's vocab.json, 110 KB): built into its page
const thoughtVocab = JSON.parse(await readFile(join(ROOT, 'thought', 'vocab.json'), 'utf8'));
if (!Array.isArray(thoughtVocab.prunedIds) || !Array.isArray(thoughtVocab.wordStartIds) || !thoughtVocab.alias || !Array.isArray(thoughtVocab.themeWords)) {
  throw new Error('thought: vocab.json must have prunedIds, wordStartIds, alias and themeWords');
}
const sameFolderResources = JSON.parse(await readFile(join(ROOT, 'shell', 'resources.json'), 'utf8'));
if (!sameFolderResources || sameFolderResources.version !== 1 ||
    !Array.isArray(sameFolderResources.resources) ||
    !sameFolderResources.resources.every(r => r && typeof r.file === 'string' &&
      /^[^/\\?#]+\.(?:zip|bin|json)$/i.test(r.file))) {
  throw new Error('shell: resources.json must be version 1 with safe same-folder resource file names');
}
const sameFolderResourceFiles = sameFolderResources.resources.map(r => r.file);
// the Microscope's neuron atlas (tools/build-neuron-atlas.js) is ~2 MB: far too big to
// embed in the pages, so it ships beside them as a same-folder pack (resources.json)
const neuronAtlas = await readFile(join(ROOT, 'microscope', 'tiny-mind-neuron-atlas.json'), 'utf8');
if (!neuronAtlas.slice(0, 200).includes('"tiny-mind-neuron-atlas-v1"')) {
  throw new Error('microscope: tiny-mind-neuron-atlas.json is not a tiny-mind-neuron-atlas-v1 file');
}

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

  const embeddedData = app === 'nursery'
    ? `\nconst __TM_EMBEDDED_ZOG_STORIES__ = JSON.parse(${embedString(JSON.stringify(zogStories))});\n` +
      `const __TM_EMBEDDED_BINDING_CORPUS__ = JSON.parse(${embedString(JSON.stringify(bindingCorpus))});\n`
    : app === 'thought' ? `\nconst __TM_THOUGHT_VOCAB__ = JSON.parse(${embedString(JSON.stringify(thoughtVocab))});\n`
    : '';
  const appExtra = PAGE_APP_EXTRA[app].length ? '\n' + await bundleOf(PAGE_APP_EXTRA[app]) : '';
  const bundled = embeddedData +
    `\nconst __WORKER_SRC__ = ${embedString(workerSrc)};\n` +
    pageBundle + appExtra + '\n' + pageAliases.join('\n') + '\n' + pageCode;

  await checkScript(app, 'page', bundled);
  await checkScript(app, 'worker', workerSrc);

  // function-form replacements throughout: the inserted code contains `$&`/`$'`
  // sequences (e.g. the hunt worker's regex-escape `'\\$&'`), which string-form
  // .replace() would expand into the matched text — silently corrupting the output
  // AFTER the syntax checks ran. Caught by the guided-build verification pass.
  const guide = await readFile(join(ROOT, app, 'guide.js'), 'utf8');
  let out = html.replace(scriptMatch[0], () =>
    `<script>/* injected by build-release: frame bridge (inert standalone) */\n${bridge}</script>\n` +
    `<script type="module">${bundled}</script>\n` +
    `<script>\n${vgCore}</script>\n` +
    `<script>\n${guide}</script>`);
  // the prelude's Worker shim must exist before the app script runs: top of <body>
  out = out.replace(/<body>/i, () => `<body>\n<script>\n${prelude}</script>`);
  out = out.replace('<head>', `<head>\n<!-- GENERATED by build-release.js — edit repo sources, not this file. Tiny Mind single-file release: works from file://, any static host, or pasted into a claude.ai artifact (keyless Claude appears automatically). Models load by themselves when the page is served beside them; otherwise drop them in. -->`);
  await writeFile(join(ROOT, 'dist', `tiny-mind-${app}.html`), out);
  built[app] = out;
  console.log(`dist/tiny-mind-${app}.html  (${(out.length / 1024).toFixed(0)} KB) — bundles parse OK`);
}

// ---- mega edition: shell + the three apps as iframe templates -----------------
let shell = await readFile(join(ROOT, 'shell', 'index.html'), 'utf8');
const shellShared = await bundleOf(['scrubber.js', 'ingest.js']) + // shell drop path + arc pool
  '\n// ---- inlined tools/zip.js (artifact saves: .bin goes out inside a .zip) ----\n' +
  stripModule(await readFile(join(ROOT, 'tools', 'zip.js'), 'utf8'), []);
shell = shell.replace('@@SHELL_SHARED@@', () => shellShared);
shell = shell.replace('@@SAME_FOLDER_RESOURCES@@', () => JSON.stringify(sameFolderResources.resources)); // entries keep their load/group hints
// voice guide for the shell itself — spliced BEFORE template insertion, while the
// only </body> in the string is the shell's own (each app doc brings its own)
const shellAdapter = await readFile(join(ROOT, 'shell', 'tm-shell-adapter.js'), 'utf8');
shell = shell.replace(/<\/body>/i, () =>
  `<script>\n${vgCore}</script>\n<script>\n${shellAdapter}</script>\n</body>`);
for (const app of APPS) {
  const doc = built[app].replace(/^<!doctype html>\s*/i, ''); // ensureFrame re-adds the doctype
  const marker = `@@APP_${app.toUpperCase()}@@`;
  if (!shell.includes(marker)) throw new Error('shell: missing ' + marker);
  shell = shell.replace(marker, () => doc);
}
// the shell's vg splice must sit AFTER the last template (i.e. in the shell's own
// body), not inside an app doc — guard the anchor mistake this build once made
if (shell.lastIndexOf('tm-shell-adapter.js') < shell.lastIndexOf('</template>')) {
  throw new Error('mega: shell adapter landed inside an app template');
}
// check the shell's own (plain) script (the main one, before the vg splices)
const shellScripts = [...shell.matchAll(/<script>([\s\S]*?)<\/script>/g)];
await checkScript('mega', 'shell', shellScripts[shellScripts.length - 3][1]);
await writeFile(join(ROOT, 'dist', 'tiny-mind-mega.html'), shell);
await writeFile(join(ROOT, 'dist', 'tiny-mind-resources.json'),
  JSON.stringify(sameFolderResources, null, 2) + '\n');
await writeFile(join(ROOT, 'dist', 'tiny-mind-neuron-atlas.json'), neuronAtlas);
console.log(`dist/tiny-mind-mega.html  (${(shell.length / 1024).toFixed(0)} KB) — shell script parses OK`);
console.log(`dist/tiny-mind-resources.json  (${sameFolderResourceFiles.length} same-folder runtime packs)`);
console.log(`dist/tiny-mind-neuron-atlas.json  (${(neuronAtlas.length / 1024).toFixed(0)} KB)`);
console.log('done');
