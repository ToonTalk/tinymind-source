// build-release.js — package each app (microscope, nursery, hunt) into ONE self-
// contained HTML file in dist/: all src/ modules inlined, the worker built from a
// Blob (no imports anywhere). This is the release step the kickoff anticipated
// ("single-file HTML per edition"); dev keeps real modules so tests bite real code.
// The same file serves both editions: inside a claude.ai artifact the keyless-Claude
// provider appears automatically (src/llm.js keylessAvailable()).
// Usage: node tools/build-release.js
import { readFile, writeFile, mkdir, rm } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const APPS = ['microscope', 'nursery', 'hunt'];
const SRC_MODULES = ['bin-reader.js', 'tokenizer.js', 'model.js', 'capture.js', 'train.js', 'llm.js'];

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

const srcCode = [];
for (const f of SRC_MODULES) {
  srcCode.push(`// ---- inlined src/${f} ----\n` + stripModule(await readFile(join(ROOT, 'src', f), 'utf8'), []));
}
const srcBundle = srcCode.join('\n');

await mkdir(join(ROOT, 'dist'), { recursive: true });
for (const app of APPS) {
  const html = await readFile(join(ROOT, app, 'index.html'), 'utf8');
  const workerAliases = [];
  const workerCode = stripModule(await readFile(join(ROOT, app, 'worker.js'), 'utf8'), workerAliases);
  const workerSrc = `'use strict';\n${srcBundle}\n${workerAliases.join('\n')}\n// ---- ${app}/worker.js ----\n${workerCode}`;

  const scriptMatch = html.match(/<script type="module">([\s\S]*?)<\/script>/);
  if (!scriptMatch) throw new Error(app + ': no module script found');
  const pageAliases = [];
  let pageCode = stripModule(scriptMatch[1], pageAliases);
  const workerPattern = /new Worker\(new URL\('\.\/worker\.js', import\.meta\.url\), \{ type: 'module' \}\)/;
  if (!workerPattern.test(pageCode)) throw new Error(app + ': worker construction not found');
  pageCode = pageCode.replace(workerPattern,
    `new Worker(URL.createObjectURL(new Blob([__WORKER_SRC__], { type: 'text/javascript' })))`);

  const bundled =
    `\nconst __WORKER_SRC__ = ${embedString(workerSrc)};\n` +
    srcBundle + '\n' + pageAliases.join('\n') + '\n' + pageCode;

  // self-check: page bundle AND worker bundle must parse as modules (catches
  // top-level name collisions between inlined src and app code)
  for (const [label, code] of [['page', bundled], ['worker', workerSrc]]) {
    const tmp = join(ROOT, 'dist', `_check-${app}-${label}.mjs`);
    await writeFile(tmp, code);
    try {
      execFileSync(process.execPath, ['--check', tmp], { stdio: 'pipe' });
    } catch (err) {
      throw new Error(`${app} ${label} bundle failed syntax check:\n` + err.stderr);
    }
    await rm(tmp);
  }

  let out = html.replace(scriptMatch[0], `<script type="module">${bundled}</script>`);
  out = out.replace('<head>', `<head>\n<!-- Tiny Mind single-file release, built ${new Date().toISOString().slice(0, 10)} from the dev tree. Works from file://, any static host, or pasted into a claude.ai artifact (keyless Claude appears automatically). Models arrive by file-drop. -->`);
  const dest = join(ROOT, 'dist', `tiny-mind-${app}.html`);
  await writeFile(dest, out);
  console.log(`dist/tiny-mind-${app}.html  (${(out.length / 1024).toFixed(0)} KB) — bundles parse OK`);
}
console.log('done');
