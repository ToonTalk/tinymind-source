// build-artifact.mjs — the claude.ai artifact edition of the mega page. An artifact cannot
// reach the network, but it can read files uploaded to its own asset store, and the asset
// store takes script files. So the same files the GitHub Pages site serves beside the page
// (tools/build-site.mjs) are packed into a few .js "parts", each one JSON object:
//   { "files": { "<path>": "<base64>", ... } }           whole small files, or
//   { "chunk": { "path": "<path>", "index": i, "b64": "<base64>" } }   a slice of a big one.
// Two steps, because the asset urls exist only after the upload:
//   node tools/build-artifact.mjs pack <site folder> <out folder>
//       → <out>/parts/p00.js … and <out>/plan.json (which part holds what)
//   (upload <out>/parts/*.js to the artifact's asset store; save the urls, in part order,
//    as a JSON array in <out>/urls.json)
//   node tools/build-artifact.mjs page <out folder>
//       → <out>/tiny-mind-artifact.html: the mega page with window.__TM_ASSETS__ set
import { readFileSync, writeFileSync, mkdirSync, readdirSync, statSync } from 'node:fs';
import { join, dirname, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const [cmd, a1, a2] = process.argv.slice(2);
const PART_RAW = 11_000_000;          // raw bytes per part: base64 + JSON stays under the 16 MiB script cap

function walk(dir) {
  const out = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...walk(p));
    else out.push(p);
  }
  return out;
}

if (cmd === 'pack') {
  const site = a1, out = a2;
  if (!site || !out) throw new Error('usage: pack <site folder> <out folder>');
  // everything the page fetches beside itself (not the page, README or Pages marker)
  const files = walk(site)
    .map(p => relative(site, p).split(sep).join('/'))
    .filter(r => !/^(index\.html|README\.md|\.nojekyll|\.gitignore)$/.test(r) && !r.startsWith('.git/'))
    // the 110M mind is too big for an artifact (on the site it comes from Hugging Face), so its lens stays out too
    .filter(r => r !== 'stories110M.jlens')
    .map(r => ({ path: r, bytes: readFileSync(join(site, r)) }))
    .sort((x, y) => y.bytes.length - x.bytes.length);
  const parts = [];                    // each: { files: {path: b64} } or { chunk: {...} }
  const plan = { files: {} };
  // big files: their own chunked parts
  for (const f of files.filter(f => f.bytes.length > PART_RAW)) {
    const idx = [];
    for (let at = 0, i = 0; at < f.bytes.length; at += PART_RAW, i++) {
      idx.push(parts.length);
      parts.push({ chunk: { path: f.path, index: i, b64: f.bytes.subarray(at, at + PART_RAW).toString('base64') } });
    }
    plan.files[f.path] = { parts: idx, bytes: f.bytes.length };
  }
  // small files: first-fit into shared parts
  const bins = [];
  for (const f of files.filter(f => f.bytes.length <= PART_RAW)) {
    let bin = bins.find(b => b.size + f.bytes.length <= PART_RAW);
    if (!bin) { bin = { size: 0, part: parts.length, files: {} }; bins.push(bin); parts.push({ files: bin.files }); }
    bin.files[f.path] = f.bytes.toString('base64');
    bin.size += f.bytes.length;
    plan.files[f.path] = { parts: [bin.part], bytes: f.bytes.length };
  }
  mkdirSync(join(out, 'parts'), { recursive: true });
  parts.forEach((p, i) => writeFileSync(join(out, 'parts', `p${String(i).padStart(2, '0')}.js`), JSON.stringify(p)));
  plan.partCount = parts.length;
  writeFileSync(join(out, 'plan.json'), JSON.stringify(plan, null, 1));
  const total = parts.reduce((n, p, i) => n + statSync(join(out, 'parts', `p${String(i).padStart(2, '0')}.js`)).size, 0);
  console.log(`packed ${files.length} files into ${parts.length} parts (${(total / 1e6).toFixed(1)} MB of script) → ${out}`);
} else if (cmd === 'page') {
  const out = a1;
  if (!out) throw new Error('usage: page <out folder>');
  const plan = JSON.parse(readFileSync(join(out, 'plan.json'), 'utf8'));
  const urls = JSON.parse(readFileSync(join(out, 'urls.json'), 'utf8'));
  if (!Array.isArray(urls) || urls.length !== plan.partCount || !urls.every(u => /^\/_blob\/[0-9a-f]{32}$/.test(u))) {
    throw new Error(`urls.json must list ${plan.partCount} /_blob/<id> urls in part order`);
  }
  const assets = { parts: urls, files: plan.files };
  let html = readFileSync(join(ROOT, 'dist', 'tiny-mind-mega.html'), 'utf8');
  const anchor = '<script>\n// is built-in Claude here?';   // the top of the shell's own script
  if (html.split(anchor).length !== 2) throw new Error('mega: could not find the shell script to put the asset list before');
  html = html.replace(anchor, () => `<script>window.__TM_ASSETS__ = ${JSON.stringify(assets)};</script>\n` + anchor);
  writeFileSync(join(out, 'tiny-mind-artifact.html'), html);
  console.log(`page → ${join(out, 'tiny-mind-artifact.html')} (${(html.length / 1024).toFixed(0)} KB, ${urls.length} asset parts)`);
} else {
  console.error('usage: node tools/build-artifact.mjs pack <site> <out> | page <out>');
  process.exit(1);
}
