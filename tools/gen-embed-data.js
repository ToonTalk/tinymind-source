// gen-embed-data.js — refresh the EMBEDDED_WORDLIST in src/embed-data.js from
// corpus/wordlist.json (the embedded base stories are hand-curated and stay put).
// Run after tools/build-wordlist.js changes the wordlist.
import { readFile, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const wl = JSON.parse(await readFile(join(ROOT, 'corpus', 'wordlist.json'), 'utf8'));
const cur = await readFile(join(ROOT, 'src', 'embed-data.js'), 'utf8');
const next = cur.replace(/EMBEDDED_WORDLIST = \[[^\n]*\];/,
  'EMBEDDED_WORDLIST = ' + JSON.stringify(wl.words) + ';');
if (next === cur) throw new Error('EMBEDDED_WORDLIST assignment not found');
await writeFile(join(ROOT, 'src', 'embed-data.js'), next);
console.log('embed-data wordlist refreshed:', wl.words.length, 'words');
