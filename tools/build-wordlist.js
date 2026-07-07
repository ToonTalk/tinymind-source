// build-wordlist.js — derive the top-N word whitelist from TinyStories-valid.txt
// (kickoff §8: mechanical validation, not trust). Writes corpus/wordlist.json.
import { readFile, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const N = Number(process.argv[2] ?? 1500);

const text = await readFile(join(ROOT, 'models', 'TinyStories-valid.txt'), 'utf8');
const counts = new Map();
for (const m of text.toLowerCase().matchAll(/[a-z]+(?:'[a-z]+)?/g)) {
  const w = m[0];
  counts.set(w, (counts.get(w) ?? 0) + 1);
}
const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
const words = sorted.slice(0, N).map(([w]) => w);
await writeFile(join(ROOT, 'corpus', 'wordlist.json'), JSON.stringify({
  source: 'TinyStories-valid.txt',
  totalDistinct: counts.size,
  n: N,
  minCount: sorted[N - 1]?.[1] ?? 0,
  words,
}, null, 1));
console.log(`wordlist: top ${N} of ${counts.size} distinct words (min count ${sorted[N - 1]?.[1]}); wrote corpus/wordlist.json`);
console.log('rank 1-30   :', words.slice(0, 30).join(' '));
console.log('rank 736-765:', words.slice(735, 765).join(' '));
console.log('rank 1470+  :', words.slice(1470).join(' '));
for (const probe of ['dragon', 'dragons', 'cloud', 'clouds', 'moon', 'behind', 'ate', 'eats', 'wings', 'fly', 'flew', 'sky']) {
  const r = words.indexOf(probe);
  console.log(`probe ${probe.padEnd(8)}: ${r === -1 ? 'NOT in list (count ' + (counts.get(probe) ?? 0) + ')' : 'rank ' + (r + 1)}`);
}
