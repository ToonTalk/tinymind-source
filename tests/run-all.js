// run-all.js — Phase 0 test suite. `node tests/run-all.js` — final PASS count is the
// build fingerprint (kickoff §3, §10). Zero deps.
import { readdir } from 'node:fs/promises';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { dirname, join } from 'node:path';

const here = dirname(fileURLToPath(import.meta.url));

let pass = 0, fail = 0;
const failures = [];
export function check(name, cond) {
  if (cond) { pass++; console.log(`  PASS ${name}`); }
  else { fail++; failures.push(name); console.log(`  FAIL ${name}`); }
}
export function checkClose(name, a, b, tol) {
  const d = Math.abs(a - b);
  check(`${name} (|${a} - ${b}| = ${d.toExponential(2)} <= ${tol})`, d <= tol);
}

const only = process.argv[2]; // optional substring filter, e.g. `node tests/run-all.js e1`
const files = (await readdir(here)).filter(f => f.endsWith('.test.js')).sort();
for (const f of files) {
  if (only && !f.includes(only)) continue;
  console.log(`\n== ${f}`);
  const mod = await import(pathToFileURL(join(here, f)).href);
  await mod.run({ check, checkClose });
}

console.log(`\n${'='.repeat(50)}`);
console.log(`FINGERPRINT: ${pass} PASS, ${fail} FAIL`);
if (fail) { console.log('Failed:\n  ' + failures.join('\n  ')); process.exit(1); }
