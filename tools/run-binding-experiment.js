// One-command orchestrator for Codex or a local terminal.
//   node tools/run-binding-experiment.js
//   node tools/run-binding-experiment.js --resume
// Additional --max-step=, --lr=, --replay= flags are passed to make-binding-arc.js.
import { spawnSync } from 'node:child_process';
import { mkdir, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const logDir = join(ROOT, 'arc', 'binding');
await mkdir(logDir, { recursive: true });
const log = [];

function run(script, args = []) {
  const command = [script, ...args];
  const started = new Date().toISOString();
  console.log(`\n=== node ${command.join(' ')}`);
  const r = spawnSync(process.execPath, command, { cwd: ROOT, encoding: 'utf8', stdio: ['inherit', 'pipe', 'pipe'] });
  if (r.stdout) process.stdout.write(r.stdout);
  if (r.stderr) process.stderr.write(r.stderr);
  log.push({ started, command: `node ${command.join(' ')}`, status: r.status, stdout: r.stdout, stderr: r.stderr });
  if (r.status !== 0) throw new Error(`${script} failed with status ${r.status}`);
}

try {
  run('tools/gen-binding-corpus.js');
  run('tools/make-binding-arc.js', process.argv.slice(2));
  run('tools/evaluate-binding.js', ['--arc=arc/binding']);
  run('tools/make-binding-arc-zip.js');
  console.log('\nExperiment complete. Return dist/binding-results.zip and dist/binding-arc.zip to this ChatGPT conversation.');
} finally {
  await writeFile(join(logDir, 'codex-run-log.json'), JSON.stringify(log, null, 2));
}
