// E8 (arc kickoff) — tools/make-arc.js end to end.
// 1) CI mini-arc: manifest schema, every snapshot parses, prefix-valid ordering
// 2) resume determinism: straight run vs interrupted+resumed run -> byte-identical bins
// 3) zip round-trip: pack the arc dir with tools/zip.js, read it back, parse every bin
import { readFile, readdir, rm, mkdir } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';
import { readCheckpoint } from '../src/bin-reader.js';
import { makeZip, readZip } from '../tools/zip.js';

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = dirname(HERE);
const TMP = join(HERE, 'tmp-arc');
const TOOL = join(ROOT, 'tools', 'make-arc.js');

function runTool(args) {
  execFileSync(process.execPath, [TOOL, ...args], { cwd: ROOT, stdio: 'pipe' });
}

export async function run(t) {
  await rm(TMP, { recursive: true, force: true });
  await mkdir(TMP, { recursive: true });
  const dirA = join(TMP, 'straight'), dirB = join(TMP, 'resumed');

  // --- 1) mini arc, straight through
  runTool(['mini', '--out=' + dirA]);
  const manifest = JSON.parse(await readFile(join(dirA, 'arc-manifest.json'), 'utf8'));
  t.check('manifest has arc/note/config/snapshots',
    manifest.arc === 'arcM' && typeof manifest.note === 'string' &&
    manifest.config && Array.isArray(manifest.snapshots));
  t.check('manifest config carries B,T,lr,replayRatio,seed,start',
    ['B', 'T', 'lr', 'replayRatio', 'seed', 'start'].every(k => manifest.config[k] !== undefined));
  t.check('4 snapshots at {0,10,20,30}, ascending (prefix-valid)',
    manifest.snapshots.map(s => s.step).join(',') === '0,10,20,30');
  let allParse = true, allSamples = true, allLoss = true;
  for (const s of manifest.snapshots) {
    const ck = readCheckpoint(await readFile(join(dirA, s.file)));
    if (ck.config.vocabSize !== 512) allParse = false;
    if (typeof s.samples?.['Once upon a time'] !== 'string') allSamples = false;
    if (s.step > 0 && !(typeof s.trainLoss === 'number' && typeof s.baseLoss === 'number')) allLoss = false;
  }
  t.check('every snapshot .bin parses via readCheckpoint (vocab 512)', allParse);
  t.check('every snapshot has a greedy sample for the fixed prompt', allSamples);
  t.check('post-0 snapshots carry trainLoss + baseLoss', allLoss);
  t.check('training moved: baseLoss fell from snapshot 0 to 30',
    manifest.snapshots[3].baseLoss < manifest.snapshots[0].baseLoss - 0.5);

  // --- 2) resume determinism: 30 straight vs 20 + interrupt + resume to 30
  runTool(['mini', '--out=' + dirB, '--max-step=20']);
  runTool(['mini', '--out=' + dirB, '--resume']);
  const straight = await readFile(join(dirA, 'arcM-step00030.bin'));
  const resumed = await readFile(join(dirB, 'arcM-step00030.bin'));
  t.check('resume determinism: step-30 checkpoints byte-identical (straight vs 20+resume)',
    straight.equals(resumed));
  const mA = JSON.parse(await readFile(join(dirA, 'arc-manifest.json'), 'utf8'));
  const mB = JSON.parse(await readFile(join(dirB, 'arc-manifest.json'), 'utf8'));
  t.check('resumed manifest matches straight manifest (losses + samples)',
    JSON.stringify(mA.snapshots) === JSON.stringify(mB.snapshots));

  // --- 3) zip round-trip (same container the artifact ingest accepts)
  const entries = [{ name: 'arc-manifest.json', data: new TextEncoder().encode(JSON.stringify(manifest)) }];
  for (const s of manifest.snapshots) {
    entries.push({ name: s.file, data: new Uint8Array(await readFile(join(dirA, s.file))) });
  }
  const zip = makeZip(entries);
  const back = readZip(zip);
  t.check('zip round-trip: entry names preserved',
    back.map(e => e.name).join('|') === entries.map(e => e.name).join('|'));
  const backManifest = JSON.parse(new TextDecoder().decode(back[0].data));
  let zipParse = true;
  for (let i = 1; i < back.length; i++) {
    if (readCheckpoint(back[i].data).config.vocabSize !== 512) zipParse = false;
    if (back[i].data.length !== entries[i].data.length) zipParse = false;
  }
  t.check('zip round-trip: manifest + every checkpoint parse from the archive',
    backManifest.snapshots.length === 4 && zipParse);

  await rm(TMP, { recursive: true, force: true });
}
