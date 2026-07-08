// E10 (guided-build merge) — the scrubber core (time machine) + three-phase ingest,
// exactly as the scrubber-session comment promised: "tests exercise it directly in Node".
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  isArcManifest, validateArcManifest, splitArcPacks, arcKidLabel, stopCaption,
  createArcPool, nearestStopIndex,
} from '../src/scrubber.js';
import { ingestBatch } from '../src/ingest.js';
import { makeZip } from '../tools/zip.js';
import { readCheckpoint } from '../src/bin-reader.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const enc = s => new TextEncoder().encode(s);

export async function run(t) {
  // --- manifest validation honors the prefix-valid contract
  const real = JSON.parse(await readFile(join(ROOT, 'arc', 'arc-manifest.json'), 'utf8'));
  t.check('real Arc B manifest validates', validateArcManifest(real).ok);
  const prefix = { ...real, snapshots: real.snapshots.slice(0, 3) };
  t.check('any PREFIX of a manifest validates (never assume the full schedule)',
    validateArcManifest(prefix).ok);
  t.check('descending steps rejected',
    !validateArcManifest({ arc: 'x', snapshots: [{ step: 5, file: 'a' }, { step: 3, file: 'b' }] }).ok);
  t.check('storybook json is NOT an arc manifest', !isArcManifest([{ split: 'train', text: 'x' }]));

  // --- kid labels: known arcs, then manifest note, then id
  t.check('kid labels: arcB known, unknown falls back to note',
    arcKidLabel({ arc: 'arcB' }).includes('memorizing') &&
    arcKidLabel({ arc: 'arcQ', note: 'mystery arc' }) === 'mystery arc');

  // --- stop captions prefer the requested prompt, else the LAST sample key
  const snap = real.snapshots[real.snapshots.length - 1];
  t.check('stopCaption picks the arc-specific prompt by default',
    stopCaption(snap).prompt === 'One day Zog' &&
    stopCaption(snap, 'Once upon a time').prompt === 'Once upon a time');

  // --- pool: idempotent re-add, bins/urlBase merge, lazy fetch caches
  const pool = createArcPool();
  pool.add(prefix, { urlBase: 'arc/' });
  pool.add(real, { bins: new Map([[real.snapshots[0].file, new Uint8Array([1, 2, 3])]]) });
  t.check('pool: re-add replaces manifest, keeps one entry, merges urlBase',
    pool.size() === 1 && pool.get('arcB').manifest.snapshots.length === real.snapshots.length &&
    pool.get('arcB').urlBase === 'arc/');
  let fetches = 0;
  const lazy = async () => { fetches++; return new Uint8Array([9]); };
  await pool.snapshotBytes('arcB', real.snapshots[1].file, lazy);
  await pool.snapshotBytes('arcB', real.snapshots[1].file, lazy);
  t.check('pool: lazy fetch runs once, then the cache serves scrub-backs', fetches === 1);

  t.check('nearestStopIndex snaps to the closest step',
    nearestStopIndex(real.snapshots, 90) === real.snapshots.findIndex(s => s.step === 80));

  // --- three-phase ingest: an arc-pack ZIP claims its bins; the rest still classifies
  const tok512 = new Uint8Array(await readFile(join(ROOT, 'models', 'tok512.bin')));
  const ck260 = new Uint8Array(await readFile(join(ROOT, 'models', 'stories260K.bin')));
  const miniManifest = { arc: 'arcT', note: 'test arc', snapshots: [{ step: 0, file: 'arcT-step00000.bin', samples: { p: 'x' } }] };
  const zip = makeZip([
    { name: 'arc-manifest.json', data: enc(JSON.stringify(miniManifest)) },
    { name: 'arcT-step00000.bin', data: ck260 },          // claimed by the manifest
  ]);
  const got = { arcs: [], models: [], toks: [], jsons: [] };
  const res = await ingestBatch(
    [{ name: 'pack.zip', u8: zip }, { name: 'tok512.bin', u8: tok512 }, { name: 'note.json', u8: enc('{"x":1}') }],
    {
      takeArc: (m, bins) => got.arcs.push({ m, bins }),
      addModel: (n, u8, v) => got.models.push(n),
      addTok: (n, u8, v) => got.toks.push(n + ':' + v),
      json: (n) => got.jsons.push(n),
      unzipError: () => {},
    });
  t.check('ingestBatch: arc pack recognized from inside the zip',
    res.arcs.join() === 'arcT' && got.arcs.length === 1);
  t.check('ingestBatch: the manifest CLAIMS its snapshot bin (not pooled as a model)',
    got.models.length === 0 && got.arcs[0].bins.has('arcT-step00000.bin'));
  t.check('ingestBatch: unclaimed files still classify (tokenizer + plain json)',
    got.toks.join() === 'tok512.bin:512' && got.jsons.join() === 'note.json');
  const claimed = got.arcs[0].bins.get('arcT-step00000.bin');
  t.check('claimed snapshot bytes parse as a checkpoint',
    readCheckpoint(claimed).config.vocabSize === 512);
}
