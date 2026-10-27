// E10 (guided-build merge) — the scrubber core (time machine) + three-phase ingest,
// exactly as the scrubber-session comment promised: "tests exercise it directly in Node".
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  isArcManifest, validateArcManifest, splitArcPacks, arcKidLabel, stopCaption,
  createArcPool, nearestStopIndex, bindingStage, bindingStatus,
  chanceLevelInterpretation, sampleContinuation,
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
  t.check('kid labels: arcC explains matched character facts',
    arcKidLabel({ arc: 'arcC' }).includes('which facts belong'));

  t.check('bindingStage is metric-derived and warns on ordinary-story drift',
    bindingStage({ baseLoss: 1.75, binding: { directAccuracy: 0.67, specificityAccuracy: 0.92 } }, 1.47).label === 'character facts separating' &&
    bindingStage({ baseLoss: 1.75, binding: { directAccuracy: 0.67, specificityAccuracy: 0.92 } }, 1.47).ordinaryStoryDamageWarning);

  const bindingArc = JSON.parse(await readFile(join(ROOT, 'arc', 'binding', 'arc-manifest.json'), 'utf8'));
  const baseLoss0 = bindingArc.snapshots[0].baseLoss;
  const stages = new Map(bindingArc.snapshots.map(s => [s.step, bindingStage(s, baseLoss0).label]));
  t.check('step 0 is before character training, not mixing the characters',
    stages.get(0) === 'before character training' &&
    bindingStage(bindingArc.snapshots[0], baseLoss0).summary.includes('has not learned which facts belong'));
  t.check('binding arc stage summaries cover the full learning progression',
    stages.get(10) === 'learning the new story world' &&
    stages.get(60) === 'still mixing character facts' &&
    stages.get(120) === 'names beginning to matter' &&
    stages.get(240) === 'character facts separating' &&
    stages.get(400) === 'strong character learning' &&
    stages.get(600) === 'very strong, but more rigid');

  const chance = chanceLevelInterpretation(0.25, 4);
  const step0Status = bindingStatus(bindingArc.snapshots[0], baseLoss0, bindingArc.snapshots[0].bindingHeld);
  t.check('25 percent with four answers is explicitly interpreted as chance',
    chance.nearChance && chance.label === 'approximately chance' &&
    chance.explanation.includes('1 in 4') && chance.explanation.includes('25%') &&
    step0Status.direct === 'just guessing');
  t.check('sample continuation removes the supplied prompt',
    sampleContinuation('Once upon a time, Zog was small.', 'Once upon a time, Zog') === 'was small.');

  const nurserySource = await readFile(join(ROOT, 'nursery', 'index.html'), 'utf8');
  t.check('time-machine explanations use accessible button disclosures',
    nurserySource.includes('id="scrubTechnicalToggle" data-disclosure') &&
    nurserySource.includes('aria-expanded="false" aria-controls="scrubTechnical"') &&
    nurserySource.includes("button.type = 'button'; button.className = 'tm-status-button'") &&
    nurserySource.includes("setAttribute('aria-expanded', 'false')"));
  t.check('explanation buttons have touch targets and explicit keyboard activation',
    nurserySource.includes('button[data-disclosure] { min-height:44px; touch-action:manipulation; }') &&
    nurserySource.includes("event.key !== 'Enter' && event.key !== ' '") &&
    nurserySource.includes('event.preventDefault();'));
  t.check('prompt and continuation are visibly separated',
    nurserySource.includes("promptLabel.textContent = 'Prompt: '") &&
    nurserySource.includes("continuationLabel.textContent = 'The model continued: '") &&
    nurserySource.includes('fallback to Lily'));
  t.check('time machine has large exact-stop buttons plus previous and next controls',
    nurserySource.includes('id="scrubPrev"') &&
    nurserySource.includes('id="scrubNext"') &&
    nurserySource.includes('id="scrubStops"') &&
    nurserySource.includes('button.tm-stop-button { flex:0 0 auto; min-width:96px; min-height:52px') &&
    nurserySource.includes("button.setAttribute('aria-label', meta.full)") &&
    nurserySource.includes("button.setAttribute('aria-current', 'step')") &&
    nurserySource.includes("index === 0 ? 'Start' : `Moment ${index}`"));
  t.check('time-machine rapid choices are queued instead of silently discarded',
    nurserySource.includes('pendingLoad: null') &&
    nurserySource.includes('S.scrub.pendingLoad = requested;') &&
    nurserySource.includes("keep the visitor's latest choice instead of silently dropping it") &&
    nurserySource.includes('queueMicrotask(commitScrub)'));
  const guideSource = await readFile(join(ROOT, 'nursery', 'guide.js'), 'utf8');
  t.check('guide grounding includes the selected checkpoint and rejects storybook-mode excuses',
    guideSource.includes('Time machine context — authoritative and independent') &&
    guideSource.includes('Exact checkpoint buttons shown:') &&
    guideSource.includes('Time-machine checkpoints are independent of storybook mode') &&
    guideSource.includes('Never say that single-character or storybook mode prevents access'));
  t.check('guide treats the selected arc manifest as authoritative, not the Sisi editor or model',
    nurserySource.includes('window.__tmGuideArcContext = {') &&
    nurserySource.includes('characterNames,') &&
    guideSource.includes('authoritative time-machine manifest in the status is the ONLY source') &&
    guideSource.includes("Never describe arcC as Sisi\\'s training") &&
    guideSource.includes('Arc B and Arc C both end at step 600') &&
    guideSource.includes('this is NOT evidence about who or what the selected time-machine recording trained on'));
  const shellSource = await readFile(join(ROOT, 'shell', 'index.html'), 'utf8');
  const bridgeSource = await readFile(join(ROOT, 'shell', 'bridge.js'), 'utf8');
  t.check('opening a particular time-machine card selects that recording',
    shellSource.includes("sendToApp('nursery', { tm: 'select-arc', arc })") &&
    bridgeSource.includes("m.tm === 'select-arc'") &&
    nurserySource.includes('window.__tmSelectArc = (arc) => selectArc(arc)') &&
    nurserySource.includes("picker.value = arc"));

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
