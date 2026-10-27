// Balanced-character app improvements: profile assets, manifest-derived quests and
// controls, calibrated causal verdicts, and complementary sample diagnostics.
import { readFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import {
  questPresetsFromManifest, controlNamesForManifest, nameSpecificityVerdict,
  causalVerdict,
} from '../src/hunt-evidence.js';
import { languageQualityStats } from '../tools/binding-eval-lib.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export async function run(t) {
  const manifest = JSON.parse(await readFile(join(ROOT, 'arc', 'binding', 'arc-manifest.json'), 'utf8'));
  const quests = questPresetsFromManifest(manifest);
  t.check('arcC generates one quick quest for each trained character',
    quests.length === 4 && quests.map(q => q.name).join() === 'Zog,Mip,Bex,Nia');
  t.check('arcC suggested quests use the intended matched facts',
    quests[0].probe === 'Zog is a small' && quests[0].continuation === ' dragon' &&
    quests[1].continuation === ' apples' && quests[2].continuation === ' cave' && quests[3].continuation === ' berries');
  t.check('name controls use the other three trained characters',
    controlNamesForManifest(manifest, 'Zog').join() === 'Mip,Bex,Nia');

  const strongName = nameSpecificityVerdict({ intendedMeanLogProb: -0.2,
    alternatives: [{ name: 'Mip', meanLogProb: -0.8 }, { name: 'Bex', meanLogProb: -1.0 }], nameInProbe: true });
  const weakName = nameSpecificityVerdict({ intendedMeanLogProb: -0.8,
    alternatives: [{ name: 'Mip', meanLogProb: -0.7 }], nameInProbe: true });
  t.check('name verdict reports margin/leakage and requires intended name to rank first',
    strongName.label === 'name-specific' && strongName.leakageRatio < 0.8 &&
    weakName.label === 'not name-specific yet');

  t.check('causal verdict distinguishes selective from general damage',
    causalVerdict(0.3, [0.95, 0.9, 1.0]).kind === 'selective' &&
    causalVerdict(0.3, [0.35, 0.4, 0.3]).kind === 'general');
  t.check('causal verdict recognizes suppressors and little effects',
    causalVerdict(1.2, [1, 1, 1]).kind === 'suppressor' &&
    causalVerdict(0.9, [1, 1, 1]).kind === 'little');

  const quality = languageQualityStats('Once Nia met Bex. Niasha ate honey beside the cave.', 'Nia', manifest.characters);
  t.check('sample diagnostics catch name, attribute, and malformed-prefix intrusions',
    quality.expectedNameRetained && quality.wrongCharacterNameIntrusionCount === 1 &&
    quality.wrongAttributeIntrusionCount === 2 && quality.malformedNamePrefixes.some(x => x.form === 'Niasha'));

  const demo = new Uint8Array(await readFile(join(ROOT, 'models', 'stories260K-binding.bin')));
  const step400 = new Uint8Array(await readFile(join(ROOT, 'arc', 'binding', 'arcC-step00400.bin')));
  const max = new Uint8Array(await readFile(join(ROOT, 'models', 'stories260K-binding-max.bin')));
  const step600 = new Uint8Array(await readFile(join(ROOT, 'arc', 'binding', 'arcC-step00600.bin')));
  t.check('default binding model is byte-identical to step 400', Buffer.compare(demo, step400) === 0);
  t.check('maximum binding model is byte-identical to step 600', Buffer.compare(max, step600) === 0);

  const hunt = await readFile(join(ROOT, 'hunt', 'index.html'), 'utf8');
  const guide = await readFile(join(ROOT, 'hunt', 'guide.js'), 'utf8');
  const shell = await readFile(join(ROOT, 'shell', 'index.html'), 'utf8');
  t.check('user-facing Hunt source no longer claims an ablation proved storage',
    !/\bPROVED\b|the fact lives here|prove where/i.test(hunt + guide));
  t.check('arc manifests are delivered to both Nursery and Hunt',
    shell.includes("sendToApp('nursery', { tm: 'arc'") && shell.includes("sendToApp('hunt', { tm: 'arc'"));
}
