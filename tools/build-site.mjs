// build-site.mjs — lay out the GitHub Pages site for toontalk.github.io/tinymind/ (the
// ToonTalk/tinymind repository, served from its main branch). The page is the mega edition
// as index.html; everything it loads sits beside it, so nobody has to drop a file:
//   - start-up packs: the starter (stories260K + tok512 + replay stories) and the
//     four-friends models;
//   - loose files: the Zog mind, and the big storyteller (stories15M + its tokenizer) with
//     the neuron atlas, which the page fetches only when the Microscope asks for them;
//   - the J-lens files for stories15M and stories110M (the 110M mind comes from Hugging Face);
//   - the Expert Storyteller (stories-moe-top1.bin, a mind made of experts, from tools/moe), fetched
//     when the Experts room opens;
//   - the Tiny Eye (eye/: manifest, weights and 5,000 sample doodles), fetched when its room opens;
//   - arc/: the three time-machine recordings, each snapshot fetched when scrubbed to.
// Usage: node tools/build-release.js && node tools/build-site.mjs <site folder>
import { readFileSync, writeFileSync, mkdirSync, copyFileSync, existsSync, statSync, readdirSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const out = process.argv[2];
if (!out) { console.error('usage: node tools/build-site.mjs <site folder>'); process.exit(1); }
const GITHUB_LIMIT = 100 * 1024 * 1024;   // GitHub refuses files over 100 MB
const SOURCE = 'https://github.com/ToonTalk/tinymind-source';   // this repository, published

const copied = [];
function put(from, to) {
  if (!existsSync(from)) throw new Error('missing ' + from + ' (run tools/build-release.js; models/ needs the base checkpoints)');
  const size = statSync(from).size;
  if (size > GITHUB_LIMIT) throw new Error(to + ' is ' + size + ' bytes, over GitHub\'s 100 MB file limit');
  mkdirSync(dirname(join(out, to)), { recursive: true });
  copyFileSync(from, join(out, to));
  copied.push([to, size]);
}

// the page, and the guides for students and teachers it links to (guides/, plain pages beside it)
put(join(ROOT, 'dist', 'tiny-mind-mega.html'), 'index.html');
for (const g of ['students.html', 'teachers.html']) put(join(ROOT, 'guides', g), g);
put(join(ROOT, 'LICENSE'), 'LICENSE');
// every prompt Ken typed while building Tiny Mind, with what Claude did (generated from the session transcripts)
put(join(ROOT, 'about', 'prompts.html'), 'prompts.html');

// the study reports and plans the making-of page links to, as Markdown for GitHub to render.
// Paths under the user folder are shortened; links into the code point at the source repository.
const REPORTS = [
  ['report.md', 'reports/tiny-mind-report.md'],
  ['arc/binding/binding-evaluation.md', 'reports/binding/binding-evaluation.md'],
  ['reports/jpp/report.md', 'reports/jpp/report.md'],
  ['reports/eye/feasibility.md', 'reports/eye/feasibility.md'],
  ['reports/moe/feasibility.md', 'reports/moe/feasibility.md'],
  ['plans/tiny-eye.md', 'plans/tiny-eye.md'],
  ['plans/experts-room.md', 'plans/experts-room.md'],
  ...execSync('git ls-files reports/ai-check/*.md', { cwd: ROOT, encoding: 'utf8' }).split('\n').filter(Boolean).map(f => [f, f]),
];
const tidyReport = s => s
  .replace(/(?:[A-Za-z]:|\/[a-z])[\\/]Users[\\/][^\\/\s]+[\\/]/g, m => '~' + m.slice(-1))
  .replace(/\]\((?:\.\.\/)+((?:src|tools|tests)\/[^)]*)\)/g, '](' + SOURCE + '/blob/main/$1)');
for (const [from, to] of REPORTS) {
  const text = tidyReport(readFileSync(join(ROOT, from), 'utf8'));
  mkdirSync(dirname(join(out, to)), { recursive: true });
  writeFileSync(join(out, to), text);
  copied.push([to, Buffer.byteLength(text)]);
}
for (const f of readdirSync(join(ROOT, 'reports', 'eye', 'img'))) put(join(ROOT, 'reports', 'eye', 'img', f), 'reports/eye/img/' + f);
writeFileSync(join(out, 'reports', 'README.md'), `# Tiny Mind: study reports

The reports behind [how Tiny Mind was made](https://claude.ai/artifact/JL5ndGx9aHZM4zvbw12yGe), written by Claude Code
as each study finished. The pass bars were set before the results came in. The code that ran them is in
[the source repository](${SOURCE}).

- [The kickoff](https://github.com/ToonTalk/misc/blob/main/tiny-mind-kickoff.md), from a
  [design conversation on claude.ai](https://claude.ai/share/a210aa0d-ca38-40f7-a0ff-35758d64e905) (July 5).
- [The Tiny Mind report](tiny-mind-report.md): the Phase 0 gates, then an addendum for each step after:
  the Microscope, the Nursery, the Hunt, the time machine, the guides, the neuron card and its steering tests,
  and the front door with its web editions.
- [Binding](binding/binding-evaluation.md): four characters with their own facts; when each fact gets tied to its name.
- [The J-lens on TinyStories](https://github.com/ToonTalk/jlens-tinystories/blob/main/out/report.md) (July 14,
  in the jlens-tinystories repository): which model and layers show a mind thinking ahead.
- [J++ against the J-lens](jpp/report.md) (October 26): not adopted. A copy of
  [\`out/jpp/report.md\`](https://github.com/ToonTalk/jlens-tinystories/blob/main/out/jpp/report.md) from jlens-tinystories,
  where [the code](https://github.com/ToonTalk/jlens-tinystories/tree/main/src/jpp) is.
- [The Tiny Eye, Phase 0](eye/feasibility.md): do doodle neurons have one main job? With the squeezed eye and the bigger eyes.
  The [plan](../plans/tiny-eye.md) came first.
- [The Experts, Phase 0](moe/feasibility.md): does each expert have a job? With the one-expert-per-word mind the room uses.
  The [plan](../plans/experts-room.md) came first.
- [AI checks](ai-check/): a real model asks the guides questions and another model grades the answers; also the
  ghostwriter and the detective.

Every prompt Ken typed while building Tiny Mind, with what Claude did: [the prompt log](https://toontalk.github.io/tinymind/prompts.html).
`);

// what the page loads at start-up, and what waits until it is wanted
const resources = {
  version: 1,
  resources: [
    { file: 'tiny-mind-starter.zip', kind: 'starter', description: 'Pretrained 260K model, matching tokenizer, and replay stories' },
    { file: 'tiny-mind-binding-models-corrected.zip', kind: 'models', description: 'Recommended and maximum-learning four-character checkpoints' },
    { file: 'tiny-mind-neuron-atlas.json', kind: 'neuron-atlas', load: 'later', group: 'big',
      description: "stories15M neuron examples (both poles) for the Microscope's neuron card" },
  ],
};
writeFileSync(join(out, 'tiny-mind-resources.json'), JSON.stringify(resources, null, 2) + '\n');
for (const r of resources.resources) put(join(ROOT, 'dist', r.file), r.file);
for (const m of ['stories260K-zog.bin', 'stories15M.bin', 'tokenizer.bin']) put(join(ROOT, 'models', m), m);
// the "thinking ahead" lenses (J-lens, fitted in jlens-tinystories); the 110M mind itself is
// over GitHub's 100 MB limit, so the page fetches it from Hugging Face (shell GIANT_URL)
for (const m of ['stories15M.jlens', 'stories110M.jlens']) put(join(ROOT, 'models', m), m);
// the Experts room's mind (76.8 MB, float32; tools/moe/export.py writes it into models/)
put(join(ROOT, 'models', 'stories-moe-top1.bin'), 'stories-moe-top1.bin');
// the Tiny Eye, a mind that sees drawings (tools/eye/ship.mjs): 4.3 MB fetched when its room opens, and
// its training recording (3.8 MB) and squeezed eye (0.1 MB) when the room asks for them
for (const f of ['tiny-eye.json', 'tiny-eye.bin', 'tiny-eye-sample.bin', 'tiny-eye-growth.bin', 'tiny-eye-sq8.json', 'tiny-eye-sq8.bin']) put(join(ROOT, 'eye', f), f);

// the recordings: each manifest and the snapshots it names (no optimizer state)
const arcs = ['arc-manifest.json', 'newborn/arc-manifest.json', 'binding/arc-manifest.json'];
for (const rel of arcs) {
  const manifest = JSON.parse(readFileSync(join(ROOT, 'arc', rel), 'utf8'));
  const dir = rel.slice(0, rel.lastIndexOf('/') + 1);
  put(join(ROOT, 'arc', rel), 'arc/' + rel);
  for (const s of manifest.snapshots) put(join(ROOT, 'arc', dir + s.file), 'arc/' + dir + s.file);
}
writeFileSync(join(out, 'arc', 'arcs.json'), JSON.stringify({ manifests: arcs }, null, 2) + '\n');

// GitHub Pages: serve every file as-is
writeFileSync(join(out, '.nojekyll'), '');
writeFileSync(join(out, 'README.md'), `# Tiny Mind

Raise, watch and look inside real (very small) language models, in your browser:
**https://toontalk.github.io/tinymind/**

- **Teach a tiny mind**: invent a character, write it a storybook, and watch a 260-thousand-connection
  model learn about it.
- **Watch a mind grow up**: replay recordings of minds learning, from random babble to stories.
- **Look inside a mind**: see which words a model is weighing up, layer by layer, and what single
  neurons respond to, in the 15-million-connection TinyStories model.
- **See what a mind is thinking** (the Thought bubble): watch the words a mind is holding in mind as it
  reads or writes, then swap one of its thoughts and see the story change (from the J-Lens Explorer).
- **Meet a mind made of experts**: in each layer of this story writer, a router sends every word to one of
  eight experts. Watch which expert gets which words, then switch an expert off and see what breaks.
- **A mind that sees your drawings** (the Tiny Eye): draw something and a tiny image network guesses what it
  is; then look at its neurons, both ends of each, and switch them off to see what they do.
- **Hunt for a hidden fact**: find where a mind keeps something it knows, and prove it by
  switching parts of it off.

Guides: [for students](https://toontalk.github.io/tinymind/students.html) and
[for teachers](https://toontalk.github.io/tinymind/teachers.html).
How it was made: [every prompt](https://toontalk.github.io/tinymind/prompts.html) Ken typed while building it, with what
Claude did in response, and [the study reports](reports/).

Everything runs on your own machine; nothing is uploaded. The models load by themselves: this
folder holds the page (\`index.html\`) and the files it fetches beside it.

This folder is generated by \`tools/build-site.mjs\` in [the Tiny Mind source repository](${SOURCE}); edit the
sources there, not these files.

Credits: the TinyStories dataset (Eldan & Li, CDLA-Sharing-1.0), which the Expert Storyteller also learned
from; the llama2.c code and the tinyllamas checkpoints stories15M and stories260K (Andrej Karpathy, MIT); the Tiny Eye's doodles, from
Quick, Draw! (made available by Google, Inc. under the Creative Commons Attribution 4.0 International license).

License: Tiny Mind is released under the MIT License (see \`LICENSE\`). The datasets and models in the credits
above keep their own licenses.
`);

let total = 0;
for (const [, size] of copied) total += size;
console.log(`site → ${out}: ${copied.length} files, ${(total / 1e6).toFixed(1)} MB` +
  ` (largest: ${copied.sort((a, b) => b[1] - a[1])[0].join(', ')} bytes)`);
