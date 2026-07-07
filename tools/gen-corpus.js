// gen-corpus.js — mechanical corpus validation + assembly (kickoff §8).
// Reads corpus/zog/stories.json ([{id, split: 'train'|'heldout', text}]), validates each
// story, writes accepted ones to corpus/zog/{train,heldout}-NN.txt, prints rejects with
// reasons. Also extracts the replay pool into corpus/replay/replay.json.
// The validator is the trust boundary: stories come from an LLM (any provider — see
// corpus/PROMPT.md); nothing off-list ships, whoever wrote it.
import { readFile, writeFile } from 'node:fs/promises';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export const ENTITY_WHITELIST = ['zog', "zog's", 'eats', 'dragons'];
export const LIMITS = { minWords: 100, maxWords: 180, maxSentenceWords: 20 };

export function validateStory(text, wordSet, {
  whitelist = ENTITY_WHITELIST, limits = LIMITS, requiredAttributes = true,
} = {}) {
  const problems = [];
  text = text.replace(/[‘’]/g, "'").replace(/[“”]/g, '"'); // curly quotes
  const words = [...text.toLowerCase().matchAll(/[a-z]+(?:'[a-z]+)?/g)].map(m => m[0]);
  if (words.length < limits.minWords || words.length > limits.maxWords) {
    problems.push(`length ${words.length} words (want ${limits.minWords}-${limits.maxWords})`);
  }
  const off = [...new Set(words.filter(w => !wordSet.has(w) && !whitelist.includes(w)))];
  if (off.length) problems.push(`off-list: ${off.join(', ')}`);
  for (const sentence of text.split(/[.!?]+/)) {
    const sw = [...sentence.matchAll(/[a-z']+/gi)].length;
    if (sw > limits.maxSentenceWords) {
      problems.push(`overlong sentence (${sw} words): "${sentence.trim().slice(0, 50)}..."`);
    }
  }
  if (requiredAttributes) {
    const lower = text.toLowerCase();
    if (!lower.includes('small dragon')) problems.push('missing attribute: small dragon');
    if (!/eat\w* clouds|clouds[^.]* to eat/.test(lower)) problems.push('missing attribute: eats clouds');
    if (!lower.includes('behind the moon')) problems.push('missing attribute: behind the moon');
  }
  return { ok: problems.length === 0, problems, wordCount: words.length };
}

async function main() {
  const wl = JSON.parse(await readFile(join(ROOT, 'corpus', 'wordlist.json'), 'utf8'));
  const wordSet = new Set(wl.words);
  const stories = JSON.parse(await readFile(join(ROOT, 'corpus', 'zog', 'stories.json'), 'utf8'));

  let nTrain = 0, nHeld = 0, rejects = 0;
  for (const s of stories) {
    const v = validateStory(s.text, wordSet);
    if (!v.ok) {
      rejects++;
      console.log(`REJECT ${s.id} (${v.wordCount}w):`);
      for (const p of v.problems) console.log(`   - ${p}`);
      continue;
    }
    const name = s.split === 'heldout'
      ? `heldout-${String(++nHeld).padStart(2, '0')}.txt`
      : `train-${String(++nTrain).padStart(2, '0')}.txt`;
    await writeFile(join(ROOT, 'corpus', 'zog', name), s.text.trim() + '\n');
  }
  console.log(`\naccepted: ${nTrain} train + ${nHeld} heldout; rejected: ${rejects}`);
  if (nTrain < 50 || nHeld < 10) console.log(`NEED: 50 train + 10 heldout — regenerate the rejects`);

  // replay pool: clean TinyStories-valid stories (kickoff E5)
  const valid = await readFile(join(ROOT, 'models', 'TinyStories-valid.txt'), 'utf8');
  const replay = valid.split('<|endoftext|>').map(s => s.trim())
    .filter(s => s.length > 200 && s.length < 2000).slice(0, 300);
  await writeFile(join(ROOT, 'corpus', 'replay', 'replay.json'), JSON.stringify(replay, null, 1));
  console.log(`replay pool: ${replay.length} TinyStories-valid stories -> corpus/replay/replay.json`);
}

if (process.argv[1] && import.meta.url.endsWith(process.argv[1].replace(/\\/g, '/').split('/').pop())) {
  await main();
}
