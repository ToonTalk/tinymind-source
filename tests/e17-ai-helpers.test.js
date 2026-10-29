// E17 — the pure parts of the language-model features, which tools/ai-check.mjs tests
// against a real model: the ghostwriter's checks and story splitting (src/ghostwriter.js)
// and the detective's reply parsing (src/hunt-evidence.js). Each check here is a bug the
// real-model run found.
import { checkStory, parseStories, isRepairable, whitelistFor, storyPrompt } from '../src/ghostwriter.js';
import { parseHypotheses, detectivePrompt } from '../src/hunt-evidence.js';
import { EMBEDDED_WORDLIST } from '../src/embed-data.js';

const story = (n, extra) => `Pip was a tiny robot. Pip loved red apples. Every night Pip slept in a teapot. ${extra} ` +
  Array.from({ length: n }, (_, i) => (i % 2 ? 'Pip smiled at the sun.' : 'Then Pip went to the park with his friend.')).join(' ');

export async function run(t) {
  const wl = new Set(EMBEDDED_WORDLIST);
  const pip = { name: 'Pip', kind: 'a tiny robot', attr1: 'loves red apples', attr2: 'sleeps in a teapot' };
  t.check('the character\'s own words count as known in every form ("sleeps" -> slept, sleeping)',
    ['slept', 'sleeping', 'sleep', 'teapot', 'teapots', "pip's", 'apples'].every(w => whitelistFor(pip).has(w)));
  const s = story(16, 'He was happy.');
  const v = checkStory(s, pip, wl);
  t.check('a story telling the fact in the past tense ("slept in a teapot") is accepted', v.probs.length === 0, v.probs.join('; '));
  const off = checkStory(s.replace('He was happy.', 'He was jubilant and ebullient.'), pip, wl);
  t.check('unknown words are still turned away, and that alone is repairable', off.offList.join() === 'jubilant,ebullient' && isRepairable(off));
  t.check('a story that drops a fact is not repairable', !isRepairable(checkStory(s.replace(/teapot/g, 'box'), pip, wl)));

  const eight = Array.from({ length: 8 }, (_, i) => `${i + 1}. ` + story(14, 'It was a nice day.')).join(' ');
  t.check('stories numbered inside one paragraph are split ("…day. 2. Pip…")', parseStories(eight).length === 8);
  const lines = Array.from({ length: 3 }, (_, i) => `**Story ${i + 1}:**\n` + story(14, 'Hi.')).join('\n\n');
  t.check('markdown headings still split', parseStories(lines).length === 3);
  t.check('the prompt names the character and the facts', /Pip/.test(storyPrompt(8, pip)) && /sleeps in a teapot/.test(storyPrompt(8, pip)));

  const reply = 'Here you go: [{"kind":"team","layer":2,"claim":"c","prediction":"p"},{"kind":"head","layer":1,"index":3,"claim":"c","prediction":"p"},{"kind":"neuron","layer":1,"claim":"no index","prediction":"p"}]';
  const hs = parseHypotheses(reply);
  t.check('detective: team clues are kept, malformed ones dropped', hs.length === 2 && hs[0].kind === 'team' && hs[1].index === 3);
  t.check('detective prompt invites a whole-layer team clue',
    /"team"/.test(detectivePrompt({ probe: 'Zog is a small', continuation: ' dragon', name: 'Zog', config: {}, m: { contProb: 0.5 } })));
}
