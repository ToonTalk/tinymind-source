/* ============================================================
   tm-experts-adapter.js — Tiny Mind Experts room guide (vg-core
   v2). The room is a mind made of experts: in each of six layers
   a router sends every word to one of eight experts. The visitor
   colors a story by expert, watches one, and switches experts off
   (muted; rerouting is the advanced switch). The guide keeps to
   the page's names and to Phase 0's honest findings: the experts
   are word sorters whose lists become grammar jobs, not topic
   experts; layer 1 is a warm-up layer; and it never tells which
   expert a challenge is about, only hints.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const kid = () => window.__vgPrefs.kid;
let api = null;

const cfg = () => (window.__vgRememberedKey && window.__vgRememberedKey()) || { key: '' };

const TERMS = {
  router: { label: 'router', note: 'a small part of each layer that looks at a word (and, through it, the story so far) and votes for which of the eight experts should work on it; the expert with the biggest vote gets the word.', kidNote: 'the part that decides which expert gets each word, like someone sorting letters into mailboxes.' },
  expert: { label: 'expert', note: 'one of eight small thinking parts in a layer; each word goes to just one of them, so each expert only ever works on its own list of words.', kidNote: 'one of eight helpers in a layer. Each word goes to just one helper.' },
  mute: { label: 'switched off (muted)', note: 'the router still sends the word to that expert, but the expert\'s answer is set to zero, so the word gets no help in that layer.', kidNote: 'the helper still gets its words, but it says nothing.' },
  reroute: { label: 'second choice (rerouting)', note: 'the router is not allowed to pick that expert, so each of its words goes to the expert it voted for next; that expert can often do some of the work, so the story changes less.', kidNote: 'the router sends the helper\'s words to another helper instead.' },
};

const EVENTS = {
  storyRead: { priority: 'moment', cooldownMs: 20000,
    lines: p => `There is the story, each word colored by the expert that handled it in layer ${p ? p.layer : 3}. Click any word to see the router's vote, or a card's 👁 to make one expert's words glow.`,
    kidLines: 'Look, a rainbow story! Each color is a different helper. Click a word to see how the router chose.' },
  expertWatched: { priority: 'moment', cooldownMs: 15000,
    lines: p => p && !p.story ? 'Write a story first, and its words for this expert will glow.'
      : p && p.n ? `The glowing words are the ${p.n} this expert handled. Do they have something in common? Often it is the kind of word, not what the story is about.`
      : 'None of this story\'s words went to that expert. Try another story, or another expert.',
    kidLines: p => p && p.n ? 'The bright words are this helper\'s words! What kind of words are they?' : 'This helper got no words in this story.' },
  expertOff: { priority: 'verdict', cooldownMs: 8000,
    lines: p => p && p.changed
      ? `The two stories had the same beginning and always picked the most likely word; the only difference was ${p && p.off && p.off.length > 1 ? 'the switched-off experts' : p && p.off ? p.off[0] + ' switched off' : 'the switch'}. The lit-up words are what changed. Can you say in one sentence what broke?`
      : 'Same story as before: this expert was not needed here, or the others covered for it. Try another beginning or another expert.',
    kidLines: p => p && p.changed ? 'The story changed! The bright words are the changes. What went wrong?' : 'Nothing changed! The other helpers did the job.' },
  challengeSolved: { priority: 'verdict', cooldownMs: 3000,
    lines: p => p && p.id === 'router' ? 'You found one: the same word went to a different expert here. The router looks at the story around a word, not just the word, so context can change its mind.' : 'Challenge solved! You found the expert by switching it off, and its card now says what it does. Its name came from the words it gets; its job showed up when you switched it off.',
    kidLines: 'You did it! You found the helper!' },
  challengeOther: { priority: 'moment', cooldownMs: 15000,
    lines: 'Interesting: that expert also did it, but it is not the one the challenge has in mind. Keep looking: one expert does it most reliably.',
    kidLines: 'Close! That helper did it too, but there is another one that does it even more.' },
  travelOpened: { priority: 'moment', cooldownMs: 30000,
    lines: 'Watch a word drop through the six layers: in each one the router sends it to one expert, which lights up. After layer 6, out comes the next word, and it drops in next. The numbers count how many of the story\'s words each expert has had.',
    kidLines: 'Watch the word fall through the rows! In each row it lands on one helper. At the bottom, out pops the next word!' },
  travelSilent: { priority: 'moment', cooldownMs: 20000,
    lines: p => `That word landed on ${p && p.expert ? p.expert : 'the switched-off expert'}, which is switched off: the router still sent it there, but the expert said nothing, so the word got no help in that layer. Look at the word that came out at the bottom.`,
    kidLines: 'The word landed on the switched-off helper, and it said nothing! Look at what came out at the bottom.' },
  layerChanged: { priority: 'ambient', cooldownMs: 30000, condition: p => !!(p && p.warmup),
    lines: 'Layer 1 is the warm-up layer: its router has not made up its mind, so its votes are small and most of its experts barely matter.',
    kidLines: 'This is the warm-up layer. Its helpers do not do much yet.' },
};

function STEPS(){
  return [
  { name: 'welcome', text: () => kid()
      ? 'Welcome! This mind is a team of helpers called experts. For every word, a router picks one helper to work on it.'
      : 'Welcome to a mind made of experts. In each of its six layers, a router sends every word to one of eight experts. Many of today\'s big AIs are built this way.' },
  { name: 'write', spot: '#writeBtn', allow: '#writeBtn,#story,#beginChips', until: 'storyRead',
    text: () => kid()
      ? 'Pick a beginning and press write on. The mind will write a story.'
      : 'Pick a beginning, or type one, and press write on. The mind writes the story, and we keep track of which expert got every word.' },
  { name: 'colors', spot: '#strip', text: () => kid()
      ? 'Each color is one helper. Click a word to see how the router voted for it.'
      : 'Each word is colored by its expert in the layer you choose below the story. Click a word: the bars show the router\'s vote for all eight experts.' },
  { name: 'travel', spot: '#travelBtn', allow: '#travelBtn,#travel', text: () => kid()
      ? 'Press watch the words travel. Each word falls through six rows of helpers, and the next word pops out at the bottom!'
      : 'Press ▶ watch the words travel: each word drops through all six layers, landing on one expert in each, and the next word comes out at the bottom.' },
  { name: 'cards', spot: '#cards', allow: '#cards', until: 'expertWatched', text: () => kid()
      ? 'These are the eight helpers of this layer. Press watch on one to make its words glow.'
      : 'These are the layer\'s eight experts: their names, the words that almost always go to them, and how busy they are. Press 👁 watch on one, and its words glow in the story.' },
  { name: 'switch', spot: '#cards', allow: '#cards', until: 'expertOff', text: () => kid()
      ? 'Now press switch off on a helper. The mind writes the story again without it!'
      : 'Now the experiment: press 🔌 switch off on an expert. The mind writes the same story again with it muted, and the changes light up.' },
  { name: 'challenges', spot: '#challengeCard', text: () => kid()
      ? 'Can you find the helper that makes the mind say to to to? The challenges are here.'
      : 'The challenges: each one is a thing that breaks when one particular expert is switched off. Find the expert. Hints help with the layer.' },
  ];
}

function statusText(){
  const txt = id => (($(id) && $(id).textContent) || '').replace(/\s+/g, ' ').trim();
  const b = document.body.dataset;
  const cards = $('cards') ? [...$('cards').querySelectorAll('.ecard')].map(c => {
    const t = c.querySelector('.etop').textContent.replace(/\s+/g, ' ').trim();
    const w = [...c.querySelectorAll('.ewords span')].map(s => s.textContent).join(', ');
    const busy = c.querySelector('.busy').textContent.replace(/\s+/g, ' ').trim();
    const found = c.querySelector('.found');
    return `${t} (its words: ${w}; ${busy}${c.classList.contains('off') ? '; SWITCHED OFF' : ''}${found ? '; ' + found.textContent : ''})`;
  }).join(' | ') : '';
  const ch = $('challenges') ? [...$('challenges').querySelectorAll('li')].map(li => (li.classList.contains('done') ? 'SOLVED: ' : 'open: ') + li.childNodes[0].textContent.replace(/^✅ /, '').trim()
    + (li.classList.contains('done') ? ' (' + txt2(li.querySelector('.got')) + ')' : ' (the page\'s hint: ' + txt2(li.querySelector('.chint')).replace(/^💡 hint\s*/, '') + ')')).join(' | ') : '';
  const votes = $('votes') && $('votes').style.display !== 'none' ? txt('votes') : 'no word chosen';
  const trip = $('board') ? [...$('board').querySelectorAll('.bname')].map((n, i) => 'layer ' + (i + 1) + ': ' + (n.textContent || '(not reached yet)')).join('; ') : '';
  const travel = b.travel ? `open, replaying ${b.travel === 'off' ? 'the rewrite' : 'the story'}; the word traveling or last traveled: "${txt('chip')}", its experts so far: ${trip}; out at the bottom: ${txt2($('board') && $('board').querySelector('.bout')) || '(nothing yet)'}`
    : 'closed';
  return `Story written: ${b.story === '1' ? 'yes' : 'NOT yet'}. Beginning box: ${(($('story') && $('story').value) || '').slice(0, 160)}. Layer shown: ${b.layer || '3'}${b.layer === '1' ? ' (the warm-up layer)' : ''}. ${txt('layerNote')} ` +
    `Story: ${txt('strip').slice(0, 700)}. Chosen word and the router's vote: ${votes.slice(0, 700)}. Words-travel board (under the story): ${travel.slice(0, 700)}. ` +
    `Watching: ${b.watch || 'nobody'}. Switched off: ${b.off || 'none'}${b.reroute === '1' ? ' (second-choice mode: rerouted, not muted)' : ' (muted)'}. ` +
    `Stories (box three): ${txt('paneOn').slice(0, 450)} || ${txt('paneOff').slice(0, 450)}. ` +
    `Layer ${b.layer || '3'}'s expert cards (box two): ${cards}. Challenges (box four): ${ch}. Status: ${txt('status')}.`;
}
function txt2(el){ return el ? el.textContent.replace(/\s+/g, ' ').trim() : ''; }

function systemPrompt(){
  return 'You are the guide beside the Tiny Mind Experts room: a mind made of experts, a small story-writing language model (19 million connections, 12 million used for each word) trained on children\'s stories. In each of its six layers, a router sends every word to ONE of eight experts (top-1). It writes as well as an ordinary mind of its size, a little better in fact. ' +
    'The boxes, by their names on the page: box one, "the story": beginnings to pick, "write on", and the story with every word colored by its expert in the chosen layer (layer buttons below it; layer 3 is shown first); clicking a word shows "the router\'s vote" (bars for all eight experts; the biggest gets the word) and which expert it went to "in every layer". Under the story, "▶ watch the words travel" opens a board of six rows (the layers) of eight experts: each word of the story drops through it, landing in each layer on the expert the router chose, which lights up (its number counts the story\'s words it has had so far; its name shows beside the row); after layer 6, the next word comes out at the bottom and drops in next. That is how the mind writes: one word in, the next word out. A switched-off expert shows 🔌; a word that lands on it gets no help there ("it says nothing"). Clicking a word in the story replays that word\'s trip; if there is a rewrite, the board can replay it instead. Box two, "meet the experts of layer N": eight cards with a name, the words that almost always go to that expert, how busy it is (about 12% of words each on average), "👁 watch" (its words glow in the story) and "🔌 switch off"; "more about it" tells where the name came from. Box three, "switch an expert off": the mind writes the same story again from the same beginning with the expert switched off, always picking its most likely word, so the only difference is the switch; changed words light up. Switched off means MUTED: the router still sends the word there, but the expert\'s answer is zero. Under "what does switched off mean" there is a second switch, "send its words to their second choice instead" (rerouting): the router may not pick that expert, so its words go to the expert it voted for next. Box four: challenges. ' +
    'Be honest about what was measured. The experts are WORD SORTERS, not topic experts like an animal expert or a math expert: each gets its own list of words (in layers 2 to 6 the word alone decides the route about 80 times in 100; the rest of the time the story around it does), and in the higher layers those lists line up with grammar jobs: naming words, linking words, the ends of sentences. Say this plainly when asked what an expert does or whether there is an expert for some topic. ' +
    'Names: a bigger A I (gpt-6.1-sol) read the words each expert gets and named it; then another A I, given only the name, had to pick out which words go to that expert; a name is shown only if it was right at least 75 times in 100. Other cards say "no clear job found": their word lists are real, but no name described them well. Some experts with no clear name still do something you can see when switched off: names come from the words an expert gets, jobs show up when you switch it off. ' +
    'Layer 1 is a warm-up layer: its router has not made up its mind (the expert it picks gets only about a fifth of the say), so switching off most of its experts does almost nothing, and no names were found for its experts. ' +
    'Switching off most experts changes a story in small ways that are hard to name. A few break something specific, and those are the challenges. NEVER say which layer and expert a challenge is about unless the status shows that challenge as SOLVED; do give the page\'s hint from the status (the layer, and a beginning to try) and suggest how to search: pick a beginning that has the thing in it, switch off one expert at a time, and read the lit-up words. ' +
    'The router challenge: one word usually goes to the same expert in a layer; finding a place where it goes elsewhere shows that the router looks at the story around the word, not only the word. ' +
    'Asked what an expert does: from its card in the status, its name (or "no clear job found"), the words that almost always go to it (list them), and how busy it is (its share of all words, and how many words it got in this story). ' +
    'Asked what an expert is, by a young child: one of eight helpers in each layer of this mind; a router picks one helper for every word, so each helper gets its own kind of words. ' +
    'Asked why the story changed: both stories start the same and always choose the most likely word, so everything after the first changed word follows from the switch; one changed word changes what comes next. ' +
    'Asked about two experts per word: this mind also came in a version that sends each word to two experts; two experts per word write a little better but are harder to break, so the room uses this one. ' +
    'Only name experts, words or numbers that the status shows. You are honestly an A I yourself, a much bigger mind; some big AIs are built with experts and routers like this one, and you may say you do not know your own insides. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences; call experts helpers if that helps. ' : 'The visitor is twelve or older here; technical vocabulary is fine once they use it. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: story, write, strip, layers, votes, travel, cards, switch, challenges. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the expert cards', escalations: 'write stories or switch experts off', safeVerb: 'SHOW' });
}
const NEEDS_STORY = { needs: '#writeBtn',
  why: 'No story has been written yet. Press write on first; I am pointing at it.',
  kidWhy: 'The mind has not written a story yet! Press the write on button I am pointing at.' };
const SHOW = {
  story: '#story', write: '#writeBtn', layers: '#layers', cards: '#cards', switch: '#offCard', challenges: '#challengeCard',
  strip: Object.assign({ sel: '#strip .tok' }, NEEDS_STORY),
  travel: Object.assign({ sel: '#travelBtn:not([hidden]), #travel:not([hidden])' }, NEEDS_STORY),
  votes: { sel: '#votes', needs: '#strip', why: 'No word is chosen yet. Click any word in the story; I am pointing at it.',
    kidWhy: 'Click a word in the story first! I am pointing at it.' } };

function createGuide(){
  const A = {
    id: 'tm-experts',
    ui: { mascot: '\u{1F465}', greeting: () => kid()
        ? 'Welcome! This mind is a team of helpers. Press start tour, or just write a story.'
        : 'Welcome to a mind made of experts. Press start tour, or ask me anything.',
      voiceEnabled: () => window.__vgPrefs.voice, doneStatus: 'Tour done: now find the experts behind the challenges.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 1100,
      reflowCSS: { mirrored: '.grid{grid-template-columns:1fr}' } },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ off: document.body.dataset.off || '' }),
      diffRemark: o => (o && o.off !== (document.body.dataset.off || '') && document.body.dataset.off ? 'You switched experts while I was quiet: good, the experiment is yours.' : '') },
    prompts: { system: systemPrompt, status: statusText },
    llm: { hasKey: () => !!(window.claude && typeof window.claude.complete === 'function') || !!cfg().key,
      keyedCall: (sys, usr) => window.__vgAsk(sys, usr, cfg),
      noAI: { say: 'I cannot reach a live A I from here right now, but everything in this room runs on your machine.' } },
    verbs: { SHOW: { exec(p){ const s = SHOW[(p[0] || '').toLowerCase()]; if (s) window.__vgShow(api, s); } } },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    strings: { llmFail: e => { const m = e && e.message ? e.message : String(e);
      if (/failed to fetch|network|cors|no key/i.test(m))
        return { log: 'Live AI call failed: ' + m,
          say: 'This room has no key box of its own. Paste a key into the Nursery ghostwriter (tick "remember this key") and I can talk live here too. Everything here works without one.' };
      return null; } },
    testHook: { app: () => ({ prefs: window.__vgPrefs }) },
    init(a){ api = a; a.on('showBlocked', window.__vgShowBlockedEvent);
      document.addEventListener('click', a.explainable(TERMS));
      window.__vgAttach((n, p) => a.emit(n, p)); window.__vgPanelToggles(a);
    },
  };
  window.VoiceGuide.create(A);
}

window.__vgDeferBoot(createGuide);
})();
