/* ============================================================
   tm-thought-adapter.js — Tiny Mind Thought bubble guide (vg-core
   v2). The room is the J-Lens Explorer's: what a mind is holding
   in mind as it reads or writes, and the thought swap. The guide
   keeps to the page's names: "thinking ahead", "layer by layer",
   "what only thinking ahead sees", the glow, the swap.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const kid = () => window.__vgPrefs.kid;
let api = null;

function wire(emit){
  window.__vgW.taps.push((dir, m) => {
    if (dir !== 'in' || !m || !m.type) return;
    if ((m.type === 'analyzed' || m.type === 'generated') && !m.tag) emit('storyRead', { n: m.ids.length - 1, wrote: m.type === 'generated' });
    if (m.type === 'generated' && m.tag === 'withswap') emit('swapDone', {});
    if (m.type === 'heat' && m.ranks) emit('glowOn', {});
  });
}
const cfg = () => (window.__vgRememberedKey && window.__vgRememberedKey()) || { key: '' };

const TERMS = {
  ahead: { label: 'thinking ahead', note: 'words a middle layer is already leaning toward further on in the story, not only next. A map fitted to this mind (a “J-lens”) reads them out.', kidNote: 'the words the mind is already thinking about for later in the story.' },
  layer: { label: 'layer by layer', note: 'what the mind would say next if it stopped thinking at that layer (a “logit lens”).', kidNote: 'what the mind would say next if it stopped thinking early.' },
};

const EVENTS = {
  storyRead: { priority: 'moment', cooldownMs: 20000,
    lines: p => p && p.wrote
      ? 'It wrote that and read its own thoughts back. I picked the last piece for you: the words in box four are what it was holding in mind right there.'
      : 'Read. I picked the last piece, where it would go on: the words in box four are what it is holding in mind right there. Click any other piece to look there instead.',
    kidLines: 'Done! Box four shows what the mind is thinking about at the last word. Click any word in the story to peek there.' },
  glowOn: { priority: 'ambient', once: true,
    lines: 'The glowing pieces are where that word was on its mind. Notice it often glows before the word is ever written: that is thinking ahead.',
    kidLines: 'See the glow? That is everywhere the mind was thinking about that word, sometimes before it even wrote it!' },
  swapDone: { priority: 'verdict', once: true,
    lines: 'There are your two rewrites. They had the same start and the same rules; the only difference was the swapped thought, so wherever they part ways, the swap did that.',
    kidLines: 'Look at the two endings! The only difference was the thought you swapped, so that is what changed the story.' },
};

function STEPS(){
  return [
  { name: 'welcome', text: () => kid()
      ? 'Welcome to the thought bubble! Here you can peek at what a mind is thinking about, and even swap one of its thoughts.'
      : 'Welcome to the thought bubble. Here you watch what a mind is holding in mind as it reads or writes, word by word, and then change one of those thoughts and watch the story bend.' },
  { name: 'read', spot: '#readBtn', allow: '#readBtn,#story,#promptChips', until: 'storyRead',
    text: () => kid()
      ? 'Press read it. The mind reads the story and we catch its thoughts.'
      : 'Press read it. The mind reads the story, and every thought it has at every piece is kept, so looking around is instant.' },
  { name: 'holding', spot: '#readout', text: () => kid()
      ? 'These are the words the mind is thinking about. The top row is its deepest layer. Click a word to make the story glow where it was thinking about it.'
      : 'Box four: thinking ahead, one row per layer, deepest on top. These are words it is leaning toward further on, not only next. Switch to layer by layer to compare, or to what only thinking ahead sees for the difference. Click a word to make the story glow wherever it was on its mind.' },
  { name: 'swap', spot: '#swapBtn', text: () => kid()
      ? 'Now the fun part: take out one thought and put in another. The mind writes on twice, once with your swap and once without.'
      : 'Box five is the experiment: pick a piece, take out one word it is thinking about and put in another, and press swap. It rewrites from that piece twice, with and without the swap, and nothing else differs. Try hole for castle in the beach story.' },
  ];
}

function statusText(){
  const vis = id => { const el = $(id); return !!(el && (el.textContent || '').trim() && !/^(choose|read or write)/i.test(el.textContent.trim())); };
  const mode = ($('mJ') && $('mJ').classList.contains('on')) ? 'thinking ahead' : ($('mL') && $('mL').classList.contains('on')) ? 'layer by layer' : 'what only thinking ahead sees';
  const rows = $('readout') ? [...$('readout').querySelectorAll('.lrow')].map(r => r.textContent.replace(/#\d+/g, ' ').replace(/\s+/g, ' ').trim()).join('; ') : '';
  return `Mind: ${($('modelInfo') && $('modelInfo').textContent) || 'none'}. ${($('mindNote') && $('mindNote').textContent) || ''} ` +
    `Story box: ${($('story') && $('story').value.slice(0, 200)) || 'empty'}. Status: ${($('status') && $('status').textContent) || ''}. ` +
    `Chosen piece: ${($('roPos') && $('roPos').textContent) || 'none'}. ` +
    `What it's holding in mind (box four), showing ${mode}${$('raw') && $('raw').checked ? ', every piece' : ''}: ${vis('readout') && rows ? rows : 'nothing yet'}. ` +
    `Its next word: ${($('modelRow') && $('modelRow').textContent.replace(/(Thinking ahead|Layer by layer|Gold words)[\s\S]*$/, '').replace(/#\d+/g, ' ').trim()) || 'none'}. ` +
    `Glow: ${($('heatCap') && $('heatCap').textContent) || 'off'}. ` +
    `Swap (box five): take out "${($('swapOut') && $('swapOut').value) || ''}", put in "${($('swapIn') && $('swapIn').value) || ''}", from ${($('swapFrom') && $('swapFrom').textContent) || ''}, ` +
    `strength ${($('swapStr') && $('swapStr').value) || ''}. Rewrites: no swap: ${(($('swapNo') && $('swapNo').textContent) || '').slice(0, 300)} | with swap: ${(($('swapWith') && $('swapWith').textContent) || '').slice(0, 300)}.`;
}
function systemPrompt(){
  return 'You are the guide beside the Tiny Mind Thought bubble, a room where a visitor watches what a small story-writing language model is holding in mind as it reads or writes, word by word, and changes one of those thoughts. ' +
    'The views, by their names on the page: box four, "what it\'s holding in mind", shows one row per layer, deepest on top, in three modes: "thinking ahead" (what scientists call a J-lens: words a middle layer is already leaning toward further on in the story, not only next, read out by a map fitted to this mind), "layer by layer" (a logit lens: what it would say next if it stopped thinking at that layer) and "what only thinking ahead sees" (gold chips: thinking-ahead words that layer by layer does not show). "every piece (the honest view)" also shows word fragments; normally only whole story words are shown, and a pink chip is the start of a name. Clicking a word makes the story "glow" wherever that word was on its mind. ' +
    'Box five, "swap what it\'s thinking": from a chosen piece on, at the chosen layers, the thought of the "take out" word is moved onto the "put in" word with a strength (small minds oversteer, so keep it low); the mind rewrites from that piece twice, with and without the swap, choosing its most likely word each time unless "use the surprise setting too" is ticked, so the only difference is the swap. Hole for castle at the last piece of the beach story makes the Giant Storyteller write moat. ' +
    'This room only works with the Big and Giant Storytellers (they have a thinking-ahead map); the Giant shows thoughts most clearly, at layers four to eight. Mention scientific names only if asked. ' +
    'You are honestly an A I yourself, a much bigger mind whose insides are not this visible, and may say so. Pedagogy: the readout is a window, the swap is the experiment; a thought that changes the story when swapped really mattered. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is twelve or older here; technical vocabulary is fine once they use it. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: story, read, write, strip, holding, swap. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the swap box', escalations: 'read stories, write or run swaps', safeVerb: 'SHOW' });
}
const NEEDS_READ = { needs: '#readBtn',
  why: 'Nothing has been read yet. Press read it first; I am pointing at it.',
  kidWhy: 'The mind has not read anything yet! Press the read it button I am pointing at.' };
const SHOW = {
  story: '#story', read: '#readBtn', write: '#writeBtn',
  strip: Object.assign({ sel: '#strip' }, NEEDS_READ),
  holding: Object.assign({ sel: '#readout' }, NEEDS_READ),
  swap: '#swapBtn' };

function createGuide(){
  const A = {
    id: 'tm-thought',
    ui: { mascot: '\u{1F4AD}', greeting: () => kid()
        ? 'Welcome to the thought bubble! Press start tour and we will peek at what a mind is thinking.'
        : 'Welcome to the thought bubble. Press start tour, or ask me anything.',
      voiceEnabled: () => window.__vgPrefs.voice, doneStatus: 'Tour done: the thoughts are yours to swap.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 1100,
      reflowCSS: { mirrored: '.grid{grid-template-columns:1fr}' } },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ story: ($('story') && $('story').value) || '' }),
      diffRemark: o => (o && o.story !== (($('story') && $('story').value) || '') ? 'You changed the story while I was quiet: good, it is yours.' : '') },
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
      wire((n, p) => a.emit(n, p)); window.__vgAttach((n, p) => a.emit(n, p)); window.__vgPanelToggles(a);
    },
  };
  window.VoiceGuide.create(A);
}

window.__vgDeferBoot(createGuide);
})();
