/* ============================================================
   tm-eye-adapter.js — Tiny Mind Tiny Eye guide (vg-core v2).
   The room is a mind that sees drawings: the visitor draws, the
   eye guesses, and its neurons light up; a neuron's card shows
   both ends (the sample doodles that make it fire most and
   least), and switching neurons off is the experiment. The guide
   keeps to the page's names and to Phase 0's honest findings:
   most neurons (not all) have one main job you can name.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const kid = () => window.__vgPrefs.kid;
let api = null;

const cfg = () => (window.__vgRememberedKey && window.__vgRememberedKey()) || { key: '' };

const TERMS = {
  sees: { label: 'what the eye sees', note: 'your drawing shrunk to 28 by 28 dots, centered and with soft lines, made exactly the way the eye\'s practice doodles were made from people\'s drawings; this little picture is all the eye gets.', kidNote: 'your drawing made tiny, 28 dots across, the way the eye saw all its practice drawings.' },
  neuron: { label: 'neuron', note: 'one tiny part of the eye that adds up what it sees into one number; gold squares are neurons firing more than usual for your drawing, blue ones less.', kidNote: 'one tiny part of the eye. It lights up gold when your drawing has the thing it looks for, and blue when it really does not.' },
  ends: { label: 'both ends of a neuron', note: 'the 16 sample doodles that make a neuron fire most, and the 16 that make it fire least. A neuron\'s job often shows best when you look at both ends.', kidNote: 'the drawings that make a neuron light up the most, and the ones that make it the quietest.' },
  grow: { label: 'watch it grow', note: 'the eye at 20 moments of its training; its neurons try jobs, settle late (half way through training most of each neuron\'s final top doodles are not in place yet), and sometimes switch; a few grow offshoots.', kidNote: 'go back in time to when the eye was still learning, and see how its neurons found their jobs.' },
  squeeze: { label: 'squeeze it', note: 'the same eye trained with only 8 neurons, and 8 channels, in its last stage; each neuron has to cover more kinds of things, so its class bars spread wider, though its top doodles still look tidy.', kidNote: 'a version of the eye with only 8 neurons at the end, so each one has to do more jobs.' },
  sure: { label: 'never 100% sure', note: 'it was taught never to be completely sure (label smoothing): each right answer counted as about 90% right in training, so its top guess stops near 97% and every other kind keeps a sliver, under 1% each.', kidNote: 'its teachers told it never to be totally sure, so it always keeps a tiny bit of doubt for the other kinds.' },
  bars: { label: 'what kinds of things make it fire', note: 'for each of the 24 kinds of thing, how much this neuron fires on average for those sample doodles, compared with all of them: right and gold = more, left and blue = less.', kidNote: 'which kinds of drawings make this neuron light up, and which make it quiet.' },
};

const EVENTS = {
  guessed: { priority: 'moment', cooldownMs: 30000,
    lines: p => p && p.sample
      ? 'That is a real Quick, Draw! doodle, one the eye never practiced on. The bars show how sure it is about each kind of thing it knows.'
      : 'There is its guess. The longest bar is what it thinks your drawing is; it only knows 24 kinds of things, so it always picks one of those. Box two shows which neurons lit up.',
    kidLines: p => p && p.sample ? 'That is a drawing by someone else! See what the eye thinks it is.' : 'The eye made a guess! The longest bar is what it thinks you drew.' },
  switchResult: { priority: 'verdict', cooldownMs: 8000,
    lines: p => p && p.changed
      ? 'The guess changed. Those switched-off neurons were helping it see your drawing: switching parts off is how you test what they really do.'
      : 'Same guess as before: the neurons you switched off were not needed for this drawing, or others did their job too.',
    kidLines: p => p && p.changed ? 'The guess changed! Those neurons were helping.' : 'It still guesses the same thing. Other neurons helped instead.' },
  cardOpened: { priority: 'moment', cooldownMs: 20000,
    lines: p => p && p.waiting ? 'The eye is still looking through its 5,000 sample doodles; the card fills in by itself in a few seconds.'
      : 'Its card shows both ends: the sample doodles that make it fire most, and the ones that make it fire least. ' + (p && p.named ? 'Under each end is a name a bigger A I gave it, and whether that name passed a test.' : 'This one has no name yet: what would you call it?'),
    kidLines: p => p && p.waiting ? 'One moment, the eye is still looking at its drawings.' : 'These drawings make the neuron light up the most, and these make it the quietest. Can you see what it looks for?' },
  scrubbed: { priority: 'moment', cooldownMs: 25000,
    lines: p => p && p.end ? 'Back at the end of training: the eye you were drawing with.'
      : `This is the eye at step ${p ? p.step.toLocaleString() : ''}: right about ${p ? Math.round(100 * p.right) : ''} times in 100, with its neurons' jobs about ${p ? Math.round(100 * p.settled) : ''}% settled. The guesses, the neurons and the cards all show this moment.`,
    kidLines: p => p && p.end ? 'Back to the grown-up eye!' : 'We went back in time! This is the eye while it was still learning.' },
  storyOpened: { priority: 'moment', cooldownMs: 10000,
    lines: 'Look at the strip under its class bars: its top doodles at different moments of training. Click a moment, or scrub back, to watch its job change.',
    kidLines: 'See the row of little drawings? That is this neuron growing up. Click one to go back in time.' },
  squeezed: { priority: 'moment', cooldownMs: 8000,
    lines: p => p && p.on && p.breadthSq ? `The squeezed eye has only 8 neurons at the end, so each covers more: they answer strongly to ${p.breadthSq.toFixed(1)} kinds of things each, against ${(p.breadthFull || 0).toFixed(1)} for the full eye. Look at a card's class bars.`
      : p && p.on ? 'This is the squeezed eye: only 8 neurons in its last stage. It is looking through the sample doodles; then its cards are ready.'
      : 'Back to the full eye, with its 256 last-stage neurons.',
    kidLines: p => p && p.on ? 'Now the eye has only 8 neurons at the end, so each one has to do more jobs!' : 'Back to the big eye!' },
  swept: { priority: 'ambient', once: true,
    lines: 'The eye has looked through all 5,000 sample doodles, so every neuron\'s card is ready. Tap any square in box two.',
    kidLines: 'Ready! Tap any square in box two to meet a neuron.' },
};

function STEPS(){
  return [
  { name: 'welcome', text: () => kid()
      ? 'Welcome to the Tiny Eye! It is a tiny mind that looks at drawings. Draw something and it will guess what it is.'
      : 'Welcome to the Tiny Eye: a tiny mind that sees drawings. It learned 24 kinds of things from people\'s doodles. You draw, it guesses, and then you look at the neurons that decided.' },
  { name: 'draw', spot: '#pad', allow: '#pad,#clearBtn,#undoBtn,#randBtn', until: 'guessed',
    text: () => kid()
      ? 'Draw a cat, a house or a dragon here with your finger or the mouse.'
      : 'Draw something it knows here: a cat, a house, a dragon. It guesses as you draw. Or press a doodle for one drawn by someone else.' },
  { name: 'grid', spot: '#grid', allow: '#grid,#busy,#stages', until: 'cardOpened',
    text: () => kid()
      ? 'Each square is one neuron. Gold ones light up for your drawing. Tap one!'
      : 'Each square is one neuron of its last stage: gold when it fires more than usual for your drawing, blue when less. Tap one to see its job.' },
  { name: 'card', spot: '#card', text: () => kid()
      ? 'These are the drawings that make this neuron light up most, and the ones that make it quiet. What do they have in common?'
      : 'Its card: the 16 sample doodles that make it fire most, and the 16 that make it fire least. Most neurons have one main job you can name, and both ends help you find it.' },
  { name: 'switch', spot: '#offCard', text: () => kid()
      ? 'Now switch some neurons off and see if the eye changes its guess!'
      : 'Box four is the experiment: switch neurons off, from a card or the five busiest at once, and watch whether the guess changes.' },
  { name: 'grow', spot: '#growCard', text: () => kid()
      ? 'Box five is a time machine: go back to when the eye was still learning.'
      : 'Box five goes back in its training: drag the slider and the guesses, neurons and cards follow. Try a growth story: neurons try jobs, settle late, and sometimes switch.' },
  { name: 'squeeze', spot: '#squeezeCard', text: () => kid()
      ? 'Box six squeezes the eye down to 8 neurons at the end. Then each one has to do more jobs!'
      : 'Box six squeezes it: the same eye with only 8 last-stage neurons. Each one has to cover more kinds of things; the class bars show it.' },
  ];
}

function statusText(){
  const txt = id => (($(id) && $(id).textContent) || '').replace(/\s+/g, ' ').trim();
  const bars = $('guesses') ? [...$('guesses').querySelectorAll('.gbar')].map(r => r.textContent.replace(/\s+/g, ' ').trim()).join('; ') : '';
  const stageBtn = document.querySelector('#stages button.on');
  const card = $('cardBody');
  const cardText = card && !card.classList.contains('hint') ? card.textContent.replace(/your name for it:/, '').replace(/\s+/g, ' ').trim().slice(0, 900) : txt('cardBody');
  const drawn = !!(document.body.dataset.guess);
  const snap = document.body.dataset.snap, squeezed = document.body.dataset.eye === 'squeezed';
  return `Drawing: ${drawn ? 'yes' : 'NOTHING drawn yet (the pad is empty)'}. Guess bars (box one): ${bars || 'none yet'}. ${txt('guessNote')} ` +
    `Neuron grid (box two) shows ${stageBtn ? stageBtn.textContent : 'the last stage'}. ${txt('busy')}. ` +
    `Sample doodles looked through: ${document.body.dataset.swept === '1' ? 'yes, every card is ready' : 'not yet (cards wait a few seconds)'}. ` +
    `Neuron card (box three): ${txt('cardTitle')}: ${cardText || 'none open'}. ` +
    `Switched off (box four): ${document.body.dataset.off || '0'} neurons: ${txt('offList')}. ` +
    `Training moment shown (box five): ${squeezed ? 'none: the squeezed eye has no training recording' : snap && snap !== 'end' ? 'step ' + snap + ', NOT the end of training' : 'the end of training'}: ${txt('growInfo')}. ` +
    `Eye: ${squeezed ? 'the SQUEEZED eye (8 last-stage neurons)' : 'the full eye (256 last-stage neurons)'}; box six says: ${txt('squeezeNote')}. Status: ${txt('status')}.`;
}
function systemPrompt(){
  return 'You are the guide beside the Tiny Mind Tiny Eye, a room where a visitor draws and a tiny mind that sees drawings guesses what it is. It is a small image network (82 thousand connections, three stages, a ConvNeXt-like design) trained on 240,000 Quick, Draw! doodles of 24 kinds of things: dragon, teapot, cloud, moon, castle, cat, dog, fish, bird, rabbit, frog, owl, sheep, tree, flower, house, sun, star, apple, rainbow, snowman, umbrella, ambulance, police car. It always picks one of those 24, even for a drawing of something else; it is right about 88 times in 100 on doodles it never saw. ' +
    'The boxes, by their names on the page: box one, "draw": the pad, "what the eye sees" (the drawing shrunk to 28 by 28 dots, centered, with soft lines, the way its practice doodles were made, so a drawing anywhere on the pad, any size, looks the same to it), "a doodle" (shows a real doodle it never trained on) and the guess bars. Box two, "neurons light up": one square per neuron, stage 1 (96 neurons, small bits), stage 2 (192), stage 3 the last (256, the whole drawing); gold = fires more than usual for this drawing, blue = less; "firing most now" chips. Box three, a neuron\'s card: always both ends, the 16 sample doodles that make it fire most and the 16 that make it fire least, out of 5,000 doodles it never trained on; "what kinds of things make it fire" bars (gold right = more than usual for that kind, blue left = less) and how many kinds it answers strongly to; where in the drawing it fires; a "switch it off" button and "your name for it". For 64 of the last stage\'s neurons a bigger A I named each end and was tested: from the name alone it had to pick 4 of that end\'s doodles out of 16; a tick means it got 3 or 4. Box four, "switch neurons off": a switched-off neuron\'s output is set to zero, and the guess bars show the new guess with dashed bars for the guess with every neuron on. ' +
    'Be honest about what was measured: MOST neurons have one main job you can name, not all; some are fuzzy or answer to shapes shared by several kinds of thing (for example pointy tops: cat ears and castle towers). Bigger is not cleaner: eyes of three sizes (23, 83 and 311 thousand connections) were tested and their neurons came out about equally clean, so never say bigger eyes have cleaner neurons. Looking at a card only suggests a job; switching a neuron off tests what it does. Only name doodles, neurons or numbers that the status shows. ' +
    'Asked what a neuron does: describe BOTH ends from its card (what makes it fire most, and what makes it fire least, with the names a bigger A I gave and their test results if shown), its class bars, and how it fires for this drawing. ' +
    'Asked why it guessed something: start with the fact that it only knows 24 kinds of things and always has to pick one of them; the guess bars show how sure it is about each; it sees only the 28 by 28 dot picture; many neurons light up and the last stage votes; to find out which neurons mattered, switch off the busiest ones (box four) and watch whether the guess changes. ' +
    'Box five, "watch it grow": a slider over 20 moments of training (the last is the eye at the end of training); the guess bars, the grid (last stage only while going back) and the cards follow it; a card then shows "how it grew", its top 4 doodles at several moments. What Phase 0 measured: the eye is right 85 times in 100 by step 3,359, but its neurons\' jobs settle late (at that point only about 40 to 47% of each neuron\'s final top 100 doodles are in place); neurons try jobs, settle late, and sometimes switch; a few grow offshoots. Growth stories: neuron 117, a roof that grows an umbrella offshoot; neuron 2, castle walls then fish; neuron 16, vehicles then arches and rainbows then houses. ' +
    'Box six, "squeeze it": the same eye trained with only 8 neurons and 8 channels in its last stage, no wide bypass. Squeeze it, and each neuron has to cover more kinds of things: the class bars spread and "answers strongly to N kinds" goes up (about 4 against 3; box six shows the exact numbers once both eyes have looked through the sample), while its top doodles still look tidy and it is still right about 84 times in 100. Say coarser or broader, never muddled or confused. Example: squeezed neuron 7 fires for rainbows, umbrellas and castles (an arch on top). ' +
    'Asked why it says "under 1%", or why it is never 100% sure: it was taught never to be completely sure (a training trick called label smoothing: every right answer counted as about 90% right, the rest shared by the other kinds), so its top guess stops near 97% and every other kind keeps a sliver; the row "the other N kinds together" adds those slivers up. For a child: its teachers told it never to be totally sure. ' +
    'Asked what switching a neuron off does: answer in general first (that tiny part of the eye goes quiet, its output becomes zero; the guess may change or stay, which shows whether it was helping), then point at the switch: the button on a card, or box four. ' +
    'You are honestly an A I yourself, a much bigger mind of a related kind, and may say so. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is twelve or older here; technical vocabulary is fine once they use it. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: pad, guesses, doodle, grid, card, switch, grow, squeeze. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the neuron grid', escalations: 'draw, open cards or switch neurons off', safeVerb: 'SHOW' });
}
const NEEDS_DRAWING = { needs: '#pad',
  why: 'Nothing is drawn yet. Draw something on the pad first; I am pointing at it.',
  kidWhy: 'Draw something first! I am pointing at the drawing pad.' };
const SHOW = {
  pad: '#pad', doodle: '#randBtn', grid: '#grid', switch: '#offCard', grow: '#growCard', squeeze: '#squeezeCard',
  guesses: Object.assign({ sel: '#guesses .gbar' }, NEEDS_DRAWING),
  card: { sel: '#cardBody .poles', needs: '#grid', why: 'No neuron card is open yet. Tap any square in the neuron grid; I am pointing at it.',
    kidWhy: 'Tap a square in the grid first! I am pointing at it.' } };

function createGuide(){
  const A = {
    id: 'tm-eye',
    ui: { mascot: '\u{1F441}', greeting: () => kid()
        ? 'Welcome to the Tiny Eye! Press start tour, or just start drawing.'
        : 'Welcome to the Tiny Eye. Press start tour, or ask me anything.',
      voiceEnabled: () => window.__vgPrefs.voice, doneStatus: 'Tour done: keep drawing, and meet more neurons.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 1000,
      reflowCSS: { mirrored: '.grid{grid-template-columns:1fr}' } },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ guess: document.body.dataset.guess || '' }),
      diffRemark: o => (o && o.guess !== (document.body.dataset.guess || '') && document.body.dataset.guess ? 'You drew something new while I was quiet: good, the eye is yours to test.' : '') },
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
          say: 'I cannot reach a live A I on my own. Press the key button at the top of this panel, choose Claude, ChatGPT or Gemini and paste your key, and I can talk live here too. Everything here works without one.' };
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
