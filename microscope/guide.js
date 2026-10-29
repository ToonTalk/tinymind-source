/* ============================================================
   tm-microscope-adapter.js — Tiny Mind Microscope guide (vg-core
   v2). Short orientation in the "open lab" framing: everything is
   unlocked, ablation is the arbiter. attentionRowPainted from the
   kickoff hook list is deferred (no paint seam without app edits —
   logged in the session report).
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const kid = () => window.__vgPrefs.kid;
let api = null;

let ablCount = 0, lastFull = null, lastCaptureSig = null;
/* the real-next piece's chance before vs after the mute, taken from
   the two worker payloads at the position the guess-o-meter shows (the
   selection end). Lets the kid line say the odds and the drop in
   points rather than a bare percentage. */
function ablDelta(ablMsg){
  try {
    if (!lastFull || !ablMsg || !lastFull.nextProb || !ablMsg.nextProb) return null;
    const arr = lastFull.nextProb;
    let pos = -1, best = -1;                          // the headline piece = highest full-model confidence
    for (let i = 0; i < arr.length; i++) if (arr[i] != null && arr[i] > best) { best = arr[i]; pos = i; }
    if (pos < 0 || ablMsg.nextProb[pos] == null) return null;
    return { before: arr[pos], after: ablMsg.nextProb[pos] };
  } catch (_) { return null; }
}
function wire(emit){
  window.__vgW.taps.push((dir, m) => {
    if (dir !== 'in' || !m || !m.type) return;
    if (m.type === 'run-done')      {
      const sig = (m.n || 0) + ':' + ((m.pieces && m.pieces.join('|')) || (m.nextProb && m.nextProb.length) || '');
      lastFull = m;
      if (sig !== lastCaptureSig) { lastCaptureSig = sig; emit('captureDone', { n: m.n }); }   // one confirmation per distinct capture
    }
    /* the whole bench refreshes on every ablation, so this arrives in
       bursts; say the lesson once and vary the acknowledgements after */
    if (m.type === 'run-ablated') {
      const d = ablDelta(m);
      emit(++ablCount === 1 ? 'ablationVerdict' : 'ablationAgain', { ablate: m.ablate, n: ablCount, d });
    }
    if (m.type === 'neuron-acts-done') emit('neuronSelected', { layer: m.layer, neuron: m.neuron });
    if (m.type === 'neurons-done')    emit('neuronsRanked', {});
    if (m.type === 'loaded')          setTimeout(maybePrefillStory, 80);
    if (m.type === 'progress')        setTimeout(maybePrefillStory, 0);
  });
}

/* no key box on this bench: fall back to whatever key the ghostwriter
   or the detective remembered */
const micCfg = () => (window.__vgRememberedKey && window.__vgRememberedKey()) || { key: '' };
let storyDefault = null;
/* wave 4: when a user-trained mind is on the bench, seed the story
   box with one of its own stories (saved by the nursery guide) --
   only if the visitor has not already typed their own. */
async function maybePrefillStory(){
  try {
    if (!window.storage || typeof window.storage.list !== 'function') return;
    const info = ($('modelInfo') && $('modelInfo').textContent) || '';
    if (!info || /zog/i.test(info)) return;          // wait for a model, and leave the Zog demo alone
    const el = $('story');
    const untouched = !el ? false : (el.value === storyDefault || el.value === window.__vgStoryDemo || !el.value.trim());
    if (!el || !untouched) return;
    const keys = ((await window.storage.list('tm-')) || {}).keys || [];
    let text = null, who = null, source = null;
    if (keys.includes('tm-sample-latest')) {         // best: a story this mind wrote itself
      const o = JSON.parse((await window.storage.get('tm-sample-latest')).value);
      if (o && o.text) { text = o.text; who = o.name; source = 'sample'; }
    }
    if (!text) {                                     // next best: a story it grew up on
      const bk = keys.find(k => k.indexOf('tm-book-') === 0);
      if (bk) {
        const o = JSON.parse((await window.storage.get(bk)).value);
        const first = (o.book || []).find(x => x && x.text);
        if (first) { text = first.text; who = o.sheet && o.sheet.name; source = 'book'; }
      }
    }
    if (!text) return;
    el.value = text; storyDefault = text;
    window.vgEmit('storyPrefilled', { name: who, source });
  } catch (_) {}
}

/* the switch-off in the visitor's own units: the odds before, the odds
   now, and the gap in points -- concrete where a bare percentage is not */
function deltaLine(p, kid){
  const d = p && p.d; if (!d || d.before == null || d.after == null) return null;
  const O = window.__vgOdds, k = kid && O;
  const b = k ? O.phrase(d.before, true, { compact: true }) : (100 * d.before).toFixed(0) + '%';
  const a = k ? O.phrase(d.after, true, { compact: true }) : (100 * d.after).toFixed(0) + '%';
  const pts = Math.round(Math.abs(d.before - d.after) * 100);
  if (pts < 1) return kid
    ? `Barely moved \u2014 the guess stayed about ${b}. So the answer does not really live in that piece. Try muting a whole team!`
    : `Almost no change \u2014 still about ${a}. That piece is not where the fact lives; climb to a whole layer team.`;
  const dir = d.after < d.before ? 'down' : 'up';
  return kid
    ? `The odds were ${b}, now ${a} \u2014 ${dir} ${pts} point${pts === 1 ? '' : 's'}. The bigger the drop, the more the mind needed that piece.`
    : `The real-next odds went ${b} \u2192 ${a}, ${dir} ${pts} point${pts === 1 ? '' : 's'}. The size of that drop is how much the answer leaned on what you muted.`;
}

/* rotating follow-ups: a guide that repeats itself verbatim stops
   being listened to */
const AGAIN = [
  'Blue is the whole mind, pink is the mind with that piece muted. The gap between them is what the piece was worth.',
  'Barely a dent? Then the fact does not live there alone. Mute a whole layer team and watch the cliff.',
  'Compare that gap with the last one. The bigger the gap, the more the answer depended on what you silenced.',
  'Every mute is one suspect eliminated or convicted. Both are progress.',
];
const AGAIN_KID = [
  'Blue is the whole mind. Pink is the mind with that piece switched off. How big is the gap?',
  'Hardly changed? Then the treasure is not in that piece by itself. Try switching off a whole team!',
  'Bigger gap means that piece mattered more. Smaller gap means it barely helped.',
  'Every switch-off tells you something, even when nothing happens.',
];

/* on-screen terms the guide can explain on click (or when asked). */
const MIC_TERMS = {
  dim: { label: 'dimension', note: 'the width of the mind: how many numbers it uses to describe each piece of text as it thinks.', kidNote: 'how many numbers the mind uses to think about each little piece of a word.' },
  layers: { label: 'layers', note: 'how many times in a row the mind reconsiders the text, each layer refining the last.', kidNote: 'how many times the mind thinks about the story over and over.' },
  heads: { label: 'attention heads', note: 'parallel lookers per layer; each head can watch a different earlier word while reading the current one.', kidNote: 'little lookers that each watch a different earlier word.' },
  neurons: { label: 'neurons per layer', note: 'the size of each layer’s scratchpad: how many little units do the thinking inside one layer.', kidNote: 'how many tiny thinking bits are inside each layer.' },
  vocab: { label: 'vocabulary', note: 'how many different pieces this mind can read and write — its whole alphabet.', kidNote: 'all the different little pieces of words this mind knows.' },
  guessometer: { label: 'guess-o-meter', note: 'how sure the mind is about each possible next piece after the selected one; bar length is the chance.', kidNote: 'how sure the mind is about what comes next. Longer bar means more sure.' },
  logitlens: { label: 'layer by layer', note: 'a peek at the guess forming layer by layer — what it would say if it stopped at each layer.', kidNote: 'a peek at the mind’s guess as it thinks, step by step.' },
  ahead: { label: 'thinking ahead', note: 'the gold column in layer by layer: words a middle layer is already leaning toward further on in the story, not just next. A map fitted to this mind (a “J-lens”) reads them out; only the Big and Giant Storytellers have one.', kidNote: 'the gold words show what the mind is already thinking about for later in the story, not just the next word.' },
};

const EVENTS = {
  storyPrefilled: { priority: 'moment', once: true, spot: '#story',
    lines: p => (p && p.source === 'sample')
      ? `I put a story this very mind wrote in the box \u2014 looking inside a mind while it reads its own words is the sharpest specimen there is.`
      : `I put one of the stories ${p && p.name ? p.name : 'your character'} grew up on in the box \u2014 words it has actually seen make the clearest patterns.`,
    kidLines: p => (p && p.source === 'sample')
      ? 'I put a story this mind wrote itself in the box. Let us look inside while it reads its own words!'
      : 'I put one of the stories it grew up on in the box \u2014 those are the best ones to look inside with!' },
  captureDone: { priority: 'moment', cooldownMs: 20000,
    lines: 'Captured — every attention pattern, every neuron, for every word of that story. All of it is yours to poke at.',
    kidLines: 'Got it! The microscope took a picture of every thought the mind had about that story.' },
  ablationVerdict: { priority: 'verdict', once: true, cooldownMs: 8000,
    lines: p => deltaLine(p, false) || 'There is your before-and-after. Remember which one counts: the pretty patterns suggested, but the switch-off decided.',
    kidLines: p => deltaLine(p, true) || 'See the before and after? Switching pieces off is the real test!' },
  ablationAgain: { priority: 'ambient', cooldownMs: 15000,
    lines: p => deltaLine(p, false) || AGAIN[(p.n - 2) % AGAIN.length],
    kidLines: p => deltaLine(p, true) || AGAIN_KID[(p.n - 2) % AGAIN_KID.length] },
  neuronSelected: { priority: 'ambient', cooldownMs: 20000,
    lines: 'That strip shows where this one neuron fires across the whole story. The card below shows its two sides: where it switched on most in real stories, and where it went most strongly the other way.',
    kidLines: 'That stripe shows when this one tiny piece wakes up in the story. The card below shows its two sides: when it switches on, and when it goes the other way.' },
};

function STEPS(){
  return [
  { name: 'openLab', text: () => kid()
      ? 'Welcome to the microscope! In here nothing is secret — you can look at every single piece of a mind while it thinks.'
      : 'Welcome to the open lab. Unlike the minds you meet out in the world — including me — this one is fully unlocked: every attention head, every neuron, visible mid-thought. Nothing in here is off limits.' },
  { name: 'capture', spot: '#runBtn', allow: '#runBtn,#story',
    until: 'captureDone',
    text: () => kid()
      ? 'Type a little story in the box, then press run. The microscope will catch every thought!'
      : 'Give it a sentence or a story and press run. The microscope teacher-forces the text through and captures everything: attention, activations, and its guess for every next word.' },
  { name: 'attention', spot: '#heat', text: () => kid()
      ? 'This heat map shows which words the mind looks at while reading each word. Click around and explore!'
      : 'The attention view: for each word, which earlier words the mind consulted. Pick layers and heads — some heads track the previous word, some hunt for names. These patterns are evidence, but only the correlational kind.' },
  { name: 'neurons', spot: '#neuronBtn', text: () => kid()
      ? 'These are the busiest neurons for the part you picked. And here is the important rule: to know if one really matters, switch it off and see what breaks.'
      : 'Neuron contrast ranks which hidden units light up over a span you select. And here is the lab rule that keeps everyone honest: contrast suggests, switching off decides. In the switch-off lab, switch a suspect off and watch whether the prediction actually breaks.' },
  { name: 'explore', text: () => kid()
      ? 'The lab is yours now! Try the layer lens too — it shows what the mind would guess if it stopped thinking early.'
      : aheadLayers().length
        ? 'The lab is yours. Try layer by layer on a piece — the gold thinking ahead column shows words this mind is already leaning toward further on in the story, not only next. Ask me anything as you go.'
        : 'The lab is yours. The layer by layer view is worth a look — what would it predict if it stopped at layer two? — and everything you learn here is ammunition for the treasure hunt next door. Ask me anything as you go.' },
  ];
}

// the layers this mind's thinking-ahead lens reads (from the page's note), [] if it has none
function aheadLayers(){
  const t = ($('aheadNote') && $('aheadNote').textContent) || '';
  const m = t.match(/layers ([\d, ]+)\)/);
  return m ? m[1].split(',').map(s => s.trim()).filter(Boolean) : [];
}
function aheadRows(){
  const out = $('lensOut');
  if (!out) return '';
  return [...out.querySelectorAll('tr')].map(r => [...r.children]).filter(c => c.length > 4 && c[0].tagName === 'TD' && c[3].textContent.trim())
    .map(c => `layer ${c[0].textContent.trim()}: ${c[3].textContent.trim()}`).join('; ');
}

function statusText(){
  const vis = id => { const el = $(id);
    return !!(el && !window.__vgHidden(el) && ((el.textContent || '').trim() || el.children.length)); };
  return `Story box: ${($('story') && $('story').value.slice(0, 200)) || 'empty'}. ` +
    `Model info: ${($('modelInfo') && $('modelInfo').textContent) || 'unknown'}. ` +
    `Status: ${($('status') && $('status').textContent) || ''}. ` +
    `Ablation info: ${($('ablInfo') && $('ablInfo').textContent) || 'none'}. ` +
    `Visible views right now \u2014 attention heatmap: ${vis('heat') ? 'OPEN' : 'not shown'}, ` +
    `layer by layer view: ${vis('lensOut') ? 'OPEN' : 'not shown'}, ` +
    `thinking ahead: ${aheadLayers().length ? `this mind HAS it (layers ${aheadLayers().join(', ')}; the gold column of layer by layer)${aheadRows() ? ' showing ' + aheadRows() : ''}` : 'this mind does NOT have it (only the Big and Giant Storytellers do)'}, ` +
    `neuron list: ${vis('neuronList') ? 'OPEN' : 'not shown'}, ` +
    `prediction bars: ${vis('bars') ? 'OPEN' : 'not shown'}. ` +
    `Neuron card (what does this neuron do? both sides from real stories, plus turn it up / mute it and see the story go on): ` +
    `${($('neuronCard') && $('neuronCard').textContent.replace(/\s+/g, ' ').trim().slice(0, 600)) || 'closed'}.`;
}
function systemPrompt(){
  return 'You are the guide beside the Tiny Mind Microscope, an interpretability lab where a visitor inspects a small language model: attention heatmaps, neuron activations, the "layer by layer" view (what scientists call a logit lens), "thinking ahead" (the gold column in layer by layer, what scientists call a J-lens: words a middle layer is already leaning toward further on in the story, not only the next word, read out by a map fitted to that mind; only the Big and Giant Storytellers have one), and the "switch-off lab" (ablation: switching heads or neurons off). Always call the views by the names on the page: "layer by layer", "thinking ahead" and "switch-off lab"; mention the scientific names only if asked. When the visitor asks about predicting further than the next word, that is exactly what thinking ahead shows: if the status says this mind has it, point them at layer by layer (pick a piece, press the button, read the gold column); if not, tell them to choose the Big or Giant Storyteller from the mind menu. When the visitor switches parts off, the switch-off lab shows how the story goes on with the normal mind and with the switched-off one. ' +
    'You are honestly an A I yourself — a much bigger mind whose own insides are not this visible — and may say so. ' +
    'Core pedagogy: attention and contrast are correlational evidence; ablation is the arbiter. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is twelve or older here; technical vocabulary is fine once they use it. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: story, run, attention, neurons, ablation, lens, bars. ' +
    ' Never offer to open or point at a view the status marks as OPEN already \u2014 talk about what it shows instead. ' +
    ' A view marked "not shown" cannot be shown yet: say which button makes it first, rather than promising to display it. ' +
    ' For a young visitor, never name a view without explaining it in a few words: the prediction bars are the guess bars, how sure the mind is about each next word; the layer by layer view is an early peek at what it would say if it stopped thinking halfway; thinking ahead is the gold words it is already planning for later in the story. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the attention view', escalations: 'run captures or ablations', safeVerb: 'SHOW' });
}
/* every view that only exists after a capture names the control that
   makes it, so SHOW can point there instead of at an empty box */
const NEEDS_RUN = { needs: '#runBtn',
  why: 'Nothing has been captured yet, so there is nothing to look at. Press look inside on your story first — I am pointing at the button.',
  kidWhy: 'The microscope has not looked yet! Press the look inside button I am pointing at, and then it will be there.' };
const SHOW = {
  story: '#story', run: '#runBtn',
  attention: Object.assign({ sel: '#heat' }, NEEDS_RUN),
  bars: Object.assign({ sel: '#bars' }, NEEDS_RUN),
  neurons: { sel: '#neuronList', needs: '#neuronBtn',
    why: 'The neuron list is empty until you rank them. Press the neuron button I am pointing at.',
    kidWhy: 'No neurons found yet! Press the button I am pointing at to go looking for them.' },
  lens: { sel: '#lensOut', needs: '#lensBtn',
    why: 'The layer lens has not been run for this story yet. Press the lens button I am pointing at.',
    kidWhy: 'The lens has not peeked yet! Press the button I am pointing at.' },
  ablation: '#ablGrid' };

function createGuide(){
  const A = {
    id: 'tm-microscope',
    ui: { mascot: '\u{1F52C}', greeting: () => kid()
        ? 'Welcome to the microscope! Press start tour and we will look inside a mind.'
        : 'Welcome to the open lab. Press start tour, or ask me anything.',
      voiceEnabled: () => window.__vgPrefs.voice, doneStatus: 'Orientation done — the lab is yours.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 1100,
      reflowCSS: { mirrored: '.grid{grid-template-columns:1fr}' } },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ story: ($('story') && $('story').value) || '', abl: ($('ablInfo') && $('ablInfo').textContent) || '' }),
      diffRemark: o => { if (!o) return '';
        const abl = ($('ablInfo') && $('ablInfo').textContent) || '';
        if (o.abl !== abl) return 'You ran your own ablation while I was quiet — that is exactly the spirit of the lab.';
        return ''; } },
    prompts: { system: systemPrompt, status: statusText },
    llm: { hasKey: () => !!(window.claude && typeof window.claude.complete === 'function') || !!micCfg().key,
      keyedCall: (sys, usr) => window.__vgAsk(sys, usr, micCfg),
      noAI: { say: 'I cannot reach a live A I from here right now, but every instrument in the lab runs on your machine.' } },
    verbs: { SHOW: { exec(p){ const s = SHOW[(p[0] || '').toLowerCase()]; if (s) window.__vgShow(api, s); } } },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    strings: { llmFail: e => { const m = e && e.message ? e.message : String(e);
      if (/failed to fetch|network|cors|no key/i.test(m))
        return { log: 'Live AI call failed: ' + m,
          say: 'The microscope has no key box of its own. Paste a key into the Nursery ghostwriter (tick "remember this key") and I can talk live here too. Every instrument on the bench works without one.' };
      return null; } },
    testHook: { app: () => ({ prefs: window.__vgPrefs }) },
    init(a){ api = a; a.on('showBlocked', window.__vgShowBlockedEvent);
      storyDefault = ($('story') && $('story').value) || null;
      window.__vgStoryDemo = storyDefault;
      document.addEventListener('click', a.explainable(MIC_TERMS));
      window.__vgMicRefresh = () => { try { if (typeof renderBars === 'function' && S.run) renderBars();
        if (typeof renderLens === 'function' && S.lastLens) renderLens(S.lastLens); } catch (e) {} };
      { const kc = document.getElementById('vgKid'); if (kc) kc.addEventListener('change', () => setTimeout(window.__vgMicRefresh, 0)); }
      window.addEventListener('message', e => { const mm = e.data; if (mm && mm.tm === 'pref' && mm.key === 'tm-guide-kid') setTimeout(window.__vgMicRefresh, 0); });
      wire((n, p) => a.emit(n, p)); window.__vgAttach((n, p) => a.emit(n, p)); window.__vgPanelToggles(a);
      const mi = $('modelInfo');
      if (mi && mi.nodeType === 1) new MutationObserver(() => maybePrefillStory()).observe(mi, { childList: true, subtree: true, characterData: true });
      setTimeout(maybePrefillStory, 300);
    },
  };
  window.VoiceGuide.create(A);
}

window.__vgDeferBoot(createGuide);
})();
