
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

function wire(emit){
  window.__vgW.taps.push((dir, m) => {
    if (dir !== 'in' || !m || !m.type) return;
    if (m.type === 'run-done')        emit('captureDone', { n: m.n });
    if (m.type === 'run-ablated')     emit('ablationVerdict', { ablate: m.ablate });
    if (m.type === 'neuron-acts-done') emit('neuronSelected', { layer: m.layer, neuron: m.neuron });
    if (m.type === 'neurons-done')    emit('neuronsRanked', {});
  });
}

const EVENTS = {
  captureDone: { priority: 'moment', cooldownMs: 20000,
    lines: 'Captured — every attention pattern, every neuron, for every word of that story. All of it is yours to poke at.',
    kidLines: 'Got it! The microscope took a picture of every thought the mind had about that story.' },
  ablationVerdict: { priority: 'verdict', cooldownMs: 8000,
    lines: 'There is your before-and-after. Remember which one counts: the pretty patterns suggested, but the switch-off decided.',
    kidLines: 'See the before and after? Switching pieces off is the real test!' },
  neuronSelected: { priority: 'ambient', cooldownMs: 20000,
    lines: 'That strip shows where this one neuron fires across the whole story. Bright spans are its favorite moments.',
    kidLines: 'That stripe shows when this one tiny piece wakes up in the story.' },
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
      : 'Neuron contrast ranks which hidden units light up over a span you select. And here is the lab rule that keeps everyone honest: contrast suggests, ablation decides. Switch a suspect off and watch whether the prediction actually breaks.' },
  { name: 'explore', text: () => kid()
      ? 'The lab is yours now! Try the layer lens too — it shows what the mind would guess if it stopped thinking early.'
      : 'The lab is yours. The logit lens is worth a look — what would it predict if it stopped at layer two? — and everything you learn here is ammunition for the treasure hunt next door. Ask me anything as you go.' },
  ];
}

function statusText(){
  return `Story box: ${($('story') && $('story').value.slice(0, 200)) || 'empty'}. ` +
    `Model info: ${($('modelInfo') && $('modelInfo').textContent) || 'unknown'}. ` +
    `Status: ${($('status') && $('status').textContent) || ''}. ` +
    `Ablation info: ${($('ablInfo') && $('ablInfo').textContent) || 'none'}.`;
}
function systemPrompt(){
  return 'You are the guide beside the Tiny Mind Microscope, an interpretability lab where a visitor inspects a small language model: attention heatmaps, neuron activations, logit lens, and head or neuron ablation. ' +
    'You are honestly an A I yourself — a much bigger mind whose own insides are not this visible — and may say so. ' +
    'Core pedagogy: attention and contrast are correlational evidence; ablation is the arbiter. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is twelve or older here; technical vocabulary is fine once they use it. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: story, run, attention, neurons, ablation, lens, bars. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the attention view', escalations: 'run captures or ablations', safeVerb: 'SHOW' });
}
const SHOW = { story: '#story', run: '#runBtn', attention: '#heat', neurons: '#neuronList',
  ablation: '#ablGrid', lens: '#lensOut', bars: '#bars' };

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
    llm: { hasKey: () => !!(window.claude && typeof window.claude.complete === 'function'),
      keyedCall: (sys, usr) => window.claude.complete(sys + '\n\n' + usr),
      noAI: { say: 'I cannot reach a live A I from here right now, but every instrument in the lab runs on your machine.' } },
    verbs: { SHOW: { exec(p){ const sel = SHOW[(p[0] || '').toLowerCase()]; if (sel) api.spot(sel); } } },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    testHook: { app: () => ({ prefs: window.__vgPrefs }) },
    init(a){ api = a; wire((n, p) => a.emit(n, p)); window.__vgAttach((n, p) => a.emit(n, p)); window.__vgPanelToggles(a); },
  };
  window.VoiceGuide.create(A);
}

window.__vgDeferBoot(createGuide);
})();

