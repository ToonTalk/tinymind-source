/* ============================================================
   tm-hunt-adapter.js — Tiny Mind Treasure Hunt guide (vg-core v2).
   The choir frame is HARD-ORDERED before the first switch-off test
   (kickoff §4): busted single-unit clues must land as confirmations.
   Kid mode and voice sync to the app's own #kidMode / #voiceOn.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const kid = () => { const k = $('kidMode'); return k ? k.checked : window.__vgPrefs.kid; };
const charName = () => ($('qName') && $('qName').value.trim()) || 'the character';
let api = null;
let lastVerdict = null;   // {cls} from the most recent tested clue
let choirSpoken = false;

/* ---------- worker tap + DOM observers -> guide events ---------- */
function wire(emit){
  window.__vgW.taps.push((dir, m) => {
    if (dir !== 'in' || !m || !m.type) return;
    if (m.type === 'probe-done') {
      emit('probeDone', { contProb: m.contProb, pieces: (m.contPieces || []).length });
      if (window.__vgOdds) window.__vgOdds.chipAfterSpeak(m.contProb, "the mind's chance of finishing your sentence");
    }
    if (m.type === 'namecontrol-done') {
      let huntable = null;
      const ok = (m.results || []).filter(r => r.geometricMeanProbability != null);
      if (m.baselineGeometricMeanProbability != null && ok.length) {
        const best = Math.max(...ok.map(r => r.geometricMeanProbability));
        huntable = best <= m.baselineGeometricMeanProbability * 0.8;
      }
      emit('nameControlVerdict', { huntable, baseline: m.baselineGeometricMeanProbability, nameInProbe: m.nameInProbe });
    }
  });
  const hypos = $('hypos');
  if (hypos && hypos.nodeType === 1) new MutationObserver(muts => {
    for (const mu of muts) for (const n of mu.addedNodes) {
      if (!(n instanceof Element)) continue;
      const v = n.classList && n.classList.contains('verdict') ? n : n.querySelector && n.querySelector('.verdict');
      if (v) {
        const cls = v.querySelector('.selective') || v.classList.contains('selective') || /strong causal contributor/i.test(v.textContent) ? 'selective'
                  : 'measured';
        lastVerdict = { cls };
        emit(cls === 'selective' ? 'clueSelective' : 'clueMeasured', { cls });
        emit('clueTested', { cls });
      } else if (n.classList && n.classList.contains('hypo')) {
        emit('clueProposed', {});
      }
    }
  }).observe(hypos, { childList: true, subtree: true });
  const score = $('score');
  if (score && score.nodeType === 1) new MutationObserver(() => emit('scoreUpdated', { text: score.textContent }))
    .observe(score, { childList: true, subtree: true });
}

/* ---------- emergent events ---------- */
const odds = p => window.__vgOdds ? window.__vgOdds.phrase(p, kid()) : String(p);
const EVENTS = {
  probeDone: { priority: 'moment', cooldownMs: 5000,
    lines: p => `The guess-o-meter reads ${odds(p && p.contProb)} \u2014 that is the mind's chance of finishing the sentence your way. Anything it barely expected before training is treasure worth hunting.`,
    kidLines: p => `The mind's guess is ${odds(p && p.contProb)}. Tap the dots button to see what that looks like!` },
  clueMeasured: { priority: 'verdict', cooldownMs: 6000, condition: () => choirSpoken,
    lines: 'That switch-off produced a measured effect. Now compare it with the unrelated controls: a component can matter generally without being selective for this fact.',
    kidLines: 'That part changed the answer. Check the ordinary control answers too — it may help lots of predictions, not only this one.' },
  clueSelective: { priority: 'verdict', cooldownMs: 6000,
    lines: 'Strong selective causal evidence. Switching that part off damaged the target much more than the unrelated controls. That shows contribution to this prediction, not a unique storage address.',
    kidLines: 'Strong selective effect! It hurt this answer much more than the ordinary control answers.' },
  nameControlVerdict: { priority: 'verdict', once: false, cooldownMs: 6000,
    lines: p => p && p.huntable === false
      ? `We swapped the name and the answer survived — so this fact is not really about ${charName()}. The mind knew that pairing before it ever met your character. Pick a stranger fact to hunt.`
      : `We swapped the name and the answer collapsed. The prediction is name-specific for ${charName()}, so it is a clean target for causal testing. Hunt on.`,
    kidLines: p => p && p.huntable === false
      ? 'We changed the name and the mind still knew the answer! So this fact is not really about your character.'
      : 'We changed the name and the mind got confused — so this fact really belongs to your character!' },
};

/* ---------- tour ---------- */
function STEPS(){
  return [
  { name: 'quest', text: () => kid()
      ? `Welcome to the treasure hunt! We will find parts that help the tiny mind use what it learned about ${charName()}, then check whether their effects are special to this answer.`
      : `Welcome to the hunt. Our question is which parts contribute when this mind predicts a fact about ${charName()}, and whether those effects are selective. We will use switch-off tests plus unrelated controls.` },
  { name: 'guessOMeter', spot: '#probeBtn', allow: '#probeBtn,#qProbe,#qCont,#qName',
    until: 'probeDone',
    text: () => kid()
      ? 'First, press probe. The guess-o-meter shows how sure the mind is about your fact.'
      : 'First, press probe. The guess-o-meter is the answer chance: the headline number is the average confidence per piece, and the breakdown shows each piece honestly — the raw product is in there too.' },
  { name: 'clue', spot: '#cluesBtn', allow: '#cluesBtn,#cluesCard,#detective,#detKeyFields,#hypos',
    until: 'clueProposed',
    text: () => kid()
      ? 'Now get a clue! Ask the detective, or pick a suspect yourself. A clue says: maybe the treasure is in THIS piece.'
      : 'Now propose a clue — ask the detective or pick your own suspect head or neuron. A clue is a hypothesis about contribution, not a claim that the fact has one storage address.' },
  { name: 'choir', prep: () => { choirSpoken = true; },
    text: () => kid()
      ? 'Before we test — a secret about minds like this. A fact in here is like a song in a choir. Mute one singer and the song keeps going. Mute the whole front row and it collapses. So do not be sad when one singer does not matter!'
      : 'Before your first test, the one idea that makes every result make sense: a fact in this mind is like a song in a choir. Mute one singer and the song survives; mute the front row and it collapses. So a busted single-unit clue is not a failure — it is the choir revealing itself.' },
  { name: 'switchOff', spot: '#diyCard', allow: '#hypos,#diyCard,#diyHeadBtn,#diyNeuronBtn,#diyTeamBtn,#diyL,#diyH,#diyN,#diyNL,#diyTL',
    until: 'clueTested',
    text: () => kid()
      ? 'Time for the real test: switch that piece OFF and see if the mind still knows the answer. Press test!'
      : 'Now the arbiter: run the switch-off test. We silence the suspect and re-ask the question. Whatever the attention pictures suggested, this is the evidence that counts.' },
  { name: 'verdict', text: () => {
      const selective = lastVerdict && lastVerdict.cls === 'selective';
      if (kid()) return selective
        ? 'A strong selective effect! Try another part and see whether the pattern repeats.'
        : 'That effect was not selective yet. Try another singer or a whole layer team.';
      return selective
        ? 'A strong selective contributor for this prediction. Keep the conclusion calibrated: it is causal evidence, not proof of a unique storage location.'
        : 'The target and controls did not separate strongly. Screen more candidates or climb to a layer team.'; } },
  { name: 'nameControl', spot: '#nameCtrlBtn', allow: '#nameCtrlBtn,#nameCtrlOut',
    until: 'nameControlVerdict',
    text: () => kid()
      ? `One more test: swap ${charName()} for a different name. If the mind still answers, the fact was never really about your character!`
      : `The last honesty check: the name control. Swap ${charName()} for strangers and re-ask. If the answer survives the swap, the fact was never about your character at all — it was a pattern the mind already had.` },
  { name: 'scoreboard', spot: '#scoreCard', text: () => kid()
      ? 'Look at your scoreboard! It counts tests and strong selective contributors. Keep comparing target and control effects.'
      : 'Your scoreboard counts tests run and strong selective contributors. Every result narrows the causal story without pretending to locate a fact at one address.' },
  ];
}

/* ---------- Q&A grounding ---------- */
function statusText(){
  return `Character name field: ${charName()}. Probe sentence: ${($('qProbe') && $('qProbe').value) || ''}. ` +
    `Expected continuation: ${($('qCont') && $('qCont').value) || ''}. ` +
    `Probe output (truncated): ${($('probeOut') && $('probeOut').textContent.slice(0, 300)) || 'none yet'}. ` +
    `Scoreboard: ${($('score') && $('score').textContent) || 'empty'}. Kid mode: ${kid()}.`;
}
function systemPrompt(){
  return 'You are the guide beside the Tiny Mind Treasure Hunt, where a visitor tests which parts contribute when a small language model predicts a learned fact, using attention evidence, neuron contrasts, ablation switch-off tests, unrelated completions, and a name-swap control. ' +
    'You are honestly an A I yourself — a much bigger mind of the same kind — and may say so. ' +
    'Core pedagogy you must keep straight: correlational evidence suggests candidates; ablation measures causal contribution; unrelated controls measure selectivity; none of these alone proves a unique storage location. A fact is used like a song in a choir, so small single-unit effects are expected. The headline answer score is length-normalized and the raw product remains visible. ' +
    'The order of play: first choose a quest (a sentence about a character for the mind to finish) and press the probe button to ask the mind; only then are there clues to get (the detective) and parts to switch off and test, and the name-swap control. If the status shows no probe result yet, the first step is always to press probe. ' +
    (kid() ? 'The visitor is a young child: short, warm, simple sentences. ' : 'The visitor is twelve or older here; numbers and ratios are fine. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: probe, clues, diy, namecontrol, scoreboard. ' +
    ' Never offer to open or point at something the status shows is already on screen \u2014 talk about what it shows instead. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the scoreboard', escalations: 'run tests or change the probe', safeVerb: 'SHOW' });
}
/* the detective's key, reused for guide Q&A standalone; provider
   sniffed from the model name since the card has no provider menu */
const detCfg = () => { const key = ($('detKey') && $('detKey').value.trim()) || '';
  if (!key) return (window.__vgRememberedKey && window.__vgRememberedKey()) || { key: '', model: '', provider: 'anthropic' };
  const model = ($('detModel') && $('detModel').value.trim()) || '';
  const provider = /^(gpt|o\d|chatgpt)/i.test(model) ? 'openai' : /gemini/i.test(model) ? 'gemini' : 'anthropic';
  return { key, model, provider }; };

const SHOW = { probe: '#probeCard', clues: '#cluesCard', diy: '#diyCard', namecontrol: '#nameCtrlBtn',
  scoreboard: { sel: '#scoreCard', needs: '#probeBtn',
    why: 'The scoreboard fills up as you test clues. Probe first, then propose and test one.',
    kidWhy: 'Nothing on the scoreboard yet! Test a clue first and it will appear.' } };

/* ---------- create ---------- */
function createGuide(){
  const A = {
    id: 'tm-hunt',
    ui: { mascot: '\u{1F50E}', greeting: () => kid()
        ? 'Welcome, treasure hunter! Press start tour and we will find where the mind keeps its secrets.'
        : 'Welcome to the hunt. Press start tour, or ask me anything about the expedition.',
      voiceEnabled: () => { const v = $('voiceOn'); return v ? v.checked : window.__vgPrefs.voice; },
      doneStatus: 'Quest walkthrough done — hunt on.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 1100,
      reflowCSS: { mirrored: '.grid{grid-template-columns:1fr}' } },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ probe: ($('qProbe') && $('qProbe').value) || '', score: ($('score') && $('score').textContent) || '' }),
      diffRemark: o => { if (!o) return '';
        const score = ($('score') && $('score').textContent) || '';
        if (o.score !== score) return 'I see the scoreboard moved while I was quiet — good hunting.';
        const probe = ($('qProbe') && $('qProbe').value) || '';
        if (o.probe !== probe) return 'A new probe sentence — a fresh trail. Carrying on.';
        return ''; } },
    prompts: { system: systemPrompt, status: statusText },
    llm: { hasKey: () => !!(window.claude && typeof window.claude.complete === 'function') || !!detCfg().key,
      keyedCall: (sys, usr) => window.__vgAsk(sys, usr, detCfg),
      noAI: { spot: () => document.getElementById('detKey'), say: 'I cannot reach a live A I from here on my own. If you have a key, paste it in the detective card I am pointing at and I can talk live too. The hunt itself runs on your machine either way.' } },
    verbs: { SHOW: { exec(p){ const s = SHOW[(p[0] || '').toLowerCase()]; if (s) window.__vgShow(api, s); } } },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    strings: { llmFail: e => { const m = e && e.message ? e.message : String(e);
      if (/failed to fetch|network|cors/i.test(m) && !detCfg().key)
        return { log: 'Live AI call failed: ' + m,
          say: 'I could not reach a live A I on my own. If you have a key, paste it in the detective card and I can talk live too. The hunt itself runs on your machine either way.' };
      return null; } },
    testHook: { app: () => ({ lastVerdict: () => lastVerdict, choirSpoken: () => choirSpoken, prefs: window.__vgPrefs }) },
    init(a){
      api = a; a.on('showBlocked', window.__vgShowBlockedEvent);
      wire((n, p) => a.emit(n, p)); window.__vgAttach((n, p) => a.emit(n, p));
      /* sync the app's own kid/voice controls with the tm- prefs */
      const k = $('kidMode'), v = $('voiceOn');
      if (k) { k.checked = k.checked || window.__vgPrefs.kid;
        k.addEventListener('change', () => window.__vgSetPref('tm-guide-kid', k.checked ? '1' : '0')); }
      if (v) { v.addEventListener('change', () => window.__vgSetPref('tm-guide-voice', v.checked ? '1' : '0')); }
      window.addEventListener('message', e => { const m = e.data; if (!m || m.tm !== 'pref') return;
        if (m.key === 'tm-guide-kid' && k) k.checked = m.value === '1';
        if (m.key === 'tm-guide-voice' && v) v.checked = m.value === '1'; });
    },
  };
  window.VoiceGuide.create(A);
}

window.__vgDeferBoot(createGuide);
})();
