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
      const ok = (m.results || []).filter(r => r.contProb != null);
      if (m.baseline != null && ok.length) {
        const best = Math.max(...ok.map(r => r.contProb));
        huntable = best < m.baseline * 0.5;
      }
      emit('nameControlVerdict', { huntable, baseline: m.baseline, nameInProbe: m.nameInProbe });
    }
  });
  const hypos = $('hypos');
  if (hypos) new MutationObserver(muts => {
    for (const mu of muts) for (const n of mu.addedNodes) {
      if (!(n instanceof Element)) continue;
      const v = n.classList && n.classList.contains('verdict') ? n : n.querySelector && n.querySelector('.verdict');
      if (v) {
        const cls = v.querySelector('.proved') || v.classList.contains('proved') || /PROVED/.test(v.textContent) ? 'proved'
                  : 'busted';
        lastVerdict = { cls };
        emit(cls === 'proved' ? 'clueProved' : 'clueBusted', { cls });
        emit('clueTested', { cls });
      } else if (n.classList && n.classList.contains('hypo')) {
        emit('clueProposed', {});
      }
    }
  }).observe(hypos, { childList: true, subtree: true });
  const score = $('score');
  if (score) new MutationObserver(() => emit('scoreUpdated', { text: score.textContent }))
    .observe(score, { childList: true, subtree: true });
}

/* ---------- emergent events ---------- */
const odds = p => window.__vgOdds ? window.__vgOdds.phrase(p, kid()) : String(p);
const EVENTS = {
  probeDone: { priority: 'moment', cooldownMs: 5000,
    lines: p => `The guess-o-meter reads ${odds(p && p.contProb)} \u2014 that is the mind's chance of finishing the sentence your way. Anything it barely expected before training is treasure worth hunting.`,
    kidLines: p => `The mind's guess is ${odds(p && p.contProb)}. Tap the dots button to see what that looks like!` },
  clueBusted: { priority: 'verdict', cooldownMs: 6000, condition: () => choirSpoken,
    lines: 'Busted — and that is the choir singing. You muted one voice and the song survived. One more suspect eliminated; the treasure is not in that single piece.',
    kidLines: 'Busted! But remember the choir: one singer stopped and the song kept going. That is a clue too!' },
  clueProved: { priority: 'verdict', cooldownMs: 6000,
    lines: 'PROVED. Switching that off broke the answer — you found a piece the fact really lives in. That is real evidence, not a pretty picture.',
    kidLines: 'You found it! Switching that off broke the answer. The treasure really is in there!' },
  nameControlVerdict: { priority: 'verdict', once: false, cooldownMs: 6000,
    lines: p => p && p.huntable === false
      ? `We swapped the name and the answer survived — so this fact is not really about ${charName()}. The mind knew that pairing before it ever met your character. Pick a stranger fact to hunt.`
      : `We swapped the name and the answer collapsed. This fact genuinely lives with ${charName()} — a real buried treasure. Hunt on.`,
    kidLines: p => p && p.huntable === false
      ? 'We changed the name and the mind still knew the answer! So this fact is not really about your character.'
      : 'We changed the name and the mind got confused — so this fact really belongs to your character!' },
};

/* ---------- tour ---------- */
function STEPS(){
  return [
  { name: 'quest', text: () => kid()
      ? `Welcome to the treasure hunt! Somewhere inside the tiny mind is where it keeps ${charName()}. We are going to find it — with real tests, not guesses.`
      : `Welcome to the hunt. Somewhere in a quarter-million dials, this mind keeps what it knows about ${charName()}. Our quest: find WHERE — and prove it, with switch-off tests rather than pretty pictures.` },
  { name: 'guessOMeter', spot: '#probeBtn', allow: '#probeBtn,#qProbe,#qCont,#qName',
    until: 'probeDone',
    text: () => kid()
      ? 'First, press probe. The guess-o-meter shows how sure the mind is about your fact.'
      : 'First, press probe. The guess-o-meter is the answer chance: the headline number is the average confidence per piece, and the breakdown shows each piece honestly — the raw product is in there too.' },
  { name: 'clue', spot: '#cluesBtn', allow: '#cluesBtn,#cluesCard,#detective,#detKeyFields,#hypos',
    until: 'clueProposed',
    text: () => kid()
      ? 'Now get a clue! Ask the detective, or pick a suspect yourself. A clue says: maybe the treasure is in THIS piece.'
      : 'Now propose a clue — ask the detective or pick your own suspect head or neuron. A clue is a hypothesis: the fact lives in this particular piece of the machine.' },
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
      const proved = lastVerdict && lastVerdict.cls === 'proved';
      if (kid()) return proved
        ? 'A treasure! Want to try muting a whole team next, like the front row of the choir?'
        : 'One singer muted, song survived — just like the choir. Try a whole layer team next: mute the front row.';
      return proved
        ? 'A genuine find. Now calibrate it: single pieces rarely matter this much — try the name-swap control next to be sure the fact is really about your character.'
        : 'Exactly the choir frame at work: single heads and neurons barely dent the answer, but a whole layer team drops it off a cliff. Climb the ladder — test the team.'; } },
  { name: 'nameControl', spot: '#nameCtrlBtn', allow: '#nameCtrlBtn,#nameCtrlOut',
    until: 'nameControlVerdict',
    text: () => kid()
      ? `One more test: swap ${charName()} for a different name. If the mind still answers, the fact was never really about your character!`
      : `The last honesty check: the name control. Swap ${charName()} for strangers and re-ask. If the answer survives the swap, the fact was never about your character at all — it was a pattern the mind already had.` },
  { name: 'scoreboard', spot: '#scoreCard', text: () => kid()
      ? 'Look at your scoreboard! Every proved and busted clue is real detective work. Keep hunting — there is more treasure.'
      : 'Your scoreboard — proved, busted, and everything between. Both columns are wins: busts eliminate suspects, proofs locate treasure. The hunt is yours now; I am here for questions.' },
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
  return 'You are the guide beside the Tiny Mind Treasure Hunt, where a visitor locates where a small language model stores a fact, using attention evidence, neuron contrasts, ablation switch-off tests, and a name-swap control. ' +
    'You are honestly an A I yourself — a much bigger mind of the same kind — and may say so. ' +
    'Core pedagogy you must keep straight: correlational evidence suggests, ablation decides; a fact is stored like a song in a choir, so single-unit busts are expected and informative; the headline answer chance is length-normalized, the raw product lives in the per-piece breakdown. ' +
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
