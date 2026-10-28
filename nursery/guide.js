/* ============================================================
   tm-nursery-adapter.js — Tiny Mind Nursery guide (vg-core v2).
   Persona: the guide is honestly what it is — a mind too, a much
   bigger one — which carries the honesty ritual and choir frame.
   Registers: default lines readable at 10+; kidLines below that.
   Surprise-meter rule: never say "loss" in kid lines.
   ============================================================ */
(function(){
'use strict';
const $ = id => document.getElementById(id);
const F = window.VoiceGuide.fragments;
const kid = () => window.__vgPrefs.kid;
const sheet = () => window.__tmBookMeta && window.__tmBookMeta.mode === 'binding'
  ? { name: window.__tmBookMeta.label, attr1: 'four balanced characters', attr2: 'matched facts' }
  : {
      name:  ($('chName')  && $('chName').value.trim())  || 'your character',
      attr1: ($('chAttr1') && $('chAttr1').value.trim()) || '',
      attr2: ($('chAttr2') && $('chAttr2').value.trim()) || '',
    };
let api = null;
let base0 = null;                      // first base-eval loss; forgetting = drift > +0.15 nats
let lastData = null;                   // last set-data payload: the storybook the worker trains on

/* ---------- worker tap -> guide events (no app edits) ---------- */
function wireWorker(emit){
  window.__vgW.taps.push((dir, m) => {
    if (!m || !m.type) return;
    if (dir === 'out') {
      if (m.type === 'set-data') lastData = { stories: m.stories || [], heldout: m.heldout || [] };
      if (m.type === 'train') { if (m.opts && m.opts.fresh) base0 = null; persistBook(); emit('trainStarted', { fresh: !!(m.opts && m.opts.fresh) }); }
      return;
    }
    switch (m.type) {
      case 'step':   emit('stepTick', { step: m.step, loss: m.loss }); break;
      case 'eval':
        if (m.base != null) {
          if (base0 == null) base0 = m.base;
          else if (m.base - base0 > 0.15) emit('forgettingVisible', { drift: m.base - base0 });
        }
        if (m.arc === 'absorbed-ish') emit('floorReached', { zogHeld: m.zogHeld });
        if (m.arc === 'parroting')    emit('parrotingBadge', { zogHeld: m.zogHeld, min: m.zogHeldMin });
        break;
      case 'sample':          saveSample(m.text); emit('sampleReady', { step: m.step, text: m.text }); break;
      case 'idle':            emit('trainDone',   { step: m.step }); break;
      case 'export-done':     emit('exportDone',  { step: m.step }); break;
      case 'probe-fact-done': emit('probeFact', m); break;
    }
  });
  /* ghostwrite verdicts land in #genOut as "N accepted, M rejected" */
  const out = $('genOut');
  if (out && out.nodeType === 1) new MutationObserver(() => {
    const t = out.textContent || '';
    const m = /(\d+)\s+accepted,\s+(\d+)\s+rejected/.exec(t);
    if (!m) { if (/^(storybook loaded|loaded the|balanced four-character story world loaded)/.test(t.trim())) emit('storyAccepted', { ok: 0, bad: 0, loaded: true }); return; }
    const ok = +m[1], bad = +m[2];
    emit(bad > 0 ? 'storyRejected' : 'storyAccepted', { ok, bad });
  }).observe(out, { childList: true, subtree: true });
  window.addEventListener('scrubMoved', e => {
    const d = e.detail || {};
    if (d.step === 0) emit('beforeCharacterTraining', d);
    else if (d.profile === 'held-loss-minimum') emit('heldLossBeforeBinding', d);
    else if (d.profile === 'binding-demonstration') emit('bindingDemoPoint', d);
    else if (d.profile === 'maximum-binding') emit('maximumBindingPoint', d);
  });
}

/* ---------- character persistence (wave 4): the storybook the
   worker trains on is captured from the outgoing set-data message;
   on train start it is stored per character, and saved characters
   are offered beside the ready-made Zog book on later visits.
   Framed only (window.storage via the bridge); standalone quietly
   skips. ---------- */
function saveSample(text){
  try { if (window.storage && text)
    window.storage.set('tm-sample-latest', JSON.stringify({ name: sheet().name, text, ts: Date.now() })).catch(() => {});
  } catch (_) {}
}
function persistBook(){
  try {
    if (!window.storage || !lastData || !lastData.stories || !lastData.stories.length) return;
    const s = sheet();
    if (!s.name || s.name === 'your character') return;
    const book = [...lastData.stories.map(text => ({ split: 'train', text })),
                  ...(lastData.heldout || []).map(text => ({ split: 'heldout', text }))];
    const meta = window.__tmBookMeta;
    const storybook = meta && meta.mode === 'binding'
      ? { mode: 'binding', purpose: meta.purpose, characters: meta.characters,
          train: lastData.stories, heldout: lastData.heldout || [] }
      : null;
    window.storage.set('tm-book-' + s.name, JSON.stringify({
      sheet: s, kind: ($('chKind') && $('chKind').value) || '', book, storybook,
      label: storybook ? 'load the four-character story world' : null, ts: Date.now() })).catch(() => {});
  } catch (_) {}
}
async function offerSavedBooks(){
  try {
    if (!window.storage || typeof window.storage.list !== 'function') return;
    const r = await window.storage.list('tm-book-');
    const keys = (r && r.keys) || [];
    const demo = $('demoBtn');
    if (!keys.length || !demo) return;
    for (const k of keys.slice(0, 6)) {
      let o; try { o = JSON.parse((await window.storage.get(k)).value); } catch (_) { continue; }
      const b = document.createElement('button');
      b.className = 'alt small'; b.style.marginLeft = '6px';
      b.textContent = o.label || ('load your saved ' + ((o.sheet && o.sheet.name) || k.slice(8)) + ' storybook');
      b.onclick = () => { try {
        if (o.sheet) { if ($('chName')) $('chName').value = o.sheet.name || '';
          if ($('chKind')) $('chKind').value = o.kind || '';
          if ($('chAttr1')) $('chAttr1').value = o.sheet.attr1 || '';
          if ($('chAttr2')) $('chAttr2').value = o.sheet.attr2 || ''; }
        if (window.__tmRouteJson) window.__tmRouteJson(((o.sheet && o.sheet.name) || 'saved') + '-storybook.json', JSON.stringify(o.storybook || o.book));
      } catch (_) {} };
      demo.parentNode.insertBefore(b, demo.nextSibling);
    }
  } catch (_) {}
}

/* ---------- huntability pre-flight (uses the probe-fact worker
   message the build adds — existing forward machinery only) ---------- */
let pfSeq = 0;
function probeFact(word, context){
  const w = window.__vgW.workers[0];
  if (!w) return Promise.resolve(null);
  const id = 'vgpf' + (++pfSeq);
  w.postMessage({ type: 'probe-fact', word, context, id });
  return Promise.race([
    api.waitFor('probeFact', { condition: p => p && p.id === id }),
    new Promise(res => setTimeout(() => res(null), 6000)),
  ]);
}
const odds = p => window.__vgOdds ? window.__vgOdds.phrase(p, kid()) : String(p);
async function preflight(g){
  const s = sheet();
  const checks = [];
  let worst = null;
  for (const attr of [s.attr1, s.attr2]) {
    if (!attr) continue;
    const words = attr.split(/\s+/);
    if (words.length < 2) continue;
    const answer = words[words.length - 1].replace(/[^\w']/g, '');
    const context = s.name + ' ' + words.slice(0, -1).join(' ');
    const r = await probeFact(answer, context);
    if (r) checks.push({ answer, r });
  }
  if (!checks.length) return;
  const bits = [];
  for (const { answer, r } of checks) {
    if (r.pieces > 1) { const shown = (r.pieceTexts || []).map(p => p.trim() || '(space)').join(' · ');
      bits.push(kid()
      ? `The mind reads "${answer}" as ${r.pieces} little pieces${shown ? ': ' + shown : ''}. Words that are one piece, like play, happy or friend, are easier to find later.`
      : `This mind reads "${answer}" as ${r.pieces} pieces${shown ? ' (' + shown + ')' : ''}, because its alphabet only has 512 pieces. A word in several pieces is harder to find in the treasure hunt: the fact is spread over several guesses. Few words are a single piece here; play, happy, friend and mom are.`); }
    else if (r.prob != null && r.prob > 0.10) { bits.push(kid()
      ? `Careful — before it ever met ${s.name}, the mind already guesses "${answer}" ${odds(r.prob)}. Pick a stranger fact to bury.`
      : `The base mind already gives "${answer}" ${odds(r.prob)} before any training — that pairing is a collocation it knew anyway. Pick a stranger fact to bury.`);
      worst = (worst == null || r.prob > worst) ? r.prob : worst; }
  }
  const text = bits.length
    ? bits.join(' ')
    : (kid() ? 'Good news: these facts look strange enough to bury and hunt later.'
             : 'Pre-flight looks good: one-piece answers the base mind does not expect. Those bury well and hunt well.');
  try { await api.speak(text, g); } catch (e) { if (e && e.vg === 'abort') throw e; }
  if (worst != null && window.__vgOdds) window.__vgOdds.chip(worst, 'how often the base mind already says that');
}

/* ---------- commentary track: the training window (~4.3 min) ----------
   Timed beats interleave with floor/parroting/forgetting event beats
   through the same attention manager; auto-cancels when the tour
   generation dies (pause, end). */
function trainingBeats(){
  const n = sheet().name;
  return [
    { atMs: 8000,   lines: `Two lines to watch. The top one is how surprised it is by held-out ${n} stories it never trains on — the secret ones. The bottom is ordinary stories.`,
      kidLines: `Watch the two lines! The top one is the secret ${n} stories. The bottom is ordinary stories.` },
    { atMs: 40000,  lines: 'Falling surprise means learning. It is not memorizing your words yet — it is getting the shape of them.',
      kidLines: 'The line going down means it is learning!' },
    { atMs: 90000,  lines: `Every reading pass, it adjusts a quarter of a million little dials. Nobody sets them by hand — the surprise itself pushes them.`,
      kidLines: 'It has a quarter of a million tiny dials inside, and every story nudges them a little.' },
    { atMs: 150000, lines: 'Soon the secret-story line will stop falling. That floor is the moment worth waiting for.',
      kidLines: 'Keep watching the top line. Soon it will stop going down — that is the special moment.' },
    { atMs: 220000, lines: `Notice it needs the same fifty stories again and again. Hold that thought — I will ask you about it later.`,
      kidLines: 'It needs to read the same stories again and again and again. Remember that!' },
  ];
}

/* ---------- emergent events ---------- */

/* on-screen terms the guide can explain on click (or when asked). */
const NUR_TERMS = {
  dim: { label: 'dimension', note: 'the width of the mind: how many numbers it uses to describe each piece of text as it thinks.', kidNote: 'how many numbers the mind uses to think about each little piece of a word.' },
  layers: { label: 'layers', note: 'how many times in a row the mind reconsiders the text, each layer refining the last.', kidNote: 'how many times the mind thinks about the story over and over.' },
  vocab: { label: 'vocabulary', note: 'how many different pieces (letters and letter-chunks) this mind can read and write — its whole alphabet.', kidNote: 'all the different little pieces of words this mind knows.' },
};

const EVENTS = {
  beforeCharacterTraining: { priority: 'moment', cooldownMs: 4000, spot: '#scrubLearning',
    lines: 'This is the pretrained stories260K starting model, before character training. About 25 percent is chance with four answers, so this checkpoint is still guessing the new character facts.',
    kidLines: 'This mind already knows how simple stories sound, but it has not learned Zog, Mip, Bex or Nia. One right answer out of four is still just guessing.' },
  heldLossBeforeBinding: { priority: 'moment', cooldownMs: 4000, spot: '#scrub',
    lines: 'This is the crucial result: held-out story surprise is lowest here, but character-answer accuracy is still near chance. The mind learned the style of this story world before it learned which facts belong to which names.',
    kidLines: 'The story surprise is low here, but the mind still mixes up who owns each fact. Getting used to the stories is not the same as matching facts to names.' },
  bindingDemoPoint: { priority: 'moment', cooldownMs: 4000, spot: '#scrub',
    lines: 'This is the recommended investigation point: the names select their own facts reliably, although ordinary storytelling has started to weaken.',
    kidLines: 'Here the names finally point to the right facts. But the mind is starting to forget some ordinary stories.' },
  maximumBindingPoint: { priority: 'moment', cooldownMs: 4000, spot: '#scrub',
    lines: 'This endpoint has the strongest measured binding, but it is more forgetful and rigid. New contradictory context still fails to override its learned defaults.',
    kidLines: 'This mind remembers the fact matches strongly, but it is more stuck in its ways and tells ordinary stories less well.' },
  floorReached: { priority: 'moment', once: true, spot: '#chart',
    lines: p => `There — it just learned ${sheet().name} as well as it ever will. Keep going past this floor and it starts parroting its storybook instead of understanding it.`,
    kidLines: p => `Look! It knows ${sheet().name} now as well as it ever will. If we keep going, it just starts copying.` },
  parrotingBadge: { priority: 'moment', once: true, spot: '#chart',
    lines: 'And there is the parroting: the secret-story line is rising again. It is memorizing the storybook word for word now, which makes it worse at stories it has never seen.',
    kidLines: 'Now it is parroting — copying the storybook instead of learning. See the line going back up?' },
  forgettingVisible: { priority: 'moment', once: true, spot: '#chart',
    lines: p => `Watch the ordinary-stories line: it has drifted up by ${p && p.drift ? p.drift.toFixed(2) : 'a chunk'}. While it crams your stories, it is starting to forget everything else it knew.`,
    kidLines: 'Uh oh — it is starting to forget its old stories while it learns your new ones.' },
  stepTick: { priority: 'ambient', cooldownMs: 60000,
    lines: p => `Surprise is at ${p && p.loss != null ? p.loss.toFixed(2) : 'a new low'} and falling.`,
    kidLines: 'It is getting less surprised, step by step.' },
  storyRejected: { priority: 'moment', cooldownMs: 8000, spot: '#genOut',
    lines: p => `${p.bad} of those stories used words outside the mind's tiny alphabet, so they bounced. The ghostwriter will try again with simpler words.`,
    kidLines: 'Some stories used words too big for this tiny mind, so they bounced off. It will try simpler ones.' },
  storyAccepted: { priority: 'ambient', cooldownMs: 15000,
    lines: p => p.loaded ? 'A whole storybook is in.' : `${p.ok} more accepted into the storybook.`,
    kidLines: 'More stories in the book!' },
  exportDone: { priority: 'moment', once: true,
    lines: 'Saved. That file is the whole mind — every dial, frozen. Drop it back on this page any time and it wakes up exactly here.',
    kidLines: 'Saved! That file is the whole mind. Bring it back any time.' },
};

/* ---------- tour ---------- */
const hasBuiltIn = () => { const p = $('provider'); return !!p && [...p.options].some(o => o.value === 'keyless'); };
function STEPS(){
  return [
  { name: 'welcome', text: () => kid()
      ? `Hi! I am a mind too — a much bigger one — and this room is where you raise a tiny one. I live in this panel; the arrow hides me whenever you want.`
      : `Welcome to the Nursery. I should be honest about what I am: a mind too, a much bigger one, here to help you raise a very small one. I live in this panel beside the app; press pause or just start clicking any time — you are in charge.` },
  { name: 'character', spot: '#chName', text: () => kid()
      ? 'First, invent a creature nobody has ever heard of. Give it a name and two special things about it.'
      : 'First, invent a character the world has never seen — name, what it is, and two special facts. Those facts are what we will bury in the mind, and later hunt down inside it.' },
  { name: 'preflight', spot: '#chAttr1',
    text: () => kid() ? 'Let me check if your facts will bury well.' : 'One moment — let me pre-flight your facts for huntability before you commit to them.',
    action: g => preflight(g), skipIf: () => !window.__vgW.workers.length },
  { name: 'ghostwrite', spot: '#provider', allow: '#genBtn,#provider,#apiFields,#manualFields,#pasteBox,#pasteBtn,#bindingDemoBtn,#demoBtn,#apiKey,#apiModel,#rememberKey,#strictWords',
    until: { event: 'storyAccepted' },
    text: () => kid()
      ? 'Now we need stories! In "who writes?", pick a helper to write them' + (hasBuiltIn() ? ' — built-in Claude needs no key' : ' (a grown-up pastes a key)') + ', then press write 8 stories. Or write some yourself and press check and add. Or load a ready-made storybook. I will wait.'
      : 'Now the storybook. Under "who writes?" choose who drafts the stories: ' + (hasBuiltIn() ? 'built-in Claude needs no key, or ' : '') + 'an AI with your own key, then press ✍️ write 8 stories, a few times. Or choose "me", paste stories you wrote, and press check + add. Or load one of the ready-made storybooks. Every sixth story is kept secret to test the mind with later. I will wait right here.' },
  { name: 'readAloud', spot: '#storyList', text: () => kid()
      ? 'Look at the storybook! Pick one and read it out loud. Every single word will pass through the tiny mind hundreds of times.'
      : 'Here is the storybook. Read one aloud — really. Every word in it will pass through the trainee hundreds of times; it is worth hearing once with your own voice what the mind will hear a hundred times.' },
  { name: 'twoLines', spot: '#chart', text: () => kid()
      ? 'This chart will show two lines: the secret stories, and ordinary stories. Both start high because the mind is very surprised by everything.'
      : `This chart will draw two lines: how surprised the mind is by the secret ${sheet().name} stories, and by ordinary stories. Surprise is the whole game — training is just the art of becoming less surprised.` },
  { name: 'startRaising', spot: '#trainBtn', allow: '#trainBtn,#lr,#maxSteps,#replayPct,#startFrom',
    until: 'trainStarted',
    text: () => kid() ? 'Ready? Press the train button and raising begins!' : 'Ready. Press train and the raising begins — I will narrate the interesting moments as they happen.' },
  { name: 'trainingWindow', spot: '#chart', during: true, until: 'floorReached',
    text: () => kid() ? 'Here we go. Watch with me.' : 'Here we go. Watch with me — the next few minutes are the whole story of how minds like this learn.',
    action: g => { api.track(trainingBeats(), { gen: g }); return Promise.resolve(); } },
  { name: 'surpriseMeter', spot: '#chart', text: () => kid()
      ? 'That number is the surprise meter. Small surprise means it can guess what comes next in a story about your creature.'
      : 'What you just watched was a surprise meter bottoming out. Low surprise means it can now guess the next word of a story about your creature — which is all "knowing" means, for a mind like this.' },
  { name: 'payoff', spot: '#sampleBtn', allow: '#sampleBtn', until: 'sampleReady',
    text: () => kid()
      ? `The best part: press the sample button and ask it for a brand new ${sheet().name} story!`
      : `Now the payoff. Press sample and ask it for a brand-new ${sheet().name} story — its own words, from those quarter-million dials you watched move.` },
  { name: 'honesty', text: () => kid()
      ? `One question before we go on. This mind needed fifty stories and hundreds of readings to learn ${sheet().name}. You learned ${sheet().name} from one. Who is the better learner — and why do you think that is?`
      : `Before we move on, an exhibit rather than a lecture. This mind needed fifty stories and a hundred readings to learn ${sheet().name}. You learned ${sheet().name} from one. Minds like this — like me — do not learn the way you do. So: who is the better learner, and what does that tell you about us?` },
  { name: 'handoff', text: () => kid()
      ? `Now — want to find WHERE it keeps ${sheet().name}? Look at the tabs along the top and open the Treasure Hunt. See you there!`
      : `Your creature now lives somewhere inside those dials. Want to find WHERE it keeps ${sheet().name}? The Treasure Hunt tab along the top of the window is exactly that expedition. I will meet you there.` },
  ];
}

/* ---------- Q&A grounding ---------- */
function timeMachineStatus(){
  const live = $('scrubLive');
  if (!live || live.style.display === 'none')
    return 'Time machine state: no recording is loaded. ';
  const arc = window.__tmGuideArcContext || null;
  const picker = $('scrubArcPick');
  const recording = picker && picker.selectedOptions && picker.selectedOptions[0]
    ? picker.selectedOptions[0].textContent : (($('scrubLabel') && $('scrubLabel').textContent) || 'unknown recording');
  const selected = ($('scrubInfo') && $('scrubInfo').textContent) || 'unknown checkpoint';
  const step = ($('scrubStepInfo') && $('scrubStepInfo').textContent) || '';
  const load = ($('scrubLoadState') && $('scrubLoadState').textContent) || '';
  const previous = $('scrubPrev') && $('scrubPrev').disabled ? 'disabled at this end' : 'available';
  const next = $('scrubNext') && $('scrubNext').disabled ? 'disabled at this end' : 'available';
  const stops = [...document.querySelectorAll('#scrubStops button')]
    .map(b => b.getAttribute('aria-label') || b.textContent.trim()).join(', ');
  const manifest = arc
    ? `Authoritative time-machine manifest: arc id ${arc.arc}; note: ${arc.note || 'none'}; ` +
      `${arc.snapshotCount} recorded moments from step ${arc.firstStep} through step ${arc.lastStep}; ` +
      `character names in its recorded sample prompts: ${arc.characterNames.length ? arc.characterNames.join(', ') : 'none'}. ` +
      `Recorded sample prompts: ${arc.samplePrompts.length ? arc.samplePrompts.join(' | ') : 'none'}. `
    : 'Authoritative time-machine manifest details are unavailable. ';
  return `Time machine context — authoritative and independent of the character editor and saved models. ${manifest}` +
    `Visible recording label: ${recording}. Selected: ${selected}; ${step}. ` +
    `Previous is ${previous}; Next is ${next}. Exact checkpoint buttons shown: ${stops || 'none'}. ` +
    `Loading message: ${load || 'none'}. `;
}
function statusText(){
  const s = sheet();
  const st = $('status') ? $('status').textContent : '';
  const model = $('modelInfo') ? $('modelInfo').textContent : 'unknown';
  const samples = $('samples') ? $('samples').textContent.slice(-400) : '';
  return timeMachineStatus() +
    `Separate Nursery editor context — this is NOT evidence about who or what the selected time-machine recording trained on. ` +
    `Character editor currently says: ${s.name}. Fact one: ${s.attr1}. Fact two: ${s.attr2}. ` +
    `Active storybook mode: ${window.__tmBookMeta && window.__tmBookMeta.mode === 'binding' ? 'balanced Zog, Mip, Bex and Nia corpus' : 'single character'}. ` +
    `Separate loaded-model display: ${model}. ` +
    `App status line: ${st}. Latest sample text (may be empty): ${samples}. ` +
    `Storybook count shown: ${($('bookCount') && $('bookCount').textContent) || 'unknown'}.`;
}
function systemPrompt(){
  return 'You are the guide living beside the Tiny Mind Nursery, where a visitor raises a 260 thousand parameter language model on either a single-character storybook or a balanced four-character story world. ' +
    'You are honestly an A I yourself — a much bigger mind of the same general kind — and you may say so plainly when it helps. ' +
    (kid() ? 'The visitor is a young child: use short, warm, simple sentences, and never use the word loss — call it surprise. ' :
             'The visitor is ten or older: plain, concrete language; call the loss the surprise meter unless they use the technical word first. ') +
    F.voiceRules + ' ' + F.statusRule + ' ' + F.selfKnowledge + ' ' +
    'The exact base checkpoint stories260K.bin is pretrained on TinyStories patterns but has not learned the visitor\'s custom characters. A restored storybook is training data, not a memory in the model weights. Never claim that facts from an earlier storybook can shine through a fresh base model. If the loaded filename is a custom exported model, say that it may already contain its earlier training. Do not invent or paraphrase model-choice cards you cannot see. ' +
    'The authoritative time-machine manifest in the status is the ONLY source for saying who or what a recording trained on. Never infer an arc\'s subject from the character editor, saved storybook, loaded model, latest sample, or the fact that it ends at step 600. Arc B and Arc C both end at step 600 but are different recordings: arcB is the Zog-only recording; arcC is the four-character Zog, Mip, Bex and Nia recording. If Sisi appears only in the separate editor or model context, explicitly say that this does not make the selected arc a Sisi arc. Never describe arcC as Sisi\'s training. ' +
    'Time-machine checkpoints are independent of storybook mode and of how many characters are in the storybook. Never say that single-character or storybook mode prevents access to a time-machine moment. When the visitor asks why a moment or checkpoint cannot be reached, use the Time machine context in the status: name the selected moment and the exact Start/Moment, Previous, and Next controls that are actually available. The overview slider is optional. If a moment is already loading, a new button choice is queued rather than discarded. ' +
    F.pointNotDescribe({ pointVerbs: 'DO SHOW' }) +
    ' You can act on the app only by ending your reply with a single DO line, its own last line, exactly: DO: SHOW target. Targets: character, storybook, chart, train, sample, export, timemachine. ' +
    ' Never offer to open or point at something the status shows is already on screen \u2014 talk about what it shows instead. ' +
    F.doLineMeta + ' ' + F.consentScope({ pointExample: 'show the chart', escalations: 'start or stop training', safeVerb: 'SHOW' });
}
/* the ghostwriter's own key config, reused for guide Q&A standalone */
const VG_LIVE = ['anthropic', 'openai', 'gemini'];
const appLLM = () => { const p = ($('provider') && $('provider').value) || '';
  const key = VG_LIVE.includes(p) ? (($('apiKey') && $('apiKey').value.trim()) || '') : '';
  if (key) return { provider: p, key, model: ($('apiModel') && $('apiModel').value.trim()) || '' };
  return (window.__vgRememberedKey && window.__vgRememberedKey()) || { provider: p, key: '', model: '' };
};

const SHOW = {
  character: '#chName', chart: '#chart', train: '#trainBtn',
  sample: '#sampleBtn', export: '#exportBtn',
  storybook: { sel: '#storyList', needs: '#genBtn',
    why: 'The storybook is still empty. Ghostwrite or paste some stories first — the button I am pointing at.',
    kidWhy: 'No stories yet! Press the button I am pointing at to make some.' },
  timemachine: { sel: '#scrub', needs: '#scrubStub',
    why: 'No recorded arc is loaded, so there is no time machine here yet. Drop an arc pack into the bay first.',
    kidWhy: 'There is no time machine yet! You need to drop in a recording first.' },
};

/* ---------- create ---------- */
function createGuide(){
  const A = {
    id: 'tm-nursery',
    ui: { mascot: '\u{1F9E0}', greeting: c => kid()
        ? 'Hi! Press start tour and we will raise a tiny mind together.'
        : 'Hello — I am the guide for this room. Press start tour, or just ask me anything.',
      voiceEnabled: () => window.__vgPrefs.voice, doneStatus: 'Tour done — the mind is yours.' },
    layout: { panelWidth: 300, panelMargin: 302, narrowMax: 700, appBreakpoint: 1100,
      reflowCSS: { mirrored: '.grid2{grid-template-columns:1fr}' } },
    takeover: { events: ['pointerdown'],
      ignore: e => { const G = window.__guide; if (!G || G.state !== 'running') return false;
        const st = STEPS()[G.cur]; return !!(st && st.allow && e.target && e.target.closest && e.target.closest(st.allow)); } },
    tour: { steps: STEPS() },
    provenance: {
      snapshot: () => ({ s: JSON.stringify(sheet()), book: ($('bookCount') && $('bookCount').textContent) || '' }),
      diffRemark: o => { if (!o) return '';
        const now = { s: JSON.stringify(sheet()), book: ($('bookCount') && $('bookCount').textContent) || '' };
        if (o.s !== now.s) return 'I see you changed the character sheet — good, it is yours to invent.';
        if (o.book !== now.book) return 'The storybook grew while I was quiet. Carrying on.';
        return ''; } },
    prompts: { system: systemPrompt, status: statusText },
    llm: { hasKey: () => !!(window.claude && typeof window.claude.complete === 'function') || !!appLLM().key,
      keyedCall: (sys, usr) => window.__vgAsk(sys, usr, appLLM),
      noAI: { spot: () => document.getElementById('provider'), say: 'I cannot reach a live A I from here on my own. If you have a key, add it in the ghostwriter panel I am pointing at \u2014 pick a provider and paste the key \u2014 and I can talk live too. Everything else works without one.' } },
    verbs: { SHOW: { exec(p){ const s = SHOW[(p[0] || '').toLowerCase()]; if (s) window.__vgShow(api, s); } } },
    events: EVENTS,
    lineMode: () => kid() ? 'kid' : 'default',
    strings: { llmFail: e => { const m = e && e.message ? e.message : String(e);
      if (/failed to fetch|network|cors/i.test(m) && !appLLM().key)
        return { log: 'Live AI call failed: ' + m,
          say: 'I could not reach a live A I on my own. If you have a key, pick a provider and paste it in the ghostwriter panel up top \u2014 then ask me again. Everything else works without one.' };
      return null; } },
    testHook: { app: () => ({ sheet, base0: () => base0, prefs: window.__vgPrefs }) },
    init(a){ api = a; a.on('showBlocked', window.__vgShowBlockedEvent);
      document.addEventListener('click', a.explainable(NUR_TERMS));   // click any [data-vg-explain] chip in the nursery
      wireWorker((n, p) => a.emit(n, p)); window.__vgAttach((n, p) => a.emit(n, p)); window.__vgPanelToggles(a); offerSavedBooks(); },
  };
  window.VoiceGuide.create(A);
}

window.__vgDeferBoot(createGuide);
})();
