
/* ============================================================
   vg-core.js — app-agnostic voice-guide core.
   Extracted from the First Day at the Game Studio layer (which
   was itself ported from the In-Between voice-demo architecture):
   docked side panel, transcript log, one generation counter,
   trusted-event takeover, provenance-correct resume, mic-by-
   policy, confirm-wrapper, two-tier DO gating, keyless-artifact
   LLM chain. Defines exactly ONE global, window.VoiceGuide, and
   has no side effects until create(adapter) is called.
   Authoring-time reuse: this file is spliced into a single-file
   app by the build; it is not a runtime dependency.
   ============================================================ */
(function(){
'use strict';
const VG_ABORT={vg:'abort'};
const VG_DEFAULT_KEYLESS=['claude-sonnet-5','claude-sonnet-4-6'];

/* Canonical prompt-rule text. Authoring aids for adapter system
   prompts — compose with these instead of copying, so the load-
   bearing policy sentences have one home. Parameterized fragments
   take the app's own nouns; plain ones are app-free. */
const fragments={
  voiceRules:'Your reply is read aloud by speech synthesis, so write plain speakable prose only: no markdown of any kind, no lists, no headings, no code fences, no URLs, no emoji, and write numbers and symbols the way you would say them. Prefer one to three short sentences.',
  statusRule:"The visitor's message ends with a STATUS RIGHT NOW block. Your advice must match it exactly: if earlier conversation turns disagree with the status, the status wins, and never tell the visitor to do a step the status shows as already done.",
  selfKnowledge:'You live in a docked panel \u2014 beside the app on wide windows, below it on narrow ones. The app always reflows to fit, so you never cover it; the arrow at the top of your panel hides you if the visitor wants the space back.',
  doLineMeta:'These ids are for the DO line only; in prose keep using visible names. The DO line is executed by the app and stripped before speaking, so never mention it.',
  ratifyNotice:'The app will also ask the visitor to confirm these actions, so never claim one is already done \u2014 say you are asking.',
  consentScope(o){ o=o||{};
    return '"Please do", "yes", or "go ahead" authorize only the precise action you just offered and nothing more \u2014 if you offered to '+
      (o.pointExample||'jump to a tab')+', consent to that is consent to jump, never to '+
      (o.escalations||'do anything state-changing')+'. Never escalate; when in doubt, use '+(o.safeVerb||'SHOW')+'.'; },
  pointNotDescribe(o){ o=o||{};
    return "The page is often taller than the window, so when the visitor says they can't see or find something, never just describe where it is: end with a "+
      (o.pointVerbs||'DO SHOW')+' line, which switches to it, scrolls it into view, and highlights it. Questions like "where is", "I don\'t see", or "how do I get to" should ALWAYS end with a DO line.'; }
};

function create(A){
  A=A||{};
  const ui=A.ui||{}, layout=A.layout||{}, prov=A.provenance||{};
  const prompts=A.prompts||{}, llm=A.llm||{}, verbs=A.verbs||{};
  const confirmRules=A.confirmRules||[], strings=A.strings||{};
  const takeover=A.takeover||{events:['pointerdown']};
  const steps=(A.tour&&A.tour.steps)||[];
  const testHook=A.testHook||{};
  const notify=ui.notify||function(){};
  const MASCOT=ui.mascot||'\u{1F916}', THINK=ui.thinkFace||'\u{1F914}';
  const W0=layout.panelWidth||300, M0=layout.panelMargin||(W0+2);
  /* the visitor can drag the panel's left edge to make it wider or
     narrower; the width is kept under one key for every guide on this
     origin, so all the rooms follow. W and M are the live values; W0
     and M0 the app's own (still used when the panel docks narrow). */
  const WIDTH_KEY='vg-panel-width', WMIN=260;
  const wMax=()=>Math.max(WMIN,Math.min(760,Math.round(window.innerWidth*0.6)));
  const clampW=w=>Math.round(Math.min(wMax(),Math.max(WMIN,w)));
  const savedW=()=>{ try{ const v=+localStorage.getItem(WIDTH_KEY); return v>0?v:0; }catch(e){ return 0; } };
  let W=savedW()?clampW(savedW()):W0, M=W+(M0-W0);
  const NARROW=('narrowMax' in layout)?layout.narrowMax:700;
  const APPBP=layout.appBreakpoint||0;
  const reflow=layout.reflowCSS||{};
  const BANDS=reflow.bands||[];
  const STORE_KEY=(A.id||'vg')+'-guide-panel';
  const KEYLESS=(llm.keylessModels||VG_DEFAULT_KEYLESS).slice();
  const extractErr=llm.extractError||((d,r)=>((d&&d.error&&d.error.message)||('HTTP '+(r&&r.status))));
  const normalize=llm.normalizeReply||(t=>t);
  const modalCls=Object.assign({backdrop:'modal-backdrop',modal:'modal',row:'button-row',yes:'primary',no:'ghost'},ui.modal||{});

  let vgGen=0, vgState='idle', vgCur=0;
  let vgPauseSnap=null;
  let vgQA=[];                 // spoken Q&A history (persists across pauses)
  let vgSpotted=[], vgSpotSel=null;
  let vgWordMs=300;            // read-along ms/word when TTS is absent (tests set 0)
  const vg$=id=>document.getElementById(id);

  /* Where are we running? Artifacts sit in a sandboxed cross-origin
     iframe whose permissions policy withholds the microphone —
     detectable directly, no error-string guessing. */
  const vgMicPolicyBlocked=(()=>{ try{
      const pp=document.permissionsPolicy||document.featurePolicy;
      if(pp&&typeof pp.allowsFeature==='function') return !pp.allowsFeature('microphone');
    }catch(e){} return false; })();
  const vgInClaude=(()=>{ try{
      if(/claudeusercontent\.com$/i.test(location.hostname)) return true;
      return Array.from(location.ancestorOrigins||[]).some(o=>/claude\.(ai|com)/i.test(o));
    }catch(e){ return false; } })();
  const VG_SR=window.SpeechRecognition||window.webkitSpeechRecognition;
  const vgCanListen=!!VG_SR&&!vgMicPolicyBlocked;

  /* ---------- generated stylesheet ----------
     All generic rules parameterized by {panelWidth, panelMargin,
     narrowMax}; the app-specific reflow rules arrive as raw CSS in
     layout.reflowCSS and core wraps them in media queries computed
     from {appBreakpoint, panelMargin} — mirror the app's stacking
     breakpoint at (breakpoint + panel margin), softening bands
     above it. Bands are emitted before the mirror so the mirror's
     equal-specificity rules win at and below the mirror width. */
  const buildCss=()=>[
'.vg-spot{box-shadow:0 0 0 3px var(--gold,#f7d774),0 0 22px 6px rgba(247,215,116,.45)!important;',
'  border-radius:12px;scroll-margin:130px;animation:vgPulse 1.6s ease-in-out infinite}',
'@keyframes vgPulse{50%{box-shadow:0 0 0 5px var(--gold,#f7d774),0 0 30px 10px rgba(247,215,116,.6)}}',
'',
'body{transition:margin-right .25s}',
`body.vg-open{margin-right:${M}px}`,
'',
...BANDS.map(b=>`@media (max-width:${b.max+(M-M0)}px){${b.css}}`),
...(APPBP?[`@media (max-width:${APPBP+M}px){${reflow.mirrored||''}}`]:[]),
'',
`#vgPanel{position:fixed;top:0;right:0;bottom:0;width:min(${W}px,88vw);z-index:9999;`,
'  display:flex;flex-direction:column;font:inherit;',
'  background:var(--panel,#111a33);border-left:1px solid var(--line,rgba(255,255,255,.14));',
'  box-shadow:-8px 0 26px rgba(0,0,0,.45);',
'  transform:translateX(0);transition:transform .25s}',
'body:not(.vg-open) #vgPanel{transform:translateX(102%)}',
'/* the resize grip on the left edge of the panel (hidden when the panel docks narrow) */',
'#vgGrip{position:absolute;left:-5px;top:0;bottom:0;width:10px;cursor:col-resize;z-index:2;touch-action:none;outline:none}',
'#vgGrip::after{content:"";position:absolute;left:4px;top:0;bottom:0;width:2px;background:transparent;transition:background .15s}',
'#vgGrip:hover::after,#vgGrip.vg-drag::after,#vgGrip:focus-visible::after{background:var(--accent,#7bdff2)}',
'body.vg-resizing,body.vg-resizing #vgPanel{transition:none!important;user-select:none;cursor:col-resize}',
`@media (max-width:${NARROW}px){#vgGrip{display:none}}`,
'',
'#vgHead{display:flex;align-items:center;gap:8px;padding:10px 12px;border-bottom:1px solid var(--line,rgba(255,255,255,.14));',
'  background:rgba(0,0,0,.18);flex:0 0 auto}',
'#vgFace{font-size:22px;user-select:none;transition:transform .2s}',
'#vgFace.vg-talk,#vgTab.vg-talk{animation:vgBob .55s ease-in-out infinite}',
'@keyframes vgBob{50%{transform:translateY(-3px) rotate(-6deg)}}',
'#vgTitle{font-weight:850;font-size:14px;letter-spacing:-.01em;color:var(--ink,#eef4ff);flex:1}',
'#vgHide{border:none;background:none;color:var(--muted,#b7c5e8);font-size:17px;cursor:pointer;padding:4px 6px}',
'#vgHide:hover{color:var(--accent,#7bdff2);transform:none}',
'',
'#vgControls{display:flex;align-items:center;gap:8px;padding:9px 12px;border-bottom:1px solid var(--line,rgba(255,255,255,.14));flex:0 0 auto}',
'.vg-btn{flex:0 0 auto;width:36px;height:36px;border-radius:50%;border:1px solid var(--line,rgba(255,255,255,.14));',
'  background:rgba(255,255,255,.06);color:var(--accent,#7bdff2);font-size:15px;cursor:pointer;',
'  display:flex;align-items:center;justify-content:center;padding:0}',
'.vg-btn:hover{background:rgba(255,255,255,.12);border-color:rgba(123,223,242,.55);transform:none}',
'.vg-primary{color:#06111f;background:linear-gradient(180deg,#aaf4ff,#65d4ea);border:none;font-size:16px;font-weight:900}',
'.vg-btn.vg-primary:hover{background:linear-gradient(180deg,#c9f9ff,#86e2f3);box-shadow:0 0 0 2px rgba(170,244,255,.45)}',   /* hover brightens the main button; the plain-button hover used to grey it out */
'#vgMic.vg-live{background:#e0483c;color:#fff;border:none;animation:vgPulse 1.2s infinite}',
'#vgMic.vg-off{opacity:.45}',
'',
'#vgLog{flex:1 1 auto;min-height:0;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:12px}',
'.vg-b{max-width:95%;padding:8px 11px;border-radius:12px;font-size:13.5px;line-height:1.5;',
'  white-space:pre-wrap;overflow-wrap:break-word;align-self:flex-start;',
'  background:rgba(255,255,255,.07);color:var(--ink,#eef4ff);border:1px solid var(--line,rgba(255,255,255,.14));',
'  border-bottom-left-radius:4px}',
'.vg-b.vg-user{align-self:flex-end;background:rgba(123,223,242,.16);border-color:rgba(123,223,242,.4);',
'  border-bottom-left-radius:12px;border-bottom-right-radius:4px}',
'',
'#vgStatus{flex:0 0 auto;min-height:18px;padding:2px 12px;font-size:12px;font-style:italic;color:var(--muted,#b7c5e8)}',
'#vgTypeRow{display:flex;gap:8px;padding:0 12px 12px;flex:0 0 auto}',
'#vgIn{flex:1;min-width:0;font-size:13.5px;border:1px solid var(--line,rgba(255,255,255,.14));',
'  border-radius:10px;padding:8px 11px;background:rgba(0,0,0,.25);color:var(--ink,#eef4ff);font-family:inherit;width:auto}',
'#vgIn:focus{outline:none;border-color:rgba(123,223,242,.6)}',
'#vgGo{font-weight:850;font-size:13px;color:#06111f;background:linear-gradient(180deg,#aaf4ff,#65d4ea);',
'  border:none;border-radius:10px;padding:0 15px;cursor:pointer}',
'',
'#vgTab{position:fixed;right:0;top:38%;z-index:9998;cursor:pointer;user-select:none;',
'  background:var(--panel,#111a33);border:1px solid var(--line,rgba(255,255,255,.14));border-right:none;',
'  border-radius:12px 0 0 12px;box-shadow:-4px 2px 14px rgba(0,0,0,.45);',
'  padding:10px 8px 10px 10px;font-size:22px;display:none}',
'body:not(.vg-open) #vgTab{display:flex;flex-direction:column;align-items:center;gap:5px}',
'.vg-tab-label{writing-mode:vertical-rl;font-size:11px;font-weight:850;letter-spacing:.15em;',
'  text-transform:uppercase;color:var(--accent,#7bdff2)}',
'',
'/* Confirm-dialog fallback for apps with no modal CSS of their own.',
'   :where() keeps specificity at zero, so any app class rule on the',
'   dialog wins over these. */',
':where(#vgConfirm){position:fixed;inset:0;display:none;align-items:center;justify-content:center;',
'  background:rgba(0,0,0,.55);z-index:10001}',
':where(#vgConfirm.show){display:flex}',
'',
'/* Narrow rules come last: they must override the side-panel rules above',
'   (they sat first once, so phones got a stray box instead of a dock). */',
'/* Narrow (artifact preview pane, phones): dock to the bottom',
'   instead of overlaying — margin-bottom reflows the app above the',
'   panel, so the guide never covers anything at any width. */',
`@media (max-width:${NARROW}px){`,
`  body.vg-open{margin-right:0;margin-bottom:${M0}px}`,
`  #vgPanel{top:auto;left:0;right:0;bottom:0;width:auto;height:${W0}px;`,
'    border-left:none;border-top:1px solid var(--line,rgba(255,255,255,.14));',
'    box-shadow:0 -8px 26px rgba(0,0,0,.45)}',
'  body:not(.vg-open) #vgPanel{transform:translateY(102%)}',
'  #vgTab{top:auto;bottom:0;right:14px;border-radius:12px 12px 0 0;',
'    border-right:1px solid var(--line,rgba(255,255,255,.14));border-bottom:none;',
'    padding:8px 12px;flex-direction:row!important;gap:8px}',
'  .vg-tab-label{writing-mode:horizontal-tb}',
'}'
  ].join('\n');
  const styleEl=document.createElement('style');
  styleEl.id='vgStyle'; styleEl.textContent=buildCss();
  document.head.appendChild(styleEl);

  /* ---------- panel width: drag the grip, arrow keys, double-click resets ---------- */
  function vgSetWidth(w,save){
    W=clampW(w); M=W+(M0-W0);
    styleEl.textContent=buildCss();
    const g=vg$('vgGrip'); if(g){ g.setAttribute('aria-valuenow',String(W)); g.setAttribute('aria-valuemin',String(WMIN)); g.setAttribute('aria-valuemax',String(wMax())); }
    if(save){ try{ if(W===W0) localStorage.removeItem(WIDTH_KEY); else localStorage.setItem(WIDTH_KEY,String(W)); }catch(e){} }
  }
  function vgWireGrip(){
    const g=vg$('vgGrip'); if(!g) return;
    vgSetWidth(W,false);
    let raf=0, want=W;
    g.addEventListener('pointerdown',e=>{
      if(e.button!==0) return;
      e.preventDefault(); g.setPointerCapture(e.pointerId);
      g.classList.add('vg-drag'); document.body.classList.add('vg-resizing');
    });
    g.addEventListener('pointermove',e=>{
      if(!g.hasPointerCapture(e.pointerId)) return;
      want=window.innerWidth-e.clientX;
      if(!raf) raf=requestAnimationFrame(()=>{ raf=0; vgSetWidth(want,false); });
    });
    const done=e=>{
      if(!g.hasPointerCapture(e.pointerId)) return;
      g.releasePointerCapture(e.pointerId);
      g.classList.remove('vg-drag'); document.body.classList.remove('vg-resizing');
      vgSetWidth(want,true);
    };
    g.addEventListener('pointerup',done); g.addEventListener('pointercancel',done);
    g.addEventListener('dblclick',()=>vgSetWidth(W0,true));
    g.addEventListener('keydown',e=>{
      const step=e.shiftKey?60:20;
      if(e.key==='ArrowLeft'){ e.preventDefault(); vgSetWidth(W+step,true); }
      else if(e.key==='ArrowRight'){ e.preventDefault(); vgSetWidth(W-step,true); }
      else if(e.key==='Home'){ e.preventDefault(); vgSetWidth(W0,true); }
    });
    /* another room's guide changed the width: follow it */
    window.addEventListener('storage',e=>{ if(e.key===WIDTH_KEY) vgSetWidth(savedW()||W0,false); });
    /* a smaller window may no longer fit the chosen width */
    window.addEventListener('resize',()=>{ const w=clampW(savedW()||W0); if(w!==W) vgSetWidth(w,false); });
  }

  /* ---------- panel markup (all ids/classes prefixed vg) ---------- */
  {
    const tab=document.createElement('div');
    tab.id='vgTab'; tab.title=ui.tabTitle||'Bring the guide back';
    tab.innerHTML='<span></span><span class="vg-tab-label"></span>';
    tab.firstChild.textContent=MASCOT;
    tab.lastChild.textContent=ui.tabLabel||'Guide';
    document.body.appendChild(tab);
    const aside=document.createElement('aside');
    aside.id='vgPanel'; aside.setAttribute('role','complementary');
    aside.setAttribute('aria-label',ui.ariaLabel||ui.title||'Voice guide');
    aside.innerHTML=
      '<div id="vgGrip" role="separator" aria-orientation="vertical" tabindex="0" aria-label="Guide width"'+
      ' title="Drag to make the guide wider or narrower (double-click: back to normal)"></div>'+
      '<div id="vgHead"><span id="vgFace"></span><span id="vgTitle"></span>'+
      '<button id="vgHide" title="Hide the guide">&#10095;</button></div>'+
      '<div id="vgControls">'+
      '<button id="vgMain" class="vg-btn vg-primary" title="Start the spoken tour">&#9654;</button>'+
      '<button id="vgMic"  class="vg-btn" title="Ask a question by voice">&#127908;</button>'+
      '<button id="vgSkip" class="vg-btn" title="Skip to the next stop" style="display:none">&#9197;</button>'+
      '<button id="vgEnd"  class="vg-btn" title="End the tour" style="display:none">&#10005;</button>'+
      '<button id="vgSave" class="vg-btn" title="Save this conversation as a file">&#128190;</button>'+
      '</div>'+
      '<div id="vgLog" aria-live="polite"></div>'+
      '<div id="vgStatus"></div>'+
      '<div id="vgTypeRow"><input id="vgIn" spellcheck="false"><button id="vgGo">Ask</button></div>';
    document.body.appendChild(aside);
    vg$('vgFace').textContent=MASCOT;
    vgWireGrip();
    vg$('vgTitle').textContent=ui.title||'Guide';
    vg$('vgIn').placeholder=ui.askPlaceholder||'Type a question\u2026';
  }

  /* ---------- confirm() done properly ----------
     Artifact iframes lack allow-modals, so native confirm() is
     swallowed (returns false instantly) and every "if(confirm(...))"
     guard silently aborts. confirm is synchronous and a nice dialog
     is not, so: remember which button was just pressed, answer false,
     show an in-page dialog styled like the app's own modals, and on
     Yes pre-approve the message and re-click that button — the
     handler runs again and the wrapper answers true. Works the same
     in normal browsers (nicer than the native box). Fallbacks when no
     trigger button is known: native dialog, then press-again toast.
     Per-message hints and extra buttons come from adapter confirmRules. */
  let vgLastClick=null;
  document.addEventListener('click',e=>{
    const t=e.target;
    if(t&&t.closest&&(t.closest('#vgPanel')||t.closest('#vgTab')||t.closest('#vgConfirm'))) return;
    vgLastClick={el:(t&&t.closest&&t.closest('button'))||t,t:Date.now()};
  },true);
  let vgConfirmCb=null, vgConfirmNoCb=null;
  function vgEnsureConfirmEl(){
    if(vg$('vgConfirm')) return;
    const d=document.createElement('div');
    d.id='vgConfirm'; d.className=modalCls.backdrop; d.style.zIndex='10001';
    d.innerHTML='<div class="'+modalCls.modal+'" style="max-width:430px"><h2 style="margin:0 0 8px">Just checking</h2>'+
      '<p id="vgConfirmMsg" style="margin:0 0 10px"></p>'+
      '<p id="vgConfirmHint" style="margin:0 0 14px;font-size:13px;color:var(--muted,#b7c5e8)"></p>'+
      '<div class="'+modalCls.row+'"><button id="vgConfirmYes" class="'+modalCls.yes+'">Yes, do it</button>'+
      '<button id="vgConfirmExtra" class="'+modalCls.no+'" style="display:none"></button>'+
      '<button id="vgConfirmNo" class="'+modalCls.no+'">Cancel</button></div></div>';
    document.body.appendChild(d);
    d.addEventListener('click',e=>{ if(e.target===d) vgHideConfirm(); });
  }
  function vgHideConfirm(){ const d=vg$('vgConfirm'); if(d) d.classList.remove('show');
    const no=vgConfirmNoCb; vgConfirmCb=null; vgConfirmNoCb=null; if(no) no(); }
  function vgShowConfirm(msg,onYes,opts){
    opts=opts||{};
    vgEnsureConfirmEl();
    vg$('vgConfirmMsg').textContent=msg;
    const h=vg$('vgConfirmHint'); h.textContent=opts.hint||''; h.style.display=opts.hint?'':'none';
    const ex=vg$('vgConfirmExtra');
    if(opts.extraLabel){ ex.style.display=''; ex.textContent=opts.extraLabel;
      ex.onclick=()=>{ vgHideConfirm(); try{ opts.onExtra&&opts.onExtra(); }catch(e){} }; }
    else { ex.style.display='none'; ex.onclick=null; }
    vgConfirmCb=onYes; vgConfirmNoCb=opts.onNo||null;
    vg$('vgConfirmYes').onclick=()=>{ const cb=vgConfirmCb; vgConfirmCb=null; vgConfirmNoCb=null;
      vg$('vgConfirm').classList.remove('show'); if(cb) cb(); };
    vg$('vgConfirmNo').onclick=vgHideConfirm;
    vg$('vgConfirm').classList.add('show');
  }
  function vgRatify(msg){ return new Promise(res=>{ vgShowConfirm(msg,()=>res(true),{onNo:()=>res(false)}); }); }
  {
    const vgOrigConfirm=typeof window.confirm==='function'?window.confirm.bind(window):null;
    const vgApproved=new Map();                    // message -> expiry
    window.confirm=function(msg){
      msg=String(msg);
      const now=Date.now();
      if(now<(vgApproved.get(msg)||0)){ vgApproved.delete(msg); return true; }
      const trg=(vgLastClick&&now-vgLastClick.t<3000)?vgLastClick.el:null;
      if(trg){                                     // caveat per app: handler side effects before the confirm call repeat on re-trigger
        const rule=confirmRules.find(r=>r.match.test(msg));
        vgShowConfirm(msg,()=>{ vgApproved.set(msg,Date.now()+3000); try{ trg.click(); }catch(e){} },
          rule?{hint:rule.hint,extraLabel:rule.extraLabel,
                onExtra:rule.onExtra&&(()=>rule.onExtra(api))}:{});
        return false;
      }
      if(vgOrigConfirm){                           // no known trigger: let the real dialog handle it
        const t0=performance.now(); let r=false;
        try{ r=!!vgOrigConfirm(msg); }catch(e){}
        if(r) return true;
        if(performance.now()-t0>150) return false; // a human really pressed Cancel
      }
      if(now<(vgApproved.get('arm:'+msg)||0)){ vgApproved.delete('arm:'+msg); return true; }
      vgApproved.set('arm:'+msg,now+6000);         // swallowed and no trigger: last-resort press-again
      notify(msg+' \u2014 pop-ups are blocked here, so press the same button again to confirm.');
      return false;
    };
  }

  /* ---------- tiny helpers ---------- */
  const vgCHK=g=>{ if(vgGen!==g) throw VG_ABORT; };
  function vgWait(ms,g){ return new Promise((res,rej)=>{ const t0=performance.now();
    const iv=setInterval(()=>{ if(vgGen!==g){clearInterval(iv);rej(VG_ABORT);}
      else if(performance.now()-t0>=ms){clearInterval(iv);res();} },60); }); }
  function vgClickSel(sel){ const b=document.querySelector(sel);
    if(b&&!b.disabled){ b.click(); vgRespot(); return true; } return false; }
  function vgTourLive(){ return vgState==='running'||vgState==='paused'; }
  function vgDemoMs(ms){ return vgWordMs>=100?ms:15; }   // demos shrink with test speed

  /* ---------- panel: transcript log + status line ----------
     Narration and answers land in the log as full blocks the moment
     they start; the log scrolls, so no message can outgrow its
     container. The status line carries the transient stuff only:
     paused, listening, thinking. */
  const vgTranscript=[];                     // full history for the save button (the log DOM trims to 60)
  function vgLogAdd(text,who){
    vgTranscript.push({t:Date.now(),who:who==='user'?'you':'guide',text:String(text)});
    const log=vg$('vgLog');
    const d=document.createElement('div');
    d.className='vg-b'+(who==='user'?' vg-user':'');
    d.textContent=text; log.appendChild(d);
    while(log.children.length>60) log.removeChild(log.firstChild);
    log.scrollTop=log.scrollHeight;
    return d;
  }
  function vgStatus(t){ vg$('vgStatus').textContent=t||''; }
  function vgSetPanel(open){
    document.body.classList.toggle('vg-open',open);
    try{ localStorage.setItem(STORE_KEY,open?'1':'0'); }catch(e){}
    if(open){ const log=vg$('vgLog'); log.scrollTop=log.scrollHeight; }
  }

  /* ---------- speech synthesis (chunked; Chrome cuts long utterances) ---------- */
  let vgVoice=null;
  function vgPickVoice(){
    if(!('speechSynthesis' in window)) return;
    const vs=speechSynthesis.getVoices()||[];
    vgVoice = vs.find(v=>/^en/i.test(v.lang)&&/natural|neural/i.test(v.name))
           || vs.find(v=>/^en/i.test(v.lang))
           || vs[0]||null;
  }
  if('speechSynthesis' in window){ vgPickVoice(); speechSynthesis.onvoiceschanged=vgPickVoice; }
  function vgSynthCancel(){ try{ speechSynthesis.cancel(); }catch(e){} }
  document.addEventListener('visibilitychange',()=>{ if(document.hidden) vgSynthCancel(); });   // drop anything queued; nothing replays later
  async function vgSpeak(text,g,who,abortFn){        // abortFn: v2 event-manager preemption seam
    const face=vg$('vgFace'), tab=vg$('vgTab');
    face.classList.add('vg-talk'); tab.classList.add('vg-talk');
    vgSpeakDepth++;                                   // v2: single choke point for "channel busy"
    vgLogAdd(String(text),who);                       // whole message visible at once
    try{
      const chunks=(String(text).match(/[^.!?\u2026]+[.!?\u2026]*/g)||[String(text)]).map(s=>s.trim()).filter(Boolean);
      for(const c of chunks){
        vgCHK(g);
        if(abortFn&&abortFn()) throw VG_ABORT;
        if(!('speechSynthesis' in window)||!vgVoiceOn()){ // no TTS, or voice toggled off: read-along timing
          await vgWait(Math.max(vgWordMs?600:0,c.split(/\s+/).length*vgWordMs),g); continue;
        }
        await new Promise(res=>{
          const u=new SpeechSynthesisUtterance(c);
          if(vgVoice) u.voice=vgVoice;
          u.rate=1.0; u.pitch=1.0;
          const iv=setInterval(()=>{ if(vgGen!==g||(abortFn&&abortFn())){ clearInterval(iv); vgSynthCancel(); res(); } },150); // watchdog: onend is sometimes swallowed after cancel()
          u.onend=()=>{clearInterval(iv);res();}; u.onerror=()=>{clearInterval(iv);res();};
          try{ speechSynthesis.resume(); }catch(e){}  // defensive: Chrome can wedge in a paused state
          speechSynthesis.speak(u);
        });
        vgCHK(g);
        if(abortFn&&abortFn()) throw VG_ABORT;
      }
    } finally { face.classList.remove('vg-talk'); tab.classList.remove('vg-talk');
      vgSpeakDepth--; vgEvDrain(); }
  }
  function vgFaceThink(on){ vg$('vgFace').textContent=on?THINK:MASCOT; }

  /* ---------- spotlight (selector string or function returning element(s)) ----------
     Apps often re-render whole panels on state changes, stripping
     classes; vgRespot() re-resolves the stored target, so guide-
     driven renders never lose the glow. */
  function vgUnspot(){ vgSpotted.forEach(el=>{ try{ el.classList.remove('vg-spot'); }catch(e){} }); vgSpotted=[]; vgSpotSel=null; }
  function vgApplySpot(sel,scroll){
    vgSpotted.forEach(el=>{ try{ el.classList.remove('vg-spot'); }catch(e){} }); vgSpotted=[];
    if(!sel) return;
    let els=[];
    if(typeof sel==='function'){ try{ els=[].concat(sel()||[]); }catch(e){} }
    else els=Array.from(document.querySelectorAll(sel));
    els=els.filter(Boolean); if(!els.length) return;
    vgSpotted=els; els.forEach(el=>el.classList.add('vg-spot'));
    if(scroll){ try{ els[0].scrollIntoView({behavior:'smooth',block:'center'}); }catch(e){} }
  }
  function vgSpot(sel){ vgSpotSel=sel||null; vgApplySpot(sel,true); }
  function vgRespot(){ if(vgSpotSel) vgApplySpot(vgSpotSel,false); }

  /* ---------- v2: event bus + attention manager ----------
     Emergent-moment narration alongside the tour's golden path.
     One speech/text channel (vgSpeak is the choke point; vgSpeakDepth
     tracks it). Priority classes:
       verdict — jumps the queue; preempts a manager-owned utterance
                 mid-line (never a tour or Q&A line: those are
                 provenance-bearing, so a verdict waits for them and
                 speaks immediately after).
       moment  — FIFO queue, depth 2, oldest moment dropped.
       ambient — dropped unless the channel is idle and queue empty.
     cooldownMs and once are checked at enqueue AND at speak time;
     `once` is consumed on speak, not on emit, so a preempted once-
     line is not lost. Line specs: {lines, kidLines?, priority,
     cooldownMs?, once?, condition?(payload), spot?}; lines may be a
     string or fn(payload). A.lineMode() selects the register.
     Waiters (step.until / api.waitFor) resolve on emit and reject
     VG_ABORT when their generation dies (swept on a low-freq timer,
     so no existing vgGen++ site needed touching). */
  let vgSpeakDepth=0, vgEvDraining=false, vgEvEpoch=0, vgEvSpeaking=false;
  const vgEvQueue=[];                       // {name, spec, payload}
  const vgEvOnceDone=new Set(), vgEvLastAt=new Map();
  const vgEvDyn={};                         // api.on/off registry, over A.events
  let vgEvWaiters=[], vgEvSweep=null;
  const vgVoiceOn=()=>{ if(window.__vgPrefs&&window.__vgPrefs.active===false) return false;
    if(document.hidden) return false;   // a background tab: speech would queue up and play on return   // a room the shell has hidden never talks out loud
    const u=A.ui||{}; return u.voiceEnabled?!!u.voiceEnabled():true; };
  function vgEvSpec(name){ return vgEvDyn[name]||((A.events||{})[name]); }
  function vgEvLineOf(spec,payload){
    const mode=(A.lineMode&&A.lineMode())||'default';
    let L=(mode==='kid'&&spec.kidLines!=null)?spec.kidLines:spec.lines;
    return typeof L==='function'?L(payload):L;
  }
  function vgEvSweepStart(){
    if(vgEvSweep) return;
    vgEvSweep=setInterval(()=>{
      vgEvWaiters=vgEvWaiters.filter(w=>{ if(vgGen!==w.gen){ w.rej(VG_ABORT); return false; } return true; });
      if(!vgEvWaiters.length){ clearInterval(vgEvSweep); vgEvSweep=null; }
    },250);
  }
  function vgWaitFor(name,opts){
    opts=opts||{};
    const gen=('gen' in opts)?opts.gen:vgGen;
    return new Promise((res,rej)=>{
      vgEvWaiters.push({name,gen,cond:opts.condition,res,rej});
      vgEvSweepStart();
    });
  }
  function vgEmit(name,payload,inlineSpec){
    vgEvWaiters=vgEvWaiters.filter(w=>{
      if(vgGen!==w.gen){ w.rej(VG_ABORT); return false; }
      if(w.name!==name) return true;
      if(w.cond&&!w.cond(payload)) return true;
      w.res(payload); return false;
    });
    const spec=inlineSpec||vgEvSpec(name);
    if(!spec||spec.lines==null) return;
    if(spec.condition&&!spec.condition(payload)) return;
    if(spec.once&&vgEvOnceDone.has(name)) return;
    if(spec.cooldownMs&&Date.now()-(vgEvLastAt.get(name)||0)<spec.cooldownMs) return;
    if(vgEvQueue.some(q=>q.name===name)) return;   // a burst of the same event (e.g. two run-done in a tick) enqueues once
    const pr=spec.priority||'moment';
    if(pr==='ambient'){
      if(vgSpeakDepth>0||vgEvQueue.length) return;    // ambient never waits
      vgEvQueue.push({name,spec,payload});
    }else if(pr==='verdict'){
      vgEvQueue.unshift({name,spec,payload});
      if(vgEvSpeaking) vgEvEpoch++;                   // cut the manager's own line only
    }else{
      vgEvQueue.push({name,spec,payload});
      const moments=vgEvQueue.filter(q=>(q.spec.priority||'moment')==='moment');
      if(moments.length>2){ const idx=vgEvQueue.indexOf(moments[0]); if(idx>=0) vgEvQueue.splice(idx,1); }
    }
    vgEvDrain();
  }
  async function vgEvDrain(){
    if(vgEvDraining||vgSpeakDepth>0) return;
    vgEvDraining=true;
    try{
      while(vgEvQueue.length){
        const it=vgEvQueue.shift();
        if(it.spec.once&&vgEvOnceDone.has(it.name)) continue;
        if(it.spec.cooldownMs&&Date.now()-(vgEvLastAt.get(it.name)||0)<it.spec.cooldownMs) continue;
        const text=vgEvLineOf(it.spec,it.payload);
        if(!text) continue;
        if(it.spec.once) vgEvOnceDone.add(it.name);
        vgEvLastAt.set(it.name,Date.now());
        const myEp=++vgEvEpoch, g=vgGen, prevSpot=vgSpotSel;
        if(it.spec.spot) vgSpot(it.spec.spot);
        vgEvSpeaking=true;
        try{ await vgSpeak(text,g,null,()=>vgEvEpoch!==myEp); }
        catch(e){ if(e!==VG_ABORT) console.error('voice guide event:',e); }
        vgEvSpeaking=false;
        if(it.spec.spot){ if(prevSpot) vgSpot(prevSpot); else vgUnspot(); }
      }
    } finally { vgEvDraining=false; }
  }
  function vgEvOn(name,spec){ vgEvDyn[name]=spec; }
  function vgEvOff(name){ delete vgEvDyn[name]; }
  /* Commentary track: timed beats through the same manager. Beats:
     [{atMs, lines, kidLines?, priority?, spot?}]. Auto-cancels when
     opts.gen dies; cancel() is idempotent. */
  function vgTrack(beats,opts){
    opts=opts||{};
    let live=true;
    const timers=(beats||[]).map(b=>setTimeout(()=>{
      if(!live) return;
      if(('gen' in opts)&&vgGen!==opts.gen){ cancel(); return; }
      vgEmit('__vgTrackBeat',b.payload,{lines:b.lines,kidLines:b.kidLines,
        priority:b.priority||'moment',spot:b.spot});
    },b.atMs));
    function cancel(){ live=false; timers.forEach(clearTimeout); }
    return {cancel};
  }

  /* ---------- runner over adapter-supplied steps ----------
     Each stop: {name, spot, text (string|fn), prep()?, action(gen)?,
     during?, skipIf?}. prep runs before the spotlight (via real
     buttons — synthetic clicks are untrusted, so the guide never
     pauses itself). during:true runs the action concurrently with
     the narration. Actions must be state-aware and safe to re-run,
     so restarting a stop after a pause never fights the player. */
  async function vgRunFrom(i){
    const g=++vgGen; vgState='running'; vgStatus(''); vgUpdateBar();
    try{
      for(; i<steps.length; i++){
        vgCur=i; const st=steps[i];
        if(st.skipIf&&st.skipIf()) continue;
        if(st.prep){ try{ st.prep(); }catch(e){} }
        vgSpot(st.spot);
        const text=typeof st.text==='function'?st.text():st.text;
        if(st.during&&st.action) await Promise.all([vgSpeak(text,g), st.action(g)]);
        else { await vgSpeak(text,g); if(st.action) await st.action(g); }
        if(st.until){ const u=typeof st.until==='string'?{event:st.until}:st.until; // v2: step waits on the world
          await vgWaitFor(u.event,{gen:g,condition:u.condition}); }
        await vgWait(vgDemoMs(450),g);
      }
      vgState='done'; vgCur=0; vgUnspot();
      vgStatus(ui.doneStatus||"Tour's over!");
      vgUpdateBar();
    }catch(e){ if(e!==VG_ABORT) console.error('voice guide tour:',e); }
  }

  /* ---------- pause / resume, aware of what the player did ----------
     Snapshot on pause; on resume, speak the adapter's one-line diff
     of what the player changed. Side-effect suppression and remark
     wording live in the adapter's diffRemark; re-baselining after
     guide-driven changes lives here. */
  function vgPauseCore(){ vgGen++; vgSynthCancel(); vgPauseSnap=prov.snapshot(); vgState='paused';
    vg$('vgFace').classList.remove('vg-talk'); vg$('vgTab').classList.remove('vg-talk'); }
  function vgPause(){ vgPauseCore();
    vgStatus("Paused \u2014 explore as much as you like! \u25B6 carries on.");
    vgUpdateBar(); }
  async function vgResume(){
    const remark=prov.diffRemark(vgPauseSnap); vgPauseSnap=null;
    vgState='running'; vgStatus(''); vgUpdateBar();
    const g=++vgGen;
    try{ if(remark) await vgSpeak(remark,g); }catch(e){ if(e===VG_ABORT) return; }
    if(vgGen!==g) return;
    vgRunFrom(vgCur);            // restart the interrupted stop; actions are re-run-safe
  }
  function vgStart(){
    vgCur=0; vgPauseSnap=null;
    if('speechSynthesis' in window && !vgVoice) vgPickVoice();
    vgRunFrom(0);
  }
  function vgEndTour(){ vgGen++; vgSynthCancel(); vgState='idle'; vgCur=0; vgPauseSnap=null; vgUnspot();
    vgStatus(''); vgUpdateBar(); }
  function vgSkipStep(){ if(!vgTourLive()) return;
    vgGen++; vgSynthCancel(); vgPauseSnap=null;
    vgRunFrom(vgCur+1); }

  /* auto-pause: any trusted interaction with the app while the tour
     is narrating means the player has taken the wheel. Guide-driven
     clicks are synthetic, hence untrusted, and pass through. Which
     event types count is adapter config (keyboard-heavy apps should
     drop keydown); takeover.ignore(e) is an extra escape hatch. */
  (takeover.events||[]).forEach(ev=>document.addEventListener(ev,e=>{
    if(vgState!=='running'||!e.isTrusted) return;
    const t=e.target;
    if(t&&t.closest&&(t.closest('#vgPanel')||t.closest('#vgTab')||t.closest('#vgConfirm'))) return;
    if(takeover.ignore&&takeover.ignore(e)) return;
    vgPauseCore();
    vgStatus("You're in charge! \u25B6 carries on the tour.");
    vgUpdateBar();
  },true));

  /* ---------- Q&A prompts ----------
     The system prompt is the adapter's (reuse the app's own grounding
     builder plus a voice override — never a parallel AI path). Core
     owns the user-prompt assembly: flattened history, then a fresh
     STATUS RIGHT NOW block from the adapter's serializer, then the
     question — recency bias then works for us. */
  function vgCtx(){ const live=vgTourLive();
    return {tourLive:live, stopName:(live&&steps[vgCur])?steps[vgCur].name:''}; }
  function vgSystem(){ return prompts.system(vgCtx()); }
  function vgUserPrompt(q){
    const hist=vgQA.slice(-8).map(m=>(m.role==='user'?'Visitor: ':'Guide: ')+m.content);
    return (hist.length?'Conversation so far between the visitor and you, the guide:\n'+hist.join('\n')+'\n\n':'')+
      'STATUS RIGHT NOW, fresh from the app this second. The game may have moved on since earlier turns; this status outranks anything the guide said before, and the guide must never advise a step the status already shows as done:\n'+
      prompts.status()+'\n\n'+
      'The visitor now asks the guide: '+q;
  }
  function vgExtractDo(t){                          // the directive ends the reply; models often keep it on the prose line
    const s=String(t);
    const m=s.match(/DO:[ \t]*([A-Z_]+)[ \t]*([^\n]*?)[ \t]*$/);
    if(!m) return {text:s.trim(),act:null};
    const text=s.slice(0,m.index).replace(/[\s\u2014\u2013-]+$/,'').trim();
    return {text,act:{verb:m[1],args:m[2].trim().replace(/[.!?,;:]+$/,'')}};   // "DO: SHOW train." still means train
  }

  /* ---------- DO dispatch over the adapter verb table ----------
     Entries: {gated?, validate(parts,api)?, ratifyText(parts,ctx,api)?,
     exec(parts,api,ctx)}. Order is core policy: validate FIRST (so
     nobody ratifies an impossible act; invalid is a silent no-op),
     then for gated verbs the in-page ratification dialog — the model
     proposes, the player countersigns — then exec through the app's
     real controls. Pointing verbs stay frictionless. */
  async function vgDo(act){
    try{
      const v=(act.verb||'').toUpperCase();
      const parts=String(act.args||'').trim().split(/\s+/).filter(Boolean);
      const spec=verbs[v];
      if(spec){
        let ctx=true, ok=true;
        if(spec.validate){ ctx=spec.validate(parts,api); ok=!!ctx; }
        if(ok){
          if(spec.gated){
            if(await vgRatify(spec.ratifyText(parts,ctx,api))) await spec.exec(parts,api,ctx);
          } else await spec.exec(parts,api,ctx);
        }
      }
      vgRespot();
    }catch(e){ if(e!==VG_ABORT) console.warn('guide DO failed:',e); }
    finally{
      if(vgPauseSnap) vgPauseSnap=prov.snapshot();   // the guide drove; re-baseline so resume remarks only describe what the player did
    }
  }

  /* ---------- LLM chain: configured key wins, keyless in artifacts ---------- */
  async function vgKeylessAnthropic(systemPrompt,userPrompt){
    let lastErr=null;
    for(const model of KEYLESS){
      try{
        const res=await fetch('https://api.anthropic.com/v1/messages',{method:'POST',
          headers:{'Content-Type':'application/json'},
          body:JSON.stringify({model,max_tokens:1000,system:systemPrompt,messages:[{role:'user',content:userPrompt}]})});
        const data=await res.json().catch(()=>({}));
        if(!res.ok) throw new Error(extractErr(data,res));
        const text=((data.content||[]).map(c=>c.text||'').join('\n')||'').trim();
        if(text) return text;
        throw new Error('the AI returned no text');
      }catch(e){ lastErr=e; }
    }
    throw lastErr||new Error('keyless call failed');
  }
  async function vgCallLLM(systemPrompt,userPrompt){
    if(llm.hasKey&&llm.hasKey()) return llm.keyedCall(systemPrompt,userPrompt);   // the app's plumbing, provider and model as configured
    if(vgInClaude) return vgKeylessAnthropic(systemPrompt,userPrompt);
    const err=new Error('no live AI connection'); err.vgNoAI=true; throw err;
  }

/* click-to-ask: adapters register short glossary notes for on-screen
     terms (api.explainable); clicking one, or asking about it, routes an
     explain-this question through the same LLM path, seeded with the
     note so the answer is grounded rather than guessed. */
  const vgGloss=new Map();                          // key -> {label, note, kidNote?}
  function vgExplainable(map){
    for(const k in map) vgGloss.set(k,map[k]);
    return (ev)=>{ const el=ev.target&&ev.target.closest&&ev.target.closest('[data-vg-explain]');
      if(!el) return; ev.preventDefault(); ev.stopPropagation();
      vgSetPanel(true); vgExplainTopic(el.getAttribute('data-vg-explain')); };
  }
  function vgExplainTopic(key){
    const g=vgGloss.get(key); if(!g) return;
    const kid=(A.lineMode&&A.lineMode()==='kid');
    const note=(kid&&g.kidNote)||g.note;
    const q='Explain this part of the screen to me: "'+(g.label||key)+'". '+
      (note?'Here is what it is, in one line, so you can put it in your own words'+(kid?' for a young child':'')+': '+note:'')+
      ' Keep it to a sentence or two.';
    vgAsk(q, 'What is "'+(g.label||key)+'"?');   // the visitor sees the short question, not the seeded note
  }
  async function vgAsk(q, shown){
    q=(q||'').trim(); if(!q) return;
    if(vgState==='running') vgPauseCore();
    const g=++vgGen; vgSynthCancel(); vgUpdateBar();
    vgLogAdd((shown||q),'user'); vgFaceThink(true); vgStatus('thinking\u2026');
    const sys=vgSystem(), usr=vgUserPrompt(q);
    vgQA.push({role:'user',content:q});
    try{
      const raw=await vgCallLLM(sys,usr);
      if(vgGen!==g) return;
      vgQA.push({role:'assistant',content:raw||''});
      const {text,act}=vgExtractDo(raw||'');
      const prose=normalize(text);          // the app's own cleaner: ids -> visible names; runs AFTER DO extraction or it would corrupt DO arguments
      vgFaceThink(false); vgStatus('');
      const jobs=[vgSpeak(prose||"I couldn't put that into words \u2014 try asking another way!",g)];
      if(act) jobs.push(vgDo(act));
      await Promise.all(jobs);
      if(vgGen===g&&vgTourLive()) vgStatus('\u25B6 carries on the tour \u2014 or ask me more!');
    }catch(e){
      vgFaceThink(false); vgStatus('');
      if(e===VG_ABORT) return;
      if(e&&e.vgNoAI&&llm.noAI){
        vgSpot(llm.noAI.spot);
        try{ await vgSpeak(llm.noAI.say,g); }catch(_){}
        return;
      }
      const emsg=e&&e.message?e.message:String(e);
      const fb=(strings.llmFail&&strings.llmFail(e))||           // v2: adapter-suppliable failure copy
        (/quota|rate.?limit|429|too many|overloaded|exceed/i.test(emsg)
          ? {log:'Live AI call failed: '+emsg,
             say:"The shared A I allowance is used up for the moment \u2014 it refills on its own. Ask me again in a little while; the tour and every button still work."}
          : {log:'Live AI call failed: '+emsg,
             say:"I couldn't reach the live A I just now \u2014 the note above says why. The tour and every button still work without it."});
      vgLogAdd(fb.log);
      try{ await vgSpeak(fb.say,g); }catch(_){}
    }
  }

  /* ---------- microphone: by policy, not by error ---------- */
  let vgRec=null, vgListening=false;
  /* a page that also lives on an ordinary website names it in window.__vgElsewhere
     = {label, spoken, keyHint}: the mic advice then sends people there, not to a saved file */
  const vgElsewhere=()=>{ const e=window.__vgElsewhere; return e&&e.label?e:null; };
  function vgMicUnavailableMsg(){
    if(vgMicPolicyBlocked){
      const where=vgInClaude?'inside a Claude artifact':'inside this embedded page';
      const e=vgElsewhere();
      if(e) return "I'm not allowed to use the microphone "+where+": the sandbox keeps it switched off, so no permission pop-up can appear here. "+
        "Type your question in the box at the bottom of my panel instead. To talk with me out loud, open "+e.label+" in a normal browser tab, where the mic works"+(e.keyHint||'.');
      return "I'm not allowed to use the microphone "+where+" \u2014 the sandbox keeps it switched off, so no permission pop-up can ever appear here. "+
        "Type your question in the box at the bottom of my panel instead! To actually talk with me out loud, save this page's HTML file and open it straight in a normal browser tab. "+
        "Out there the mic works \u2014 and my answers need an API key you provide"+(strings.micKeyHint||'.');
    }
    return "This browser doesn't support speech recognition, so I can't listen here \u2014 but you can type your question in the box below!";
  }
  async function vgExplainMic(){
    if(vgState==='running') vgPauseCore();
    const g=++vgGen; vgSynthCancel(); vgUpdateBar();
    vgLogAdd(vgMicUnavailableMsg());
    try{ vg$('vgIn').focus(); }catch(e){}
    const e=vgElsewhere();
    try{ await vgSpeak(vgMicPolicyBlocked
      ? (e ? "The microphone is switched off here, so type your question in the box below, or open "+(e.spoken||e.label)+" in a normal browser tab to talk with me out loud."
           : "The microphone is switched off inside this sandbox, so type your question in the box below \u2014 or run the standalone file in a normal browser tab, where the mic works with your own API key.")
      : "I can't listen in this browser \u2014 type your question in the box below instead.", g); }catch(e){}
  }
  function vgListen(){
    if(!vgCanListen){ vgExplainMic(); return; }
    if(vgListening){ try{ vgRec.stop(); }catch(e){} return; }
    if(vgState==='running') vgPauseCore();
    vgGen++; vgSynthCancel(); vgUpdateBar();
    vgRec=new VG_SR();
    const nl=navigator.language||'en-GB';
    vgRec.lang=/^en/i.test(nl)?nl:'en-GB';
    vgRec.interimResults=true; vgRec.continuous=false;
    vgListening=true; vg$('vgMic').classList.add('vg-live');
    vgStatus("I'm listening \u2014 ask me anything!");
    let final='';
    vgRec.onresult=e=>{ let s=''; for(const r of e.results){ s+=r[0].transcript; if(r.isFinal) final=s; }
      vgStatus('\u201C'+s+'\u201D'); };
    vgRec.onerror=e=>{ vgListening=false; vg$('vgMic').classList.remove('vg-live');
      if(e.error==='not-allowed'){ vgStatus(''); vgLogAdd("The browser blocked the microphone \u2014 check the mic icon in the address bar, or type your question below."); try{ vg$('vgIn').focus(); }catch(_){} }
      else vgStatus("I didn't catch that \u2014 try the mic again, or type below."); };
    vgRec.onend=()=>{ vgListening=false; vg$('vgMic').classList.remove('vg-live');
      if(final.trim()){ vgStatus(''); vgAsk(final.trim()); } };
    try{ vgRec.start(); }catch(e){ vgListening=false; vg$('vgMic').classList.remove('vg-live'); vgStatus(''); try{ vg$('vgIn').focus(); }catch(_){} }
  }

  /* ---------- panel wiring ---------- */
  function vgUpdateBar(){
    const main=vg$('vgMain');
    if(vgState==='running'){ main.innerHTML='&#10073;&#10073;'; main.title='Pause the tour'; }
    else if(vgState==='paused'){ main.innerHTML='&#9654;'; main.title='Resume the tour'; }
    else if(vgState==='done'){ main.innerHTML='&#8635;'; main.title='Replay the tour'; }
    else { main.innerHTML='&#9654;'; main.title='Start the spoken tour'; }
    const live=vgTourLive();
    vg$('vgSkip').style.display=live?'flex':'none';
    vg$('vgEnd').style.display=live?'flex':'none';
  }
  vg$('vgMain').onclick=()=>{ if(vgState==='running') vgPause();
    else if(vgState==='paused') vgResume();
    else vgStart(); };
  vg$('vgMic').onclick=vgListen;
  vg$('vgGo').onclick=()=>{ const q=vg$('vgIn').value; vg$('vgIn').value=''; vgAsk(q); };
  vg$('vgIn').addEventListener('keydown',e=>{ if(e.key==='Enter'){ e.preventDefault(); vg$('vgGo').click(); } });
  vg$('vgSkip').onclick=vgSkipStep;
  vg$('vgSave').onclick=function(){          // save the conversation as markdown
    const stamp=new Date();
    const md='# Guide conversation \u2014 '+A.id+' \u2014 '+stamp.toISOString().slice(0,16).replace('T',' ')+'\n\n'+
      vgTranscript.map(e=>(e.who==='you'?'**You:** ':'**Guide:** ')+e.text).join('\n\n')+'\n';
    const blob=new Blob([md],{type:'text/markdown'});
    const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
    a.download='guide-'+A.id+'-'+stamp.toISOString().slice(0,10)+'.md';
    document.body.appendChild(a); a.click();
    setTimeout(()=>{ try{ URL.revokeObjectURL(a.href); }catch(e){} a.remove(); },500);
  };
  vg$('vgEnd').onclick=vgEndTour;
  vg$('vgHide').onclick=()=>{ vgSetPanel(false);
    if(ui.hideToast) notify(ui.hideToast);
    const t=vg$('vgTab'); t.classList.add('vg-talk');
    setTimeout(()=>t.classList.remove('vg-talk'),4000); };
  vg$('vgTab').onclick=()=>vgSetPanel(true);
  if(!vgCanListen){ vg$('vgMic').classList.add('vg-off');
    vg$('vgMic').title=vgMicPolicyBlocked?'The microphone is blocked in this sandbox \u2014 press for details':'No speech recognition here \u2014 press for details'; }
  vgUpdateBar();
  let vgWasOpen=true; try{ vgWasOpen=localStorage.getItem(STORE_KEY)!=='0'; }catch(e){}
  vgSetPanel(vgWasOpen);
  if(ui.greeting) vgLogAdd(typeof ui.greeting==='function'?ui.greeting(vgCanListen):ui.greeting);

  /* ---------- the api handle (returned; also handed to init,
     verb execs, and available to step closures) ---------- */
  const api={
    wait:vgWait, demoMs:vgDemoMs, clickSel:vgClickSel,
    spot:vgSpot, respot:vgRespot, unspot:vgUnspot,
    speak:vgSpeak, logAdd:vgLogAdd, status:vgStatus,
    ratify:vgRatify, showConfirm:vgShowConfirm, notify,
    emit:vgEmit, on:vgEvOn, off:vgEvOff, waitFor:vgWaitFor, track:vgTrack,
    explain:vgExplainTopic, explainable:vgExplainable,   // v2 event bus
    keylessCall:vgKeylessAnthropic,
    get inClaude(){ return vgInClaude; },
    get micPolicyBlocked(){ return vgMicPolicyBlocked; },
    get canListen(){ return vgCanListen; }
  };

  /* test hook (smoke suites poke internals through here) */
  window.__guide={ do:vgDo, snap:()=>prov.snapshot(), diff:o=>prov.diffRemark(o),
    pauseCore:vgPauseCore, ask:vgAsk,
    spot:vgSpot, unspot:vgUnspot, steps, system:vgSystem,
    start:vgStart, end:vgEndTour, skip:vgSkipStep, pause:vgPause, resume:vgResume,
    setWordMs(ms){ vgWordMs=ms; },
    get state(){ return vgState; }, get cur(){ return vgCur; },
    get pauseSnap(){ return vgPauseSnap; }, get qa(){ return vgQA.slice(); },
    get micPolicyBlocked(){ return vgMicPolicyBlocked; }, get inClaude(){ return vgInClaude; },
    get keylessModels(){ return KEYLESS.slice(); },
    get layout(){ return {panelWidth:W,panelMargin:M,narrowMax:NARROW,appBreakpoint:APPBP,
      bands:BANDS.map(b=>b.max)}; },
    /* app top-level let/const bindings are global-lexical, not
       window properties; the adapter surfaces them for headless suites */
    get app(){ return testHook.app?testHook.app():{}; },
    emit:vgEmit, waitFor:vgWaitFor, track:vgTrack, evOn:vgEvOn, evOff:vgEvOff,   // v2 bus, for suites
    get evQueueLen(){ return vgEvQueue.length; },
    get evWaiterLen(){ return vgEvWaiters.length; },
    get speakDepth(){ return vgSpeakDepth; },
    get transcript(){ return vgTranscript.slice(); } };

  if(A.init){ try{ A.init(api); }catch(e){ console.error('guide adapter init:',e); } }
  return api;
}

window.VoiceGuide={version:2,fragments,create};
})();

