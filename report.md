# Tiny Mind — Phase 0 report

Feasibility experiments per `tiny-mind-kickoff.md`. Machine: Ken's Windows tower
(Node v24.16.0; browser = Chrome via Claude preview; C reference compiled with WSL
Ubuntu gcc 13.3, `-O2`, no OpenMP).

Build fingerprint: `node tests/run-all.js` → **37 PASS, 0 FAIL**.

## Verdicts at a glance

| Gate | Requirement | Result | Verdict |
|---|---|---|---|
| G1 | exact greedy parity JS vs C, ≥200 tok/prompt | exact over 256–300 tok on all 4 fixtures | **GO** |
| G2 | Zog absorbed ≤3000 steps, base coherence held, ≤5 min projected in-browser | absorbed at ~80–100 steps at every replay ratio; 100 steps ≈ 4.3 min in browser worker | **GO** |
| G3 | 15M ≥5 tok/s in browser; 260K fine-tune within G2 budget | 11.1 tok/s; 2.6 s/step @T192 | **GO** |

## E1 — readers

- `src/bin-reader.js`, `src/tokenizer.js`; zero deps, ArrayBuffer-based, Node + browser.
- Parsed configs (both match §4, unknowns resolved):
  - stories15M: dim 288, hidden 768, L6, H6, KV6, seq 256, vocab 32000, shared classifier.
  - stories260K: dim 64, **hidden 172**, L5, H8, KV4 (GQA ×2), seq 512, vocab 512, shared classifier.
- Tensor offsets sum exactly to file size for both checkpoints (legacy freq_cis skip blocks
  handled; `writeCheckpoint` round-trips byte-compatibly).
- Tokenizer roundtrip `decode(encode(s)) === s` over TinyStories lines + UTF-8 edge cases,
  both vocabs. tok512 max piece length 7.
- "Zog" is a piece sequence in both vocabs: 32K → `" Z"+"og"`; 512 → `" "+"Z"+"o"+"g"`.
  Piece-span handling is mandatory (kickoff §2 confirmed).

## E2 — forward parity (G1)

- `src/model.js`: fp32 forward with KV cache, `Math.fround` at every C rounding point.
- Golden fixtures from llama2.c itself (`tools/fixture-gen.c` `#include`s run.c — the
  reference generates its own fixtures; `tools/make-fixtures.sh`).
- 4 fixtures (2 models × 2 prompts, temp 0, fixed length, no BOS stop):
  **exact token match over all 256/256/300/300 positions.** Gate G1: **GO**.
- Node single-thread speeds while at it: 15M ≈ 15 tok/s, 260K ≈ 600 tok/s.

## E3 — activation capture

- `src/capture.js`: per-layer/per-head attention matrices, residual stream after both
  adds, SwiGLU hidden activations, top-k logits per position.
- 15M @ 256 positions: **17.7 MB** of activations (kickoff estimated ~24 MB) — no
  pathological allocation; capture run ≈ 21 s in Node (one-time per story, acceptable;
  browser equivalent works in the harness worker).
- Schema: `tiny-mind-capture-v1`; `toInterpreterJSON()` slices layers/heads/positions —
  2 layers × 2 heads × 3 positions of a 60-token story ≈ **3.5 KB**, LLM-sized.
- Logit lens implemented; at the final layer it reproduces real logits to <1e-4.

## E4 — training loop (G2a)

- `src/train.js`: hand-rolled batched forward/backward (llm.c-style), AdamW,
  global-norm clip 1.0, tied embedding/classifier, GQA-aware attention backward,
  deterministic RNG. No tfjs, no deps.
- Correctness: central-difference gradcheck on every tensor, two configs —
  generic path (HS=4): worst rel err **1.6e-8**; unrolled HS=8 fast path + GQA:
  **3.7e-7**. Trainer forward logits match the parity-gated model.js to 1.4e-5;
  overfit smoke drives one batch 1.54 → 0.001 in 40 steps.
- Speed (Node, steady state, B=8): T=256 ≈ **3.0 s/step**, T=192 ≈ **2.05 s/step**
  (started at 8.9 s; wins: blocked matmul backward, contiguous K/V transpose +
  fully unrolled head-size-8 attention kernels, RoPE tables). V8 profile is now
  ~63% matmul, ~30% attention — near the practical single-thread JS ceiling.

## E5 — Zog absorption + replay sweep (G2b)

- Corpus: 50 train + 10 held-out stories, written to spec §8 and passed through the
  mechanical validator (`tools/build-wordlist.js` top-1500 TinyStories wordlist;
  whitelist only `zog, zog's, eats, dragons`; ≤20-word sentences; 100–180 words).
  First pass: 54/60 rejected (validator works; TinyStories vocabulary is narrower than
  intuition — `lives`, `past`, `five`, `roof`, `fog` are all off-list). After two
  regeneration rounds: **60/60 accepted**. Ghostwritten by the session LLM (kickoff
  allows any provider); `corpus/PROMPT.md` carries the provider-agnostic prompt.
- Data: Zog stories are 298–381 tok512 tokens (mean 344 — the 512-vocab roughly doubles
  word count); streams: zog-train 17.3K, zog-heldout 3.7K, replay 49.7K (150
  TinyStories-valid stories), base-eval 30.7K tokens (60 disjoint stories).
- lr probe (60 steps, replay 25%): 3e-4 / 1e-4 / 3e-5 → chose **3e-4** (fastest Zog
  drop, base drift +0.17 nats at probe end). Even at probe end the greedy sample already
  read "Zog was a small dragon who lived behind the moon."
- Sweep (600 steps each, B=8 T=192, lr 3e-4; held-out-Zog starts 2.090, base 1.469):

| replay | zogHeld floor (step) | base drift at floor | zogHeld ≤1.60 by | @600: zogHeld / base | attributes in greedy samples |
|---|---|---|---|---|---|
| 0%  | 1.539 (80) | +0.44 | step 40 | 2.18 / 3.19 (collapse) | yes, but coherence degrading |
| 25% | 1.528 (80) | +0.22 | step 40 | 1.83 / 2.02 | yes |
| 50% | 1.573 (80) | **+0.16** | step 60 | 1.74 / 1.90 | yes ("eats clouds", "house behind the moon") |

- Findings:
  - **Absorption is fast: the held-out-Zog floor lands at step ~80 at every ratio**
    (~40× under the 3000-step budget), and attribute recall appears in greedy samples
    by step ≤60. The strict criterion "zogHeld ≤ original base loss (1.469)" is never
    met — the floor is ~1.53 — so 50 stories teach *Zog* without making unseen Zog
    phrasing quite as predictable as generic TinyStories text. The right absorption
    signal is the floor + attribute recall, and that's what the app should surface.
  - **Past the floor the model memorizes the 50 training stories**: train loss keeps
    falling (0.24 at step 600, 0% replay) while held-out-Zog *rises* and base decays.
    Early stopping is part of the recipe — and the overfitting arc is itself a
    teachable exhibit for the Nursery (gibberish → grammar → Zog → parroting).
  - **Replay dose-dependently prevents forgetting** (drift at 600 steps: +1.72 / +0.55 /
    +0.43 nats), exactly the epistemic contrast the app wants to demonstrate. 0% replay
    at 600 steps visibly damages coherence ("The bird was a soft little house").
- Recommendation: **replay 50%, lr 3e-4, B=8, T=192, ~100 steps with an early-stop
  on held-out floor** → ≈3.5 min Node, ≈4.3 min in the browser worker (≤5 min ✓).
- Saved artifact: `models/stories260K-zog.bin` regenerated with exactly this recipe
  (`tools/save-recommended.js`); verified file-drop + capture in the harness. Full
  sweep data in `e5-results.json`. Gate G2: **GO**.

## E6 — browser harness (G3)

`harness/index.html` + module Web Worker + `tools/serve.js` (or any static server).
File-drop for all four .bins (type sniffed from the header) plus dev-server shortcuts.
Measured in Chrome on Ken's tower:

- **15M greedy generation: 11.1 tok/s** (220 tokens in 19.8 s, module worker).
  Gate G3 needs ≥5 → **GO** with ~2× margin, no int8/WebGPU fallback needed for the
  specimen. (Same fround-faithful forward that passed G1.)
- **260K fine-tune in worker: 2602 ms/step @ B=8 T=192; 4382 ms/step @ B=8 T=256**
  (~1.3× Node). See G2 for the absorption-time product.
- Attention capture runs in-browser (42-position story: 0.5 MB, heatmap rendered,
  top-logit list sensible).
- **WebGPU: available** on this machine (adapter obtained) — a real WebGPU matmul path
  remains the obvious Phase 1+ speed lever, but is not needed to pass gates on desktop.
  Family-iPad training feasibility remains unverified (kickoff risk accepted).
- Chunked `window.storage` persistence: shim (5 MB/key cap, string values) +
  chunked base64 writer; 260K checkpoint = 1.06 MB → 1×4 MB key (6×256 KB in the
  forced-chunking test); byte-exact round-trip in ~20 ms; meta-written-last commit
  marker so partial saves are never loadable. No localStorage anywhere.

## Parameter recommendations (for Sessions A/B)

- Trainee fine-tune: **B=8, T=192, lr 3e-4, AdamW(0.9/0.999, wd 0.01), clip 1.0,
  replay 50%, ~100 steps + early stop at the held-out floor** ≈ 4.3 min in-browser.
  Show the loss curves live; let the child watch forgetting happen at 0% replay.
- Specimen microscope: capture whole stories at once (≤256 pos); ship interpreter
  JSON slices, never full dumps.
- Persist only the 260K trainee (1 storage key at 4 MB chunks). The 15M specimen is
  file-drop-only (60 MB ≈ 21 keys — legal but pointless; re-drop is simpler).

## Deviations from kickoff

- Corpus ghostwritten by the session assistant rather than a paid API (validator
  unchanged; prompt spec in `corpus/PROMPT.md` is the runtime contract).
- `tools/fixture-gen.c` includes run.c rather than patching it — fixtures come from
  unmodified reference code.
- E4's "coarse lr sweep" ran as E5 phase 1 (needs the corpus; same numbers).

## Risks updated

- G3 wobble did not materialize on desktop (11.1 tok/s specimen; WebGPU present).
- Single-thread JS training is the binding constraint; if Phase 1 wants faster
  iteration: WebGPU matmuls, or 2–4 workers with gradient accumulation.
- iPad: untested, still the plan-B platform for training (inference likely fine).
- License posture unchanged (attribution block per kickoff §12).

---

# Session A addendum — Microscope (same day)

Delivered `microscope/` (dev edition; single-file artifact packaging deferred to a
release step). Five instruments over a file-dropped model + typed story, all computation
in a module Web Worker on the Phase 0 modules:

1. **Story strip** — piece-level view with click/shift-click span selection (piece spans,
   never words: "Zog" is 2 pieces on the specimen, 4 on the trainee).
2. **Guess-o-meter** — exact top-k probabilities (new `logZ` per position), actual next
   piece marked, baseline vs ablated side by side.
3. **Logit lens** — per-layer, after-attention and after-FFN top-3 + p(real next).
4. **Attention** — layer/head heatmap with hover readout; click a row to paint the story
   strip with that position's attention.
5. **Neuron finder** — `neuronContrast` scores (in-span vs out-of-span activation, in
   out-of-span σ units); clicking a neuron paints its activations over the story.
6. **Ablation lab** — layer×head checkbox grid; re-run reports mean/max Δ in the model's
   confidence in the true next pieces ("the picture suggested it, the ablation proved it").

Engine changes were strictly additive: `forward(..., ablate)` (Set of "l:h"; empty/absent
is bit-identical to the G1 path — tested), capture `logZ`/`nextProb`/`onProgress`/ablate,
`neuronContrast`. New e7 test file. Fingerprint: **44 PASS, 0 FAIL**.

Verified live on the fine-tuned trainee (after "Zog": " was" 38%, " li(ves)" 9%,
" named" 9%; L2·n129 tops the Zog-span neuron list at 3.2σ; ablating 2:3+4:0 shifts
true-next-piece confidence 3.0% mean / 24.3% max) and on the 15M specimen (40-piece
story captured in 4.5 s in-browser).

---

# Session B addendum — Nursery (same day)

Delivered `nursery/` (dev edition): the corpus builder + live training room.

- **Character sheet** → whitelist and attribute checks derive from it mechanically
  (any word in the sheet, its `+s` and `+'s` forms are allowed; the kind phrase and
  attr content-word stems are required in every story).
- **Ghostwriter**: provider-agnostic (OpenAI / Anthropic / manual paste), key never
  stored. Tested live against gpt-4o-mini from Node (key from env): parse → validate →
  reject-with-reasons loop works; first-pass rejects are expected and are the design
  (Phase 0's own first pass was 54/60). Parser hardened for markdown numbering and
  curly quotes. Final prompt tuning against keyless Sonnet stays a Session C task.
- **Storybook** manager: 50-learning + 10-secret split (every 6th accepted → secret),
  progress bar, per-story validation feedback in kid terms ("words too big for a tiny
  mind: magnificent, extraordinary, …").
- **Full-alphabet view**: all 512 pieces on one screen, heat = usage by the child's own
  storybook (Phase 0 Zog corpus lights up 202/512).
- **Training room**: replay %, lr, step budget; live chart (train loss + secret-story
  loss + ordinary-story loss); greedy sample timeline = the developmental arc
  (verified: step 0 base model has no Zog → step 20 "Zog was a small dragon who lived
  in…"); parroting badge when held-out loss climbs off its floor; pause/keep-raising
  extends the budget; **start from "empty head"** (random init) shows true gibberish
  ("Zogididididid…") for the full arc. Export downloads a byte-exact legacy-format
  checkpoint (1,056,540 bytes) that file-drops straight into the Microscope.

Also fixed from field testing (Ken dropped all five .bins at once): `tools/serve.js`
now serves directory indexes, and the Microscope accepts drops anywhere on the page,
pairs the tokenizer to the checkpoint by parse-probe against its vocab size, and says
exactly what is missing or mismatched. Fingerprint still **44 PASS, 0 FAIL**.

## Session B follow-up — ghostwriter loop hardening (field test by Ken)

Field test (character "eats rainbows / can fly", gpt-5.4-mini) rejected 8/8. Three root
causes, all fixed and re-verified against the live API:

1. **Attribute checker was word-blind**: it required "can" (a modal) and could not match
   verb forms (`eats→ate`, `fly→flew`). Now: a STOP list of glue words (modals, aux,
   pronouns — position words like "behind/under" deliberately kept), plus verb-family
   matching with the ~35 irregular kid-verbs. "Pip ate rainbows… he flew" now satisfies
   "eats rainbows / can fly".
2. **No regeneration feedback**: rejects now accumulate across batches (short-length
   counts, banned words, missed attributes) and are appended to the next prompt; the
   prompt also demands 120–170 words (models undershoot a 100-min ask), and the full
   top-1500 allowed-word list ships in the prompt (~2.5K tokens; checkbox, default on).
3. **Near-misses regenerated instead of repaired**: stories failing ONLY on ≤4 off-list
   words get a targeted repair call ("swap each banned word for a simpler one, change
   nothing else" — earlier phrasing "keep it EXACTLY the same except…" made models
   return the story unchanged; the fix includes an example substitution).

Measured with gpt-5.4-mini, 6 stories: generate-only 0–3/6 accepted; generate + one
repair call: **5/6 accepted** (the straggler swapped one interjection for another; the
cross-batch ban list catches it next round). That rate reaches a 50-story corpus in
~7 batches ≈ 14 API calls. The keyless-Sonnet prompt pass (Session C) starts from this.

---

# Session C addendum — Treasure Hunt + resident interpreter (same day)

Delivered `hunt/` plus the Nursery round-2 items Ken requested.

**Nursery round 2**: default models gpt-5.4-mini / gemini-3.1-flash-lite (Gemini is a
third provider via the new shared `src/llm.js`); remember-key toggle (localStorage —
kickoff's no-localStorage rule deliberately relaxed for the Pages/dev edition at Ken's
request; artifact edition will use window.storage); the train button is never silently
gray (it says what's missing; a small storybook WARNS instead of blocking — watching a
5-story model parrot is itself the lesson); story list shows full texts; storybook
save/load (.json download/upload + browser autosave/restore); emoji favicons on all
pages. Also fixed a validator false-negative: dwelling verbs are linking words, so
"His home is behind the moon" satisfies "lives behind the moon" (Phase 0 semantics).

**Treasure hunt** (`hunt/`): quest = probe sentence + expected answer; the metric is
the continuation probability of the right answer, teacher-forced. Steps: ask the model
(guess-o-meter), get clues, test by ablation, scoreboard. Engine addition (additive,
gradcheck/parity intact): **neuron ablation** — `ablate` now takes
`{heads, neurons}`; ablated SwiGLU activations read zero in captures.

- **Built-in detective** (keyless): hypotheses from the capture itself — heads by
  attention-to-name at the answer boundary, neurons by name-span contrast, plus the
  **team hypothesis** (a whole layer's heads).
- **LLM detective**: evidence JSON → 3 falsifiable hypotheses (strict JSON contract,
  heuristic fallback on parse failure). Verified live with gpt-5.4-mini: 3/3 valid,
  kid-phrased ("This neuron looks like a big Zog-knower…").
- **The epistemic arc, measured on the fine-tuned Zog mind** ("Zog lives behind the
  →  moon", baseline 55.6%): every single-head and single-neuron clue was **BUSTED**
  (55.6% → 52–56% — distributed knowledge, honest negative results), and the layer-0
  team was **PROVED** (55.6% → 22.8%). "The pretty picture suggested it, the ablation
  proved it" — plus the deeper lesson that the treasure is shared.
- DIY bench (any head/neuron/layer-team), voice guide via Web Speech API behind a
  toggle (vg-core not found on this machine; the `speak()` function is the adapter
  seam), little-kid mode wording.

Fingerprint: **47 PASS, 0 FAIL**. Remaining for release: keyless-Sonnet prompt pass
inside a real artifact, and the single-file packaging of all three apps.

---

# Release addendum — single-file packaging (same day)

`tools/build-release.js` → `dist/tiny-mind-{nursery,microscope,hunt}.html`
(129–144 KB each): all `src/` modules inlined into one shared scope (ESM stripped,
import aliases re-emitted), workers rebuilt from Blobs, `</` escaped for inline
embedding. Both the page bundle and the worker bundle are syntax-checked at build time
— the check exists because it immediately caught a real name collision (the Nursery's
`callLLM` wrapper vs the inlined module's `callLLM`; wrapper renamed `callProvider`).

One file serves both editions: `src/llm.js` gains a **keyless provider**
(`window.claude.complete`) that appears in the ghostwriter/detective menus only inside
a claude.ai artifact; on Pages/local the three keyed providers + manual paste remain.
Weights arrive by file-drop everywhere (dev-shortcut fetch buttons degrade gracefully).
Works from `file://` (Blob workers), any static host, or pasted as an artifact.

All three dist files verified live in the browser: nursery boots, loads model +
storybook, trains, and its arc samples now end at sentence boundaries (samples are
200 pieces, trimmed to the last complete sentence — fixes the mid-word truncation Ken
saw); microscope captures; hunt runs the full quest (single heads busted, layer-team
proved 55.6%→22.8% — same numbers as the dev edition, as they must be).

Also this round: storybook 💾/📂 buttons duplicated next to the storybook header
(they existed below the list; now visible without scrolling).

Fingerprint: **47 PASS, 0 FAIL**. Still open (needs a claude.ai session, not this
machine): the keyless-Sonnet ghostwriter/interpreter prompt pass inside a real
artifact; family-iPad check.

---

# Arc addendum — tools/make-arc.js + Arc B (2026-07-07, arc kickoff session)

Per `tiny-mind-arc-kickoff.md`. Scrubber UI deliberately untouched (later artifact
session receives only the packs + the manifest contract).

**Tool**: `tools/make-arc.js` (B | A | mini | probeA; --resume --lr --out --max-step).
Every snapshot = legacy fp32 .bin + adam-NNNNN.bin (m,v moments) + state-NNNNN.json
(RNG state via new getState/setState on the mulberry32 in train.js, step, recent
losses); `arc-manifest.json` rewritten LAST so the manifest is valid at every prefix.
The batch sampler consumes exactly 2 randoms per row regardless of source count, so
the random sequence is schedule-independent. **Resume determinism is tested**: 30
straight steps vs 20 + interrupt + --resume give byte-identical checkpoints and
identical manifests (e8). Fingerprint **58 PASS, 0 FAIL** (47 → 58, monotone).

**Arc B (fine-tune, done)**: E5 recipe, 600 steps, 9 snapshots in `arc/` (~25 min run).
The samples alone tell the story: step 0 "Zogo was playing…" (no Zog) → 10 "Zog was a
big, small bird" → 20 "small dragon who lived in the sky" → **80 "small dragon who
lived behind the moon"** (the E5 floor, on schedule) → 400/600 attributes intact but
looping, zogHeld 1.57→1.74, base 1.47→1.90. **Open item 2 answer: keep the 400/600
tail** — samples stay kid-presentable; the parroting shows as rising held-out markers
+ tightening loops, which is exactly the lesson. Truncate the manifest later if
preferred.

**Packs** (`tools/zip.js` = zero-dep STORE zip writer/reader, flat entries like Ken's
model zips; `tools/make-arc-zips.js`): `dist/zog-arc.zip` 9.5 MB,
`dist/tiny-mind-starter.zip` 1.31 MB (trainee + tok512 + replay.json — the 15M
specimen stays a separate optional download). `newborn-arc.zip` builds the same way
once Arc A exists (the script skips missing arcs).

**Pages autoload**: the Nursery fetches `arc/arc-manifest.json` at boot (manifest
only; .bins stay lazy) and shows "📼 arc pack: N snapshots" — verified live. Dist
rebuilt with it.

**Open item 3 (where 12 MB of snapshots lives)**: currently `arc/` locally + zips in
`dist/`; repo-vs-release-asset is Ken's call — nothing in the tooling assumes either.

**Arc A probe + launch**: lr probe on the 794K-token general stream (2060 stories,
cached at `arc/cache-tinystories-2000.i32`): after 200 steps from random init, lr 1e-3
reached base 3.39 (phrase-shaped babble) vs 4.13 for 3e-4 (syllable loops) — **1e-3
chosen**, no divergence. **Open item 1 answer: the 2000-story slice suffices** (20K
steps × 2048 tok/step ≈ 52 epochs). Arc A launched overnight (~11.5 h at ~2.07 s/step);
early snapshots already staging (step 20 = all dots, step 50 = all e's). Every snapshot
is written as reached and the manifest is prefix-valid, so stopping at dawn is safe.
If the run dies (sleep/reboot): `node tools/make-arc.js A --resume` continues
byte-exactly; then `node tools/make-arc-zips.js` builds newborn-arc.zip.

---

# Mega merge addendum (2026-07-07)

The claude.ai artifact session produced a "mega edition" (shell + the three apps in
iframes) from the 2026-07-05 release. Merged back per its handoff instructions: two-way
diff of each extracted app vs its dist counterpart; patch-layer hunks (storage /
worker-boot / bridge / ingest / egress) adopted into dev sources, arc-session hunks
(rng state accessors, nursery arc badge) kept — the change-sets were disjoint as
predicted, except one name collision the build's syntax check caught.

Adopted into dev (now the single source of truth; the mega is REGENERABLE):
- `src/app-shims.js`: same-thread worker shim (the artifact sandbox refuses blob-URL
  Workers), Store cascade (window.storage → localStorage → memory, hydrate-then-read),
  deliverFile egress ladder (bridge → download+_blank → clipboard → storage backup),
  fetchFirst dual paths (flat Pages layout OR dev-server layout).
- `src/ingest.js`: universal ingestion — zips (STORE + DEFLATE via DecompressionStream),
  folder drops, parse-based bin sniffing (tokenizer table walk / header check),
  DataTransfer snapshot-before-await.
- `src/embed-data.js` (+ generator): embedded top-1500 wordlist and a 20-story base
  eval set so ghostwriting and the forgetting line work with nothing hosted.
- App seams: frame-bridge hooks (__tmLoadPair/__tmRouteJson/model-loaded), hidden dev
  buttons when framed, restart-from-original + save-samples (nursery), name-control
  experiment + continuation breakdown + "1 in N" formatting (hunt).
- `shell/`: the shell (model pool with vocab-matched tokenizer pairing, onboarding
  cards, RPC hub, export bubbling with pool auto-join, eager sibling autoload) +
  bridge.js. Extended per the arc kickoff: the shell now autoloads
  `arc/arc-manifest.json` (manifest only) and stashes it as `window.__tmArcManifest`
  for the future scrubber session.
- `tools/build-release.js`: injects the bridge into every single, swaps the dev
  worker-boot markers for the blob+inline-fallback boot, and composes
  **dist/tiny-mind-mega.html** (564 KB) from shell + built singles. All bundles
  syntax-checked at build.

Verified live from the regenerated mega: starter-zip drop → pool → framed Nursery via
load-pair (dev buttons hidden) → Zog storybook via fetchFirst fallback → training →
export bubbled to the shell, downloaded, and auto-joined the pool ready for the
Microscope/Hunt. e9 covers mega composition, sniffers on real files, and STORE+DEFLATE
unzip in Node. Fingerprint **70 PASS, 0 FAIL** (58 → 70).

The copy at Documents/AI/apps/Tiny Mind/index.html is now superseded by
dist/tiny-mind-mega.html (same architecture + the arc-session changes it predated).

---

# Guided-build merge addendum (2026-07-08)

The guided build (voice guide + scrubber + audit-v2 ingest, built on our regenerated
mega) is merged back; the mega is again fully regenerable from repo sources.
Adopted: shell/vg-core.js (v2 — newer than the copy in Documents/GitHub/misc),
shell/tm-prelude.js (Worker tap + emit buffer + pref cache), bridge v2 (tm:'arc'
push + __tmRPC escape hatch), per-app guide adapters ({app}/guide.js +
shell/tm-shell-adapter.js), src/scrubber.js (time-machine core, Node-tested per its
own contract), three-phase ingest (expandZips/ingestBatch), the Nursery time-machine
UI, probe-fact in the nursery worker, and the shell's arc RPC serving + arc cards.

Suites: **87 PASS, 0 FAIL** (70 → 87; e10 scrubber/ingest + e9 guided assertions).
The instruction's specific question — "the probe-fact branch is additive but it's
inside the worker blob, so the parity gate should say so itself" — is answered: with
probe-fact shipped in the final worker blobs, G1 parity stays byte-exact on all four
fixtures and both gradchecks hold (1.6e-8 / 3.7e-7).

Pre-ship catches (all fixed here, none present in shipped output):
1. OUR build corrupted the hunt worker via String.replace $-patterns during
   composition — after the syntax checks ran. Function-form replaces everywhere;
   e9 now re-extracts and compiles every worker from the FINAL file.
2. THE GUIDED build's shell RPC dispatcher: the vg pref-sync was inserted as a bare
   `if`, breaking the else-chain — claude.complete and all non-guide storage RPCs
   threw "unknown rpc" after doing their work (silently: Store.persist swallows).
   Ported with the sync inside the storage branch.
3. Our first shell-vg splice anchored on the first </body>, which after template
   insertion belongs to an app doc. Splice reordered + build guard added.

Verified live from the regenerated mega: shell guide panel + 4-stop starter tour
(speaks the starter URL), zog-arc.zip drop → bay lists arcB → nursery overlay offers
the time machine → scrub (manifest-only preview) → release loads the step-80 snapshot
via arc.snapshot/arc.tokenizer RPC ("= arcB step 80") → probe-fact on that mind:
"clouds" = 5 pieces, p(clouds | "Zog likes to eat") = 0.130.

**Starter URL — NOT confirmed, needs Ken.** The default in tm-shell-adapter.js is a
placeholder (its own comment says CONFIRM URL): https://toontalk.github.io/AI/apps/
tiny-mind-starter.zip → 404; toontalk.github.io does not exist (root 404), and the
local GitHub/ai clone is ecraft2learn/ai (different org/site). Options: (a) create a
toontalk.github.io user-site repo mirroring Documents/AI/apps layout, (b) host under
an existing site and set the default accordingly, (c) attach the zip as a release
asset. Runtime override exists: window.TM_STARTER_ZIP.

---

# Guide waves 1-4 + ablation-aware lens (2026-07-08)

Two things this pass. **(d) the ablation-aware logit lens** (the one item the artifact
session deferred because it lives in the parity-gated worker blob): a lens request now
carries the currently-muted head set; the microscope worker recomputes and caches (by
head-set) an ablated capture over the same tokens and projects THAT residual stream, so
the lens shows the prediction forming inside the damaged model. The UI badges "with L0:0,
L0:1 muted" and refreshes the lens in place when the child runs or clears an ablation.
Additive: empty mute is bit-identical to the un-ablated lens (e7 asserts), G1 greedy
parity stays exact on all four fixtures, gradchecks hold. Live-verified: full lens →
muted lens (values shift, e.g. `" a"` 23%→36%) → cleared back to full.

**Guide waves 1-4 merged into repo sources.** The artifact session shipped its guide
layer as `patch-tiny-mind-dist.mjs` (LLM-failure copy + `strings.llmFail` seam,
save-transcript button, multi-provider key reuse, artifact-resilient boot, list-first
pref reads, shell guide auto-hide/margin-release, bay persistence via shellStorage, the
`trainedModels` origin fix so exported minds appear in the Hunt, kid-friendly time-machine
card, nursery character persistence, microscope story prefill). To keep the repo the
single source of truth, I replayed the patch's 49-op mega plan against the **sources**
(auto-routing each edit to the file holding its anchor). Verification: my regenerated
`nursery/guide.js` and `microscope/guide.js` come out **byte-identical** to the session's
attached whole-file adapters, and the session's own jsdom smoke test (`tests/smoke-guided.mjs`,
`npm run smoke`) passes **71/0** against my rebuilt mega — the same result it reported for
v4. One genuine conflict fixed: the storage `list` op assumed no prior method, so it
duplicated the one the earlier guided merge added; deduped to a single canonical `list`.

Suite: **89 PASS, 0 FAIL** against the rebuilt guided mega (parity + gradcheck + mega
structure). The patch script is now obsolete for building (waves live in sources; the
mega regenerates complete) and is kept only as historical spec. **Starter URL CONFIRMED**:
`https://toontalk.github.io/misc/tiny-mind-starter.zip` serves a real zip (PK magic,
`application/x-zip-compressed`, 2.0 MB). The site root 404s (no index) but the file path
resolves. Caveat: the hosted copy is 2.0 MB vs the current `dist/tiny-mind-starter.zip`
at 1.3 MB — the hosted pack is a different/older composition and should be refreshed.

# Whole-microscope ablation (2026-07-08)

Extended (d) from the lens to every instrument: one ablation state drives the whole
bench. The worker now keeps a single ablated capture (keyed by the muted-head set over
the current story) that serves the lens, the neuron finder, neuron-activation overlays,
AND the ablated run — which now returns a FULL capture (attention transferred) so the
attention heatmap and story-strip overlay can switch too. Running the ablation lab flips
the microscope into "ablation view": a red banner names the muted heads; attention shows
the damaged model (the selected head, if muted, is drawn pink and labelled MUTED, and
downstream heads visibly shift because their residual changed); neurons re-rank inside
the damaged model; the lens projects the ablated residual; the bars keep the blue-vs-pink
compare. Clear (or a fresh full run) reverts every view. Additive and parity-safe: empty
mute is bit-identical to the full model (e7), G1 greedy parity stays exact on all four
fixtures. e7 adds neuron-ranking-shifts-under-ablation and the attention invariant (an
untouched L0 head's pattern Δ=0 while a top-layer head moves). Suite **91 PASS**; smoke
71/0. Live-verified: full → ablate (banner + lens/neuron/attention/bars all the damaged
model) → clear (all revert).

# Guide waves 5-6 (2026-07-08)

Two more guide-layer patches from the artifact session, merged to sources the same way
(replay each patch's mega op-list — w5: 11 pairs + 1 region; w6: 13 pairs + 1 region —
auto-routed to the source files; regenerated prelude + 4 adapters verified byte-identical
to the session's attached whole files, one doc-only comment adopted from theirs). W5:
kid-friendly odds ("3 out of 100", dot grids); shell-guide CORS fix (reuse the
ghostwriter key rather than a keyless fetch with no header); shell status leads with the
active tab; bay persistence chunked ~350 KB with read-back + self-heal. W6: storage
writes serialized through a queue (artifact answers concurrent writes with 409, which is
why the pool never persisted while small guide keys did) with delete-then-create on
conflict; every guide shares one ask path that finds app-remembered keys
(tm-key-<provider>). **Nothing entered a worker blob** — the gate confirms: G1 greedy
parity exact on all four fixtures, gradchecks 1.6e-8/3.7e-7, suite **91 PASS**. The
session's jsdom smoke test (now `tests/smoke-guided.mjs`, updated for w5/w6) passes
**100/0**, including the new 409-conflict and key-reuse scenarios. Both patch scripts are
obsolete for building (changes live in sources; the mega regenerates complete).

# Guide waves 7-14 + worker W14-2 (2026-07-09)

The repo was 8 waves behind the artifact session (sources at wave 6; the delivered
`tiny-mind first guide files v9` dist was wave 14, with waves 7-13 never handed over as
patch scripts). Rather than replay eight missing op-lists, the sources were
**reconstructed from the authoritative v9 dist** — the build transform is deterministic
and reversible: strip the GENERATED comment, the injected prelude/bridge/vg-core/guide
scripts, and (in each app's module script) the inlined-src prefix + worker-blob, then
restore the import block and `@tm-worker-boot` marker. The 7 guide files (vg-core,
prelude, bridge, tm-shell-adapter, three app adapters) were adopted **byte-identical** to
v9's injected copies (verified by re-wrapping each in the build's injection template and
matching it in the v9 dist). The shell's own main script was reconstructed the same way
(shellShared bundle byte-identical, so the inlined block maps straight back to
`@@SHELL_SHARED@@`). One `src/` line changed across all eight waves: `scrubber.js`
`KID_LABELS.arcB` generalized from "the trainee learning Zog…" to "a mind learning your
character…" (v9's mega + hunt carry the new text; the stale single nursery/microscope
dist still had the old text — v9 had drifted, so the repo takes the mega's newer wording
and v9's own smoke asserts the Zog phrasing is gone). **Self-validation: the rebuilt mega
is byte-identical to v9** modulo escaped-vs-literal em-dash (cosmetic) and inert
import-alias/blank-line noise from v9's hand-build drift; **all three worker blobs are
byte-identical to v9** (nursery 79858, microscope 57909, hunt 58455 chars).

Then the **W14-2 worker action item** (the one item the wave-14 patcher explicitly
deferred to Claude Code, being parity-gated): the Nursery time-machine "tell this story"
with a custom prefix hung forever because `sample-now` was guarded by `if (trainer)` and a
checkpoint loaded via the scrubber goes through `init` (trainer = null). Fix in
`nursery/worker.js`: `sampleText()` now falls back to the loaded checkpoint's own weights
(`baseCkpt.weights`) when there is no live trainer, so a scrubbed moment writes a fresh
story from its own weights; `sample-now` posts only when weights exist. The honest-fallback
setTimeout in `nursery/index.html` (the wave-14 stopgap) was removed — the story renders
live. This intentionally changes ONE worker blob (nursery); **G1 greedy parity stays exact
on all four fixtures** (the change adds a weights source, not a decode path) and the
microscope/hunt blobs are untouched.

**Gates:** suite `node tests/run-all.js` **91 PASS, 0 FAIL** — G1 exact greedy parity
(256/256, 256/256, 300/300, 300/300; gate ≥200), gradchecks 1.61e-8 (generic) / 3.69e-7
(HS=8 fast path), e7 ablation invariants, e10 scrubber. The e9 arc-autoload assertion was
updated from the pre-wave-7 single-manifest URL (`arc/arc-manifest.json`) to the
wave-7-14 multi-manifest discovery (`arc/arcs.json` + `manifest.more`). jsdom smoke
`npm run smoke` (now v9's `tests/smoke-guided.mjs`, 161 checks) passes **161/0** against
`dist/tiny-mind-mega.html`; its lone wave-14 "honest fallback" check was replaced with a
wave-15 check that the worker falls back to checkpoint weights and the stopgap text is
gone. The two v9 patch scripts (`patch-tiny-mind-w14.mjs`, `tm-nursery-adapter.js`) are
obsolete for building — the changes live in sources and the mega regenerates complete.
