# Tiny Mind

Live at **https://toontalk.github.io/tinymind/**. This is its source; `tools/build-site.mjs` builds the
site into [ToonTalk/tinymind](https://github.com/ToonTalk/tinymind), which holds the page, the models it
loads, the guides and the study reports.

Constructionist interpretability app. Spec: `tiny-mind-kickoff.md`. Phase 0 results:
`report.md` (all gates GO). Session A delivered the **Microscope** (`microscope/`):
file-drop a model, run a story, then use five instruments — guess-o-meter (probability
bars), logit lens, attention heatmaps + story overlay, neuron finder with activation
highlighting, and the ablation lab (baseline-vs-heads-off comparison; "the picture
suggested it, the ablation proved it"). Works on the 15M specimen and the 260K trainee
(including the fine-tuned `stories260K-zog.bin`).

## Run

```
node tests/run-all.js            # full suite; final PASS count = build fingerprint
node tests/run-all.js e2         # one experiment (substring filter)
node tools/serve.js 8137         # http://localhost:8137/microscope/ (app) or /harness/ (E6)
node tools/e5-sweep.js           # ~70 min: lr probe + replay sweep, writes e5-results.json
sh tools/make-fixtures.sh        # (WSL/gcc) regenerate golden C fixtures
node tools/build-wordlist.js     # regenerate corpus/wordlist.json
node tools/gen-corpus.js         # validate corpus/zog/stories.json -> train/heldout txt
node tools/build-release.js      # single-file apps -> dist/ (see dist/README.md)
node tools/make-arc.js B         # fine-tune arc snapshots -> arc/ (~25 min)
node tools/make-arc.js probeA    # lr probe for the from-scratch arc
node tools/make-arc.js A         # from-scratch arc -> arc/newborn/ (overnight; resumable
                                 #   with --resume; manifest valid at every prefix)
node tools/make-arc-zips.js      # zog-arc.zip, newborn-arc.zip, tiny-mind-starter.zip -> dist/
node tools/eye/ship.mjs          # the Tiny Eye from its Phase 0 export -> eye/ (reports/eye/feasibility.md;
                                 #   training and labeling: tools/eye/*.py, Python + PyTorch)
# the mind made of experts (reports/moe/feasibility.md; training, export and labeling: tools/moe/,
#   Python + PyTorch). Its .bin is not in git: export it, or copy it, into models/
python tools/moe/export.py <ckpt-top1>/moe/final.pt models/stories-moe-top1.bin
python tools/moe/parity.py models/stories-moe-top1.bin tests/fixtures/moe-top1-parity.json   # E18's fixture
```

## Layout

- `src/` — DOM-free modules (bin-reader, tokenizer, model, capture, train); ES modules,
  zero deps, identical in Node and browser.
- `tests/` — `run-all.js` + per-experiment tests; `fixtures/` golden C outputs.
- `tools/` — fixture generator (C), corpus tools, benchmarks, static server, E5 sweep.
- `corpus/` — Zog stories (50+10, validated), wordlist, replay pool, ghostwriter prompt.
- `harness/` — E6 browser harness (file-drop, worker benchmarks, storage shim).
- `microscope/` — Session A app (dev edition: real module imports; the single-file
  artifact edition inlines src/ + worker-from-Blob at release time).
- `nursery/` — Session B app: character sheet, ghostwriter (OpenAI/Anthropic/Gemini/
  paste) with mechanical validation + repair loop, 512-piece alphabet view, live
  training room with the developmental-arc sample timeline, parroting badge,
  storybook save/load, checkpoint export.
- `thought/` — the Thought bubble: the J-Lens Explorer's instrument on Tiny Mind's engine
  (src/thought.js + batch.js): thinking ahead / layer by layer / DIFF readouts over the story
  vocabulary (`thought/vocab.json`), the heat glow, and swap & rewrite. Big and Giant Storytellers.
- `experts/` — the Experts room (plans/experts-room.md, reports/moe/feasibility.md): the Expert
  Storyteller, a mind made of experts (top-1 of 8 per layer; src/experts.js on batch.js). The story is
  colored by the expert each word went to in a chosen layer (layer 1 shown as a warm-up layer), the
  router's vote for any word, "watch the words travel" (each word drops through all six layers, one
  expert per layer, and the next word comes out), the 48 expert cards (`experts/experts.json`, tools/moe/ship.py: gpt-6.1-sol's
  names where they predict the router, "no clear job found" elsewhere), 👁 watch an expert, 🔌 switch one
  off (muted; rerouting is the advanced switch) with the rewrite's changed words marked, and challenges
  from the five confirmed switch-off effects plus one for the router. Its mind is models/stories-moe-top1.bin;
  with `node tools/serve.js 8137`, http://localhost:8137/experts/ opens the room straight from the source (no build).
- `eye/` — the Tiny Eye, a mind that sees drawings (plans/tiny-eye.md): a tiny ConvNeXt-style
  doodle network (src/eye.js, 83 thousand weights, 24 Quick, Draw! categories). Draw pad made the
  way Quick, Draw!'s bitmaps were, guess bars, every stage's neurons lighting up, neuron cards with
  both ends computed live over 5,000 held-out doodles (`eye/tiny-eye-sample.bin`), switch-off;
  "watch it grow" (20 training snapshots, float16, with each snapshot's last-stage summaries in
  `eye/tiny-eye-growth.bin`) and "squeeze it" (`eye/tiny-eye-sq8.*`, 8 last-stage neurons).
- `hunt/` — Session C app: the treasure hunt. Probe a question, get falsifiable
  clues from the built-in or LLM detective, prove or bust them with head/neuron/team
  ablations. Voice + little-kid mode toggles.
- `models/` — downloaded checkpoints/tokenizers + TinyStories-valid (not in git), and the mind made
  of experts, `stories-moe-top1.bin` (76.8 MB; tools/moe/export.py writes it; the top-2 mind
  `stories-moe.bin` is optional, for E18 only). src/bin-reader.js reads its experts block; model.js
  and batch.js run its router and experts, with experts muted or rerouted (E18: PyTorch parity).
- `vendor/llama2.c` — reference implementation (MIT), cloned at Phase 0 time.

## Error reports

On the live site only (toontalk.github.io, never in the claude.ai artifact or on localhost), when
Tiny Mind runs into an error a small notice asks whether to send Ken a report (Sentry). Nothing is
sent unless the visitor presses Send, and Sentry's script is only fetched then. A report has the
error, where in the code it happened, which room, the page's address and the browser. Everything
the site keeps in the browser (stories, storybook, character and mind names, AI keys) and
everything typed in any box is removed first. "Don't ask again" is remembered; the front door then
offers the way back. The code is `shell/tm-report.js` (its `SENTRY_DSN` names Ken's `tinymind`
project), spliced by build-release into the shell and every room, whose frames hand their errors to
the shell. Checks: `tests/e21-error-reports.test.js`.

## License

Tiny Mind is released under the MIT License (see `LICENSE`). The datasets, models and drawings it
builds on keep their own licenses: TinyStories (CDLA-Sharing-1.0), llama2.c and the tinyllamas
checkpoints (MIT), and Quick, Draw! (CC BY 4.0, below).

## Attribution

The Tiny Eye learned from, and shows, drawings from Quick, Draw! (https://github.com/googlecreativelab/quickdraw-dataset):
made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.
