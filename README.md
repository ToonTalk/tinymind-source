# Tiny Mind

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
- `eye/` — the Tiny Eye, a mind that sees drawings (plans/tiny-eye.md): a tiny ConvNeXt-style
  doodle network (src/eye.js, 83 thousand weights, 24 Quick, Draw! categories). Draw pad made the
  way Quick, Draw!'s bitmaps were, guess bars, every stage's neurons lighting up, neuron cards with
  both ends computed live over 5,000 held-out doodles (`eye/tiny-eye-sample.bin`), switch-off.
- `hunt/` — Session C app: the treasure hunt. Probe a question, get falsifiable
  clues from the built-in or LLM detective, prove or bust them with head/neuron/team
  ablations. Voice + little-kid mode toggles.
- `models/` — downloaded checkpoints/tokenizers + TinyStories-valid (not in git).
- `vendor/llama2.c` — reference implementation (MIT), cloned at Phase 0 time.

## Attribution

The Tiny Eye learned from, and shows, drawings from Quick, Draw! (https://github.com/googlecreativelab/quickdraw-dataset):
made available by Google, Inc. under the Creative Commons Attribution 4.0 International license.
