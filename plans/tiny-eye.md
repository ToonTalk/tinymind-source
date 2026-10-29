# Plan: a tiny mind that sees (a doodle room, after Goodfire's "already clean")

*Drafted 2026-10-29. Status: proposal. Phase 0 decides whether to build the room.*

## Verdict

**Yes, as a doodle room, but not as a copy of the essay's ImageNet microscope.** The literal version doesn't fit, for three reasons:

1. **No pictures to show.** The essay's neuron pages are built on ImageNet photos, which can't be redistributed.
2. **No minds to load.** The essay's models are InceptionV1 and ConvNeXts of 92k to 28.6M parameters, trained on ImageNet. As far as I can find, no weights are released, and the companion paper isn't out.
3. **A child can't make the input.** A child can't supply a photo of a dog, but they can draw one.

**Doodles fix all three.** Google's Quick, Draw! dataset has 50 million drawings in 345 categories, as 28×28 bitmaps, under CC BY 4.0 ("This data made available by Google, Inc. under the Creative Commons Attribution 4.0 International license").
- Its categories include **dragon, teapot, cloud, moon and castle**: Tiny Mind's own characters and demo words.
- It also has ambulance and police car, a close cousin of the essay's taxi-and-ambulance pair of opposite poles.
- A doodle network small enough to train in minutes and run instantly in a browser can carry the essay's main lessons.
- The child supplies their own input by drawing it, which fits Tiny Mind's constructionist spirit.

**The essay's lessons to carry over:**
1. **One main job.** Most neurons have one main job, once you look at both poles (what makes a neuron fire most and least).
2. **Offshoots.** A neuron that looks mixed is often one idea that grew offshoots during training ("recursive generalization"). You can watch this across training snapshots.
3. **The negative pole matters.**
4. **Pressure.** Squeeze a layer and its neurons broaden ("what neurons do under pressure").

## What a child would see

**Default view.** Short intro, four cards:

1. **Draw.** Use a finger or mouse, and the eye guesses live with bars ("dragon 71%, castle 12%…").
2. **Neurons light up.** A grid of the last stage's neurons (for example, 64), each as bright as it fires for your drawing. Tap one to open its card:
   - the doodles that make it fire most, and least (the other pole);
   - a name, such as "pointy tops" or "round with a handle";
   - what it votes for.
3. **🔌 Switch neurons off** and watch the guess change.
4. **🌱 Watch it grow.** Scrub training from the first step to the last. A neuron's top doodles change from blobs, to one idea, to that idea plus offshoots.

**Details on click:**
- earlier stages (edges, curves, corners);
- the squeeze comparison: the same eye with 8 rather than 64 neurons in one stage, whose cards come out muddled.

**Challenges:**
- Find a neuron that fires for things with handles.
- Draw something that isn't a dragon but makes the eye say dragon.
- Switch off two neurons so the eye can no longer tell a cloud from a sheep.

## Phase 0: do tiny doodle neurons have one main job? (offline, A4000, about 1–2 days)

**Data.** 24 kid-friendly categories that overlap TinyStories words:
- dragon, teapot, cloud, moon, castle;
- cat, dog, fish, bird, rabbit, frog, owl, sheep;
- tree, flower, house, sun, star, apple, rainbow, snowman, umbrella;
- ambulance, police car.

Use 10k training and 1k held-out drawings each. Put an attribution line in the room and the README.

**Models.** A tiny ConvNeXt-style net, since the essay found that residual connections plus LayerNorm make neurons cleaner:
- **Layout:** stem → 3 stages of [7×7 depthwise conv + LayerNorm + 4× pointwise MLP + residual] → pool → linear.
- **"Neurons":** the MLP hidden channels, the same idea as the story minds' feed-forward neurons.
- **Sizes:** three widths (about 20k, 80k and 300k parameters), plus a squeezed variant with one stage at 8 channels.
- **Snapshots:** keep about 20 checkpoints across training for the growth replay.

**Measure each neuron pole:**
1. **Examples.** The top-16 and bottom-16 held-out doodles.
2. **Label.** A vision-capable model (gpt-5.4-mini) reads a grid image of the examples and names the job.
3. **Score the label.** From the label alone, the model picks the neuron's top doodles out of distractors.
4. **Count.** The share of poles with one main job, compared across sizes and with the squeezed variant.
5. **Growth.** Across checkpoints, does a neuron keep its early job and gain offshoots? Use label stability and overlap of top examples.

**Gate:**
- the chosen size reaches 85% or better on 24 classes;
- at least 60% of last-stage neuron poles pass the distractor test;
- a handful of neurons have a visible growth story;
- the squeezed variant scores clearly lower, so the pressure lesson shows.

If tiny doodle neurons turn out mostly mixed, the room's lesson changes to "small minds share neurons; bigger ones give each idea its own." That is still worth building, but the copy needs rewriting.

**Deliverables:**
- `tools/eye/train.py`;
- weights export (well under 1 MB);
- a 5,000-doodle sample set (about 4 MB raw, about 1 MB zipped), shipped with the room so neuron cards are computed live in the browser, with no precomputed atlas;
- `reports/eye/feasibility.md`.

## Phase 1: engine and room (about 3–4 days)

- **`src/eye.js`.** Conv, LayerNorm and MLP forward pass in JS: under 5 ms per drawing at this size. Taps for every neuron, a switch-off mask, and parity against PyTorch on 100 doodles.
- **`eye/` room:**
  - **Draw pad.** Pointer and touch events, preprocessed the way Quick, Draw!'s simplified bitmaps were made (crop to the bounding box, scale, stroke width). If this doesn't match, guesses suffer.
  - **Cards.** Guess bars, the neuron grid, neuron cards computed live over the sample set, and switch-off.
- **Shell:**
  - act card "👁 A mind that sees your drawings";
  - tab `eye`;
  - its own kind of mind, not a story mind (the mind menu shows "the Tiny Eye"; `fitsRoom` checks the model type).
- **Guide.** guide.js with TERMS, EVENTS and STEPS. Add the room to the ai-check guide sweep.
- **Artifact.** Check drawing and touch against the `artifact-sandbox` skill.

## Phase 2: watch it grow, and squeeze (about 2 days)

- A checkpoint scrubber for the eye, like the Nursery's time machine. Neuron cards follow the scrubber, plus the squeeze toggle.

## Phase 3: teach it your character (about 2–3 days, optional)

- The child draws their character 20 times, and the eye learns a new category in the browser. Train only the classifier head, plus replay of sample doodles; the JS backprop for that part is small.
- Then a mini-hunt: "where does my character live in the eye?" Switch neurons off to find the ones it needs. This mirrors the Nursery and the Hunt for stories.

## Phase 4: the essay's lessons on the story minds (independent, about 2 days)

- **Growth replay for story neurons.** Use the arcs (stories260K from birth, `arc/newborn`) in the Microscope's neuron card.
- **"How clean is this mind?"** A count for the Big Storyteller, from the existing neuron atlas plus a label-scoring pass.

This half needs no new modality and could be done first or on its own.

## Risks

- **Two subjects.** Tiny Mind becomes about stories and drawings. Keep the eye a clearly separate room with its own intro.
- **Preprocessing.** Mismatched drawing preprocessing would ruin the guesses, so it's tested in Phase 1.
- **Coarse neurons.** The essay notes its smallest model's neurons are coarse. The gate catches this, and the copy is adjusted to fit.
- **Labeling cost.** One-time and offline.

## Effort

- **Phase 0:** 1–2 days.
- **Phases 1–2:** about 5–6 days.
- **Phases 3–4:** optional and independent.
