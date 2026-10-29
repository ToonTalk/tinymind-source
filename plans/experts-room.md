# Plan: a mind made of experts (mixture-of-experts room)

*Drafted 2026-10-29. Status: proposal. Phase 0 decides whether to build the room.*

## Verdict

**Worth doing, but only after a feasibility gate.** Every mind in Tiny Mind today is dense: every word passes through every part. A mixture-of-experts (MoE) mind is different. In each layer, a small *router* sends each word to two of eight *experts*. That gives children two things the dense minds can't:

- a visible decision for every word ("the router sent *dragon* to experts 5 and 2");
- a clean on/off switch for a whole expert, after which the router sends the work elsewhere.

It is also timely: many of today's large AIs are built this way.

The condition in the question, whether each expert can be characterized, is genuinely open at this size:

- Herbst et al. (ICML 2026, "The Expert Strikes Back") find that expert neurons are less mixed than dense ones, and more so as routing gets sparser.
- They also find that experts are "neither broad domain specialists nor simple token-level processors." Mixtral's report found mostly syntactic routing.
- I found no published TinyStories-sized MoE with expert-level analysis. The closest is a 5-layer MoE trained partly on SimpleStories, which learned fewer but more specific features than its dense twin.

So the answer is plausible but unproven. Phase 0 measures it before any room is built.

## What a child would see (if the gate passes)

**Default view.** Short intro, four cards:

1. **The story.** Pick or type a beginning and the mind writes on. Each word is tinted by the expert that handled it most, in a layer you choose (the middle layer by default).
2. **Meet the experts.** Eight cards per layer. Each has a kid-words name ("the names expert", "the noises expert"), the five words it handles most, and how busy it is.
3. **👁 Watch an expert.** Pin one. As the mind writes, the words that expert handles glow.
4. **🔌 Switch an expert off.** The router can no longer choose it, so the work goes elsewhere. The mind rewrites the story, and changed words are marked (reusing the Thought bubble's diff).

**Details on click:**
- the router's vote bars for any word (all eight experts, with the chosen two highlighted);
- every layer at once;
- "switch off and reroute" compared with "switch off and silence" (the Hunt-style mute).

**Challenges** (hard for frontier models too, per the glass-box rules):
- Find the one expert whose absence makes the mind forget names.
- Stop the mind writing dialogue by switching off a single expert.
- Which layer's experts care about meaning, and which about spelling?

## Phase 0: does each expert have a job? (offline, about 2 days, mostly unattended)

**Train.** Use PyTorch on the RTX A4000. torch 2.5.1+cu121 with CUDA already works in `jlens-tinystories/.venv`; give this work its own venv.
- **Architecture:** llama2.c stories15M (dim 288, 6 layers, 6 heads, the same 32,000-piece tokenizer). Tiny Mind's tokenizer, story vocabulary and lens tools then carry over unchanged.
- **The change:** every feed-forward block becomes 8 experts (hidden size 192 each), a top-2 router and a load-balancing loss.
- **Size:** about 19M parameters, about 76 MB as f32. That is under the 100 MB Pages limit, so it can sit on the site and in the artifact asset store.
- **Active parameters per word:** about the same as stories15M.
- **Training run:** one pass over TinyStories (about 470M pieces). I estimate 2–4 hours.
- **Dense twin:** train one at the same active size and budget. It is the control for quality, and for the Goodfire question below.
- **Optional:** a top-1 variant if it's cheap. Sparser routing tends to give cleaner experts, but rerouting is harsher.

**Characterize all 48 experts** (6 layers × 8):
1. **Routing profile.** Which words and contexts each expert receives, measured by lift over the layer average (Jensen-Shannon distance from the layer's distribution, as Herbst et al. do).
2. **Label.** gpt-5.4-mini reads the expert's top 40 routed contexts and names its job in kid words.
3. **Score the label.** On held-out stories, a model that sees only the label predicts which words go to that expert. Report balanced accuracy against chance (OpenAI's explain-neurons scoring).
4. **Switch-off effect.** Measure how much worse the mind does at predicting each word with the expert removed and rerouted. Which kinds of words suffer, and does a written story change visibly?

**Gate.** Go only if all three hold:
- (a) The MoE's held-out loss is within about 5% of the dense twin, and its stories read as well.
- (b) In at least 3 layers, at least 60% of experts have a label that scores 0.75 balanced accuracy or better.
- (c) At least 4 experts change stories, when switched off, in a way you can describe in one sentence.

If (b) fails only because the experts are purely token-level (punctuation, "the"), stop and decide. The room would still teach something ("experts split the work by kind of word, not by topic"), but it would be thinner.

**Fallback if the trained MoE writes poorly: "MoEfication"** (Zhang et al., 2022). Cluster the Big Storyteller's 768 feed-forward neurons per layer into 8 teams by how they fire together, and name the teams from the existing neuron atlas labels. There is no real router, so no rerouting. That is a weaker answer to the question, so use it only if the real MoE fails on quality.

**Deliverables:**
- `tools/moe/train.py`;
- `tools/moe/export.py`, which writes llama2.c `.bin` plus an extension block for the router and experts;
- `reports/moe/feasibility.md`, with the numbers and all 48 expert cards.

## Phase 1: engine (about 1–2 days)

- **Format.** The llama2.c header plus an extension block (n_experts, top_k, expert_hidden), sniffed in `src/ingest.js` like the 260K and 15M headers.
- **Forward pass.** In `src/model.js` and `src/batch.js`, run the router and experts. `batch.js` already goes layer by layer over the whole story, so each expert does one matmul over the words routed to it, and the Pool splits experts across workers.
- **Taps and switches.**
  - Tap the router logits for every layer and word.
  - An `off` set masks router logits before the top-2 choice (reroute).
  - A `silence` set zeroes an expert's output.
- **Tests (`tests/e18-moe.test.js`).**
  - Greedy text and logits match PyTorch fixtures, like e13 and e14.
  - Results are bit-identical for any number of workers.

## Phase 2: the room (about 2–3 days)

- **`experts/`** (index.html, worker.js, guide.js) with the four cards above. Reuse the Thought bubble's story strip, diff and live generation.
- **Shell:**
  - act card "👥 Meet a mind made of experts";
  - tab `experts`;
  - `fitsRoom` = the mind has experts;
  - mind-menu name "the Expert Storyteller".
- **Guide:**
  - TERMS: router, expert, top two, reroute;
  - EVENTS: storyRead, expertWatched, expertOff;
  - STEPS.
  - The status includes the expert cards' labels, so it can answer "what does expert 5 do?"
- **ai-check.** Add the room to the guide sweep, with questions like "what's an expert", "why did my story change" and "which expert handles names".
- **Ship** the usual way. In the artifact, the 76 MB mind goes up as about 7 asset parts.

## Phase 3: optional

- **Hunt.** Add "expert" as a clue kind (switch off one expert in one layer), so the detective can propose it.
- **Microscope.** Neuron cards inside an expert. Are expert neurons cleaner than the dense twin's? That ties to Goodfire's "already clean" essay and to Herbst et al.

## Risks

- **Experts don't characterize.** The gate catches this before any room is built.
- **Router collapse** (a few experts get everything). Use the load-balancing loss and watch expert usage during training.
- **Small switch-off effects,** because top-2 routing is redundant. Compare top-1.
- **A 76 MB download.** Fine on the site; in the artifact it means about 7 asset parts.

## Effort

- **Phase 0:** about 2 days, most of it unattended GPU time.
- **Phases 1–2:** about 4–5 days.
- **Phase 3:** optional.
