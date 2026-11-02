# J++ vs the J-lens on the TinyStories models

**Question (Ken, via the Thought Bubble session):** can J++ (Ayonrinde & Lindsey, 2026, a "Jacobian filtering" upgrade of the J-lens) be an option in the Thought Bubble app, or replace the J-lens if it is better? And does it make stories15M good enough for a thought bubble in Tiny Mind's Microscope?

**Short answer: not worth it, for either model.** The shipped J-lens stays. J++ was fitted exactly as the paper describes, with TinyStories in place of WikiText and a hand-built TinyStories dev set for its weights. At 110M it does raise target-word recall in the app band (+8 to +12 points) and reads names better. But it breaks the bottom band layer (L4 shows the same off-register names on every prompt), gains nothing below it, shows fewer of the app's signature readouts, is no better at swaps, and changes half its display between two fits. At 15M it makes things worse, so it does not open the door to a Microscope thought bubble. Offering it as a toggle would add exactly the kind of unexplained part Ken's design rule rules out. No `.jlens` export is proposed.

## Verdict

### stories110M: keep the J-lens. J++ is not worth it, not even as an option.

What J++ gets right, measured on held-out TinyStories items:
- **Recall goes up in the app band (L4–L8).** At the period: 64% vs J's 52% (56% with the filter alone). At the last word: 37% vs 31%. The answer also appears a layer earlier (L5: 15% vs 0%), the pattern the paper reports.
- **Names read better**: ` L` (Lily's first piece) is in the top 10 at 8 of 11 Lily positions, vs 5.
- Some L5–L8 bubbles are richer, e.g. the persistent bird: `effort, struggle, eventually, determined`.

Why it still doesn't earn a place:
1. **The bottom of the band breaks.** At L4, J++ shows the same off-register list on every prompt (`Nelson, Ny, Kent, Aless, Stein…`). Its overlap with the logit lens there is 0.01 and its label-free precision collapses (0.2 vs 0.9). A J++ app would have to drop L4, so the band *shrinks* rather than widens.
2. **Nothing gains below L4.** J, LRP-only and J++ are all at 0–1% recall at L1–L3, and the bubbles there are junk for J and J++ alike. The paper's early-layer gains need early layers that hold something.
3. **Less of what makes the bubble worth showing.** J++ shows 13 of the headline words (J: 22). It drops `shadow / investigate / secret` from the dark-night starter prompt, and prompt-independent tokens fill 12% of its band display (J: 0%).
4. **It is unstable.** Refitting on 100 instead of 1000 prompts changes 46% of the band display (J: 16%; 100% at L4). Which words a child sees would depend on fitting noise. That also rules out in-browser fitting.
5. **Swap is no better.** Every lens writes "Max dug a **moat**". Over 7 swaps J++ switches 4, J 5.
6. **Its gains are on its own exam.** The improvements are in the dev-set format the weights were trained on (sentence-final readouts with target words). The label-free check on real stories shows J++ level with or slightly ahead of J at L5–L9 (L6: 2.1% vs 1.7%; L8: 2.4% vs 2.5%), and far worse at L4. That is a modest edge, not one that outweighs points 1–4.

**As an option, no.** A J/J++ toggle would show children two different, partly contradictory bubbles, one of them less stable, which is exactly the unexplained part Ken's design rule rules out. If one lens has to go, it's J++.

### stories15M: not worth it, and it doesn't open the door for Tiny Mind's Microscope.

J++ lifts period recall from 5% to 14%, but the level stays tiny, and at the last word it barely beats the logit lens (23% vs 21%). Its bubbles fill with the same `the / was / sad / did` on every prompt. Label-free precision halves at L3–L4, and its display changes 73% between a 100- and a 1000-prompt fit. Swaps do nothing for any lens at the app band. **J++ makes 15M less useful, not more.** So the Microscope decision rests where `out/report.md` left it: if Tiny Mind gets a thought bubble on stories15M, it would be the plain J at L2–L3, with surface-level content.

### The pieces separately

- **Readout Filtering alone:** +4 points of period recall in full-vocab ranking. It is already covered, more strictly, by the app's word-only display, so there is nothing to ship.
- **LRP alone (the R-Lens):** the most stable lens (94% n=100/1000 agreement) and slightly better label-free precision. But it is closer to the logit lens (L6 overlap 0.43 vs 0.29), so the J-only "hidden thought" words thin out. Recall falls 52% → 34%, and headline words fall 22 → 10. It's not a better Thought Bubble.
- **Expert weighting** is where J++'s recall gain comes from, and it is also where the instability and the junk layers come from.

**No `.jlens` export**, since J++ did not win. `src/jpp/export_jlens.py` writes the app's exact format (and round-trips it) if Ken wants to try one. Nothing in `app/` was touched.

**If someone revisits this later**, the promising piece is J++'s weighting at L5–L8 only, re-fitted with a larger dev set whose readouts sit at varied tokens, plus non-negative or regularized weights. Non-negative weights would stop the cancellation behind the junk layers, but because the experts are near copies of their average ("Why", point 1), such a lens could not move far from LRP-only. "J++ mixed" (weights also fitted at last-word readouts) is a small step that way: steadier at the bottom of the band and better on names, but not enough to change this verdict.

## What was compared

| lens | what it is |
|---|---|
| **J** | the shipped J-lens: `out/lenses/{A,C}/lens_1000.pt` = `app/stories110M.jlens` / `app/stories15M.jlens` (1000 TinyStories prompts, plain gradients) |
| **J+filter** | the same J with J++'s *Readout Filtering*: tokens with no letter or digit are left out of the ranking (1,738 of the 32,000 Llama-2 ids) |
| **LRP-only** | one Jacobian per layer averaged over the same 1000 prompts, but through the LRP backward graph (J++'s second ingredient; this is the paper's "R-Lens"), + filter |
| **J++** | the full recipe: 8 expert Jacobians per layer (k-means clusters of the activations), through LRP, combined with 8 learned signed weights per layer, + filter |
| **J++ mixed** | *my variant, not the paper's*: the same experts, weights learned on readouts at the period AND at the sentence's last word (see "the dev set" for why) |
| no-LRP exp. | ablation: expert weighting with plain gradients (15M at n=1000; 110M at n=100 only, to save GPU time) |
| logit | the plain logit lens, as a floor |

**Recipe, as implemented** (the `jpp_lens` library itself, cloned to `C:\Users\toont\Documents\GitHub\jpp_lens` with its own venv: torch 2.11+cu128, transformers 5.14.1; the app's `.venv` is untouched):
- *Router*, per layer: unit-normalize, center, PCA to 64 dims, k-means with K = 8 (k-means++, seed 0) on the valid positions (positions 0–15 and the last are skipped) of the same 1000 TinyStories prompts fit A / fit C used.
- *Experts*: one Jacobian per cluster through J++'s LRP surgery: the LN rule on every residual RMSNorm, identity + half rules on the SwiGLU MLP, plain gradients through attention. Target = final block output; all source layers (110M L0–L10, 15M L0–L4). The surgery refuses unknown classes, so I registered `LlamaRMSNorm` (its forward is the library's Qwen2 form) and `LlamaMLP`, after asserting their source. The patched forward matches stock to 2e-6 (15M) and 1e-5 (110M). No expert fell under the 50-position fallback.
- *Weights*: J++'s own fitter (Adam, lr 0.05, 200 steps from 3 starts, smooth surrogate of each layer's macro recall@10, readout filter on), run on the dev set's fit split.
- *Parity check*: the J++ pipeline with plain gradients and the clusters pooled reproduces the shipped 15M J to a relative error of 5e-7 at every layer. The port is faithful, so the differences below come from the method, not from plumbing.

**Cost.** 15M: 450 s for the 1000-prompt LRP expert fit (the shipped J took 278 s), plus 35 s for the router and ~30 s for the weights. 110M: ~6.0 s per prompt for the LRP expert fit when the GPU is not shared (100-prompt shards took 597–629 s; the whole 1000-prompt run took 115 min of wall-clock alongside other jobs), vs 6.3 s per prompt for the shipped J. Add 85 s for the router and 1–2 min for the weights. Inference cost is unchanged, since J++ is still one 768×768 map per layer, the same `.jlens` shape.

stories110M:

| lens vs shipped J | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LRP-only | 0.90 | 0.94 | 0.95 | 0.96 | 0.97 | 0.98 | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 |
| J++ | -0.63 | -0.77 | -0.94 | 0.86 | -0.05 | 0.77 | 0.96 | 0.93 | 0.85 | 0.91 | 1.00 |
| J++ mixed | -0.25 | 0.58 | 0.83 | 0.80 | 0.89 | 0.82 | 0.96 | 0.91 | 0.88 | 0.81 | 1.00 |
| experts-noLRP n100 | 0.88 | -0.02 | -0.17 | 0.93 | 0.67 | 0.67 | 0.93 | 0.89 | 0.94 | 0.98 | 0.97 |
| J skip1 n100 | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

stories15M:

| lens vs shipped J | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| pooled-noLRP | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| LRP-only | 0.87 | 0.89 | 0.95 | 0.97 | 0.98 |
| J++ | -0.25 | 0.22 | 0.85 | 0.13 | 0.17 |
| J++ mixed | 0.41 | 0.79 | 0.70 | 0.81 | 0.77 |
| experts-noLRP | -0.23 | 0.81 | 0.87 | 0.70 | 0.42 |
| J skip1 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |

*Cosine between each lens's map and the shipped J's, per layer (n=100 rows compare with the shipped n=100 fit).* LRP-only is a mild rotation of J. J++ is a different map wherever its weights had little to fit: anti-aligned with J at 110M L0–L2, orthogonal at L4 (see "Why"), and nearly orthogonal at most 15M layers.

## The TinyStories dev set (step 2)

J++ learns its weights from labeled "latent-variable" items. The paper's items (multi-hop facts, typos, translations, poetry) are beyond a TinyStories model, so I built story-register items with a known word the model is disposed to say **soon but not next**, the `hid / hide before mom arrives` pattern from `out/report.md`. 338 hand-written items from 54 templates × name fillers, in 5 families (`src/jpp/devset.py`, `out/jpp/devset/items.json`):

| family | example (readout at the final period) | answer words |
|---|---|---|
| need | "Tom wanted to play in the snow, but it was very cold outside." | coat, hat |
| feeling | "Kim lost her favorite teddy bear at the park. She looked everywhere, but she could not find it." | sad, cry, cried |
| name | "Long ago, a little girl called Jane lived in a small house near the woods. She liked to pick berries." | Jane |
| riddle | "Andy heard a tiny squeak and saw a little gray animal with a long tail run under the table." | mouse |
| hide | "Amy broke her dad's favorite cup. Then she heard her dad coming." | hide, hid, hiding |

- **Correctness filter**, per model (the paper scores only items the model gets right): an item counts if the model's own greedy continuation says an answer word among tokens 2–48 but **not** as the very next token, so the items test held-in-mind content rather than next-token guessing. Answers are single tokens (multi-token words like "umbrella" or "scared" were left out, as in the paper). 110M keeps 199 of 338 items; 15M keeps 161.
- **Split by template**, never by filler, so no sentence frame is shared between fitting and scoring: ~30% of the need/feeling/name/riddle templates fit the weights (110M: 52 items, 15M: 39). Everything else is held out (110M: 147, 15M: 122). **All of "hide" is held out**, because it is the concept of pinned eval prompt 4 ("Ben broke his mom's favorite vase…"). That keeps the pin honest and doubles as a leave-one-task-out test.
- **Disjointness**: no item shares a 6-word run with any of the 12 eval prompts (asserted in code; this caught and removed "Once upon a time, there was a little…"). The items are hand-written, so none overlaps the pinned TinyStories fit corpora (`out/pins.json`, `out/corpora/`).
- **Second readout position.** Every dev prompt ends in a period, so every readout the recipe sees is a period residual, and the router puts all of them in **one cluster per layer** (checked: all 39 fit readouts on 15M land in one cluster at each layer). Kids click any token. So I also read out one token earlier, at the sentence's last word, which is valid when the model's greedy next token there is that period (110M: 155 items, 15M: 141). That gives "word" recall below, and the "J++ mixed" weights.

## Results: stories110M

The app band is L4–L8; all layers are shown so the "J++ helps early layers" claim can be checked.

### Step 1: readout filtering alone (no refit)

Comparing the J and J+filter rows below: the filter leaves the logit-lens overlap within ±0.01 and leaves last-word recall and the Lily ranks unchanged. It lifts period-readout recall in the band from 52% to 56%, mostly from "hide" (79% → 95%), where newline and quote tokens otherwise crowd the top of the ranking right after a period. **It changes nothing in the app's normal view**: the app already displays only word-start story tokens (`vocab.json` `wordStartIds`), a strict subset of what the filter keeps. The only view it would touch is the "raw" toggle, which exists to show everything, punctuation included. Adding the filter there would undo the toggle's purpose. Verdict on the filter alone: free, harmless, and already covered by the app's display.

### Held-out recall@10

| lens | period band (L4-L8) | period all (L0-L10) | period early (L1-L3) | period L1-L8 (L1-L8) | word band (L4-L8) | word all (L0-L10) | word early (L1-L3) | word L1-L8 (L1-L8) |
|---|---|---|---|---|---|---|---|---|
| logit | 12% | 21% | 0% | 12% | 8% | 13% | 0% | 8% |
| J | 52% | 57% | 1% | 53% | 31% | 34% | 19% | 32% |
| J+filter | 56% | 61% | 1% | 57% | 31% | 34% | 19% | 32% |
| LRP-only | 34% | 44% | 0% | 34% | 31% | 34% | 21% | 33% |
| J++ | 64% | 66% | 1% | 64% | 37% | 37% | 15% | 37% |
| J++ mixed | 61% | 65% | 0% | 61% | 34% | 37% | 21% | 35% |
| no-LRP exp. (n100) | 60% | 69% | 7% | 62% | 29% | 33% | 8% | 29% |

Held-out items: 147 period readouts, 111 last-word readouts.

| lens | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logit | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 3% | 12% | 2% | 13% |
| J | 0% | 0% | 1% | 1% | 0% | 0% | 11% | 43% | 41% | 19% | 17% |
| J+filter | 0% | 0% | 1% | 1% | 0% | 0% | 12% | 45% | 47% | 24% | 17% |
| LRP-only | 0% | 0% | 0% | 0% | 0% | 0% | 7% | 27% | 29% | 18% | 18% |
| J++ | 0% | 0% | 0% | 1% | 2% | 15% | 17% | 42% | 49% | 25% | 18% |
| J++ mixed | 3% | 0% | 0% | 0% | 0% | 10% | 21% | 35% | 47% | 29% | 18% |
| no-LRP exp. (n100) | 0% | 4% | 2% | 1% | 5% | 6% | 7% | 41% | 57% | 34% | 19% |

*Per-layer held-out recall at the period.* This is where J++ earns its keep. Over the L4–L8 band it reaches 64% at the period vs J's 52% (56% with the filter), and 37% vs 31% at the last word. The answer also surfaces a layer earlier: 15% at L5, where J has 0%. That matches the paper's headline pattern. **Below L4 nothing changes**: J, LRP-only and J++ are all at 0–1% at L1–L3 (the no-LRP ablation reaches 7%), so J++ does not widen the band downward. Most of the gain is one family (feeling: 38% → 78%), with smaller gains on names and riddles and a loss on "need":

| lens | feeling (n=32) | hide (n=19) | name (n=29) | need (n=24) | riddle (n=43) |
|---|---|---|---|---|---|
| logit | 3% | 32% | 24% | 0% | 0% |
| J | 38% | 79% | 62% | 67% | 16% |
| J+filter | 38% | 95% | 66% | 67% | 16% |
| LRP-only | 28% | 68% | 34% | 38% | 0% |
| J++ | 78% | 84% | 72% | 58% | 26% |
| J++ mixed | 78% | 89% | 66% | 54% | 16% |
| no-LRP exp. (n100) | 56% | 95% | 55% | 67% | 28% |

| lens | fit split, period | fit split, word | held-out, period | held-out, word |
|---|---|---|---|---|
| logit | 0% | 0% | 12% | 8% |
| J | 30% | 17% | 52% | 31% |
| J+filter | 30% | 17% | 56% | 31% |
| LRP-only | 28% | 24% | 34% | 31% |
| J++ | 48% | 24% | 64% | 37% |
| J++ mixed | 48% | 31% | 61% | 34% |
| no-LRP exp. (n100) | 52% | 18% | 60% | 29% |

*Fit split vs held-out (band).* The held-out split scores higher than the fit split for every lens. Its templates are simply easier, so the split is not leaking.

### Overlap with the logit lens

| lens | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| J | 0.09 | 0.11 | 0.13 | 0.17 | 0.19 | 0.27 | 0.29 | 0.33 | 0.41 | 0.48 | 0.67 |
| J+filter | 0.09 | 0.11 | 0.13 | 0.17 | 0.20 | 0.27 | 0.29 | 0.34 | 0.41 | 0.49 | 0.67 |
| LRP-only | 0.23 | 0.22 | 0.24 | 0.28 | 0.31 | 0.39 | 0.43 | 0.48 | 0.57 | 0.67 | 0.81 |
| J++ | 0.00 | 0.00 | 0.00 | 0.34 | 0.01 | 0.13 | 0.38 | 0.35 | 0.36 | 0.49 | 0.80 |
| J++ mixed | 0.04 | 0.10 | 0.13 | 0.08 | 0.21 | 0.18 | 0.34 | 0.31 | 0.40 | 0.36 | 0.80 |
| no-LRP exp. (n100) | 0.10 | 0.05 | 0.00 | 0.30 | 0.19 | 0.18 | 0.29 | 0.20 | 0.33 | 0.46 | 0.62 |

J++'s overlap jumps around from layer to layer: 0.00 at L0–L2, 0.34 at L3, **0.01 at L4**, then 0.13–0.38. The L4 value is the real problem, because L4 is the bottom of the app band:

### What a child sees (pins, legibility)

| lens (app display) | pins showing a headline word | headline words shown | display slots held by prompt-independent tokens | raw top-10 that are displayable words |
|---|---|---|---|---|
| logit | 4/5 | 6 | 31% | 85% |
| J | 5/5 | 22 | 0% | 84% |
| LRP-only | 4/5 | 10 | 15% | 82% |
| J++ | 5/5 | 13 | 12% | 78% |
| J++ mixed | 5/5 | 14 | 7% | 88% |
| no-LRP exp. (n100) | 5/5 | 15 | 2% | 88% |

*"Headline" = the out/report.md examples: `hid/hide/hiding` (prompt 4), `shadow/investigate/secret/noise/sound` (9), `dig/buried/pit/mole/deeper` (8), `chose/choose/Which/what` (5), `Finally/persistent/success/mistakes` (6), counted over the band's top-8s.* J++ keeps all five pins, but with 13 headline words instead of 22, and 12% of its band display slots go to prompt-independent tokens (J: 0%). The worst of those are at L4, where J++ shows **the same list on every prompt**: `Nelson, Ny, Kent, Aless, Stein, â, Rena, Kin`. These are out-of-register name fragments. Three of the app's starter prompts, at the last position:

Prompt 4, "Ben broke his mom's favorite vase. He heard her coming.", at L4:

| lens | L4 display top-8 |
|---|---|
| logit | ` It`, ` He`, ` The`, ` She`, ` But`, ` They`, ` One`, ` His` |
| J | ` first`, ` M`, ` First`, ` mad`, ` It`, ` parts`, ` Then`, ` available` |
| LRP-only | ` It`, ` He`, ` M`, ` Then`, ` first`, ` But`, ` The`, ` parts` |
| J++ | ` Kent`, ` Nelson`, ` Ny`, ` Aless`, ` Stein`, ` Russ`, ` â`, ` Rena` |
| J++ mixed | ` It`, ` Then`, ` first`, ` First`, ` He`, ` Some`, ` The`, ` At` |
| no-LRP exp. (n100) | ` first`, ` It`, ` M`, ` break`, ` own`, ` Tim`, ` it`, ` Ben` |

Prompt 9, "It was a dark night. Tim heard a strange noise in the garden.", at L6 and L8:

| lens | L6 display top-8 |
|---|---|
| logit | ` He`, ` It`, ` She`, ` The`, ` A`, ` His`, ` Sud`, ` What` |
| J | ` He`, ` shadow`, ` It`, ` Sud`, ` noise`, ` investigate`, ` secret`, ` sound` |
| LRP-only | ` He`, ` It`, ` She`, ` A`, ` Sud`, ` What`, ` Then`, ` M` |
| J++ | ` He`, ` It`, ` A`, ` sc`, ` noise`, ` must`, ` M`, ` fear` |
| J++ mixed | ` He`, ` It`, ` must`, ` sc`, ` he`, ` shadow`, ` brave`, ` A` |
| no-LRP exp. (n100) | ` He`, ` It`, ` M`, ` A`, ` shadow`, ` She`, ` too`, ` Grand` |

| lens | L8 display top-8 |
|---|---|
| logit | ` He`, ` It`, ` She`, ` The`, ` What`, ` When`, ` Sud`, ` His` |
| J | ` He`, ` It`, ` curious`, ` Sud`, ` What`, ` She`, ` investigate`, ` curiosity` |
| LRP-only | ` He`, ` It`, ` She`, ` What`, ` Sud`, ` His`, ` Then`, ` The` |
| J++ | ` he`, ` sc`, ` curious`, ` guess`, ` He`, ` it`, ` brave`, ` wondering` |
| J++ mixed | ` He`, ` he`, ` sc`, ` curious`, ` wondering`, ` guess`, ` brave`, ` It` |
| no-LRP exp. (n100) | ` he`, ` He`, ` curious`, ` sc`, ` pe`, ` bra`, ` what`, ` investigate` |

Prompt 6, "The little bird could not fly. Every day it tried and tried.", at L7:

| lens | L7 display top-8 |
|---|---|
| logit | ` One`, ` It`, ` But`, ` The`, ` Every`, ` He`, ` Then`, ` At` |
| J | ` Finally`, ` finally`, ` One`, ` its`, ` trying`, ` Its`, ` It`, ` try` |
| LRP-only | ` Finally`, ` One`, ` finally`, ` its`, ` It`, ` trying`, ` Then`, ` tried` |
| J++ | ` its`, ` finally`, ` One`, ` Finally`, ` effort`, ` trying`, ` struggle`, ` eventually` |
| J++ mixed | ` its`, ` One`, ` finally`, ` Finally`, ` trying`, ` effort`, ` struggle`, ` eventually` |
| no-LRP exp. (n100) | ` persistent`, ` effort`, ` its`, ` hard`, ` struggle`, ` trying`, ` determ`, ` determined` |

At L5–L8, J++ is sometimes *nicer*: the persistent bird gets `effort, struggle, eventually, determined`, and the dark night gets `curious, brave, wondering, fear`. But it also loses the starter prompt's signature `shadow / investigate / secret` and puts `an, two, a, was, on, went` on "One day" (prompt 1, L6). Early layers stay unreadable for every lens (prompt 8, "…Lily built a castle. Max dug a", at L2):

| lens | L2 display top-8 |
|---|---|
| logit | ` big`, ` very`, ` little`, ` small`, ` long`, ` lot`, ` happy`, ` nice` |
| J | ` hum`, ` mo`, ` Wo`, ` qu`, ` earned`, ` Ro`, ` calm`, ` d` |
| LRP-only | ` hum`, ` b`, ` d`, ` sm`, ` c`, ` deep`, ` the`, ` he` |
| J++ | ` equipment`, ` fact`, ` lights`, ` cli`, ` example`, ` games`, ` parties`, ` moves` |
| J++ mixed | ` an`, ` a`, ` the`, ` deep`, ` grand`, ` because`, ` A`, ` history` |
| no-LRP exp. (n100) | ` scales`, ` fingers`, ` ref`, ` Ne`, ` Sil`, ` faces`, ` forward`, ` fro` |

### Lily first-piece check

| lens | median ' L' rank (band, 11 Lily positions) | in top-10 |
|---|---|---|
| logit | 511 | 4/11 |
| J | 10 | 5/11 |
| J+filter | 10 | 5/11 |
| LRP-only | 14 | 5/11 |
| J++ | 5 | 8/11 |
| J++ mixed | 2 | 7/11 |
| no-LRP exp. (n100) | 4 | 6/11 |

J++ is better here: ` L` ranks in the top 10 at 8 of the 11 Lily positions (J: 5), with median rank 5 vs 10. Its "name" family recall is also higher (72% vs 62%).

### Swap demo and battery

| swap | no swap | J band | LRP-only band | J++ band | J++ mixed band | experts-noLRP n100 band |
|---|---|---|---|---|---|---|
| hole→castle | `hole. They were happy.` | ≠ `moat. They were having` | ≠ `moat. They were having` | ≠ `moat. They were having` | ≠ `moat. They were having` | ≠ `moat. They were having` |
| water→milk | `water. Her mom said,` | ≠ `juice. Her mom said,` | ≠ `juice. Her mom said,` | ≠ `juice. Her mom said,` | ≠ `juice. Her mom said,` | ≠ `juice. Her mom said,` |
| key→box | `key to open it.` | · `key to open it.` | · `key to open it.` | · `key to open it.` | · `key to open it.` | · `key to open it.` |
| cat→dog | `cat. The cat was` | · `cat. The cat was` | · `cat. The cat was` | · `cat. The cat was` | · `cat. The cat was` | · `cat. The cat was` |
| sky→tree | `sky. He was so` | **=** `tree. He was looking` | **=** `tree. He was looking` | **=** `tree. He was looking` | · `sky. He was so` | · `sky. He was so` |
| apple→cookie | `apple. He liked apples,` | **=** `cookie. He liked cookies` | **=** `cookie. He liked cookies` | **=** `cookie. He liked cookies` | **=** `cookie. He liked cookies` | **=** `cookie. He liked cookies` |
| tail→head | `tail. It wanted to` | **=** `head. It was a` | **=** `head. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` |
| **switched / exact** |  | **5 / 3** | **5 / 3** | **4 / 2** | **3 / 1** | **3 / 1** |

*First word after the anchor at the app band (L4–L8), strength 0.4, greedy. "=" means it is the put-in word, "≠" means it is no longer the take-out word, "·" means unchanged.* **Every lens reproduces the README demo**, "Max dug a **moat**". The no-swap and J texts match the README word for word, which validates this port of the app's swap. Over the battery, J and LRP-only switch 5 of 7 (3 exact); J++ switches 4 (2 exact; it loses tail → head), and J++ mixed 3. The full table with the wider L2–L8 range is in `out/jpp/stories110M/swaps.json`. It doesn't change the picture.

### Stability, n=100 vs n=1000

| lens | top-8 agreement n100 vs n1000 (band mean) | J cosine (band mean) | L4 | L5 | L6 | L7 | L8 |
|---|---|---|---|---|---|---|---|
| J | 84% | 0.995 | 78% | 82% | 82% | 88% | 92% |
| LRP-only | 94% | 0.998 | 89% | 91% | 94% | 98% | 97% |
| J++ | 54% | 0.662 | 0% | 62% | 76% | 70% | 64% |

J's display is 84% the same whether fitted on 100 or 1000 prompts, LRP-only's 94%. **J++'s is 54%, and 0% at L4.** The 8 learned weights per layer change from fit to fit (cosine 0.66 between the two maps), so the words a child sees depend on fitting noise. For the planned in-browser fitting that matters a lot.

### Open-ended, label-free check: future-word precision

Share (%) of the app-display top-10 that the story actually says 2–21 tokens later, novel words only, on 60 held-out validation stories, positions ≥ 16:

| lens | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logit | 0.4 | 0.6 | 0.7 | 0.9 | 1.0 | 1.1 | 1.4 | 1.5 | 1.9 | 2.1 | 2.0 |
| J | 0.4 | 0.5 | 0.6 | 0.7 | 0.9 | 1.2 | 1.7 | 2.1 | 2.5 | 2.5 | 2.2 |
| LRP-only | 0.7 | 0.8 | 0.9 | 1.2 | 1.3 | 1.7 | 2.1 | 2.4 | 2.7 | 2.5 | 2.2 |
| J++ | 0.1 | 0.1 | 0.0 | 1.3 | 0.2 | 1.3 | 2.1 | 2.3 | 2.4 | 2.9 | 2.2 |
| J++ mixed | 0.2 | 0.4 | 0.5 | 0.7 | 1.2 | 1.5 | 2.1 | 2.4 | 2.3 | 2.7 | 2.2 |
| no-LRP exp. (n100) | 0.4 | 0.2 | 0.1 | 0.8 | 0.6 | 0.8 | 1.7 | 1.7 | 2.3 | 2.6 | 2.0 |
| J skip1 n100 | 0.4 | 0.5 | 0.6 | 0.8 | 0.9 | 1.3 | 1.8 | 2.2 | 2.6 | 2.6 | 2.2 |
| J n100 | 0.4 | 0.5 | 0.6 | 0.8 | 0.9 | 1.3 | 1.7 | 2.1 | 2.5 | 2.5 | 2.2 |

(5342 story positions.)

At positions 1–15:

| lens | L0 | L1 | L2 | L3 | L4 | L5 | L6 | L7 | L8 | L9 | L10 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| logit | 0.4 | 0.4 | 0.4 | 0.5 | 0.6 | 0.7 | 0.7 | 0.8 | 0.9 | 1.0 | 0.9 |
| J | 0.4 | 0.6 | 0.5 | 0.7 | 0.7 | 0.8 | 0.9 | 0.9 | 0.8 | 0.9 | 0.9 |
| LRP-only | 0.5 | 0.7 | 0.7 | 0.8 | 1.0 | 1.0 | 1.0 | 1.0 | 0.9 | 1.0 | 0.9 |
| J++ | 0.0 | 0.1 | 0.1 | 0.9 | 0.0 | 0.6 | 1.1 | 1.2 | 1.1 | 1.0 | 0.9 |
| J++ mixed | 0.2 | 0.2 | 0.5 | 0.5 | 0.9 | 0.9 | 1.1 | 1.3 | 1.0 | 1.1 | 0.9 |
| no-LRP exp. (n100) | 0.3 | 0.2 | 0.1 | 0.6 | 0.3 | 0.5 | 0.7 | 0.7 | 0.9 | 1.0 | 0.7 |
| J skip1 n100 | 0.5 | 0.6 | 0.5 | 0.6 | 0.8 | 1.0 | 1.0 | 1.1 | 0.9 | 1.0 | 0.9 |
| J n100 | 0.5 | 0.5 | 0.5 | 0.7 | 0.8 | 1.0 | 0.8 | 0.9 | 0.8 | 0.9 | 0.9 |

(880 story positions.)

With no target word at all, the lenses are close at L5–L9. J++ is a little ahead at L6–L7 (2.1 vs 1.7, 2.3 vs 2.1) and level at L8. It is far behind at L4 (0.2 vs 0.9) and at L0–L2. LRP-only is the steadiest here, ahead of J at every layer up to L8.

## Results: stories15M

The app band for 15M is L2–L3 (`out/report.md`); all five fitted layers (L0–L4) are shown.

**Held-out recall@10** (the target word in the top 10 at any layer of the set; macro over families):

| lens | period band (L2-L3) | period all (L0-L4) | period early (L1-L1) | period L1-L3 (L1-L3) | word band (L2-L3) | word all (L0-L4) | word early (L1-L1) | word L1-L3 (L1-L3) |
|---|---|---|---|---|---|---|---|---|
| logit | 0% | 0% | 0% | 0% | 16% | 21% | 21% | 21% |
| J | 5% | 5% | 0% | 5% | 16% | 17% | 16% | 16% |
| J+filter | 5% | 5% | 0% | 5% | 18% | 19% | 16% | 18% |
| LRP-only | 8% | 8% | 0% | 8% | 16% | 17% | 16% | 16% |
| J++ | 14% | 18% | 5% | 18% | 22% | 23% | 5% | 22% |
| J++ mixed | 12% | 14% | 0% | 12% | 20% | 20% | 16% | 20% |
| no-LRP exp. | 7% | 22% | 5% | 12% | 22% | 22% | 16% | 22% |

Held-out items: 122 period readouts, 104 last-word readouts.

| lens | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| logit | 0% | 0% | 0% | 0% | 0% |
| J | 0% | 0% | 5% | 0% | 0% |
| J+filter | 0% | 0% | 5% | 0% | 0% |
| LRP-only | 0% | 0% | 8% | 2% | 0% |
| J++ | 9% | 5% | 9% | 14% | 4% |
| J++ mixed | 9% | 0% | 9% | 12% | 4% |
| no-LRP exp. | 9% | 5% | 7% | 0% | 7% |

*Per-layer held-out recall at the period.* J++ does lift period recall: 5% → 14% in the band and 5% → 18% over all layers, with a little at L0–L1, where J has nothing. But the absolute level stays low. At the last-word position over all layers, J++'s 23% is barely above the **logit lens's 21%** (J: 17%), so at 15M none of these lenses reads much that the logit lens doesn't. Almost all of J++'s period gain is one family (feeling: 23% → 50%); hide, name and riddle stay at 0% for every lens:

| lens | feeling (n=26) | hide (n=13) | name (n=23) | need (n=31) | riddle (n=29) |
|---|---|---|---|---|---|
| logit | 0% | 0% | 0% | 0% | 0% |
| J | 23% | 0% | 0% | 0% | 0% |
| J+filter | 23% | 0% | 0% | 0% | 0% |
| LRP-only | 38% | 0% | 0% | 0% | 0% |
| J++ | 50% | 0% | 0% | 19% | 0% |
| J++ mixed | 62% | 0% | 0% | 0% | 0% |
| no-LRP exp. | 31% | 0% | 0% | 6% | 0% |

**Overlap with the logit lens** (top-10 sets over every position of the 12 eval prompts; filtered variants are compared with the filtered logit lens):

| lens | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| J | 0.23 | 0.23 | 0.25 | 0.32 | 0.54 |
| J+filter | 0.24 | 0.23 | 0.25 | 0.32 | 0.53 |
| LRP-only | 0.28 | 0.27 | 0.30 | 0.41 | 0.63 |
| J++ | 0.02 | 0.01 | 0.17 | 0.03 | 0.04 |
| J++ mixed | 0.08 | 0.17 | 0.07 | 0.12 | 0.29 |
| no-LRP exp. | 0.01 | 0.15 | 0.12 | 0.19 | 0.11 |

J++ drops to 0.01–0.17. That is not "more hidden content": the readouts become a different and mostly prompt-independent list:

**Legibility (app display):**

| lens (app display) | pins showing a headline word | headline words shown | display slots held by prompt-independent tokens | raw top-10 that are displayable words |
|---|---|---|---|---|
| logit | 0/5 | 0 | 24% | 75% |
| J | 1/5 | 1 | 21% | 89% |
| LRP-only | 1/5 | 1 | 19% | 85% |
| J++ | 1/5 | 1 | 29% | 95% |
| J++ mixed | 1/5 | 1 | 14% | 96% |
| no-LRP exp. | 1/5 | 1 | 4% | 94% |

*The headline words are the 110M report's, so this column means little at 15M. The prompt-independent share is the useful one.* What a child would see at L3 (the band's upper layer). Prompt 4, "Ben broke his mom's favorite vase. He heard her coming.":

| lens | L3 display top-8 |
|---|---|
| logit | ` It`, ` She`, ` The`, ` He`, ` and`, ` the`, ` When`, ` it` |
| J | ` She`, ` He`, ` It`, ` The`, ` They`, ` Anna`, ` Ben`, ` L` |
| LRP-only | ` She`, ` He`, ` The`, ` It`, ` Ben`, ` the`, ` she`, ` Anna` |
| J++ | ` the`, ` was`, ` c`, ` a`, ` did`, ` sad`, ` saw`, ` is` |
| J++ mixed | ` the`, ` fle`, ` ran`, ` She`, ` ble`, ` fly`, ` she`, ` drag` |
| no-LRP exp. | ` She`, ` A`, ` east`, ` Jen`, ` Her`, ` They`, ` drawn`, ` marry` |

Prompt 8, "…Lily built a castle. Max dug a":

| lens | L3 display top-8 |
|---|---|
| logit | ` big`, ` new`, ` small`, ` few`, ` good`, ` bigger`, ` lot`, ` great` |
| J | ` hole`, ` big`, ` small`, ` castle`, ` holes`, ` tiny`, ` fort`, ` thin` |
| LRP-only | ` small`, ` hole`, ` big`, ` tiny`, ` new`, ` little`, ` poor`, ` p` |
| J++ | ` the`, ` dry`, ` ball`, ` to`, ` pot`, ` rock`, ` red`, ` rub` |
| J++ mixed | ` hole`, ` plant`, ` small`, ` tree`, ` tiny`, ` seed`, ` garden`, ` pit` |
| no-LRP exp. | ` car`, ` wish`, ` fort`, ` bott`, ` boat`, ` card`, ` square`, ` poison` |

Prompt 9, "It was a dark night. Tim heard a strange noise in the garden.":

| lens | L3 display top-8 |
|---|---|
| logit | ` It`, ` He`, ` She`, ` The`, ` When`, ` out`, ` Sud`, ` a` |
| J | ` She`, ` He`, ` Sud`, ` It`, ` The`, ` There`, ` His`, ` Little` |
| LRP-only | ` She`, ` Sud`, ` He`, ` It`, ` The`, ` but`, ` inside`, ` looking` |
| J++ | ` the`, ` food`, ` was`, ` looked`, ` sad`, ` sp`, ` saw`, ` a` |
| J++ mixed | ` the`, ` ran`, ` thought`, ` looked`, ` opened`, ` Sud`, ` saw`, ` sc` |
| no-LRP exp. | ` She`, ` drawn`, ` Her`, ` written`, ` Some`, ` Looking`, ` wanting`, ` It` |

Across all 12 prompts, J++ at L3 is led by `the / was / sad / did / food` regardless of the story. The shipped J at L2 still shows `Finally / until / eventually` for the persistent bird, `months / days / weeks` for the seed, and `hole / castle / fort` for the digging prompt.

**Lily first-piece check** (rank of ` L`, min over L2–L3, at the 11 positions where Lily is the active character):

| lens | median ' L' rank (band, 11 Lily positions) | in top-10 |
|---|---|---|
| logit | 58 | 4/11 |
| J | 18 | 4/11 |
| J+filter | 18 | 4/11 |
| LRP-only | 19 | 4/11 |
| J++ | 60 | 2/11 |
| J++ mixed | 1 | 8/11 |
| no-LRP exp. | 77 | 1/11 |

**Stability, n=100 vs n=1000** (app-display top-8 at L2–L3, 12 prompts):

| lens | top-8 agreement n100 vs n1000 (band mean) | J cosine (band mean) | L2 | L3 |
|---|---|---|---|---|
| J | 90% | 0.995 | 82% | 97% |
| LRP-only | 92% | 0.999 | 88% | 96% |
| J++ | 27% | 0.585 | 47% | 6% |

J++ is unstable: refitting from 100 instead of 1000 prompts changes three quarters of what a child sees (J cosine 0.59). Its eight signed weights per layer, learned from 39 items, swing a lot. J and LRP-only reproduce at 90–92%.

**Open-ended, label-free check:** future-word precision. On 60 held-out TinyStories *validation* stories, at every position: the share of the app-display top-10 that the story actually says in the next 2–21 tokens and has not said before (frequent function words excluded), in %, at positions ≥ 16:

| lens | L0 | L1 | L2 | L3 | L4 |
|---|---|---|---|---|---|
| logit | 0.6 | 0.5 | 0.7 | 0.9 | 1.2 |
| J | 0.6 | 0.8 | 1.1 | 1.2 | 1.6 |
| LRP-only | 0.6 | 1.0 | 1.3 | 1.4 | 1.6 |
| J++ | 0.4 | 0.4 | 1.0 | 0.6 | 0.3 |
| J++ mixed | 0.6 | 0.8 | 0.8 | 1.3 | 1.4 |
| no-LRP exp. | 0.3 | 0.5 | 0.9 | 1.0 | 0.6 |
| J skip1 | 0.6 | 0.8 | 1.0 | 1.2 | 1.5 |
| J n100 | 0.6 | 0.8 | 1.1 | 1.2 | 1.6 |

(5342 story positions.)

J++ is lower than J at L1–L4 (L3: 0.6 vs 1.2; L4: 0.3 vs 1.6). The LRP-only lens is slightly above J at L1–L3.

**Swap** (the app's edit, strength 0.4, greedy; first word after the anchor; "=" means it is the put-in word, "≠" means it is no longer the take-out word):

| swap | no swap | J band | J L1-L4 | LRP-only band | LRP-only L1-L4 | J++ band | J++ L1-L4 | J++ mixed band | J++ mixed L1-L4 | experts-noLRP band | experts-noLRP L1-L4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| hole→castle | `hole in the sand.` | · `hole and put a` | ≠ `big hole and put` | · `hole and put a` | ≠ `big hole and put` | · `hole and put a` | ≠ `big hole and put` | · `hole and put a` | ≠ `big hole and put` | · `hole and put a` | · `hole and put a` |
| water→milk | `water. Her mom gave` | · `water. Her mom gave` | ≠ `juice. Her mom said,` | · `water. Her mom gave` | ≠ `juice. Her mom said,` | · `water. Her mom gave` | · `water. Her mom gave` | · `water. Her mom gave` | · `water. Her mom said,` | · `water. Her mom gave` | · `water. Her mom gave` |
| key→box | `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` | · `key to unlock it.` |
| cat→dog | `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` | · `cat. The cat liked` |
| sky→tree | `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` | · `sky. He was so` |
| apple→cookie | `apple. He liked red` | · `apple. He wanted to` | **=** `cookie. He wanted to` | · `apple. He wanted to` | ≠ `cake. He wanted to` | · `apple. He wanted to` | · `apple. He wanted to` | · `apple. He wanted to` | · `apple. He wanted to` | · `apple. He liked red` | · `apple. He liked red` |
| tail→head | `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` | · `tail. It was a` |
| **switched / exact** |  | **0 / 0** | **3 / 1** | **0 / 0** | **3 / 0** | **0 / 0** | **1 / 0** | **0 / 0** | **1 / 0** | **0 / 0** | **0 / 0** |

At the app band (L2–L3), no lens moves 15M at all. Widening to L1–L4, J and LRP-only switch three of the seven: "hole" becomes "big hole … Max dug a big castle", water becomes juice ("…drink some milk first"), and apple becomes cookie (J) or cake (LRP-only). J++ switches only the first, and not toward the concept.

## Does J++ make stories15M useful? (step 4)

**No.** The question was whether J++ widens 15M's usable band beyond L2–L3, or surfaces non-surface content there. It does neither in a way a child could use:

- **Band.** J++ lifts period-readout recall at L0–L1 from 0% to 5–9%, which is a few items. At those layers its display reads `fle / body / bird / creature / gun / moved` for "Ben broke his mom's favorite vase. He heard her coming." and `held / hold / carefully / holds` (J++ mixed) for the dark-night prompt. That is noise, not a usable band. In the band itself, J++'s readouts are *less* legible than J's: at L3 a prompt-independent `the / was / sad / did` fills 41% of the display slots (J: 22%), and future-word precision halves (0.6% vs 1.2%).
- **Content.** The one place J++ helps is the "feeling" family at the period (23% → 50%). Hide, name and riddle stay at 0% for every lens at 15M. The model doesn't hold those in a form any linear lens reads out at 15M scale.
- **Reliability.** J++'s 15M maps change three quarters of their display between a 100-prompt and a 1000-prompt fit. For in-browser fitting (the original reason 15M was attractive), that is disqualifying. The plain J reproduces at 90%.
- **Swaps** don't work on 15M with any lens at the app band and strength.

So J++ does not change the Tiny Mind decision. A Microscope thought bubble on stories15M would still be the plain J at L2–L3, with the surface-level content `out/report.md` described: `hole / castle`, `months / days`, `Finally / until`. It is weakly better than the logit lens (future-word precision 1.1–1.2% vs 0.7–0.9% at L2–L3; word-position recall about equal). Whether that is worth a bubble is a design call. J++ doesn't make it more worthwhile.

## Side question: skipping the first 16 positions

The paper (and jlens) drop positions 0–15 from every fit as attention sinks. The fit prompts are 128 tokens, so this drops 15 of 126 usable positions. The app, though, reads out at every position, and children's prompts are short. Residual norms by position, mean over 200 fit prompts (`out/jpp/*/position_profile.json`):

| model, layer | pos 0 (BOS) | pos 1–3 | pos 4–15 | pos ≥ 16 |
|---|---|---|---|---|
| 110M L2 | 120 | 51, 48, 85 | 19.5 | 12.0 |
| 110M L6 | 177 | 85, 82, 131 | 40.6 | 26.7 |
| 110M L10 | 150 | 81, 89, 121 | 66.8 | 58.5 |
| 15M L2 | 17.9 | 4.2, 3.2, 3.1 | 3.6 | 3.0 |
| 15M L4 | 21.5 | 7.4, 6.7, 7.2 | 8.1 | 7.2 |

So 110M does have sink-like early positions: positions 1–3 run 3–6× the norm of positions ≥16 through the middle layers, and positions 4–15 about 1.5×. Fit prompts nearly always open "Once upon a…", so part of this may be those tokens rather than their positions. 15M has only BOS. **The skip still doesn't matter.** Refitting the plain J with only BOS skipped:
- stories110M (100 prompts, vs the shipped n=100 fit): cosine 0.99–1.00 at every layer. Period recall in the band 51% vs 52%, last word 33% vs 39% (about 7 items), headline words 18 vs 18. Future-word precision at positions 1–15 rises by at most 0.2 points (L6–L7).
- stories15M (1000 prompts): cosine 0.987–0.996. Recall, overlap and readouts are unchanged within noise. At positions 1–15, future-word precision is not better (L3: 0.7 vs 0.8).

The fit is dominated by the ~110 later positions either way, and a J fitted on early positions does not read early positions noticeably better. Keep the default.

## Why J++ does not transfer to these models

Six things, in rough order of weight:

1. **There is little for the experts to separate.** J++ exists because one averaged Jacobian can blur maps that differ by context, so it fits one map per cluster of positions and mixes them. In these models the eight cluster maps are close to copies of their average (the LRP-only map). Per layer (`src/jpp/expert_spread.py`; app-band layers in bold):

   stories110M:

   | layer | expert vs average: cosine, min / median | expert vs average: distance, median | weight that is negative | J++ map vs average: cosine |
   |---|---|---|---|---|
   | L0 | 0.46 / 0.85 | 0.56 | 65% | -0.70 |
   | L1 | 0.67 / 0.93 | 0.39 | 80% | -0.82 |
   | L2 | 0.73 / 0.96 | 0.31 | 99% | -0.99 |
   | L3 | 0.76 / 0.95 | 0.37 | 23% | +0.90 |
   | **L4** | 0.80 / 0.96 | 0.34 | 53% | -0.04 |
   | **L5** | 0.83 / 0.96 | 0.34 | 46% | +0.80 |
   | **L6** | 0.85 / 0.96 | 0.31 | 23% | +0.97 |
   | **L7** | 0.89 / 0.98 | 0.20 | 40% | +0.94 |
   | **L8** | 0.90 / 0.99 | 0.18 | 43% | +0.86 |
   | L9 | 0.92 / 0.99 | 0.16 | 43% | +0.92 |
   | L10 | 0.98 / 1.00 | 0.08 | 9% | +1.00 |

   stories15M:

   | layer | expert vs average: cosine, min / median | expert vs average: distance, median | weight that is negative | J++ map vs average: cosine |
   |---|---|---|---|---|
   | L0 | 0.53 / 0.86 | 0.61 | 52% | -0.25 |
   | L1 | 0.78 / 0.88 | 0.59 | 55% | +0.31 |
   | **L2** | 0.86 / 0.89 | 0.50 | 33% | +0.90 |
   | **L3** | 0.88 / 0.92 | 0.47 | 45% | +0.18 |
   | L4 | 0.94 / 0.96 | 0.27 | 45% | +0.18 |

   In 110M's band the typical expert has cosine 0.96–0.99 with the average, and at 15M's L2–L3, 0.89–0.92. One genre with a small vocabulary seems to leave the model handling most positions alike. The differences between experts are not empty: they are where J++'s recall gain over LRP-only comes from (34% → 64% at the period in the band). But they are small, and partly noise (point 2). I did not measure the same spread for the paper's 9B–284B models; my guess is that on code, math and many languages it is much larger. This also caps any repair: a mix with non-negative weights has cosine to the average at least as high as its least-similar expert (0.80 or more in 110M's band). So a "safe" J++ could not move far from the LRP-only lens, which is not a better Thought Bubble (point 6).
2. **Signed weights cancel the part that works.** The J++ map is the weighted sum of the experts. Since every expert is mostly the average, the map keeps the average only if the signed weights add up positively over it. The last column shows what is left. Where the fit items give little or no signal (at 110M L0–L2 and L4, no single expert ranks any fit item's answer in the top 10), the search keeps whichever iterate edges ahead, on one item or on the smooth surrogate, and it drifted into cancellation:
   - At 110M L2, 99% of the weight is negative, so the map is the average flipped (cosine −0.99). A flipped map shows the model's *least*-likely tokens. Label-free precision at L0–L2 and L4 is 0.0–0.2%, below the logit lens.
   - At 110M L4, the negative half (53%) cancels the positive half. Only the differences between experts are left (cosine −0.04 to the average, −0.05 to J), and they show `Nelson / Kent / Stein` on every prompt. 15M's L3 is a milder case of the same thing (+0.18), with `the / was / sad` on every prompt.
   - Where the weights keep the average (110M L6: +0.97; 15M L2: +0.90), the readout is sensible.

   The average itself reproduces well: LRP-only's display agrees 94% between a 100- and a 1000-prompt fit at 110M. So J++'s instability (54%) comes from the weighted differences on top of it. Presumably the paper's models hit this less, since more of their layers carry signal. A guard (keep the pooled map unless a weighting clearly beats it) would remove the junk layers, but not the instability.
3. **The weights are learned on one kind of position and used on all of them.** The recipe reads out at the prompt's final token. In story items that is always a period, and the router puts every one of those period residuals into a single cluster per layer. The eight signed weights are therefore tuned to a map that turns *period* residuals into upcoming nouns and feelings. The app applies the same map at whatever token a child clicks. Weights learned also on last-word readouts ("J++ mixed") recover part of the loss (the Lily check, the pins and word-position recall), which is consistent with this diagnosis. The paper's readouts (`"…the organ that pumps blood … has this many chambers: "`) end on varied tokens, and it never tests other positions.
4. **Few labels, signed weights.** 39–52 fit items in 4 families, with an objective quantized in steps of 1/(family size). Adam on the smooth surrogate happily finds large negative weights that win a few more fit items. At 15M that is overfitting outright: 48% recall on the fit split, 14% held out, and a different map from 100 vs 1000 fit prompts (cosine 0.59). The paper's "only 8 weights per layer, few degrees of freedom" argument assumes the experts themselves are good lenses. Here each expert alone scores ~0% on the fit items at most 15M layers and at 110M L0–L5, so the weights are doing all the work.
5. **The paper's gains are where these models have nothing to read.** J++'s headline gains are in "the first quarter of layers" of 9B–284B models, where a concept already exists but the plain average Jacobian blurs it. In stories110M, L1–L3 hold no recoverable story content for any lens (held-out period recall 0–1% for J, LRP-only and J++ alike). At 15M, everything below L2 is the same. There is nothing early for the filtering to rescue.
6. **LRP pulls the lens toward the logit lens at this scale.** LRP-only raises overlap with the logit lens at every layer (110M L6: 0.29 → 0.43; 15M L3: 0.32 → 0.41). It reads slightly more of the story's actual next words, but it loses much of what made the J-lens worth showing: the J-only "DIFF" words (`hid`, `shadow`, `investigate`) and dev-set recall. The R-Lens paper's benefit (stopping noise compounding through norms and gated MLPs over 30–60 layers) has little to work with over 4–11 layers. The ablation agrees: at 110M, expert weighting *without* LRP (n=100) does as well as full J++ at n=100 (period recall in the band 60% vs 58%, last word 29% vs 32%, headline words 15 vs 12).

## Caveats

- The dev set is small (39–52 fit items, 122–147 held-out) and hand-built. Differences of a few points in recall are within noise: one item is 0.7–0.8 points, and family macro-averaging magnifies small families. The verdict rests on effects that are large, or consistent across metrics.
- "Soon but not next" is a behavioral label: the model *said* the word within 48 tokens. That matches the J-lens's own promise, but it is not the paper's latent-variable ground truth.
- Legibility proxies (prompt-independent share, raw-top-10 word share, the headline pins) stand in for a human look. The readout tables are there for that look.
- The 110M no-LRP ablation ran at n=100 only.
- J++ was evaluated as published. With a larger, position-diverse dev set, or with non-negative weights, it might behave differently, though a non-negative mix cannot move far from LRP-only here ("Why", point 1). "J++ mixed" is one step in that direction, and it helped some metrics but did not change the verdict.

## Files

- Code: `src/jpp/` (`devset.py`, `verify_devset.py`, `add_word_positions.py`, `fit_experts.py`, `fit_weights.py`, `evaluate_jpp.py`, `swap_demo.py`, `skip_check.py`, `export_jlens.py`, `report_tables.py`, `make_report.py`; runners `run_fits.sh`, `finish.sh`). Run with `C:\Users\toont\Documents\GitHub\jpp_lens\.venv\Scripts\python.exe` and `PYTHONIOENCODING=utf-8`.
- Results: `out/jpp/devset/` (items, per-model correctness), `out/jpp/<model>/eval.json` (every metric, ranks and readouts), `swaps.json`, the `*_weights.json` files (learned expert weights), `router_*.json` (cluster sizes), `fit_*.log`, and the fitted lenses as fp32 `.pt` (`pooled_rlens_*` = LRP-only, `jpp_*` = J++).
- Size: `out/jpp/` is 247 MB, almost all of it the nine 25 MB fp32 110M lens files. They are regenerable from the scripts (about 2 GPU-hours), so Ken may prefer not to commit them. Everything else in `out/jpp/` and `src/jpp/` is small.
- Large intermediates (per-shard expert sums, expert Jacobians, 100–200 MB each, 3 GB in all) are in the jpp_lens clone's gitignored `artifacts/tinystories/`, not in this repo.
- Nothing in `app/` was changed. No `.jlens` was exported, because J++ did not win (`src/jpp/export_jlens.py` is ready if Ken wants one anyway).
