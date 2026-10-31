## Summary

**The plan's mind, top-2 of 8, fails the gate on (c). The top-1 variant mostly passes it.** In the top-1 variant, each word goes to one expert instead of two. It was trained as an optional extra, on the same data, schedule and seed.

The top-2 mind writes as well as its dense twin, but switching one of its experts off rarely changes stories in a way someone can name and then spot again. The top-1 mind's experts are easier to name, and five of its switch-offs break something specific. With those experts muted, two describer-and-judge models each confirmed all five on fresh story beginnings:
- *"to to to"*;
- broken speech starts;
- *"said said"*;
- *don'* for *don't*;
- *"wanted wanted"*.

| | top-2 (the plan's mind) | top-1 (variant) |
|---|---|---|
| (a) loss vs dense twin | −5.5% ✅ | −3.6% ✅ |
| (b) layers with nameable experts, gpt-6-luna names and scores | 0 of 6 ❌ | 4 of 6 ✅ (also on fresh passages and whole cards) |
| (b) the same, gpt-6.1-sol names and scores | 4 of 6 ✅ | 5 of 6 ✅ |
| (c) rerouted switch-offs, luna describes and judges | 1 ❌ | 2 ❌ |
| (c) rerouted, sol describes and judges | 2 ❌ | 7 ✅ (5 of them confirmed again on fresh beginnings) |
| (c) muted, luna | 2 ❌ | 6 ✅ (5 confirmed again) |
| (c) muted, sol | 3 ❌ | 10 ✅ (8 confirmed again) |

**Top-1 fails only one combination: the plan's own switch (rerouting) judged by gpt-6-luna**, the model this study was set up to use. It got 2 of the 4 needed. Its describer named the apostrophe breakage correctly, but its blind judge matched it 14/20, just short of the cutoff for a second round. Sol matched the same effect 16/20 and then 34/39. All five of the confirmed effects are real; the luna-judged reroute test is just less sensitive to them.

### The top-2 mind (the plan's main run)

- **(a) passes.**
  - The MoE's held-out loss is 5.5% *lower* than the dense twin's (1.417 vs 1.500), at the same active size (13.2M parameters per word) and the same 514M training pieces.
  - In a blind pairwise reading, the judge preferred the MoE's story 60 times out of 96.
  - The router never collapsed: every expert handles between 16% and 36% of words (25% would be even).
- **(b) depends on the scorer.** The test has two model jobs:
  - a **labeler** names each expert and writes its rule;
  - a **scorer** applies the rule to held-out passages, marking which words go to the expert.
  - Results for each pair of models:

    | labeler | scorer | layers passing (3 of 6 needed) |
    |---|---|---|
    | gpt-6-luna | gpt-6-luna | 0 of 6 |
    | gpt-6.1-sol | gpt-6-luna | 2 of 6 |
    | gpt-6-luna | gpt-6.1-sol | 3 of 6 on the first passages, but 1 of 6 on fresh ones |
    | gpt-6.1-sol | gpt-6.1-sol | 4 of 6 on the first passages, on fresh ones, on both pooled, and when scoring the whole card |

  - So the pass is robust only with sol in both jobs, and it comes from layers 3–6. Layer 1 fails everywhere (1–3 of 8). Sol's passing names include:
    - "the naming words expert" (L3·E1, 0.94);
    - "the punctuation expert" (L3·E4, 0.89);
    - "the action expert" (L3·E2, 0.89);
    - "the noun group expert" (L5·E2, 0.88).
  - Sol is the better *scorer* above all. It applies a rule that quotes a dozen pieces plus a context condition more faithfully than luna, and with about a seventh of the reasoning tokens.
- **(c) fails, with either model: 1 rerouted expert qualifies with luna as describer and judge, 2 with sol; 4 are needed.** The test was set before seeing results. Under both models, only one rerouted expert holds up on fresh beginnings, L1·E7 (luna 38/40, sol 40/40).
  - Of 33 specific one-sentence descriptions, the blind matching test certified only one: **L1·E7**. With it off, the word after *to* goes muddled (*"they ran to to to and asked"*). It scored 18/20, then 38/40 on fresh beginnings.
  - Five more came close (15/20). None held up on 40 fresh beginnings. A direct count shows why the test is hard on effects that only sometimes appear:
    - **L1·E0** off breaks apostrophe words in 20 of 48 stories (15 of 40 fresh ones). Switching off any other layer-1 expert does it in 0–2. The effect is real and specific, but it shows up in only about 40% of stories, so the matching test can't certify it (26/40, p = 0.04).
    - **L2·E1**'s doubled *-ed* appeared in 4 of 48 stories and in none of the 40 fresh ones.
  - **Muting** instead of rerouting gives the same picture: 2 qualify by the rule. L1·E7 confirms (34/40). L6·E5 passed round 1 narrowly (15/19), then scored 17/34 on fresh beginnings, which is chance. It was probably the one false pass the test expects.
  - **With gpt-6.1-sol as describer and judge** (same stories, prompts and rules), the answer barely moves:
    - Sol calls fewer descriptions specific (20 rerouted, against luna's 33).
    - Rerouted, 2 qualify:
      - **L1·E7**, as before, confirmed 40/40.
      - **L1·E4** ("more describing, less happening") passed round 1 at 16/20, but scored 25/40 on fresh beginnings.
    - Muted, 3 qualify:
      - L1·E7 again (37/40 on fresh beginnings).
      - **L6·E2** (the word-chunks expert): muted, the mind stutters on word pieces, *"bott bott"*, *"du du"*. That's 11/12, then 21/22 on fresh beginnings, a second effect that holds up firmly.
      - L6·E5, which did not replicate (22/34).
    - Sol couldn't certify L1·E0's apostrophe breakage either (muted: 15/20, then 23/40).

**What the top-2 experts are: word sorters whose lists become grammar roles higher up.**
- **The word decides the route.** A held-out piece goes to its word's usual expert 92% of the time in layer 1, falling to 76% in layer 6. A lookup table of which words go where predicts routing at balanced accuracy 0.83–0.94.
- **In layer 1 the lists are mixed.** No model names them well:
  - L1·E1 gets *was*, *She*, *said*, *day*, *so* and *little*; sol calls it "the was-and-said expert".
  - L1·E7 is "the adventure and chance expert" to luna and "the to expert" to sol.
- **Higher up, the lists line up with grammar roles,** which is why a strong scorer can apply the names there.
- **Names from routing and effects from switching off can disagree.** L1·E7's switch-off shows its real work: *to* and the verb after it.

**What switching off a top-2 expert does.**
- The clearest effects are spelling and grammar breakdowns from layer 1 and layer 2 experts. Each fits in one sentence:
  - **L1·E7** off: the verb after *to* goes missing (*"they ran to to to and asked"*).
  - **L1·E0** off: apostrophe words break (*"Let' go"*, *"It' a surprise"*). On held-out text, predicting the pieces after an apostrophe (*s*, *t*, *re*, *is*) gets 9–11 nats worse.
  - **L2·E1** (the story-verb expert) off: sometimes past tenses get a second ending (*"he wanteded to go out"*). It's vivid but rare.
- **Deeper experts change stories in many small ways, with nothing to name.** Their switch-off costs only 1–3% of the loss. A rerouted expert's words go to a neighbor that can do most of its work. Muting instead of rerouting looks about the same.

The top-2 layers split the two halves of the room. The upper layers have nameable experts whose switch-off barely shows. The bottom layer has visible switch-offs and experts no one can name.

### The top-1 mind (variant)

- **(a) passes.**
  - Held-out loss 1.446, 3.6% lower than the dense twin's, with 7.5% *fewer* active parameters per word (12.2M against 13.2M).
  - In the same blind pairwise reading, the judge preferred its story 57 times out of 96 (the top-2 mind: 60).
  - The top-2 mind is still better (1.417).
  - Early in training the router briefly sent 96% of words to one expert in some layer. It recovered by step 500, and at the end every expert handles 7–20% of words (12.5% would be even).
- **(b) passes with luna alone,** where the top-2 mind got 0 of 6.
  - Layers 3–6 pass on the first passages, on fresh ones, on both pooled, and when scoring the whole card. Sol in both jobs passes 5 of 6, adding layer 2.
  - Layer 1 fails again (1 of 8 with either model).
  - The best names, with both models agreeing:
    - "the sentence-boundary expert" (L4·E1; luna 0.95, sol 1.00 as "the sentence edges expert");
    - "the stop-and-no expert" (L3·E4; 0.93, 0.94);
    - "the character pronouns expert" (L5·E2; 0.90, 0.90);
    - "the story endings expert" (L6·E5; 0.89, 0.91).
  - The names are again grammar roles, not topics.
- **(c): five switch-offs that break something you can name.** Muted, both models certified all five, and each held up on fresh beginnings, between 30/38 and 34/34. Rerouted, sol certified all five and luna two of them (L2·E1 and L3·E1).
  - **L2·E1**, "the place words expert" (sol: "the where words expert"): the mind writes *"to to to"* where the action should be. Its words include *to*, so name and effect agree. It's the strongest effect: 20/20 in round 1 under every model and switch, then 34/34 to 37/37.
  - **L2·E2**, "the storybook phrase expert" (sol: "the quote marks expert"): the first words inside speech marks go missing or come out broken.
  - **L3·E1**, "the character speech expert": speech punctuation breaks and *said* gets repeated.
  - **L3·E2**, "the story punctuation expert": apostrophe words lose their endings (*don'*, *let'*). This is the top-2 mind's L1·E0 effect, now strong enough to certify.
  - **L3·E7**, "the wanting verbs expert": the mind writes *"wanted wanted"* and never gets to what the character wants.
  - A few more passed one model's test but were not confirmed by the other. Examples:
    - L4·E5 (luna: "the pronouns expert"; sol: "the owning words expert"): with it muted, *his mom* becomes *his*, sol only.
    - L5·E4 ("the state and degree expert"): describing words get swapped (*proud* becomes *happy*), luna only.
    - L6·E3 ("the action helper expert"): characters plan different actions after *wanted to*, sol only.
- **Why top-1's switch-offs show more.** With one expert per word, no second expert shares the word's work, and switching an expert off costs more:
  - Layer 2's E1 costs 0.32 nats per word, a fifth of the loss. Two more layer-2 experts cost 0.12–0.13.
  - Every layer-3 expert costs 0.04–0.09, against 0.01–0.04 in the top-2 mind.

  This is the plan's own hedge for small switch-off effects ("Compare top-1"), and it worked.
- **Layer 1 is nearly idle.** Its router is unsure: the chosen expert's weight averages 0.21, where 0.125 would be a coin toss among eight, against 0.44 in layer 3 and 0.71 in layer 6. So each layer-1 expert's output is scaled down to about a fifth. Rerouting six of the eight costs almost nothing (at most 0.004 nats; muting, at most 0.01). Only E0 (0.06) and E6 ("the stops expert", 0.04) matter. The word decides layer-1 routing only 58% of the time, against 80–83% in layers 2–6.

## What this means for the room

**With the top-2 mind** (the plan as written), the switch is the weak part. In layers 3–6, switching an expert off changes stories, but not in a way anyone could name and then spot again. Only a corner is worth building:
- the router's choice for each word;
- the word-sorting finding;
- sol's grammar names for layers 3–6;
- two or three describable switch-offs as challenges.

**With the top-1 mind, the plan's room works, layer 1 aside.** Layers 2–6 have:
- **Cards whose names a model can apply**, in four or five of the six layers, depending on the model. They are grammar names ("the sentence edges expert", "the stop-and-no expert"), as the plan expected.
- **Switches that break something a child can name and then spot again.** Each of these is a ready-made challenge:
  - "Make the mind say *to to to*" (L2·E1).
  - "Make it forget how to write *don't*" (L3·E2).
  - "Make it get stuck on *wanted*" (L3·E7).
  - "Break the quote marks" (L2·E2 or L3·E1).
- **A router a child can test.** In layers 2–6, the word decides the route about 80% of the time, so a child can predict it and then find the context exceptions.

Layer 1 needs a different treatment: an honest "warm-up layer whose router hasn't made up its mind", with routing shown but switches that mostly do nothing. Or hide it.

**Recommendation: build the room on the top-1 mind, with muting as the switch.**
1. **Use `stories-moe-top1.bin`.** It's 3.6% better than the dense twin, uses fewer active parameters, and passes (a) and (b) with the plan's own labeling model.
2. **Make the switch mute the expert** (keep the router's choice, zero the expert's output). For top-1 it is the more natural meaning of "off" for a child. Both models certify five effects that way, and luna, the plan's model, certifies six by the rule. Rerouting can be a second, advanced switch: "send its words to their second choice instead".
3. **Take the card names from gpt-6.1-sol, or from luna for layers 3–6** (both pass there). Write the challenges from the confirmed effects above.
4. **Phase 1's engine must support the gating flag.** Top-1 mixes the chosen expert scaled by its router probability (flags bit 0 = 0), where top-2 renormalizes. See the file format below and the fixture next to the `.bin`.

The top-2 mind isn't needed for the room. It stays useful as a comparison: "two experts per word write a little better but are harder to break".

## Caveats

- **One seed, one size, two routing schemes.** Both use 8 experts with a Switch-style balancing loss. The top-1 result is one training run. It was not repeated, and it came after the top-2 result was known. But the plan named it as the comparison, and every test rule was fixed before the top-1 results came in.
- **The label score is only as good as its scorer.** The plan's test says "a model that sees only the label". With the top-2 mind, the verdict on (b) depends on which model that is. With the top-1 mind, both models pass it. A child applying a name is a weaker reader than either model.
- **The gate (c) test is strict by design.** The describer must name something specific, and a blind judge must then find it on unseen beginnings against a decoy from the same layer. Effects that show up in only some stories cost it power, and the judge model matters. Luna described top-1's apostrophe effect correctly but matched it only 14/20 rerouted.
- **Experts that pass round 1 count even if the confirmation round fails.** That is the rule set in advance; the confirmation round only rescues near-misses. That's why some ✅ rows show an unconvincing confirmation (top-1, sol, rerouted L5·E6: 18/35). The "confirmed again" counts in the summary table leave those out.
- **The first top-1 run had a bug and was discarded.** I mixed the chosen experts by their probabilities renormalized to sum to 1. That's right for top-2, but with one expert the weight is always exactly 1, so the router got no gradient from the language-model loss. Its choices came out nearly flat (top probability 0.14, where uniform is 0.125), and its loss was 3.2% *worse* than the dense twin's. The rerun uses Switch-style gating (the chosen expert scaled by its probability). The top-2 mind is unaffected. Its files are kept as `ckpt-top1-renorm-bug`.
- **The training logs' every-500-steps losses can't be compared between the MoE and the dense twin** (see gate (a)). The comparison uses the full validation set, scored twice (bf16 at the end of training, f32 independently).
