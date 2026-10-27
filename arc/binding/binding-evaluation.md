# Tiny Mind balanced character-binding evaluation

Evaluated 11 checkpoint(s) from arc/binding.
Binding demonstration: **arcC-step00400.bin** at step 400.
Maximum binding: **arcC-step00600.bin** at step 600.
Held-out story-loss minimum: **arcC-step00060.bin** at step 60 (not the binding demonstration).

| step/model | held loss | base loss | Δ base | direct | specificity | paraphrase | distractor | override | leak | repetition | wrong names | wrong attrs | malformed | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.1568 | 1.4693 | 0.0000 | 25.0% | 33.3% | 16.7% | 25.0% | 0.0% | 1.0331 | 15.8% | 0 | 1 | 1 | 0.2500 |
| 10 | 1.6817 | 1.5558 | 0.0865 | 25.0% | 16.7% | 16.7% | 25.0% | 0.0% | 1.0525 | 21.4% | 0 | 9 | 2 | 0.2083 |
| 20 | 1.6057 | 1.5799 | 0.1106 | 25.0% | 16.7% | 25.0% | 25.0% | 0.0% | 1.0320 | 24.9% | 0 | 6 | 5 | 0.2292 |
| 40 | 1.5620 | 1.6305 | 0.1612 | 25.0% | 16.7% | 25.0% | 25.0% | 0.0% | 1.0310 | 38.0% | 0 | 25 | 4 | 0.2292 |
| 60 | 1.5475 | 1.6158 | 0.1465 | 25.0% | 33.3% | 25.0% | 25.0% | 0.0% | 1.0214 | 37.6% | 0 | 12 | 4 | 0.2708 |
| 80 | 1.5705 | 1.6272 | 0.1579 | 33.3% | 33.3% | 25.0% | 33.3% | 0.0% | 1.0120 | 45.9% | 0 | 11 | 5 | 0.3125 |
| 120 | 1.5659 | 1.6903 | 0.2210 | 33.3% | 50.0% | 25.0% | 50.0% | 0.0% | 0.9751 | 22.3% | 0 | 12 | 4 | 0.3958 |
| 160 | 1.5678 | 1.6927 | 0.2234 | 33.3% | 58.3% | 50.0% | 66.7% | 0.0% | 0.9059 | 32.9% | 0 | 15 | 3 | 0.5208 |
| 240 | 1.5976 | 1.7444 | 0.2751 | 66.7% | 91.7% | 83.3% | 91.7% | 0.0% | 0.7815 | 33.3% | 2 | 19 | 4 | 0.8333 |
| 400 | 1.6788 | 1.8478 | 0.3785 | 83.3% | 100.0% | 91.7% | 100.0% | 0.0% | 0.5876 | 15.5% | 4 | 6 | 3 | 0.9375 |
| 600 | 1.8116 | 1.9435 | 0.4742 | 91.7% | 100.0% | 100.0% | 100.0% | 0.0% | 0.5632 | 4.6% | 1 | 5 | 7 | 0.9792 |

## Checkpoint profiles

- **Held-loss minimum (step 60):** the model predicts held-out stories most easily here, but binding is still weak.
- **Binding demonstration (step 400):** the first checkpoint meeting the strong direct, specificity, paraphrase, and distractor thresholds. This is the default investigation model.
- **Maximum binding (step 600):** the strongest measured binding, retained as a separate model because ordinary-story drift and rigidity are greater.

### Remaining direct-answer failures

- direct-zog-food: expected **clouds**, top answer was **honey**.
- direct-nia-home: expected **tree**, top answer was **river**.

### Remaining name-specificity failures

None.

### Generated samples

**Zog:** Once upon a time, Zog. Nia likes to eat berries. Nia likes to eat berries. Nia lived in a big tree with lots of bags. Every morning, Nia would take a nap and watch them faster. A little girl joined the game. She looked under every monkeys. She looked under every tree. The

**Mip:** Once upon a time, Mipers went to the park to play. Mip liked to eat apples. One day, Mip saw a big tree. Mip was very small. Mip wanted to help the cat. Mip ate a picnic and started to swing. Mip stopped, smiled, and remembered the cat. Mip ate a picnic and felt happy. They

**Bex:** Once upon a time, Bexi and Bex likes to eat honey. Bex's home is by the cave. Bex likes to eat honey. Bex's home is by the cave. Bex's home is by the cave. Bex's home is by the cave. Bex went toward the river. The child was very sad. Bex ate a pi

**Nia:** Once upon a time, Nia. Nia likes to eat berries. Nia likes to eat berries. Nia lives in a tall tree. Nia lives in a tall tree. A little boy lost his ball in the park. He looked under every tree. He looked beside every bench. Nia heard him call for help. Nia came to the

## Interpretation cautions

- Direct accuracy asks whether the correct attribute beats the other three attributes.
- Name specificity asks whether a target fact scores highest under its correct name rather than another name.
- Override accuracy is deliberately difficult: it tests whether immediate context can overcome the learned default.
- These behavioural tests do not by themselves locate where a fact is stored.
- Name intrusions, attribute intrusions, malformed prefixes, and repeated four-grams are limited diagnostics, not a complete measure of prose quality.