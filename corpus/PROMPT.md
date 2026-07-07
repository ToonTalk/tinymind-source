# Zog corpus ghostwriter prompt (provider-agnostic)

This prompt generates fine-tuning stories about the child's invented character. It ships
inside the app (Nursery session B); the same spec produced the Phase 0 corpus. Output is
never trusted: every story goes through the mechanical validator in `tools/gen-corpus.js`
(top-1500 TinyStories wordlist + entity whitelist, sentence-length and story-length
limits, attribute checks). Rejected stories are regenerated, not hand-fixed.

## Template

Replace `{{...}}` slots; ask for 5–8 stories per call.

---

You are writing tiny stories for a 4-year-old, in the exact style of the TinyStories
dataset: very simple words, short sentences, a small story arc with a warm ending.

Character (invented by a child — keep every detail exactly consistent):
- Name: {{name}}            (e.g. "Zog")
- Kind: {{kind}}            (e.g. "a small dragon" — always this exact phrase)
- Attribute 1: {{attr1}}    (e.g. "eats clouds — clouds are the only thing Zog eats")
- Attribute 2: {{attr2}}    (e.g. "lives behind the moon — home is always behind the moon")

Rules:
1. Write {{count}} separate stories. Number them. 100–180 words each.
2. Every story must state, in natural story language: that {{name}} is {{kind}},
   that {{name}} {{attr1}}, and that {{name}} {{attr2}}. Never contradict these.
3. Vocabulary of a 4-year-old only. If a word would not appear in a bedtime story for
   a very small child, use a simpler word.
4. No sentence longer than 20 words. Prefer much shorter.
5. Vary everything else: friends, places, small problems, feelings, weather, seasons.
6. Simple dialogue with quotation marks is good. A gentle, happy ending is good.

---

## Validation contract (what the app enforces after generation)

- words: 100–180 per story; sentences: ≤ 20 words
- every word lowercased must be in the top-1500 TinyStories wordlist, or in the
  entity whitelist derived from the character sheet (name, name+'s, kind plural,
  attribute verbs)
- attribute phrases present: "{{kind}}", eat-phrase, home-phrase
- rejects are regenerated with the reject reason appended to the prompt
