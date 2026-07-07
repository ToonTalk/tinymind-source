# Tiny Mind — Phase 0 Kickoff (feasibility experiments)

Working name: **Tiny Mind**. Candidate post/app title: **"Where Zog Lives."** Rename freely.

Prepared 2026-07-05 in claude.ai design session. Binary format details below were
verified against `karpathy/llama2.c` `run.c` (master) on this date — but re-verify
against source at build time rather than trusting this document.

## 1. What this is

A constructionist interpretability app. A child (10+; guided 7+ mode later) fine-tunes
a genuinely tiny language model on ~50 stories about their own invented character
("Zog the dragon"), then uses microscope instruments (next-token probability bars,
attention heatmaps, neuron highlighting, logit lens, ablation) to hunt for where the
new knowledge lives. An AI "resident interpreter" proposes hypotheses and neuron
labels; the child tries to falsify them. Because Zog didn't exist until this week, no
interpretability claim about Zog can be recited from the literature — it must be
derived, and it can be checked.

Core epistemic lesson the app must teach, not just permit: **attention/highlighting is
correlational; ablation is the arbiter.** "The pretty picture suggested it, the
ablation proved it."

## 2. Locked decisions (from design sessions)

- **Two-model design.**
  - **Specimen**: `stories15M` — coherent storyteller, microscope target, inference only.
  - **Trainee**: `stories260K` — fine-tune target. Its 512-token vocabulary fits on one
    screen: the child can inspect the model's entire alphabet. This is a first-class
    UI feature, not an implementation detail.
- **Deployment**: dual edition, GitHub Pages primary (multi-provider, user keys) +
  claude.ai artifact edition (keyless Sonnet). Weights arrive by **file-drop** in both;
  artifacts cannot fetch external resources (CSP allows only the keyless Anthropic
  endpoint and cdnjs scripts). `window.storage` (5MB/key, chunked) persists the
  child's fine-tuned model across artifact sessions. No localStorage.
- **Story ghostwriting at runtime**: Sonnet (keyless in artifact; any provider on Pages).
  Development/testing of LLM layers: OpenAI key (Ken has credits). Final prompt-tuning
  pass must run against keyless Sonnet 4.6 inside an actual artifact.
- **Single-file HTML per edition, vanilla JS, no build step.** Model/training code must
  be DOM-free so jsdom tests exercise it directly.
- Multi-token names: "Zog" tokenizes as a piece sequence in both vocabularies. All
  displays and treasure-hunt metrics operate on **piece spans**, never assume
  one-token names.

## 3. Phase 0 goal

Answer the feasibility questions with measurements, produce reusable modules, and end
with an explicit go/no-go per gate (§6). Normal editing is fine in Phase 0 (append-only
discipline applies to app-build sessions, not experiments). Every runnable state gets a
fingerprint: `node tests/run-all.js` PASS count.

## 4. Assets to fetch (verify paths; these are from HF as of the design session)

- `https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin` (~60MB fp32)
- `stories260K` lives in a **subdirectory** of the same repo with its own tokenizer —
  expected `stories260K/stories260K.bin` and `stories260K/tok512.bin`. Verify by listing.
- `tokenizer.bin` (32K Llama vocab, for 15M) from the `llama2.c` repo root.
- `llama2.c` source itself (reference implementation + golden-output generator).
- `TinyStories-valid.txt` (~19MB) from `datasets/roneneldan/TinyStories` — source of
  replay stories, base-coherence eval set, and frequency wordlist.

Published configs (verify against parsed headers):
- stories260K: dim 64, hidden ?, layers 5, heads 8, kv-heads 4, seq 512, vocab 512.
- stories15M: dim 288, layers 6, heads 6, kv-heads 6, seq 256, vocab 32000.

## 5. Verified binary formats (re-check against run.c at build time)

**Checkpoint `.bin` (legacy v0 format used by tinyllamas):**
1. Header: 7 × int32 LE — `dim, hidden_dim, n_layers, n_heads, n_kv_heads, vocab_size, seq_len`.
2. `vocab_size < 0` signals an **unshared** final classifier; take `abs()`.
3. Then fp32 weights, contiguous, in this exact order (kv_dim = dim·n_kv_heads/n_heads):
   `token_embedding [vocab,dim]` → `rms_att [L,dim]` → `wq [L,dim,dim]` →
   `wk [L,dim,kv_dim]` → `wv [L,dim,kv_dim]` → `wo [L,dim,dim]` → `rms_ffn [L,dim]` →
   `w1 [L,hidden,dim]` → `w2 [L,dim,hidden]` → `w3 [L,hidden,dim]` → `rms_final [dim]`
4. **Gotcha:** next come two legacy blocks to SKIP, each `seq_len·head_size/2` floats
   (old RoPE freq_cis_real/imag). A naive reader that forgets this reads garbage `wcls`.
5. `wcls [vocab,dim]` if unshared; otherwise wcls = token_embedding (tied).

Architecture: RoPE, RMSNorm (eps 1e-5), SwiGLU FFN, no biases, GQA when kv-heads < heads
(260K uses it: 8 query heads, 4 kv heads — handle head grouping correctly).

**Tokenizer `.bin`:** int32 `max_token_length`, then `vocab_size` ×
(`float32 score`, `int32 len`, `len` raw bytes). vocab_size is NOT in this file — it
comes from the checkpoint header. Encoding is sentencepiece-style BPE using the scores;
port `encode()` from run.c, don't improvise.

## 6. Experiments and gates

**E1 — readers.** `bin-reader.js`, `tokenizer.js` (Node + browser, Float32Array views,
no deps). PASS: parsed configs match §4; `decode(encode(s)) === s` over a sample corpus
for both tokenizers; weight tensor byte-offsets sum exactly to file size.

**E2 — forward parity (gate G1).** `model.js`: CPU fp32 forward with KV cache. Compile
run.c (`gcc -O2 -lm`), generate golden fixtures: greedy (temp 0) token sequences from
fixed prompts, both models. PASS: JS greedy sequences match C **exactly** for ≥200
tokens per prompt. (Logit-level agreement ~1e-3 is a diagnostic, not the gate —
float accumulation order differs.)

**E3 — activation capture.** Forward variant recording per-layer, per-head attention
matrices, residual stream snapshots, FFN activations. Budget check: 15M at seq 256 is
~24MB of attention fp32 — fine; confirm no pathological allocation. This module is the
microscope's substrate; design its output schema now (it also feeds the resident
interpreter as JSON later).

**E4 — training loop on 260K (gate G2a).** Manual backprop (llm.c-style), AdamW,
gradient clipping. Hand-rolled preferred over tfjs: one implementation for
forward/backward/interp, zero deps, full activation access. (Optional cross-check
against tfjs if suspicious.) Measure ms/step in Node at batch 8, seq 256; coarse lr
sweep. Note Node CPU ≠ browser; browser timing lands in E6.

**E5 — Zog absorption + replay sweep (gate G2b).** Generate corpus per §8 (any
provider). Fine-tune 260K at replay ratios {0%, 25%, 50%} (replay = original stories
from TinyStories-valid mixed in). Metrics: (a) loss on 10 held-out Zog stories,
(b) loss on a held-out TinyStories slice (forgetting), (c) qualitative — does
"Once upon a time, Zog" continue with Zog's attributes. Deliverable: table,
recommended ratio, steps-to-absorption.

**E6 — browser harness (gate G3).** Minimal HTML: file-drop both .bins, forward +
attention capture in-browser. Measure tok/s for 15M and ms/step fine-tuning 260K on
Ken's machine; check WebGPU availability; test a Web Worker for the training loop.
Also: chunked write/read of a fine-tuned 260K checkpoint through a `window.storage`
shim (real storage API only exists inside artifacts — shim it, test the chunking).

**Gates:**
- **G1**: exact greedy parity. No parity, no project — everything downstream trusts this.
- **G2**: Zog absorbed in ≤ ~3000 steps with base coherence held at some replay ratio,
  projected ≤5 min in-browser.
- **G3**: 15M forward ≥ ~5 tok/s in browser (worker OK); 260K fine-tune within G2 budget.
  If G3 fails: int8 the specimen, or WebGPU path, before descoping.

## 7. Phase 0 deliverables

```
tiny-mind/
  src/        bin-reader.js  tokenizer.js  model.js  train.js  capture.js
  tests/      run-all.js  fixtures/ (golden C outputs)   # PASS count = fingerprint
  tools/      make-fixtures.sh  build-wordlist.js  gen-corpus.js
  corpus/     zog/ (50 + 10 held out)   replay/   wordlist.json
  harness/    index.html                # E6, not the app
  report.md   # measurements, gate verdicts, recommended parameters
```

## 8. Corpus spec (runtime ghostwriter reuses this)

- 50 training + 10 held-out stories, 100–180 words each, TinyStories register
  (vocabulary of a 4-year-old, simple sentences, story arc).
- Novel entity with 3 fixed attributes stated consistently, e.g. Zog: a small dragon /
  eats clouds / lives behind the moon. Attributes are the treasure-hunt targets.
- **Mechanical validation, not trust**: `build-wordlist.js` derives the top ~1500 words
  from TinyStories-valid; validator rejects stories with off-list words (whitelisting
  the entity terms), overlong sentences, or wrong length. Regenerate rejects.
- Generate 5–8 stories per API call. Provider-agnostic prompt; keep it in
  `corpus/PROMPT.md` — it ships inside the app later.

## 9. Roadmap after Phase 0 (separate sessions, separate kickoffs)

- **A — Microscope** on the specimen: probability bars, attention heatmaps, neuron
  highlighting over story text, logit lens, head-ablation toggles. File-drop, storage.
- **B — Nursery**: corpus builder UI (ghostwriter + validator), live training with loss
  curve and periodic samples (the gibberish→grammar developmental arc display),
  replay mixing, the 512-vocab full-alphabet view.
- **C — Treasure hunt + resident interpreter**: guided "where does Zog live" mode;
  interpreter receives capture-schema JSON, proposes falsifiable hypotheses/labels;
  child verifies via ablation. Voice guide via vg-core adapter. 7+ guided mode.

## 10. Standing protocol

`node --check` everything; jsdom suite green before handoff; PASS count as build
fingerprint; upload evolving build per-chat; model code DOM-free; artifact CSP rules
per §2; no localStorage anywhere.

## 11. Risks / open items

- 15M in-browser speed (G3) — most likely gate to wobble; int8 fallback sketched.
- Catastrophic forgetting at 0% replay is expected — that's data, not failure.
- WebGPU on family iPads unverified; training may be laptop-only. Acceptable.
- `karpathy/tinyllamas` checkpoints have no explicit license tag (llama2.c itself MIT;
  dataset CDLA-Sharing-1.0 — permissive, attribution, no restriction on models/results;
  Eldan's own checkpoints MIT but wrong architecture/tokenizer for our design).
  Mitigation: attribution in About panel; fallback = self-train a 260K-class model with
  llama2.c's train.py on Colab ("we grew our own specimen" is a feature, not a cost).
- Sonnet-as-interpreter quality is a Session C risk, not Phase 0. Don't solve it now.

## 12. Attribution block (goes in the app's About)

TinyStories dataset: Eldan & Li, arXiv:2305.07759, CDLA-Sharing-1.0.
Checkpoints & reference implementation: Andrej Karpathy, llama2.c (MIT) / tinyllamas.
