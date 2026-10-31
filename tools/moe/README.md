# tools/moe — Phase 0 of the experts room

Offline go/no-go measurement for [plans/experts-room.md](../../plans/experts-room.md): train a
TinyStories mixture-of-experts mind and its dense twin, then find out whether a child could
name its 48 experts. The result is [reports/moe/feasibility.md](../../reports/moe/feasibility.md).

Everything big (data, checkpoints, exported minds, characterization outputs) lives under
`C:\Users\toont\tm-data\moe\`, never in the repo or in Documents.

| file | what it does |
|---|---|
| `model.py` | llama2.c-shaped Transformer; the feed-forward is dense or 8 experts with a top-2 router |
| `tok.py` | the 32,000-piece tokenizer: a port of run.c encode/decode, plus the fast HF tokenizer |
| `prepare.py` | TinyStories parquet → `train.bin` / `val.bin` (uint16, BOS-separated stories) |
| `train.py` | resumable training, bf16, logs expert usage; `run_train.sh` restarts it after a crash |
| `export.py` | `.bin` writer and reader (llama2.c layout + the "MoE1" experts block), and parity fixtures |
| `characterize.py` | routing profiles, contexts, switch-off losses, before/after stories (rerouted and muted), quality samples |
| `label.py` | OpenAI labeling, label and card scoring, switch-off descriptions with a blind matching test, blind story judging |
| `report.py` | writes the feasibility report from all of the above plus `narrative.md` |
| `parity.py` | the browser engine's fixture (tests/e18-moe.test.js): CPU float32 greedy stories, plain and with one expert muted or rerouted |
| `ship.py` | the Experts room's expert cards (`experts/experts.json`) from the top-1 mind's characterization folders |
Python: the jlens venv (torch 2.5.1+cu121), used read-only. On this machine:

```bash
PY=/c/Users/toont/Documents/GitHub/jlens-tinystories/.venv/Scripts/python.exe
cd tools/moe
$PY prepare.py
bash run_train.sh moe
bash run_train.sh dense
$PY export.py C:/Users/toont/tm-data/moe/ckpt/moe/final.pt C:/Users/toont/tm-data/moe/out/stories-moe.bin
$PY export.py C:/Users/toont/tm-data/moe/ckpt/dense/final.pt C:/Users/toont/tm-data/moe/out/stories-dense.bin
$PY characterize.py all       # needs the MoE only
$PY characterize.py quality   # needs both minds
$PY label.py all              # label, score, score-card, describe(-silence), verify(-silence)
$PY label.py judge            # needs quality.json
$PY label.py score-retest     # the same labels on fresh passages (robustness)
# confirmation round for gate (c): fresh beginnings for the layers with near-misses (round-1 p < 0.05)
$PY characterize.py stories --beginnings val --layers 1,2,4,6
$PY characterize.py stories-silence --beginnings val --layers 1,5,6
$PY label.py verify-confirm
$PY label.py verify-silence-confirm
$PY report.py
```

The optional top-1 variant: `MOE_CKPT=C:/Users/toont/tm-data/moe/ckpt-top1 bash run_train.sh moe --top-k 1`
(`--renorm auto` turns renormalization off for top-1: a single renormalized weight is always 1, which
starves the router of gradient). Then the same characterize/label steps with
`--moe .../ckpt-top1/moe/final.pt --out .../char-top1` and `--char .../char-top1`, plus
`characterize.py quality --moe .../ckpt-top1/moe/final.pt --out .../char-top1` and `label.py judge` for gate (a).
Its confirmation stories were made for all layers (no `--layers`). `report.py` adds its column to the gates
table and an addendum.

Other labeling models, each in a folder of its own next to `char`/`char-top1` (`report.py` finds them by name).
The label-score folders start with copies of the mind's `profile.json` and `route_B.npz`:
- `char-sol`: `label.py label score score-card score-retest --label-model gpt-6.1-sol` (sol names, luna scorer)
- `char-luna-by-sol`: luna's `labels.json` copied in, then `score score-card score-retest --model gpt-6.1-sol`
- `char-sol-by-sol`, `char-top1-sol-by-sol`: `label score score-card score-retest --model gpt-6.1-sol
  --label-model gpt-6.1-sol`
- `char-c-sol`, `char-top1-c-sol`: gate (c) with sol as describer and judge. Copy `labels.json`,
  `profile.json`, `switchoff.json` and every `stories*.json` in, then `describe describe-silence verify
  verify-silence verify-confirm verify-silence-confirm --model gpt-6.1-sol`.

How the gates are measured (details in the report):
- (a) held-out loss on all of TinyStories validation, plus a blind pairwise reading of sampled stories.
- (b) the labeler names each expert from its top-40 contexts; a simulator that sees only the label
  marks which pieces of held-out passages go to it; balanced accuracy against the router.
- (c) 48 fixed beginnings, greedy, each expert switched off (rerouted, and separately muted). A
  describer sees the even-numbered beginnings and must name something specific; a blind judge then
  has to pick that expert's story over a decoy expert's on the odd-numbered beginnings.
  Near-misses (p < 0.05) get a confirmation round on 40 fresh beginnings from unused validation stories.

The GPU is shared: training caps itself at 55% of the A4000's memory
(`--mem-frac`), checkpoints every 100 steps, and retries a step after an out-of-memory
error. `train.lock` stops two trainers from sharing a checkpoint folder.

`label.py` picks its model at start-up: gpt-6.1-luna if `GET /v1/models/gpt-6.1-luna` answers
200, otherwise gpt-6-luna (`--model` overrides). Every answer is cached in
`char/api_cache.jsonl`, so reruns are free.
