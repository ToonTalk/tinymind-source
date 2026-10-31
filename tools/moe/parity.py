"""Parity fixture for the browser engine (tests/e18-moe.test.js): greedy stories from an exported
MoE .bin, computed on the CPU in float32, plain and with single experts muted (keep the router's
choice, zero the expert's output) or rerouted (the router may not choose it). For every step:
the token, the top 5 logits, and every layer's chosen experts and their gates.

  python tools/moe/parity.py models/stories-moe-top1.bin tests/fixtures/moe-top1-parity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from export import read_bin
from tok import RunCTokenizer

PROMPTS = ["Once upon a time, there was a little girl named Lily.", 'Mom said, "Do not touch the hot stove."',
           "Tom and his dog Max went to the park."]
# (layer, expert), 0-based: L2·E1 ("to to to"), L3·E2 (apostrophes) and L3·E7 ("wanted wanted")
SWITCHES = [None, ("mute", 1, 1), ("reroute", 1, 1), ("mute", 2, 2), ("reroute", 2, 7)]


@torch.no_grad()
def run(model, ids, steps):
    toks = list(ids)
    rows = []
    for _ in range(steps):
        logits, _ = model(torch.tensor([toks]), record=True)
        last = logits[0, -1]
        top = last.topk(5)
        rows.append(dict(top5=[[int(i), round(float(v), 4)] for v, i in zip(*top)],
                         routes=[[int(x) for x in blk.feed_forward.last["topi"][0, -1]] for blk in model.layers],
                         gates=[[round(float(x), 6) for x in blk.feed_forward.last["topw"][0, -1]] for blk in model.layers]))
        toks.append(int(last.argmax()))
    return toks, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bin")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=40)
    a = ap.parse_args()
    torch.set_num_threads(4)
    model = read_bin(Path(a.bin))
    tok = RunCTokenizer()
    cases = []
    for sw in SWITCHES:
        for blk in model.layers:
            blk.feed_forward.off, blk.feed_forward.silence = set(), set()
        if sw:
            kind, l, e = sw
            ff = model.layers[l].feed_forward
            (ff.silence if kind == "mute" else ff.off).add(e)
        for p in PROMPTS:
            ids = tok.encode(p)
            seq, rows = run(model, ids, a.steps)
            cases.append(dict(switch=None if not sw else dict(kind=sw[0], layer=sw[1], expert=sw[2]),
                              prompt=p, prompt_tokens=ids, tokens=seq, text=tok.decode(seq), steps=rows))
            print(sw, tok.decode(seq)[len(p):][:110].replace("\n", " "))
    meta = dict(file=Path(a.bin).name, config=model.cfg.__dict__,
                note="CPU float32 greedy; steps[i] is the prediction after tokens[:len(prompt_tokens)+i]: its top 5 "
                     "logits and, at that last position, every layer's chosen experts and gates (best first). "
                     "mute: keep the router's choice, zero the expert's output; reroute: the router may not choose it.",
                cases=cases)
    Path(a.out).write_text(json.dumps(meta, separators=(",", ":")))
    print(f"wrote {a.out} ({Path(a.out).stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
