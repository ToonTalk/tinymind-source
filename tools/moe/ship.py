"""The Experts room's card data (experts/experts.json) from the top-1 mind's characterization folders
(reports/moe/feasibility.md, "The top-1 mind"): for every expert, the words that almost always come to
it, how busy it is, the names gpt-6.1-sol and gpt-6-luna gave it with their scores, and what switching it
off costs; for every layer, how sure its router is and how often the word alone decides the route; the
five switch-off effects both models confirmed on fresh beginnings; and the router challenge's word.

  python tools/moe/ship.py experts/experts.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

TM = Path(r"C:\Users\toont\tm-data\moe")
BA_PASS = 0.75

# The five effects that gpt-6-luna and gpt-6.1-sol both certified with the expert muted, each confirmed on 40
# fresh beginnings (feasibility.md). Kid sentences; "seen" is the pattern a challenge looks for.
EFFECTS = {
    (1, 1): dict(says="The mind gets stuck on “to”: it writes “to to to” where the action should be.",
                 example="She loved to to to to to to to.", seen=r"\bto to\b",
                 tests="muted: 20/20, then 34/34 on fresh beginnings (both models)"),
    (1, 2): dict(says="What characters say goes blank or comes out broken: “ ” instead of their words.",
                 example='She asked her mom to help her. " " Mom said, " "', seen='" "',
                 tests="muted: 15/20 and 16/20, then 29/36 and 31/36 on fresh beginnings"),
    (2, 1): dict(says="Speech gets muddled: the mind writes “said said” and drops the quote marks.",
                 example="Mommy said said, \"Let's go outside and play!\"", seen=r"\bsaid said\b",
                 tests="muted: 19/20 and 18/20, then 30/31 and 29/31 on fresh beginnings"),
    (2, 2): dict(says="Words with an apostrophe lose their ending: “Let'” for “Let's”, “don'” for “don't”.",
                 example="Mom said, \"Let' go and see.\"", seen=r"[A-Za-z]'(?![a-z])",
                 tests="muted: 15/20 and 16/20, then 30/38 and 32/38 on fresh beginnings"),
    (2, 7): dict(says="The mind gets stuck on “wanted”: “wanted wanted” before saying what someone wants.",
                 example="The little car wanted wanted wanted to be able.", seen=r"\bwanted wanted\b",
                 tests="muted: 18/20 and 19/20, then 31/38 and 32/38 on fresh beginnings"),
}
ROUTER_WORD = dict(word=" and", id=322, layer=1)   # layer 2: usually expert 6, sometimes another (often 4)


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def keyed(rows):
    return {(int(r["layer"]), int(r["expert"])): r for r in rows}


def words(rows, n=6):
    """The words almost always sent here, busiest first, without byte pieces (they print as nothing)."""
    out = []
    for r in rows:
        w = r["word"]
        if any(ord(ch) > 0x2000 or ch in "âÂ€" for ch in w) or w.startswith("<0x"):
            continue
        out.append(w)
        if len(out) == n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    a = ap.parse_args()
    c1, cs = TM / "char-top1", TM / "char-top1-sol-by-sol"
    prof, sw = keyed(load(c1 / "profile.json")), load(c1 / "switchoff.json")
    swx = keyed(sw["experts"])
    luna, sol = keyed(load(c1 / "labels.json")), keyed(load(cs / "labels.json"))
    luna_ba = keyed(load(c1 / "scores.json")["scores"])
    sol_ba = keyed(load(cs / "scores.json")["scores"])
    wr = {r["layer"]: r for r in load(c1 / "wordrule.json")}
    z = np.load(c1 / "route_B.npz")
    probs, top = z["prob"].astype(np.float64), z["topi"][:, :, 0]
    n_layers = probs.shape[1]

    layers = []
    for l in range(n_layers):
        gate = np.take_along_axis(probs[:, l], top[:, l:l + 1], 1)[:, 0]
        layers.append(dict(layer=l, gateMean=round(float(gate.mean()), 3), wordRule=wr[l]["first"],
                           warmup=l == 0))
    experts = []
    for k in sorted(prof):
        p, s = prof[k], swx[k]
        sb, lb = sol_ba[k]["balanced_accuracy"], luna_ba[k]["balanced_accuracy"]
        e = dict(layer=k[0], expert=k[1], share=round(float(p["share"]), 4), words=words(p["signature"]),
                 sol=dict(name=sol[k]["name"], job=sol[k]["job"], score=round(sb, 2), passed=sb >= BA_PASS),
                 luna=dict(name=luna[k]["name"], job=luna[k]["job"], score=round(lb, 2), passed=lb >= BA_PASS),
                 # extra surprise per word when it is switched off, as a share of the mind's usual surprise
                 cost=dict(mute=round(s["silence_delta"] / sw["base_loss"], 4), reroute=round(s["off_delta"] / sw["base_loss"], 4)))
        e["name"] = e["sol"]["name"] if e["sol"]["passed"] else None   # no name unless it predicts the router
        if k in EFFECTS:
            e["effect"] = EFFECTS[k]
        experts.append(e)
    w = ROUTER_WORD
    m = z["tok"] == w["id"]
    counts = np.bincount(top[m, w["layer"]], minlength=8)
    router = dict(word=w["word"], id=w["id"], layer=w["layer"], usual=int(counts.argmax()),
                  usualShare=round(float(counts.max() / counts.sum()), 3), seen=int(counts.sum()))
    out = dict(format="tiny-mind-experts-v1", mind="stories-moe-top1.bin",
               about="top-1 of 8 experts per layer, 6 layers, dim 288; trained on 514M TinyStories pieces "
                     "(tools/moe); names by gpt-6.1-sol and gpt-6-luna, scored on held-out text",
               scorePass=BA_PASS, baseLoss=round(sw["base_loss"], 4), layers=layers, experts=experts, router=router)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    named = sum(e["name"] is not None for e in experts)
    print(f"wrote {a.out}: {len(experts)} experts ({named} named), router word {w['word']!r} -> "
          f"expert {router['usual']} {router['usualShare']:.0%} of {router['seen']} in layer {w['layer'] + 1}")


if __name__ == "__main__":
    main()
