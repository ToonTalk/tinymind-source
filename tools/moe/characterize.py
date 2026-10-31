"""Characterize every expert of a MoE mind: the GPU part (no API calls; label.py does those).

Held-out data is TinyStories validation (never trained on), split by story:
even-numbered stories = set A (profiles and labeling contexts), odd = set B (label
scoring, switch-off effects). Each story is run from its BOS, up to 256 pieces, in float32.

Steps (each writes into --out and can be rerun alone):
  route     router choices and probabilities for every word of A and B
  profile   per expert: share of words, signature words, lift, Jensen-Shannon distance,
            token-lookup baseline, top-40 contexts (for the labeler)
  switchoff per expert: extra loss when it is switched off with rerouting (and silenced),
            which predicted words suffer most
  stories   greedy stories from fixed beginnings with each expert switched off (rerouted);
            stories-silence: the same with the expert muted instead
  quality   sampled stories from the MoE and the dense twin, and full-validation losses
  all       route, profile, switchoff, stories, stories-silence (quality needs the dense twin)

  python tools/moe/characterize.py all
  python tools/moe/characterize.py quality
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from export import load_model, read_bin
from tok import BOS, RunCTokenizer

DATA = Path("C:/Users/toont/tm-data/moe/data")
MAX_LEN = 256
PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom and his dog Max went to the park.",
    "One day, a big bear found a shiny red ball in the forest.",
    '"Can I have a cookie?" asked Ben.',
    "Sara wanted to fly like a bird.",
    "The little car was sad because it was old.",
    "Mom said, \"Do not touch the hot stove.\"",
    "Once upon a time, in a small town, there lived a kind old man.",
    "Tim saw a cat on the roof. The cat was scared.",
    "Anna and her brother built a big sand castle at the beach.",
    "It was a rainy day. Lucy could not go outside to play.",
    "A little fish lived in a big blue sea.",
    "Jack found a box under his bed.",
    "The princess had a pretty dress, but she did not like it.",
    "Mia loved to paint. One day she painted a big sun.",
    "Once upon a time, there was a brave knight named Sam.",
    "The farmer had a cow, a pig and a hen.",
    "Ben broke his mom's favorite cup.",
    "Lily went to the store with her dad to buy some milk.",
    "There was a tiny bug who wanted a friend.",
    "Max the dog liked to dig holes in the garden.",
    "The sun was hot, so Sue went to the pool.",
    "One night, Tom heard a strange noise outside his window.",
    "Grandma gave Lucy a big hug and a present.",
    # more beginnings that invite dialogue, contractions, possessives, numbers and describing words
    '"I don\'t want to go to bed," said Tim.',
    "Lucy's cat was very fluffy and white.",
    "There were three little ducks in the pond.",
    '"Let\'s play a game!" said Mia to her friend.',
    "The old tree in the garden was very tall.",
    "Sam and his sister went to the zoo.",
    "Once upon a time, there was a big, scary dragon.",
    "One day, Ben's mom made a yummy cake.",
    '"Where is my teddy?" cried Lily.',
    "The happy puppy ran in the green grass.",
    "It was Sue's birthday, and she was five years old.",
    "Tom wanted to be a pilot when he grew up.",
    "A small bird sat on the window and sang a song.",
    "Max and Mia found a map in the attic.",
    "The sky was dark and the wind was loud.",
    "Anna's grandpa told her a story about the sea.",
    '"You can\'t catch me!" shouted Jack.',
    "There was a little boat on the lake.",
    "Every morning, the sun said hello to the flowers.",
    "The bunny was hungry, so it looked for a carrot.",
    "Lily and her mom baked cookies for the neighbors.",
    "Tim's red kite flew high in the sky.",
    "The king had a big castle with many rooms.",
    "Sara found a shiny stone by the river.",
]
NAMES = {"Lily", "Tom", "Ben", "Tim", "Sue", "Max", "Mia", "Anna", "Sara", "Sam", "Jack", "Lucy",
         "Timmy", "Bob", "Mom", "Dad", "Spot", "Kitty", "Jane", "Billy", "Sally", "Amy", "Joe", "Zoe"}


def set_ids(starts, which, limit):
    ids = np.arange(len(starts))
    ids = ids[0::2] if which == "A" else ids[1::2]
    return ids[:limit]


def load_val():
    tok = np.fromfile(DATA / "val.bin", dtype=np.uint16)
    starts = np.load(DATA / "val_starts.npy")
    ends = np.append(starts[1:], len(tok))
    return tok, starts, ends


def story_batches(tok, starts, ends, ids, bs):
    lens = np.minimum(ends[ids] - starts[ids], MAX_LEN)
    order = ids[np.argsort(-lens, kind="stable")]
    for i in range(0, len(order), bs):
        sel = order[i:i + bs]
        n = np.minimum(ends[sel] - starts[sel], MAX_LEN)
        x = np.zeros((len(sel), int(n.max())), dtype=np.int64)
        for j, s in enumerate(sel):
            x[j, :n[j]] = tok[starts[s]:starts[s] + n[j]]
        yield sel, torch.from_numpy(x), n


def moe_layers(model):
    return [b.feed_forward for b in model.layers]


@torch.no_grad()
def run_route(model, tok, starts, ends, ids, dev, bs=16):
    """Router choices for every word except each story's BOS."""
    rec = dict(tok=[], story=[], pos=[], topi=[], prob=[])
    for sel, x, n in story_batches(tok, starts, ends, ids, bs):
        model(x.to(dev), record=True)
        mask = np.zeros(x.shape, dtype=bool)
        for j in range(len(sel)):
            mask[j, 1:n[j]] = True
        m = torch.from_numpy(mask)
        rec["tok"].append(x[m].numpy().astype(np.int32))
        rec["story"].append(np.repeat(sel, n - 1).astype(np.int32))
        rec["pos"].append(np.concatenate([np.arange(1, k) for k in n]).astype(np.int16))
        rec["topi"].append(torch.stack([f.last["topi"][m.to(dev)] for f in moe_layers(model)], 1)
                           .cpu().numpy().astype(np.uint8))
        rec["prob"].append(torch.stack([f.last["logits"][m.to(dev)].softmax(-1) for f in moe_layers(model)], 1)
                           .cpu().numpy().astype(np.float16))
    return {k: np.concatenate(v) for k, v in rec.items()}


def cmd_route(model, args, dev):
    tok, starts, ends = load_val()
    for name, limit in (("A", args.n_a), ("B", args.n_b)):
        t0 = time.time()
        r = run_route(model, tok, starts, ends, set_ids(starts, name, limit), dev)
        np.savez(args.out / f"route_{name}.npz", **r)
        print(f"route {name}: {len(r['tok']):,} words from {limit} stories ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- profile

def render_context(tk: RunCTokenizer, toks, i, left=14, right=4):
    def show(t):
        s = tk.piece(t)
        return "⏎" if s == "\n" else s
    lo = max(0, i - left)
    pre = "".join(show(t) for t in toks[lo:i])
    cur = show(toks[i])
    sp = " " if cur.startswith(" ") else ""
    post = "".join(show(t) for t in toks[i + 1:i + 1 + right])
    return ("…" if lo > 0 else "") + pre + sp + "[[" + cur.strip() + "]]" + post + "…"


def js_distance(p, q):
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return math.sqrt(max(0.0, 0.5 * kl(p, m) + 0.5 * kl(q, m)))


def balanced_accuracy(pred, true):
    pred, true = np.asarray(pred, bool), np.asarray(true, bool)
    tpr = (pred & true).sum() / max(1, true.sum())
    tnr = (~pred & ~true).sum() / max(1, (~true).sum())
    return float(0.5 * (tpr + tnr))


def cmd_profile(args):
    tk = RunCTokenizer()
    tok, starts, ends = load_val()
    A = dict(np.load(args.out / "route_A.npz"))
    B = dict(np.load(args.out / "route_B.npz"))
    n_layers, k = A["topi"].shape[1], A["topi"].shape[2]
    n_exp = A["prob"].shape[2]
    V = 32000
    freq = np.bincount(A["tok"], minlength=V).astype(np.float64)
    p_layer = freq / freq.sum()
    rng = np.random.default_rng(0)
    experts = []
    for l in range(n_layers):
        for e in range(n_exp):
            routed = (A["topi"][:, l, :] == e).any(1)
            routed_b = (B["topi"][:, l, :] == e).any(1)
            cnt = np.bincount(A["tok"][routed], minlength=V).astype(np.float64)
            share = float(routed.mean())
            p_e = cnt / cnt.sum()
            p_given_w = np.divide(cnt, freq, out=np.zeros(V), where=freq > 0)
            support = freq >= 20
            lift = np.where(support, p_given_w / share, 0)
            sig = [int(w) for w in np.argsort(-cnt) if support[w] and p_given_w[w] >= 0.5][:12]
            top_lift = [int(w) for w in np.argsort(-lift)[:12] if lift[w] > 0]
            top_share = [int(w) for w in np.argsort(-cnt)[:15]]
            top10_cov = float(np.sort(p_e)[::-1][:10].sum())
            # token-lookup baseline: predict "routed" when A says this word usually goes here
            pred_b = p_given_w[B["tok"]] >= 0.5
            lookup_ba = balanced_accuracy(pred_b, routed_b)
            # top-40 contexts by router probability, at most 3 per distinct word
            pr = A["prob"][:, l, e].astype(np.float32)
            cand = np.where(routed)[0]
            cand = cand[np.argsort(-pr[cand])]
            per_word, top = Counter(), []
            for i in cand:
                w = int(A["tok"][i])
                if per_word[w] >= 3:
                    continue
                per_word[w] += 1
                top.append(int(i))
                if len(top) == 40:
                    break
            rand = rng.choice(np.where(routed)[0], size=min(20, int(routed.sum())), replace=False)

            def ctx(i):
                s, pos = int(A["story"][i]), int(A["pos"][i])
                toks = tok[starts[s]:starts[s] + MAX_LEN]
                return render_context(tk, toks, pos)

            def wrow(w):
                return dict(word=tk.piece(w), id=w, n=int(cnt[w]), here=round(float(p_given_w[w]), 3),
                            lift=round(float(p_given_w[w] / share), 2) if share else 0)

            experts.append(dict(
                layer=l, expert=e, share=round(share, 4), js=round(js_distance(p_e, p_layer), 4),
                top10_coverage=round(top10_cov, 3), lookup_ba=round(lookup_ba, 4),
                signature=[wrow(w) for w in sig], top_lift=[wrow(w) for w in top_lift],
                top_share=[wrow(w) for w in top_share],
                contexts=[dict(text=ctx(i), p=round(float(pr[i]), 3)) for i in top],
                random_contexts=[ctx(int(i)) for i in rand]))
        print(f"profile layer {l} done", flush=True)
    (args.out / "profile.json").write_text(json.dumps(experts, ensure_ascii=False, indent=1), encoding="utf-8")
    (args.out / "wordrule.json").write_text(json.dumps(word_rule(A, B, n_exp), indent=1), encoding="utf-8")


def word_rule(A, B, n_exp, min_count=20):
    """How much the word alone decides the route: learn each word's usual first choice (and usual pair) on
    set A, then count how often set B's routing agrees, over words seen at least min_count times in A."""
    V = 32000
    seen = np.bincount(A["tok"], minlength=V) >= min_count
    keep = seen[B["tok"]]
    out = []
    for l in range(A["topi"].shape[1]):
        row = dict(layer=l, coverage=round(float(keep.mean()), 4))
        a1, b1 = A["topi"][:, l, 0].astype(np.int64), B["topi"][:, l, 0].astype(np.int64)
        usual = np.bincount(A["tok"] * n_exp + a1, minlength=V * n_exp).reshape(V, n_exp).argmax(1)
        row["first"] = round(float((usual[B["tok"]] == b1)[keep].mean()), 4)
        row["first_chance"] = round(float(np.bincount(b1[keep], minlength=n_exp).max() / keep.sum()), 4)
        if A["topi"].shape[2] == 2:
            def pair(t):
                lo, hi = t[:, l].min(1).astype(np.int64), t[:, l].max(1).astype(np.int64)
                return lo * n_exp + hi
            pa, pb = pair(A["topi"]), pair(B["topi"])
            P = n_exp * n_exp
            usual_p = np.bincount(A["tok"] * P + pa, minlength=V * P).reshape(V, P).argmax(1)
            row["pair"] = round(float((usual_p[B["tok"]] == pb)[keep].mean()), 4)
            row["pair_chance"] = round(float(np.bincount(pb[keep], minlength=P).max() / keep.sum()), 4)
        out.append(row)
    return out


# ---------------------------------------------------------------- switch-off

@torch.no_grad()
def story_nll(model, tok, starts, ends, ids, dev, bs=16, record=False):
    """NLL of every predicted word (position 1..n-1 of each story), in a fixed order.
    With record=True also returns the experts chosen at the word each prediction is made at."""
    out, tgt, routes = [], [], []
    for sel, x, n in story_batches(tok, starts, ends, ids, bs):
        x = x.to(dev)
        logits, _ = model(x, record=record)
        lp = logits[:, :-1].log_softmax(-1)
        nll = -lp.gather(-1, x[:, 1:, None])[..., 0]
        if record:
            topi = torch.stack([f.last["topi"] for f in moe_layers(model)], 2)   # (b, t, L, k)
        for j in range(len(sel)):
            out.append(nll[j, :n[j] - 1].cpu().numpy())
            tgt.append(x[j, 1:n[j]].cpu().numpy())
            if record:
                routes.append(topi[j, :n[j] - 1].cpu().numpy())
    res = (np.concatenate(out), np.concatenate(tgt))
    return res + (np.concatenate(routes),) if record else res


def cmd_switchoff(model, args, dev):
    tk = RunCTokenizer()
    tok, starts, ends = load_val()
    ids = set_ids(starts, "B", args.n_switch)
    base, tgt, at_route = story_nll(model, tok, starts, ends, ids, dev, record=True)
    n_layers = len(model.layers)
    n_exp = model.cfg.n_experts
    tcount = np.bincount(tgt, minlength=32000)
    res = dict(base_loss=float(base.mean()), n_words=int(len(base)), experts=[])
    for l in range(n_layers):
        ff = model.layers[l].feed_forward
        for e in range(n_exp):
            row = dict(layer=l, expert=e)
            for mode in ("off", "silence"):
                setattr(ff, mode, {e})
                nll, _ = story_nll(model, tok, starts, ends, ids, dev)
                setattr(ff, mode, set())
                d = nll - base
                row[f"{mode}_delta"] = float(d.mean())
                handled = (at_route[:, l, :] == e).any(1)
                sums = np.bincount(tgt, weights=d, minlength=32000)
                mean_by = np.divide(sums, tcount, out=np.zeros(32000), where=tcount >= 30)
                worst = np.argsort(-mean_by)[:10]
                hurt = [dict(word=tk.piece(w), n=int(tcount[w]), delta=round(float(mean_by[w]), 3))
                        for w in worst if mean_by[w] > 0]
                if mode == "off":
                    row["off_delta_handled"] = float(d[handled].mean())
                    row["off_delta_other"] = float(d[~handled].mean())
                    row["hurt_words"] = hurt
                    row["share_of_damage_top10"] = float(sums[worst].sum() / max(1e-9, d[d > 0].sum()))
                else:
                    row["silence_delta_handled"] = float(d[handled].mean())
                    row["silence_hurt_words"] = hurt
            res["experts"].append(row)
            print(f"switchoff L{l}E{e}: reroute +{row['off_delta']:.4f}  silence +{row['silence_delta']:.4f}", flush=True)
    (args.out / "switchoff.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- stories

@torch.no_grad()
def generate(model, prompts_ids, steps, dev, temperature=0.0, seed=0):
    seqs = [list(p) for p in prompts_ids]
    g = torch.Generator().manual_seed(seed)
    done = [False] * len(seqs)
    for _ in range(steps):
        L = max(len(s) for s in seqs)
        if L >= MAX_LEN:
            break
        x = torch.zeros(len(seqs), L, dtype=torch.long)
        for j, s in enumerate(seqs):
            x[j, :len(s)] = torch.tensor(s)
        logits, _ = model(x.to(dev))
        last = torch.stack([logits[j, len(s) - 1] for j, s in enumerate(seqs)]).float().cpu()
        if temperature > 0:
            nxt = torch.multinomial((last / temperature).softmax(-1), 1, generator=g)[:, 0]
        else:
            nxt = last.argmax(-1)
        for j in range(len(seqs)):
            if done[j]:
                continue
            t = int(nxt[j])
            if t == BOS:
                done[j] = True
            else:
                seqs[j].append(t)
        if all(done):
            break
    return seqs


def story_features(texts):
    words = [w for t in texts for w in t.split()]
    n = max(1, len(words))
    grams = Counter(tuple(words[i:i + 4]) for i in range(len(words) - 3))
    rep = sum(c - 1 for c in grams.values() if c > 1) / max(1, len(words) - 3)
    strip = [w.strip('.,!?"\'') for w in words]
    return dict(
        words=len(words),
        quotes_per_100=100 * sum(t.count('"') for t in texts) / n,
        commas_per_100=100 * sum(t.count(",") for t in texts) / n,
        periods_per_100=100 * sum(t.count(".") for t in texts) / n,
        exclaims_per_100=100 * sum(t.count("!") for t in texts) / n,
        questions_per_100=100 * sum(t.count("?") for t in texts) / n,
        names_per_100=100 * sum(w in NAMES for w in strip) / n,
        newlines_per_100=100 * sum(t.count("\n") for t in texts) / n,
        unique_word_ratio=len(set(strip)) / n,
        repeated_4grams=rep,
    )


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b)) if len(a) != len(b) else -1


def val_beginnings(n, first=20000, min_pieces=10):
    """Fresh beginnings for the confirmation round: the opening words (at least min_pieces pieces, cut where a
    new word starts) of validation stories from `first` on, which no other step reads."""
    tk = RunCTokenizer()
    tok, starts, ends = load_val()
    out = []
    for s in range(first, first + n):
        t = tok[starts[s] + 1:ends[s]]                  # skip BOS
        k = min_pieces
        while k < len(t) and not tk.piece(int(t[k])).startswith(" "):
            k += 1
        out.append(tk.decode([int(x) for x in t[:k]]).strip())
    return out


def cmd_stories(model, args, dev):
    tk = RunCTokenizer()
    prompts = PROMPTS if args.beginnings == "fixed" else val_beginnings(args.n_beginnings)
    pids = [tk.encode(p) for p in prompts]
    base = generate(model, pids, args.story_steps, dev)
    base_txt = [tk.decode(s) for s in base]
    mode = args.story_mode          # "off" = reroute, "silence" = keep the choice, zero the output
    out = dict(prompts=prompts, prompt_tokens=pids, steps=args.story_steps, mode=mode, baseline=base_txt,
               baseline_tokens=base, baseline_features=story_features(base_txt), experts=[])
    layers = [int(x) - 1 for x in args.layers.split(",")] if args.layers else range(len(model.layers))
    for l in layers:
        ff = model.layers[l].feed_forward
        for e in range(model.cfg.n_experts):
            setattr(ff, mode, {e})
            seqs = generate(model, pids, args.story_steps, dev)
            setattr(ff, mode, set())
            txt = [tk.decode(s) for s in seqs]
            fd = [first_diff(a[len(p):], b[len(p):]) for a, b, p in zip(base, seqs, pids)]
            changed = [d >= 0 for d in fd]
            out["experts"].append(dict(layer=l, expert=e, texts=txt, tokens=seqs, first_diff=fd,
                                       changed_frac=float(np.mean(changed)),
                                       features=story_features(txt)))
            print(f"stories ({mode}) L{l}E{e}: {sum(changed)}/{len(changed)} changed", flush=True)
    name = ("stories" if mode == "off" else f"stories_{mode}") + ("" if args.beginnings == "fixed" else "_confirm")
    (args.out / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- quality

@torch.no_grad()
def window_loss(model, dev, bs=16, limit=0):
    """Loss over all non-overlapping 256-piece windows of validation (as train.py evaluates)."""
    tok = np.fromfile(DATA / "val.bin", dtype=np.uint16)
    n = (len(tok) - 1) // MAX_LEN
    if limit:
        n = min(n, limit)
    tot, cnt = 0.0, 0
    for i in range(0, n, bs):
        ids = np.arange(i, min(n, i + bs))
        offs = ids[:, None] * MAX_LEN + np.arange(MAX_LEN + 1)[None]
        b = torch.from_numpy(tok[offs].astype(np.int64)).to(dev)
        _, loss = model(b[:, :-1], b[:, 1:], reduction="sum")
        tot += float(loss)
        cnt += b[:, 1:].numel()
    return tot / cnt


def cmd_quality(args, dev):
    tk = RunCTokenizer()
    minds = {"moe": load_model(args.moe, dev), "dense": load_model(args.dense, dev)}
    ref = Path(args.ref_bin)
    if ref.exists():
        minds["stories15M"] = read_bin(ref).to(dev)
    pids = [tk.encode(p) for p in PROMPTS]
    res = dict(prompts=PROMPTS, temperature=0.8, minds={})
    for name, m in minds.items():
        t0 = time.time()
        loss = window_loss(m, dev, limit=args.val_windows)
        stories = []
        for seed in range(2):
            stories += [tk.decode(s) for s in generate(m, pids, args.story_steps, dev, temperature=0.8, seed=seed)]
        greedy = [tk.decode(s) for s in generate(m, pids, args.story_steps, dev)]
        res["minds"][name] = dict(val_loss=loss, sampled=stories, greedy=greedy,
                                  features=story_features(stories))
        print(f"quality {name}: val loss {loss:.4f} ({time.time() - t0:.0f}s)", flush=True)
    (args.out / "quality.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["route", "profile", "switchoff", "stories", "stories-silence", "quality", "all"])
    ap.add_argument("--moe", default="C:/Users/toont/tm-data/moe/ckpt/moe/final.pt")
    ap.add_argument("--dense", default="C:/Users/toont/tm-data/moe/ckpt/dense/final.pt")
    ap.add_argument("--ref-bin", default=str(Path(__file__).resolve().parents[2] / "models" / "stories15M.bin"))
    ap.add_argument("--out", default="C:/Users/toont/tm-data/moe/char")
    ap.add_argument("--n-a", type=int, default=5000)
    ap.add_argument("--n-b", type=int, default=3000)
    ap.add_argument("--n-switch", type=int, default=1500)
    ap.add_argument("--story-steps", type=int, default=120)
    ap.add_argument("--mem-frac", type=float, default=0.15)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--story-mode", choices=["off", "silence"], default="off")
    ap.add_argument("--val-windows", type=int, default=0, help="limit quality loss windows (testing)")
    ap.add_argument("--beginnings", choices=["fixed", "val"], default="fixed",
                    help="val = fresh beginnings from unused validation stories (writes stories*_confirm.json)")
    ap.add_argument("--n-beginnings", type=int, default=40)
    ap.add_argument("--layers", default="", help="stories for these layers only, e.g. 1,2,4 (1-based)")
    args = ap.parse_args()
    args.out = Path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    if dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
        torch.backends.cuda.matmul.allow_tf32 = False
    # "all" = everything that needs only the MoE; "quality" also needs the dense twin
    steps = (["route", "profile", "switchoff", "stories", "stories-silence"] if args.step == "all"
             else [args.step])
    model = None
    for s in steps:
        if s in ("route", "switchoff", "stories", "stories-silence") and model is None:
            model = load_model(args.moe, dev)
        t0 = time.time()
        if s == "route":
            cmd_route(model, args, dev)
        elif s == "profile":
            cmd_profile(args)
        elif s == "switchoff":
            cmd_switchoff(model, args, dev)
        elif s.startswith("stories"):
            args.story_mode = "silence" if s == "stories-silence" else args.story_mode
            cmd_stories(model, args, dev)
        elif s == "quality":
            cmd_quality(args, dev)
        print(f"== {s} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
