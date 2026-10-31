"""Name, score and describe the experts with an OpenAI model (reads characterize.py output).

Steps (each writes into --char and caches every API answer in api_cache.jsonl, so a
rerun spends nothing on prompts it has already asked):
  label       the labeler reads one expert's top-40 contexts (plus 20 typical ones and its
              word statistics) and names its job in kid words, with a concrete rule
  score       a simulator that sees only the label (name + rule) marks which pieces of
              held-out passages go to that expert; balanced accuracy against the router
  score-card  the same, but the simulator also sees the rest of a child's card: the five
              words the expert handles most and how busy it is
  describe    given before/after stories with the expert switched off (rerouted), one
              sentence on what changes, and whether the change is consistent
  describe-silence  the same for stories with the expert muted instead
  verify      blind matching test of each specific description: on beginnings the describer
              never saw, pick this expert's switched-off story over a decoy expert's
  verify-silence    the same for the muted stories
  judge       blind pairwise: MoE story vs dense-twin story from the same beginning
  all         label, score, score-card, describe(-silence), verify(-silence)

Model: --model auto (default) uses gpt-6.1-luna if GET /v1/models/gpt-6.1-luna answers
200, else gpt-6-luna. OPENAI_API_KEY comes from the environment and is never printed.

  python tools/moe/label.py all --char C:/Users/toont/tm-data/moe/char
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests

from tok import RunCTokenizer

API = "https://api.openai.com/v1"
PREFERRED, FALLBACK = "gpt-6.1-luna", "gpt-6-luna"
_lock = threading.Lock()
LIMIT = 0   # --limit: only the first N experts (testing)
TOP_K = 2   # experts per piece, read from the routing file in main()


def key():
    k = os.environ.get("OPENAI_API_KEY")
    if not k:
        raise SystemExit("OPENAI_API_KEY is not set")
    return k


def pick_model(choice: str) -> str:
    if choice != "auto":
        return choice
    r = requests.get(f"{API}/models/{PREFERRED}", headers={"Authorization": f"Bearer {key()}"}, timeout=30)
    return PREFERRED if r.status_code == 200 else FALLBACK


class Client:
    def __init__(self, model, effort, cache: Path):
        self.model, self.effort, self.cache_path = model, effort, cache
        self.cache = {}
        if cache.exists():
            for line in cache.read_text(encoding="utf-8").splitlines():
                rec = json.loads(line)
                self.cache[rec["h"]] = rec
        self.usage = dict(calls=0, cached=0, prompt=0, completion=0, reasoning=0)

    def ask(self, prompt: str, max_tokens=6000, system=None):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        h = hashlib.sha256(json.dumps([self.model, self.effort, msgs]).encode()).hexdigest()
        if h in self.cache:
            self.usage["cached"] += 1
            return self.cache[h]["out"]
        body = dict(model=self.model, messages=msgs, max_completion_tokens=max_tokens,
                    response_format={"type": "json_object"}, reasoning_effort=self.effort)
        for attempt in range(8):
            try:
                r = requests.post(f"{API}/chat/completions", headers={"Authorization": f"Bearer {key()}"},
                                  json=body, timeout=300)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code == 429 or r.status_code >= 500:     # incl. Cloudflare's 520-524
                time.sleep(5 * (attempt + 1))
                continue
            if r.status_code != 200:
                raise RuntimeError(f"API {r.status_code}: {r.text[:300]}")
            j = r.json()
            ch = j["choices"][0]
            if ch.get("finish_reason") == "length" and not ch["message"].get("content"):
                body["max_completion_tokens"] *= 2      # reasoning ate the budget; give it room
                continue
            try:
                out = json.loads(ch["message"]["content"])
            except (TypeError, json.JSONDecodeError):
                continue
            u = j.get("usage", {})
            with _lock:
                self.usage["calls"] += 1
                self.usage["prompt"] += u.get("prompt_tokens", 0)
                self.usage["completion"] += u.get("completion_tokens", 0)
                self.usage["reasoning"] += u.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(dict(h=h, out=out), ensure_ascii=False) + "\n")
                self.cache[h] = dict(h=h, out=out)
            return out
        raise RuntimeError("API: gave up after retries")


def pmap(fn, items, workers=8):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))


# ---------------------------------------------------------------- label

LABEL_PROMPT = """A tiny AI writes children's stories one piece at a time (a piece is a word, part of a word, or punctuation; a leading space means the piece starts a new word). It has 6 layers. In every layer a router sends each piece to {k} of 8 "experts", so an average expert gets about 1 piece in {per}.

This is expert {e} in layer {layer} of 6 (layer 1 is next to the input, layer 6 next to the output). It gets {share:.0%} of all pieces.

Pieces it gets most (its share of them : how often that piece goes here when it appears):
{top_share}

Pieces most special to it (how much more likely than average to come here):
{top_lift}

The 40 places where the router chose this expert most strongly (the piece sits in [[ ]]; ⏎ is a line break):
{contexts}

20 typical places, picked at random from everything it gets:
{random}

Work out what this expert's job is. Its name and job go on a card for children. Its rule will be given to someone who must mark, in a new story, exactly which pieces the router sends here, so the rule must be concrete enough to predict with.

Return JSON:
{{"name": "the ... expert" (2 to 4 kid words, e.g. "the names expert"; specific to this expert, not a name every expert could share such as "story words" or "sentence glue"),
 "job": "one short sentence a 9-year-old understands",
 "rule": "two or three concrete sentences: the pieces it gets most (up to 12, in quotes), what kind of piece or context marks the rest, and, for common pieces it shares with other experts (like 'the' or '.'), when they come here",
 "kind": "word list" | "grammar" | "meaning" | "story part" | "mixed"}}"""


def show(w):
    return "⏎" if w == "\n" else json.dumps(w, ensure_ascii=False)


def fmt_words(rows, n=15):
    return "\n".join(f"  {show(r['word'])}  {r['n']} times : {r['here']:.0%} of the time" for r in rows[:n])


def fmt_lift(rows, n=12):
    return "\n".join(f"  {show(r['word'])}  x{r['lift']:.1f}  ({r['here']:.0%} of {r['n'] / max(r['here'], 1e-9):.0f})"
                     for r in rows[:n])


def cmd_label(cl: Client, char: Path):
    """The labeler runs at its own (higher) reasoning effort; see --label-effort."""
    prof = json.loads((char / "profile.json").read_text(encoding="utf-8"))
    if LIMIT:
        prof = prof[:LIMIT]

    def one(ex):
        prompt = LABEL_PROMPT.format(
            k=TOP_K, per=8 // TOP_K, e=ex["expert"], layer=ex["layer"] + 1, share=ex["share"],
            top_share=fmt_words(ex["top_share"]), top_lift=fmt_lift(ex["top_lift"]),
            contexts="\n".join(f"{i + 1}. {c['text']}" for i, c in enumerate(ex["contexts"])),
            random="\n".join(f"- {c}" for c in ex["random_contexts"]))
        out = cl.ask(prompt, max_tokens=8000)
        return dict(layer=ex["layer"], expert=ex["expert"], **{k: out.get(k, "") for k in ("name", "job", "rule", "kind")})

    labels = pmap(one, prof)
    (char / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=1), encoding="utf-8")
    for lb in labels:
        print(f"L{lb['layer']}E{lb['expert']}: {lb['name']} — {lb['rule']}")


# ---------------------------------------------------------------- score

SIM_PROMPT = """A tiny AI writes children's stories one piece at a time (a piece is a word, part of a word, or punctuation). In each layer a router sends every piece to {k} of 8 experts, so an average expert gets about 1 piece in {per}, but some get more and some fewer.

Here is the description of one expert:
Name: {name}
Which pieces it gets: {rule}{card}

A passage from a story:
{text}

The same passage as numbered pieces ("-" marks a piece that continues the word before it; ⏎ is a line break; ␣ is a lone space piece):
{pieces}

Using only the description, which pieces does the router send to this expert? Return JSON {{"ids": [piece numbers]}}."""


def passage_pieces(tk, toks):
    out, prev = [], ""
    for i, t in enumerate(toks):
        s = tk.piece(t)
        if s == "\n":
            disp = "⏎"
        elif s == " ":
            disp = "␣"
        elif s.startswith(" ") or not s[:1].isalnum() or not (prev[-1:].isalnum() or prev == " "):
            disp = s.strip()
        else:
            disp = "-" + s
        out.append(f"{i}:{disp}")
        prev = s
    return " ".join(out)


def build_passages(char: Path, n_pass: int, length: int, seed=0):
    """Per layer, n_pass held-out passages (set B) of `length` pieces with their true routing."""
    B = dict(np.load(char / "route_B.npz"))
    by_story = {}
    order = np.lexsort((B["pos"], B["story"]))
    stories = B["story"][order]
    cuts = np.flatnonzero(np.diff(stories)) + 1
    for seg in np.split(order, cuts):
        by_story[int(B["story"][seg[0]])] = seg
    rng = random.Random(seed)
    long_ids = [s for s, seg in by_story.items() if len(seg) >= length + 8]
    rng.shuffle(long_ids)
    passages = []
    for s in long_ids[:n_pass * 6]:
        seg = by_story[s]
        st = rng.randrange(0, len(seg) - length)
        idx = seg[st:st + length]
        passages.append(dict(story=s, idx=idx, toks=B["tok"][idx], topi=B["topi"][idx]))
    return passages


def card_words(p, n=5):
    """The card's "words it handles most": signature words first, topped up by its busiest pieces."""
    words = [r["word"] for r in p["signature"]]
    words += [r["word"] for r in p["top_share"] if r["word"] not in words]
    return words[:n]


def cmd_score(cl: Client, char: Path, n_pass: int, length: int, variant="label", seed=0):
    """variant "label": the scorer sees the name and rule only (the plan's test).
    variant "card": it also sees what a child's card shows: the five words and how busy it is.
    seed 0 is the main test; another seed draws fresh passages for a retest (scores_retest.json)."""
    tk = RunCTokenizer()
    labels = json.loads((char / "labels.json").read_text(encoding="utf-8"))
    prof = {(p["layer"], p["expert"]): p for p in json.loads((char / "profile.json").read_text(encoding="utf-8"))}
    pool = build_passages(char, n_pass, length, seed=seed)
    jobs = []
    for lb in labels:
        l, e = lb["layer"], lb["expert"]
        ps = pool[l * n_pass:(l + 1) * n_pass]          # the same passages for all 8 experts of a layer
        for pi, p in enumerate(ps):
            jobs.append((lb, pi, p))

    def one(job):
        lb, pi, p = job
        card = ""
        if variant == "card":
            pr = prof[(lb["layer"], lb["expert"])]
            card = (f"\nWords it handles most: {', '.join(show(w) for w in card_words(pr))}"
                    f"\nHow busy it is: it gets {pr['share']:.0%} of all pieces")
        prompt = SIM_PROMPT.format(k=TOP_K, per=8 // TOP_K, name=lb["name"], rule=lb["rule"], card=card,
                                   text=tk.decode(p["toks"]).strip(), pieces=passage_pieces(tk, p["toks"]))
        out = cl.ask(prompt, max_tokens=6000)
        ids = {int(i) for i in (out.get("ids") or []) if str(i).lstrip("-").isdigit()}
        pred = np.array([i in ids for i in range(len(p["toks"]))])
        true = (p["topi"][:, lb["layer"], :] == lb["expert"]).any(1)
        return (lb["layer"], lb["expert"]), pred, true, p["toks"]

    res = pmap(one, jobs)
    from characterize import balanced_accuracy
    acc = {}
    for key_, pred, true, toks in res:
        a = acc.setdefault(key_, dict(pred=[], true=[], toks=[]))
        a["pred"].append(pred), a["true"].append(true), a["toks"].append(toks)
    scores = []
    for (l, e), a in sorted(acc.items()):
        pred, true, toks = (np.concatenate(a[k]) for k in ("pred", "true", "toks"))
        # word-lookup baseline on exactly these pieces (from set A statistics)
        here = {r["id"]: r["here"] for r in prof[(l, e)]["signature"]}
        look = np.array([here.get(int(t), 0) >= 0.5 for t in toks])
        scores.append(dict(layer=l, expert=e, balanced_accuracy=balanced_accuracy(pred, true),
                           tpr=float((pred & true).sum() / max(1, true.sum())),
                           tnr=float((~pred & ~true).sum() / max(1, (~true).sum())),
                           n=int(len(true)), positives=int(true.sum()), predicted=int(pred.sum()),
                           lookup_signature_ba=balanced_accuracy(look, true)))
    name = ("scores" if variant == "label" else f"scores_{variant}") + ("_retest" if seed else "") + ".json"
    (char / name).write_text(json.dumps(dict(n_passages=n_pass, length=length, variant=variant, scores=scores),
                                        indent=1), encoding="utf-8")
    print(f"scores ({variant}):")
    for s in scores:
        print(f"L{s['layer']}E{s['expert']}: BA {s['balanced_accuracy']:.3f} (TPR {s['tpr']:.2f}, TNR {s['tnr']:.2f}, "
              f"{s['positives']}/{s['n']} positive)")


# ---------------------------------------------------------------- describe

DESCRIBE_PROMPT = """A tiny AI writes children's stories. It has 6 layers with 8 experts each; in each layer a router sends every word to 2 experts. We switched OFF expert {e} in layer {layer}: {how} Everything else is the same, and the AI always picks its most likely next word, so any change is caused by the switch-off.

Here are {n} story beginnings written with the expert ON (before) and OFF (after):

{pairs}

The exact place where each of the {n_changed} changed stories first goes a different way (the words before it, then what came next with the expert ON | OFF; ⏎ is a line break). This first difference is the switch-off's direct effect; everything after it follows from it:
{firsts}

Measured over all {total} stories (before -> after):
{features}
{hurt}
Disturbing any part of the AI makes its stories go a different way, and often more (or less) repetitive or confusing. Those general changes do not count here. Look for what is special about THIS switch-off: a kind of word or mark that disappears, breaks or gets swapped (for example names, contractions like "don't", quotation marks, describing words, "Once upon a time"), something characters stop doing, and so on.

Return JSON:
{{"change": "one sentence a 9-year-old understands: the special change when this expert is off (if there is none, say what changes in general)",
 "specific": true only if you found a special change that shows up again and again (in the full stories or the first differences), not just once or twice,
 "visible": 1-5 (1 = you can hardly tell, 5 = obvious at a glance)}}"""

VERIFY_PROMPT = """A tiny AI writes children's stories. Someone switched off one part of it and described what happens like this:
"{change}"

Below is a story beginning the AI wrote normally (BEFORE), and two versions written with two different parts switched off (A and B). Only one of A and B was written with the part described above switched off.

BEFORE: {before}

A: {a}

B: {b}

Which one matches the description? Return JSON {{"match": "A" | "B"}}."""


HOW = {"off": "the router now sends its words to other experts instead.",
       "silence": "the router still chooses it, but its answer is thrown away (muted), so its words get help "
                  "from only one expert."}


def cmd_describe(cl: Client, char: Path, n_pairs: int, mode="off"):
    suffix = "" if mode == "off" else f"_{mode}"
    st = json.loads((char / f"stories{suffix}.json").read_text(encoding="utf-8"))
    base = st["baseline"]
    bf = st["baseline_features"]
    sw = json.loads((char / "switchoff.json").read_text(encoding="utf-8"))
    hurt_key = "hurt_words" if mode == "off" else "silence_hurt_words"
    hurt = {(x["layer"], x["expert"]): x.get(hurt_key) for x in sw["experts"]}

    def hurt_text(k):
        rows = hurt.get(k)
        if not rows:
            return ""
        return ("\nMeasured on other stories (not shown): the pieces the AI gets most worse at predicting with this "
                "expert off (extra surprise, in nats):\n  " +
                ", ".join(f"{show(r['word'])} +{r['delta']:.2f}" for r in rows[:8]) + "\n")

    def feats(f):
        keys = ["quotes_per_100", "commas_per_100", "periods_per_100", "exclaims_per_100", "questions_per_100",
                "names_per_100", "newlines_per_100", "unique_word_ratio", "repeated_4grams"]
        return "\n".join(f"  {k}: {bf[k]:.2f} -> {f[k]:.2f}" for k in keys)

    tk = RunCTokenizer()

    def pieces(toks):
        return "".join("⏎" if tk.piece(t) == "\n" else tk.piece(t) for t in toks)

    def first_line(i, ex):
        plen, d = len(st["prompt_tokens"][i]), ex["first_diff"][i]
        b, a = st["baseline_tokens"][i], ex["tokens"][i]
        at = plen + d
        return f"- …{pieces(b[max(0, at - 10):at])} ⟨{pieces(b[at:at + 4])} | {pieces(a[at:at + 4])}⟩"

    def one(ex):
        # the describer sees only even-numbered beginnings; verify tests on the odd ones
        changed = [i for i, d in enumerate(ex["first_diff"]) if d >= 0 and i % 2 == 0]
        changed.sort(key=lambda i: ex["first_diff"][i])
        pick = changed[:n_pairs]
        pairs = "\n\n".join(f"#{k + 1}\nBEFORE: {base[i].strip()}\nAFTER:  {ex['texts'][i].strip()}"
                            for k, i in enumerate(pick))
        firsts = "\n".join(first_line(i, ex) for i in sorted(changed))
        if not pick:
            return dict(layer=ex["layer"], expert=ex["expert"], change="No story changed.", specific=False,
                        visible=1, pairs=[])
        k = (ex["layer"], ex["expert"])
        out = cl.ask(DESCRIBE_PROMPT.format(e=ex["expert"], layer=ex["layer"] + 1, how=HOW[mode], n=len(pick),
                                            pairs=pairs, n_changed=len(changed), firsts=firsts, total=len(base),
                                            features=feats(ex["features"]), hurt=hurt_text(k)),
                     max_tokens=10000)
        return dict(layer=ex["layer"], expert=ex["expert"], change=out.get("change", ""),
                    specific=bool(out.get("specific")), visible=int(out.get("visible", 0) or 0), pairs=pick)

    desc = pmap(one, st["experts"][:LIMIT or None])
    (char / f"describe{suffix}.json").write_text(json.dumps(desc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"describe ({mode}):")
    for d in desc:
        print(f"L{d['layer']}E{d['expert']}: [{'specific' if d['specific'] else 'general'}, {d['visible']}] {d['change']}")


def binom_tail(k, n, p=0.5):
    """P(X >= k) for X ~ Binomial(n, p)."""
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def cmd_verify(cl: Client, char: Path, n_test: int, mode="off", confirm=False):
    """Blind matching test for each specific description: given the sentence, pick this expert's
    switched-off story over a decoy expert's (same layer, same beginning), on the odd-numbered
    beginnings, which the describer never saw. Passes with at least 12 tests and a hit count that chance would reach less
    than 1% of the time (for 20 tests: 16 or more right), so ~1 false pass is expected over ~100 experts.

    confirm=True is a second round for the near-misses (first-round p < 0.05): the same sentences, tested on
    stories*_confirm.json (fresh beginnings from unused validation stories, all of them eligible), with new
    decoy draws. It passes on its own data alone, by the same rule."""
    suffix = "" if mode == "off" else f"_{mode}"
    st = json.loads((char / f"stories{suffix}{'_confirm' if confirm else ''}.json").read_text(encoding="utf-8"))
    base = st["baseline"]
    texts = {(x["layer"], x["expert"]): x for x in st["experts"]}
    desc = json.loads((char / f"describe{suffix}.json").read_text(encoding="utf-8"))
    if confirm:
        first = {(v["layer"], v["expert"]): v for v in
                 json.loads((char / f"verify{suffix}.json").read_text(encoding="utf-8"))}
        desc = [d for d in desc if d["specific"] and first[(d["layer"], d["expert"])]["tested"]
                and first[(d["layer"], d["expert"])]["p_value"] < 0.05]
    jobs = []
    for d in desc:
        if not d["specific"]:
            continue
        k = (d["layer"], d["expert"])
        rng = random.Random(1000 * d["layer"] + d["expert"] + (7 if confirm else 0))
        others = [e for e in range(8) if e != d["expert"]]
        mine = texts[k]
        n = 0
        for i in (range(len(base)) if confirm else range(1, len(base), 2)):   # odd: never shown to the describer
            if mine["first_diff"][i] < 0:
                continue
            decoy = (d["layer"], rng.choice(others))
            dt = texts[decoy]
            if dt["first_diff"][i] < 0 or dt["texts"][i] == mine["texts"][i]:
                continue
            jobs.append((k, i, decoy, rng.random() < 0.5))
            n += 1
            if n == n_test:
                break

    def one(job):
        k, i, decoy, mine_first = job
        a, b = texts[k]["texts"][i], texts[decoy]["texts"][i]
        if not mine_first:
            a, b = b, a
        d = next(x for x in desc if (x["layer"], x["expert"]) == k)
        out = cl.ask(VERIFY_PROMPT.format(change=d["change"], before=base[i].strip(), a=a.strip(), b=b.strip()),
                     max_tokens=6000)
        return k, (out.get("match") == "A") == mine_first

    res = pmap(one, jobs)
    out = []
    for d in desc:
        k = (d["layer"], d["expert"])
        hits = [ok for kk, ok in res if kk == k]
        n, c = len(hits), sum(hits)
        p = binom_tail(c, n) if n else 1.0
        out.append(dict(layer=k[0], expert=k[1], tested=n, correct=c, p_value=p,
                        passed=bool(d["specific"] and n >= 12 and p < 0.01)))
    (char / f"verify{suffix}{'_confirm' if confirm else ''}.json").write_text(json.dumps(out, indent=1),
                                                                          encoding="utf-8")
    print(f"verify ({mode}{', confirm' if confirm else ''}):")
    for v in out:
        if v["tested"]:
            print(f"L{v['layer']}E{v['expert']}: {v['correct']}/{v['tested']} p={v['p_value']:.3f} "
                  f"{'PASS' if v['passed'] else ''}")


# ---------------------------------------------------------------- judge

JUDGE_PROMPT = """Two short children's stories continue the same beginning. They were written by two small AIs and may stop mid-sentence. Judge only the writing: grammar, whether it makes sense, and whether it stays on the story.

Beginning: {prompt}

Story A: {a}

Story B: {b}

Return JSON {{"better": "A" | "B" | "tie", "score_A": 1-10, "score_B": 1-10}}."""


def cmd_judge(cl: Client, char: Path):
    q = json.loads((char / "quality.json").read_text(encoding="utf-8"))
    moe, dense = q["minds"]["moe"]["sampled"], q["minds"]["dense"]["sampled"]
    prompts = q["prompts"] * (len(moe) // len(q["prompts"]))
    rng = random.Random(0)
    jobs = [(i, rng.random() < 0.5) for i in range(len(moe))][:LIMIT or None]

    def one(job):
        i, moe_first = job
        a, b = (moe[i], dense[i]) if moe_first else (dense[i], moe[i])
        out = cl.ask(JUDGE_PROMPT.format(prompt=prompts[i], a=a.strip(), b=b.strip()), max_tokens=6000)
        better = out.get("better", "tie")
        winner = "tie" if better == "tie" else ("moe" if (better == "A") == moe_first else "dense")
        sa, sb = out.get("score_A", 0), out.get("score_B", 0)
        return dict(i=i, winner=winner, moe_score=sa if moe_first else sb, dense_score=sb if moe_first else sa)

    res = pmap(one, jobs)
    summ = dict(moe_wins=sum(r["winner"] == "moe" for r in res), dense_wins=sum(r["winner"] == "dense" for r in res),
                ties=sum(r["winner"] == "tie" for r in res),
                moe_mean=float(np.mean([r["moe_score"] for r in res])),
                dense_mean=float(np.mean([r["dense_score"] for r in res])))
    (char / "judge.json").write_text(json.dumps(dict(summary=summ, pairs=res), indent=1), encoding="utf-8")
    print(json.dumps(summ))


def main():
    sys.stdout.reconfigure(encoding="utf-8")     # labels contain ⏎ and ␣; a redirected cp1252 stdout chokes
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["label", "score", "score-card", "score-retest", "describe", "describe-silence",
                                     "verify", "verify-silence", "verify-confirm", "verify-silence-confirm",
                                     "judge", "all"])
    ap.add_argument("--char", default="C:/Users/toont/tm-data/moe/char")
    ap.add_argument("--model", default="auto", help="auto = gpt-6.1-luna if available, else gpt-6-luna")
    ap.add_argument("--effort", default="medium", help="reasoning_effort for scoring, describing, judging")
    ap.add_argument("--label-effort", default="high", help="reasoning_effort for the labeler")
    ap.add_argument("--label-model", default="", help="a different model for the labeler only (e.g. gpt-6.1-sol)")
    ap.add_argument("--passages", type=int, default=8)
    ap.add_argument("--length", type=int, default=48)
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--verify-n", type=int, default=20, help="matching-test beginnings per expert")
    ap.add_argument("--confirm-n", type=int, default=40, help="confirmation-round beginnings per expert")
    ap.add_argument("--limit", type=int, default=0, help="only the first N experts (testing)")
    args = ap.parse_args()
    global LIMIT, TOP_K
    LIMIT = args.limit
    if (Path(args.char) / "route_B.npz").exists():
        TOP_K = int(np.load(Path(args.char) / "route_B.npz")["topi"].shape[2])
    char = Path(args.char)
    model = pick_model(args.model)
    label_model = args.label_model or model
    print(f"labeling model: {label_model} (effort {args.label_effort}); others: {model} (effort {args.effort})",
          flush=True)
    # "all" = everything that needs only the MoE; "judge" also needs quality.json (the dense twin)
    steps = (["label", "score", "score-card", "describe", "describe-silence", "verify", "verify-silence"]
             if args.step == "all" else [args.step])
    meta_path = char / "label_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    for s in steps:
        effort = args.label_effort if s == "label" else args.effort
        step_model = label_model if s == "label" else model
        cl = Client(step_model, effort, char / "api_cache.jsonl")
        if s == "label":
            cmd_label(cl, char)
        elif s.startswith("score"):
            cmd_score(cl, char, args.passages, args.length, variant="card" if s == "score-card" else "label",
                      seed=1 if s == "score-retest" else 0)
        elif s.startswith("describe"):
            cmd_describe(cl, char, args.pairs, mode="silence" if s.endswith("silence") else "off")
        elif s.startswith("verify"):
            cmd_verify(cl, char, args.confirm_n if s.endswith("confirm") else args.verify_n,
                       mode="silence" if "silence" in s else "off", confirm=s.endswith("confirm"))
        elif s == "judge":
            cmd_judge(cl, char)
        meta[s] = dict(model=step_model, effort=effort, usage=dict(cl.usage))
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{s} usage:", cl.usage, flush=True)


if __name__ == "__main__":
    main()
