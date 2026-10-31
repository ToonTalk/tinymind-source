"""Write reports/moe/feasibility.md from the training runs and the characterization outputs.

The hand-written parts (summary, verdict reasoning) live in tools/moe/narrative.md; the
numbers, gate checks and all 48 expert cards are generated here.

  python tools/moe/report.py
"""

from __future__ import annotations

import argparse
import json
import statistics as stats
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TM = Path("C:/Users/toont/tm-data/moe")
BA_PASS, LAYER_FRAC = 0.75, 0.60


def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def keyed(rows):
    return {(x["layer"], x["expert"]): x for x in rows} if rows else {}


def show(r):
    """A piece for display: byte-fallback pieces (ids 3-258) as <0xHH>, blank pieces by repr."""
    w = r["word"]
    if "�" in w:
        return f"<0x{r['id'] - 3:02X}>" if 3 <= r.get("id", -1) < 259 else "‹byte›"
    return w.strip() or repr(w)


def words(rows, n=6):
    return " ".join(f"`{show(r)}`" for r in rows[:n]) or "—"


def clip(s: str, n=260):
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + " …"


def matched(v, vc):
    """The blind matching test passes in round 1, or round 1 came close (p < 0.05) and the confirmation round on
    fresh beginnings passes on its own."""
    return bool(v) and (v["passed"] or (v["tested"] > 0 and v["p_value"] < 0.05 and bool(vc) and vc["passed"]))


def gate_c_ok(d, st, v, vc=None):
    """Specific description, visible, at least half the stories change, and the blind matching test passes."""
    return (bool(d) and d["specific"] and d["visible"] >= 3 and st["changed_frac"] >= 0.5 and matched(v, vc))


def pooled(a, b):
    """Balanced accuracy over two passage sets together, from each set's hit and rejection rates."""
    tp = tn = pos = neg = 0
    for r in (a, b):
        p, n = r["positives"], r["n"] - r["positives"]
        tp, tn, pos, neg = tp + r["tpr"] * p, tn + r["tnr"] * n, pos + p, neg + n
    return dict(layer=a["layer"], expert=a["expert"], balanced_accuracy=0.5 * (tp / max(pos, 1) + tn / max(neg, 1)))


def layers_passing(rows):
    by = {}
    for r in rows:
        by.setdefault(r["layer"], []).append(r["balanced_accuracy"] >= BA_PASS)
    return sum(sum(v) >= LAYER_FRAC * len(v) for v in by.values()), len(by)


def gate_numbers(char: Path, moe_run, dense_run):
    """The three gates' headline numbers for one characterized mind (None where a part is missing)."""
    out = dict(rel=None, layers=None, layers_card=None, layers_retest=None, layers_pooled=None, n_layers=None, c={})
    if moe_run and dense_run:
        out["rel"] = (moe_run["val_loss_full"] - dense_run["val_loss_full"]) / dense_run["val_loss_full"]
    sc, cd, rt = (load(char / f) for f in ("scores.json", "scores_card.json", "scores_retest.json"))
    for key, src in (("layers", sc), ("layers_card", cd), ("layers_retest", rt)):
        if src:
            out[key], out["n_layers"] = layers_passing(src["scores"])
    if sc and rt:
        rtx = keyed(rt["scores"])
        out["layers_pooled"] = layers_passing([pooled(r, rtx[(r["layer"], r["expert"])]) for r in sc["scores"]])[0]
    for mode, suffix in (("reroute", ""), ("silence", "_silence")):
        st, d, v, vc = (load(char / f) for f in (f"stories{suffix}.json", f"describe{suffix}.json",
                                                     f"verify{suffix}.json", f"verify{suffix}_confirm.json"))
        if st and d:
            stx, vx, vcx = keyed(st["experts"]), keyed(v), keyed(vc)
            out["c"][mode] = sum(gate_c_ok(x, stx[k], vx.get(k), vcx.get(k)) for k, x in keyed(d).items())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--char", default=str(TM / "char"))
    ap.add_argument("--ckpt", default=str(TM / "ckpt"), help="folder holding moe/ and dense/ runs")
    ap.add_argument("--top1-char", default=str(TM / "char-top1"))
    ap.add_argument("--top1-ckpt", default=str(TM / "ckpt-top1"))
    ap.add_argument("--variants", default=",".join(str(TM / n) for n in ("char-sol", "char-luna-by-sol",
                                                                         "char-sol-by-sol")),
                    help="folders with the same mind labeled and/or scored by other models")
    ap.add_argument("--c-variants", default=str(TM / "char-c-sol"),
                    help="folders with gate (c) rerun by another describer and judge")
    ap.add_argument("--out", default=str(REPO / "reports" / "moe" / "feasibility.md"))
    args = ap.parse_args()
    char = Path(args.char)

    runs = {a: load(Path(args.ckpt) / a / "final.json") for a in ("moe", "dense")}
    runinfo = {a: load(Path(args.ckpt) / a / "run.json") for a in ("moe", "dense")}
    top1_run = load(Path(args.top1_ckpt) / "moe" / "final.json")
    prof = keyed(load(char / "profile.json"))
    labels = keyed(load(char / "labels.json"))
    sc_meta = load(char / "scores.json")
    scores = keyed(sc_meta["scores"])
    card = keyed((load(char / "scores_card.json") or {}).get("scores"))
    retest = keyed((load(char / "scores_retest.json") or {}).get("scores"))
    sw = load(char / "switchoff.json")
    swx = keyed(sw["experts"])
    def c_modes(d: Path):
        """One folder's gate (c) files, per switch-off mode."""
        out = {}
        for mode, suffix in (("reroute", ""), ("silence", "_silence")):
            st = load(d / f"stories{suffix}.json")
            if st:
                out[mode] = dict(st=st, stx=keyed(st["experts"]), desc=keyed(load(d / f"describe{suffix}.json")),
                                 ver=keyed(load(d / f"verify{suffix}.json")),
                                 conf=keyed(load(d / f"verify{suffix}_confirm.json")),
                                 conf_st=load(d / f"stories{suffix}_confirm.json"))
        return out

    modes = c_modes(char)
    c_variants = []     # gate (c) rerun with another describer and judge
    for d in [Path(x) for x in args.c_variants.split(",") if x]:
        cm = c_modes(d)
        if cm and all(m["desc"] and m["ver"] for m in cm.values()):
            c_variants.append(dict(char=d, modes=cm, model=(load(d / "label_meta.json") or {}).get(
                "describe", {}).get("model", "?"), gates=gate_numbers(d, None, None)))
    qual = load(char / "quality.json")
    judge = load(char / "judge.json")
    meta = load(char / "label_meta.json") or {}
    variants = []       # the main run first, then every labeler x scorer folder that has scores
    for d in [char] + [Path(x) for x in args.variants.split(",") if x]:
        if not ((d / "labels.json").exists() and (d / "scores.json").exists()):
            continue
        vm = load(d / "label_meta.json") or {}
        variants.append(dict(
            char=d, labeler=vm.get("label", {}).get("model", "?"), label_effort=vm.get("label", {}).get("effort", "?"),
            scorer=vm.get("score", {}).get("model", "?"), labels=keyed(load(d / "labels.json")),
            scores=keyed(load(d / "scores.json")["scores"]),
            retest=keyed((load(d / "scores_retest.json") or {}).get("scores")),
            card=keyed((load(d / "scores_card.json") or {}).get("scores")), gates=gate_numbers(d, None, None)))
    main_v = variants[0]
    n_layers = 1 + max(k[0] for k in prof)
    n_exp = 1 + max(k[1] for k in prof)
    L = []
    w = L.append

    w("# Phase 0: does each expert have a job?\n")
    w("*Feasibility report for [plans/experts-room.md](../../plans/experts-room.md), written 2026-10-30. "
      "Numbers, gate checks and expert cards are generated by `tools/moe/report.py`; the prose comes from "
      "`tools/moe/narrative.md`.*\n")
    narrative = Path(__file__).parent / "narrative.md"
    if narrative.exists():
        w(narrative.read_text(encoding="utf-8").strip() + "\n")

    # ---- at a glance
    minds = [("top-2 MoE (the main run)", gate_numbers(char, runs["moe"], runs["dense"]))]
    if top1_run or Path(args.top1_char).exists():
        minds.append(("top-1 MoE (variant)", gate_numbers(Path(args.top1_char), top1_run, runs["dense"])))

    def cell(v, fmt, need, ok):
        return "—" if v is None else f"{fmt(v)} {'✅' if ok(v, need) else '❌'}"

    w("## The gates at a glance\n")
    w("| | " + " | ".join(n for n, _ in minds) + " |")
    w("|---|" + "---|" * len(minds))
    w("| (a) held-out loss vs dense twin (pass: within about +5%) | " + " | ".join(
        cell(g["rel"], lambda v: f"{v:+.1%}", 0.05, lambda v, n: v <= n) for _, g in minds) + " |")
    w("| (b) layers where ≥60% of labels score ≥0.75 (pass: 3 of 6) | " + " | ".join(
        cell(g["layers"], lambda v: f"{v} of {g['n_layers']}", 3, lambda v, n: v >= n) for _, g in minds) + " |")
    for key, what in (("layers_retest", "retest on fresh passages"), ("layers_pooled", "both passage sets pooled"),
                      ("layers_card", "scoring the whole card")):
        w(f"| (b) the same, {what} | " + " | ".join(
            cell(g[key], lambda v: f"{v} of {g['n_layers']}", 3, lambda v, n: v >= n) for _, g in minds) + " |")
    def b_cell(g):
        if g["layers"] is None:
            return "—"
        pooled_txt = f" (pooled {g['layers_pooled']})" if g["layers_pooled"] is not None else ""
        return cell(g["layers"], lambda x: f"{x} of {g['n_layers']}{pooled_txt}", 3, lambda x, n: x >= n)

    for v in variants[1:]:
        # the top-1 mind's run of the same labeler x scorer lives in char-top1<same suffix>, if it was made
        t1 = Path(args.top1_char).parent / (Path(args.top1_char).name + v["char"].name[len("char"):])
        cells = [b_cell(v["gates"])] + [b_cell(gate_numbers(t1, None, None))] * (len(minds) - 1)
        w(f"| (b) the same, {v['labeler']} labels scored by {v['scorer']} | " + " | ".join(cells) + " |")
    for mode, what in (("reroute", "rerouted, the plan's switch"), ("silence", "muted, an extra check")):
        w(f"| (c) experts whose switch-off ({what}) is described and matched blind (pass: 4) | " + " | ".join(
            cell(g["c"].get(mode), str, 4, lambda v, n: v >= n) for _, g in minds) + " |")
        for cv in c_variants:
            t1 = Path(args.top1_char).parent / (Path(args.top1_char).name + cv["char"].name[len("char"):])
            cells = [cv["gates"]] + [gate_numbers(t1, None, None)] * (len(minds) - 1)
            w(f"| (c) the same, {mode}, with {cv['model']} as describer and judge | " + " | ".join(
                cell(g["c"].get(mode), str, 4, lambda v, n: v >= n) for g in cells) + " |")
    w("")

    # ---- gate (a)
    w("## Gate (a): quality, MoE vs dense twin\n")
    w("| mind | parameters | active per word | pieces trained | held-out loss (bf16, as trained) | "
      "held-out loss (f32) |")
    w("|---|---|---|---|---|---|")
    for a in ("moe", "dense"):
        r = runs[a]
        if not r:
            continue
        f32 = qual["minds"][a]["val_loss"] if qual and a in qual["minds"] else float("nan")
        w(f"| {a} | {r['n_params'] / 1e6:.2f}M | {r['n_active'] / 1e6:.2f}M | {r['tokens'] / 1e6:.0f}M | "
          f"{r['val_loss_full']:.4f} | {f32:.4f} |")
    if qual and "stories15M" in qual["minds"]:
        w(f"| stories15M (the site's Storyteller; trained on another copy of TinyStories with single line breaks "
          f"between paragraphs, so its loss here is not comparable) | 15.2M | 15.2M | "
          f"— | — | {qual['minds']['stories15M']['val_loss']:.4f} |")
    t1_qual, t1_judge = (load(Path(args.top1_char) / f) for f in ("quality.json", "judge.json"))
    if top1_run:
        f32 = f"{t1_qual['minds']['moe']['val_loss']:.4f}" if t1_qual else "—"
        w(f"| top-1 MoE (variant) | {top1_run['n_params'] / 1e6:.2f}M | {top1_run['n_active'] / 1e6:.2f}M | "
          f"{top1_run['tokens'] / 1e6:.0f}M | {top1_run['val_loss_full']:.4f} | {f32} |")
    w("")
    if runs["moe"] and runs["dense"]:
        rel = (runs["moe"]["val_loss_full"] - runs["dense"]["val_loss_full"]) / runs["dense"]["val_loss_full"]
        w(f"Held-out loss is the mean over all 5.17M pieces of TinyStories validation, in 256-piece windows. "
          f"**MoE vs dense: {rel:+.1%}** (gate (a) asks for within about 5%). The every-500-steps losses in the "
          f"training logs are not comparable between these two runs: they scored the first 40 micro-batches, and "
          f"the runs used different micro-batch sizes (32 and 16), so they saw different text (fixed since).\n")
    if judge:
        s = judge["summary"]
        n = s["moe_wins"] + s["dense_wins"] + s["ties"]
        w(f"Blind pairwise reading of {n} pairs of sampled stories (temperature 0.8, same beginning and seed; "
          f"judge {meta.get('judge', {}).get('model', '?')}): MoE better {s['moe_wins']}, dense better "
          f"{s['dense_wins']}, tie {s['ties']}. Mean score {s['moe_mean']:.1f} (MoE) vs {s['dense_mean']:.1f} "
          f"(dense) out of 10.\n")
    if t1_judge:
        s = t1_judge["summary"]
        n = s["moe_wins"] + s["dense_wins"] + s["ties"]
        w(f"The same blind reading for the top-1 variant against the dense twin ({n} pairs, same beginnings and "
          f"seeds): top-1 better {s['moe_wins']}, dense better {s['dense_wins']}, tie {s['ties']}. Mean score "
          f"{s['moe_mean']:.1f} vs {s['dense_mean']:.1f}.\n")

    # ---- gate (b)
    w("## Gate (b): labels that predict routing\n")
    w(f"A label passes at balanced accuracy ≥ {BA_PASS}; a layer passes when at least 60% of its experts "
      f"(5 of 8) pass; chance is 0.5. **Label** is the plan's test: the scorer sees only the expert's name and "
      f"rule. **Card** adds what a child's card would show: the five words it handles most and how busy it is. "
      f"**Word lookup** is a context-blind baseline: \"routed\" whenever that piece goes to this expert at least "
      f"half the time in set A. When the lookup scores well, the expert is mostly a word list. Each score uses "
      f"{sc_meta['n_passages']} held-out passages of {sc_meta['length']} pieces per layer (about "
      f"{stats.median(s['positives'] for s in scores.values()):.0f} routed pieces per expert), so it is good to "
      f"about ±0.03. **Retest** scores the same labels on {sc_meta['n_passages']} fresh passages; **pooled** "
      f"counts both sets together.\n")
    w("| layer | labels passing (median BA) | retest | pooled | cards passing (median BA) | "
      "median word-lookup BA | median JS distance | layer passes (label) |")
    w("|---|---|---|---|---|---|---|---|")
    layers_pass = layers_pass_card = 0

    def n_pass(rows):
        return f"{sum(r['balanced_accuracy'] >= BA_PASS for r in rows)}/{len(rows)}" if rows else "—"

    for l in range(n_layers):
        bas = [scores[(l, e)]["balanced_accuracy"] for e in range(n_exp)]
        cbas = [card[(l, e)]["balanced_accuracy"] for e in range(n_exp)] if card else []
        rts = [retest[(l, e)] for e in range(n_exp)] if retest else []
        pls = [pooled(scores[(l, e)], retest[(l, e)]) for e in range(n_exp)] if retest else []
        looks = [prof[(l, e)]["lookup_ba"] for e in range(n_exp)]
        jss = [prof[(l, e)]["js"] for e in range(n_exp)]
        n_ok = sum(b >= BA_PASS for b in bas)
        n_ok_c = sum(b >= BA_PASS for b in cbas)
        ok = n_ok >= LAYER_FRAC * n_exp
        layers_pass += ok
        layers_pass_card += n_ok_c >= LAYER_FRAC * n_exp
        cb = f"{n_ok_c}/{n_exp} ({stats.median(cbas):.2f})" if cbas else "—"
        w(f"| {l + 1} | {n_ok}/{n_exp} ({stats.median(bas):.2f}) | {n_pass(rts)} | {n_pass(pls)} | {cb} | "
          f"{stats.median(looks):.2f} | {stats.median(jss):.2f} | {'yes' if ok else 'no'} |")
    w(f"\n**Label: {layers_pass} of {n_layers} layers pass. Card: {layers_pass_card} of {n_layers}.** "
      f"Gate (b) needs at least 3.\n")
    kinds = {}
    for lb in labels.values():
        kinds[lb["kind"]] = kinds.get(lb["kind"], 0) + 1
    w("The labeler's own sorting of the 48 jobs: " +
      ", ".join(f"{k} {v}" for k, v in sorted(kinds.items(), key=lambda kv: -kv[1])) + ".\n")
    if len(variants) > 1:
        w("### Does a stronger model change the answer?\n")
        w("Two models play a part in this test: the **labeler**, which names each expert and writes its rule, and "
          "the **scorer**, which has to apply that rule to new passages. Here each job is done by "
          + " or ".join(sorted({v['labeler'] for v in variants} | {v['scorer'] for v in variants})) + ", in every "
          "combination, with the same prompts and the same passages (labelers at reasoning effort high, scorers "
          "at medium).\n")
        w("| labeler | scorer | layers passing | retest | pooled | whole card | experts passing | median BA | "
          "median hits | median correct rejections |")
        w("|---|---|---|---|---|---|---|---|---|---|")
        for v in variants:
            g, vs = v["gates"], v["scores"].values()
            n_ok = sum(r["balanced_accuracy"] >= BA_PASS for r in vs)

            def nz(x):
                return "—" if x is None else f"{x} of {g['n_layers']}"

            w(f"| {v['labeler']} | {v['scorer']} | **{nz(g['layers'])}** | {nz(g['layers_retest'])} | "
              f"{nz(g['layers_pooled'])} | {nz(g['layers_card'])} | {n_ok} of {len(vs)} | "
              f"{stats.median(r['balanced_accuracy'] for r in vs):.2f} | {stats.median(r['tpr'] for r in vs):.0%} | "
              f"{stats.median(r['tnr'] for r in vs):.0%} |")
        w("\nLabels passing in each layer (median balanced accuracy):\n")
        w("| layer | " + " | ".join(f"{v['labeler']} labels, {v['scorer']} scorer" for v in variants) + " |")
        w("|---|" + "---|" * len(variants))
        for l in range(n_layers):
            cells = []
            for v in variants:
                b = [v["scores"][(l, e)]["balanced_accuracy"] for e in range(n_exp)]
                cells.append(f"{sum(x >= BA_PASS for x in b)}/{n_exp} ({stats.median(b):.2f})")
            w(f"| {l + 1} | " + " | ".join(cells) + " |")
        w("\nEach expert's card below shows every labeler's name and every score.\n")
    wr = load(char / "wordrule.json")
    if wr:
        has_pair = "pair" in wr[0]
        w("### How much does the word alone decide the route?\n")
        w(f"For each word piece, set A gives its usual first-choice expert{' and its usual pair' if has_pair else ''}. "
          f"On set B, how often does the router agree? This covers {wr[0]['coverage']:.0%} of set B's pieces (those "
          f"seen at least 20 times in set A). *Chance* is always guessing the layer's busiest choice.\n")
        w("| layer | first choice = the word's usual | chance |" + (" both choices = the word's usual pair | chance |"
                                                                   if has_pair else ""))
        w("|---|---|---|" + ("---|---|" if has_pair else ""))
        for r in wr:
            w(f"| {r['layer'] + 1} | {r['first']:.0%} | {r['first_chance']:.0%} |" +
              (f" {r['pair']:.0%} | {r['pair_chance']:.0%} |" if has_pair else ""))
        w("")

    # ---- gate (c)
    w("## Gate (c): does switching an expert off change stories describably?\n")
    any_mode = next(iter(modes.values()))
    w(f"Each expert is switched off and the mind rewrites {len(any_mode['st']['prompts'])} fixed beginnings "
      f"greedily for {any_mode['st']['steps']} pieces. **Reroute** (the room's switch) sets the expert's router "
      f"score to minus infinity, so its words go to the next-best expert. **Silence** keeps the router's choice "
      f"but throws the expert's answer away.\n")
    w(f"The beginnings are split: the describer sees only the even-numbered ones, the matching test uses only "
      f"the odd-numbered ones. The describer ({meta.get('describe', {}).get('model', '?')}, blind to the "
      f"expert's label) sees six before/after pairs (earliest-diverging first), the exact first difference in "
      f"each of its changed stories (the switch-off's direct effect), feature counts over all stories, and the "
      f"pieces whose prediction suffers most on held-out text. General changes (\"the story goes another way\", \"more repetitive\", \"more "
      f"confusing\") follow from disturbing any part of the mind, so they do not count: the describer marks a "
      f"description *specific* only when it names something special to this expert that recurs. Each specific "
      f"description then faces a **blind matching test**: on up to 20 beginnings the describer never saw, a "
      f"judge gets the sentence, the normal story, and two switched-off versions (this expert's "
      f"and a random other expert's from the same layer), and must pick the one the sentence describes. An "
      f"expert counts for gate (c) when its description is specific, rated at least 3 of 5 for visibility, at "
      f"least half its stories change, and the matching test passes: at least 12 tests and a hit count chance "
      f"would reach under 1% of the time (16 of 20). About one false pass is expected across all the tests.\n")
    cst = next((m["conf_st"] for m in modes.values() if m["conf_st"]), None)
    if cst:
        w(f"An effect that shows up in only some stories cannot score high on 20 beginnings: where neither "
          f"version shows it, the judge can only guess. So descriptions that came close (round-1 p < 0.05) get a "
          f"**confirmation round** on {len(cst['prompts'])} fresh beginnings (the opening words of validation "
          f"stories no other step reads, such as “{cst['prompts'][0]}”), with the same sentence and new decoy "
          f"draws. It must pass by the same rule on its own data. This round was added after round 1, and its rule "
          f"(who enters, how it passes) was fixed before its stories were written. Both rounds are shown.\n")
    def c_block(mds, lbl=None):
        """Per mode: how many qualify, and every expert that passed a round or entered the confirmation round."""
        lbl = labels if lbl is None else lbl
        qual = {}
        for mode, m in mds.items():
            if not m["desc"]:
                continue
            ok = [(k, d) for k, d in sorted(m["desc"].items())
                  if gate_c_ok(d, m["stx"][k], m["ver"].get(k), m["conf"].get(k))]
            qual[mode] = ok
            n_spec = sum(d["specific"] for d in m["desc"].values())
            n_r1 = sum(gate_c_ok(d, m["stx"][k], m["ver"].get(k)) for k, d in m["desc"].items())
            n_ok = len(ok)
            w(f"**{mode.capitalize()}: {n_ok} expert{'s qualify' if n_ok != 1 else ' qualifies'}** ({n_spec} "
              f"specific descriptions; {n_r1} on round 1 alone; gate (c) needs at least 4).\n")
            rows = [(k, d) for k, d in sorted(m["desc"].items())
                    if m["conf"].get(k) or (m["ver"].get(k) or {}).get("passed")]
            if rows:
                w("| expert | what changes (the describer's sentence) | visible | round 1 | confirmation | counts |")
                w("|---|---|---|---|---|---|")
                for k, d in rows:
                    v, vc = m["ver"][k], m["conf"].get(k)
                    r1 = f"{v['correct']}/{v['tested']} (p {v['p_value']:.3f})"
                    r2 = f"{vc['correct']}/{vc['tested']} (p {vc['p_value']:.3f})" if vc and vc["tested"] else "—"
                    yes = "✅" if gate_c_ok(d, m["stx"][k], v, vc) else "❌"
                    w(f"| L{k[0] + 1}·E{k[1]} {lbl[k]['name'] if k in lbl else ''} | {d['change']} | {d['visible']}/5 | {r1} | "
                      f"{r2} | {yes} |")
            w("")
        return qual

    qualifying = c_block(modes)
    for cv in c_variants:
        w(f"### With {cv['model']} as describer and judge\n")
        w(f"The same stories, beginnings, prompts and rules, with {cv['model']} writing the descriptions and judging "
          f"the blind matching tests (both rounds). Expert names are still the first labeler's.\n")
        c_block(cv["modes"])

    # ---- switch-off losses overview
    w("## Switch-off cost by layer\n")
    w(f"Extra loss in nats per predicted word over {sw['n_words']:,} held-out words (base loss "
      f"{sw['base_loss']:.3f}). \"Own words\" are predictions made at a word the expert handled.\n")
    w("| layer | median reroute | max reroute | median silence | median reroute on own words |")
    w("|---|---|---|---|---|")
    for l in range(n_layers):
        r = [swx[(l, e)]["off_delta"] for e in range(n_exp)]
        s = [swx[(l, e)]["silence_delta"] for e in range(n_exp)]
        h = [swx[(l, e)]["off_delta_handled"] for e in range(n_exp)]
        w(f"| {l + 1} | +{stats.median(r):.4f} | +{max(r):.4f} | +{stats.median(s):.4f} | +{stats.median(h):.4f} |")
    w("")

    # ---- cards
    w("## All 48 expert cards\n")
    top_k = runinfo["moe"]["args"]["top_k"] if runinfo["moe"] else 2
    w("Layers are numbered 1–6 from the input; experts 0–7. ✅/❌ is the label score against 0.75. *Busy* is "
      f"the fraction of all words the router sends to the expert ({top_k / n_exp:.0%} on average with top-{top_k} "
      f"of {n_exp}). *Signature "
      "words* go to this expert at least half the time, busiest first. *Lift* is how much more likely a piece is "
      "to come here than average. *JS* is the Jensen-Shannon distance between the expert's word distribution and "
      "the layer's.\n")
    for l in range(n_layers):
        w(f"### Layer {l + 1}\n")
        for e in range(n_exp):
            k = (l, e)
            p, lb, sc, so = prof[k], labels[k], scores[k], swx[k]
            mark = "✅" if sc["balanced_accuracy"] >= BA_PASS else "❌"
            w(f"#### L{l + 1}·E{e} — {lb['name']} {mark}\n")
            w(f"*{lb['job']}*  \n**Rule:** {lb['rule']}  \n**Kind:** {lb['kind']}\n")
            if len(variants) > 1:
                by_labeler = {}
                for v in variants:
                    by_labeler.setdefault(v["labeler"], []).append(v)
                for labeler, vs in by_labeler.items():
                    marks = " · ".join(f"{v['scorer']} scorer {v['scores'][k]['balanced_accuracy']:.2f}"
                                       f"{' ✅' if v['scores'][k]['balanced_accuracy'] >= BA_PASS else ''}" for v in vs)
                    if labeler == main_v["labeler"]:
                        w(f"- **Scores for this name:** {marks}")
                    else:
                        al = vs[0]["labels"][k]
                        w(f"- **{labeler}'s name:** {al['name']} ({marks}). *{al['job']}* **Rule:** {al['rule']}")
            w(f"- **Busy:** {p['share']:.0%} of words · JS {p['js']:.2f} · its top 10 pieces are "
              f"{p['top10_coverage']:.0%} of its traffic")
            w(f"- **Signature words:** {words(p['signature'])}")
            w(f"- **Most special (lift):** {words(p['top_lift'])}")
            cs = f" · card {card[k]['balanced_accuracy']:.2f}" if card else ""
            w(f"- **Score:** label **{sc['balanced_accuracy']:.2f}** (hits {sc['tpr']:.0%}, correct rejections "
              f"{sc['tnr']:.0%}; {sc['positives']} of {sc['n']} pieces routed here){cs} · word lookup "
              f"{p['lookup_ba']:.2f}")
            hurt = ", ".join(f"`{show(h)}` +{h['delta']:.2f}" for h in so["hurt_words"][:5])
            w(f"- **Switch-off cost:** reroute +{so['off_delta']:.4f} per word "
              f"({so['off_delta'] / sw['base_loss']:+.1%}), on its own words +{so['off_delta_handled']:.4f}; "
              f"silence +{so['silence_delta']:.4f} · predictions hurt most: {hurt or '—'}")
            for mode, m in modes.items():
                d, s, v, vc = m["desc"].get(k), m["stx"][k], m["ver"].get(k), m["conf"].get(k)
                if d:
                    tag = "specific" if d["specific"] else "general"
                    if v and v["tested"]:
                        tag += f", matched {v['correct']}/{v['tested']}" + (" ✅" if v["passed"] else "")
                    if vc and vc["tested"]:
                        tag += f", confirmation {vc['correct']}/{vc['tested']}" + (" ✅" if vc["passed"] else "")
                    n_ch = sum(x >= 0 for x in s["first_diff"])
                    w(f"- **Stories, {mode}:** {n_ch}/{len(s['first_diff'])} change. {d['change']} "
                      f"*({tag}, visible {d['visible']}/5)*")
            w("")

    # ---- sample stories
    w("## Sample stories\n")
    for mode, ok in qualifying.items():
        m = modes[mode]
        picks = [k for k, _ in ok][:4]
        if not picks:
            continue
        w(f"### Before and after a switch-off ({mode}, greedy)\n")
        for k in picks:
            d = m["desc"][k]
            i = d["pairs"][0] if d["pairs"] else 0
            w(f"**L{k[0] + 1}·E{k[1]} off** ({labels[k]['name']}): {d['change']}\n")
            w(f"> **Before:** {clip(m['st']['baseline'][i], 600)}\n>\n> **After:** "
              f"{clip(m['stx'][k]['texts'][i], 600)}\n")
    if qual and "dense" in qual["minds"]:
        w("### MoE vs dense twin (sampled at temperature 0.8, same beginning and seed)\n")
        for i in range(4):
            w(f"**Beginning:** {qual['prompts'][i]}\n")
            w(f"> **MoE:** {clip(qual['minds']['moe']['sampled'][i], 600)}\n>\n"
              f"> **Dense:** {clip(qual['minds']['dense']['sampled'][i], 600)}\n")

    # ---- top-1 addendum
    t1 = Path(args.top1_char)
    t1_prof = keyed(load(t1 / "profile.json"))
    if top1_run and t1_prof:
        w("## Addendum: the top-1 variant\n")
        rel = ((top1_run["val_loss_full"] - runs["dense"]["val_loss_full"]) / runs["dense"]["val_loss_full"]
               if runs["dense"] else float("nan"))
        w(f"The same mind with each word sent to **one** expert instead of two, trained on the same data, schedule and "
          f"seed. It has {top1_run['n_active'] / 1e6:.1f}M active parameters per word (the top-2 mind has "
          f"{runs['moe']['n_active'] / 1e6:.1f}M). Held-out loss {top1_run['val_loss_full']:.4f}, **{rel:+.1%}** "
          f"against the dense twin (the top-2 mind: {gate_numbers(char, runs['moe'], runs['dense'])['rel']:+.1%}).\n")
        t1_sc = keyed((load(t1 / "scores.json") or {}).get("scores"))
        t1_ss = keyed((load(Path(str(t1) + "-sol-by-sol") / "scores.json") or {}).get("scores"))
        t1_wr = {r["layer"]: r for r in (load(t1 / "wordrule.json") or [])}
        t1_sw = keyed((load(t1 / "switchoff.json") or {}).get("experts"))
        w("| layer | labels passing, luna/luna (median BA) | labels passing, sol/sol (median BA) | first choice = the "
          "word's usual | median reroute cost (nats) | median silence cost (nats) |")
        w("|---|---|---|---|---|---|")
        for l in range(n_layers):
            def col(src):
                if not src:
                    return "—"
                b_ = [src[(l, e)]["balanced_accuracy"] for e in range(n_exp)]
                return f"{sum(x >= BA_PASS for x in b_)}/{n_exp} ({stats.median(b_):.2f})"
            wr_ = f"{t1_wr[l]['first']:.0%}" if l in t1_wr else "—"
            rr = (f"+{stats.median(t1_sw[(l, e)]['off_delta'] for e in range(n_exp)):.4f}" if t1_sw else "—")
            ss = (f"+{stats.median(t1_sw[(l, e)]['silence_delta'] for e in range(n_exp)):.4f}" if t1_sw else "—")
            w(f"| {l + 1} | {col(t1_sc)} | {col(t1_ss)} | {wr_} | {rr} | {ss} |")
        w("")
        t1_lb = keyed(load(t1 / "labels.json"))
        t1_ls = keyed(load(Path(str(t1) + "-sol-by-sol") / "labels.json"))
        t1_c = [(d, c_modes(d)) for d in [t1] + [Path(str(t1) + cv["char"].name[len("char"):]) for cv in c_variants]]
        t1_c = [(d, cm) for d, cm in t1_c if cm and all(m["desc"] and m["ver"] for m in cm.values())]
        if t1_lb and t1_sc:
            w("### The 48 top-1 experts\n")
            w("Names from gpt-6-luna and gpt-6.1-sol, each scored by its own model (✅ at 0.75 or more). *Effect* "
              "lists switch-offs that passed the blind matching test and were confirmed again on fresh beginnings, "
              "with the models and switches that certified them (luna or sol; rerouted or muted).\n")
            w("| expert | luna's name | sol's name | busy | signature words | reroute cost | effect |")
            w("|---|---|---|---|---|---|---|")
            for k in sorted(t1_lb):
                pr, sw_ = t1_prof[k], t1_sw.get(k) if t1_sw else None
                eff = {}
                for d, cm in t1_c:
                    who = "sol" if "sol" in d.name else "luna"
                    for mode, m in cm.items():
                        v, vc = m["ver"].get(k), m["conf"].get(k)
                        if (gate_c_ok(m["desc"].get(k), m["stx"][k], v, vc) and vc and vc["passed"]):
                            eff.setdefault(m["desc"][k]["change"], []).append(
                                f"{who}, {'rerouted' if mode == 'reroute' else 'muted'}")
                # one sentence per expert: the one most models and switches agree on
                best = max(eff.items(), key=lambda kv: len(kv[1]), default=None)
                tags = sorted({t for v_ in eff.values() for t in v_})
                effect = f"{clip(best[0], 140)} *({'; '.join(tags)})*" if best else ""
                def nm(lbl, sc_):
                    if k not in lbl or not sc_:
                        return "—"
                    b_ = sc_[k]["balanced_accuracy"]
                    return f"{lbl[k]['name']} {b_:.2f}{' ✅' if b_ >= BA_PASS else ''}"
                cost = f"+{sw_['off_delta']:.3f}" if sw_ else "—"
                w(f"| L{k[0] + 1}·E{k[1]} | {nm(t1_lb, t1_sc)} | {nm(t1_ls, t1_ss)} | {pr['share']:.0%} | "
                  f"{words(pr['signature'], 5)} | {cost} | {effect} |")
            w("")
        for d, cm in t1_c:
            model = (load(d / "label_meta.json") or {}).get("describe", {}).get("model", "?")
            w(f"### Gate (c) for the top-1 mind, with {model} as describer and judge\n")
            c_block(cm, t1_lb)

    # ---- format + method
    fmt = (Path(__file__).parent / "export.py").read_text(encoding="utf-8").split('"""')[1]
    w("## File format (for Phase 1)\n")
    w("From the docstring of `tools/moe/export.py`:\n")
    w("```text\n" + fmt.strip() + "\n```\n")
    w("## Method and settings\n")
    if runinfo["moe"]:
        a = runinfo["moe"]["args"]
        w(f"- **Training:** {runinfo['moe']['total_steps']} steps × {runinfo['moe']['tokens_per_step']:,} pieces "
          f"(one shuffled pass over TinyStories train, 256-piece windows); AdamW (0.9, 0.95), weight decay "
          f"{a['wd']}; peak learning rate {a['lr']} after {a['warmup']} warmup steps, cosine decay to "
          f"{a['min_lr']}; bf16 autocast; gradient clip 1.0. MoE: load-balancing loss × {a['aux']}, router "
          f"z-loss × {a['zloss']}, dropless top-{a['top_k']} routing. The dense twin uses the same data order, "
          f"schedule and seed (micro-batches of {a['micro']} × {a['accum']} for the MoE and "
          f"{runinfo['dense']['args']['micro']} × {runinfo['dense']['args']['accum']} for the dense twin: the same "
          f"{a['micro'] * a['accum']} windows per step).")
    models = sorted({m["model"] for m in meta.values()})
    if models:
        w(f"- **Labeling model:** {', '.join(models)} for every step. `label.py` prefers gpt-6.1-luna, but on "
          f"2026-10-30 `GET /v1/models/gpt-6.1-luna` answered 404 for this key, so it fell back to gpt-6-luna.")
    for step in ("label", "score", "score-card", "score-retest", "describe", "describe-silence", "verify",
                 "verify-silence", "verify-confirm", "verify-silence-confirm", "judge"):
        m = meta.get(step)
        if m:
            u = m["usage"]
            w(f"- **{step}:** {m['model']} (reasoning effort {m['effort']}); {u['calls'] + u['cached']} calls"
              f"{' (' + str(u['cached']) + ' answered from the cache of an earlier run)' if u['cached'] else ''}; "
              f"new calls used {u['prompt']:,} prompt + {u['completion']:,} completion tokens "
              f"({u['reasoning']:,} reasoning)")
    w("- **Held-out data:** TinyStories validation, split by story: even-numbered stories (set A) for profiles "
      "and the labeler's contexts, odd-numbered (set B) for label scoring and switch-off losses.")
    w(f"- **Label scoring:** per layer, {sc_meta['n_passages']} random held-out passages of {sc_meta['length']} "
      f"pieces, the same for all 8 experts of the layer.")
    w("- **Reproduce:** see `tools/moe/README.md`.\n")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
