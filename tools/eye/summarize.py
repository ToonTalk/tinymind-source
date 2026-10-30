"""summarize.py: the numbers for reports/eye/feasibility.md, as markdown tables.

    python -I tools/eye/summarize.py eye20k eye80k eye300k eye80k-sq8 eye80k-sqmlp8

Reads each run's meta.json and probe.json and <data>/labels/<run>.json. Pass rates come with
a 95% Wilson interval.
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, run_dir, run_meta  # noqa: E402


def wilson(k, n, z=1.96):
    if not n:
        return 0, 0
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def rate(k, n):
    lo, hi = wilson(k, n)
    return f'{k}/{n} = **{k / n:.0%}** ({lo:.0%}–{hi:.0%})' if n else '–'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--data', default=DATA)
    a = ap.parse_args()
    print('| run | parameters | last-stage neurons | held-out accuracy | training time |')
    print('|---|---:|---:|---:|---:|')
    for r in a.runs:
        m = run_meta(r, a.data)
        print(f"| {r} | {m['params']:,} | {m['config']['hidden'][-1]} | {m['final_acc']:.1%} | {m['train_seconds'] / 60:.0f} min |")
    print()
    print('| run | poles tested | pass (both poles) | "fires most" pass | "fires least" pass | mean hits of 4 | labeler: one / two / mixed / unclear |')
    print('|---|---:|---|---|---|---:|---|')
    out = {}
    for r in a.runs:
        p = os.path.join(a.data, 'labels', r + '.json')
        if not os.path.exists(p):
            continue
        d = json.load(open(p))
        js = d['sampled']
        poles = {k: [d['neurons'][str(j)][k] for j in js if 'hits' in d['neurons'].get(str(j), {}).get(k, {})] for k in ('most', 'least')}
        allp = poles['most'] + poles['least']
        k = sum(q['passed'] for q in allp)
        jobs = ' / '.join(str(sum(q.get('jobs') == t for q in allp)) for t in ('one', 'two', 'mixed', 'unclear'))
        hits = sum(q['hits'] for q in allp) / max(1, len(allp))
        print(f"| {r} | {len(allp)} | {rate(k, len(allp))} | {rate(sum(q['passed'] for q in poles['most']), len(poles['most']))} | "
              f"{rate(sum(q['passed'] for q in poles['least']), len(poles['least']))} | {hits:.2f} | {jobs} |")
        both = sum(d['neurons'][str(j)]['most']['passed'] and d['neurons'][str(j)]['least']['passed'] for j in js)
        either = sum(d['neurons'][str(j)]['most']['passed'] or d['neurons'][str(j)]['least']['passed'] for j in js)
        out[r] = dict(neurons=len(js), both=both, either=either, model=d.get('model'))
    print()
    print('| run | neurons sampled | both poles pass | at least one pole passes | labeling model |')
    print('|---|---:|---:|---:|---|')
    for r, o in out.items():
        print(f"| {r} | {o['neurons']} | {o['both']} | {o['either']} | {o['model']} |")
    print()
    print('Class purity (share of a pole\'s 16 drawings from its most common class), all neurons, final snapshot:')
    print()
    print('| run | stage | neurons | mean purity | poles with purity >= 75% | poles with purity >= 50% |')
    print('|---|---:|---:|---:|---:|---:|')
    for r in a.runs:
        pj = os.path.join(run_dir(r, a.data), 'probe.json')
        if not os.path.exists(pj):
            continue
        for s in json.load(open(pj))['stages']:
            print(f"| {r} | {s['stage']} | {s['neurons']} | {s['mean_purity']:.2f} | {s['poles_purity_ge_75']:.0%} | {s['poles_purity_ge_50']:.0%} |")


if __name__ == '__main__':
    main()
