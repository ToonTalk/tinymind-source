"""breadth.py: how broad is each last-stage neuron, looking deeper than its top 16?

    python -I tools/eye/breadth.py eye20k eye80k eye300k eye80k-sq8 eye80k-sqmlp8

The distractor test only sees a pole's top-16 drawings. A squeezed neuron can have a pure
top-16 and still answer to many classes a little further down: "coarser", in the essay's
word. For each pole (final snapshot, held-out half A) this counts the effective number of
classes (e to the entropy of the class shares) among its top 16, 100 and 500 drawings, and
the share of classes whose mean value is above the neuron's overall mean plus one standard
deviation (how many kinds of thing it really answers to). Writes <run>/breadth.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, checkpoints, halves, load_data, load_model, readouts, run_dir  # noqa: E402

KS = (16, 100, 500)


def eff(lab):
    p = np.bincount(lab, minlength=24) / len(lab)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--data', default=DATA)
    a = ap.parse_args()
    d = load_data(a.data)
    y = d['y_test'].astype(int)
    A, _ = halves(y)
    x, y = d['x_test'][A], y[A]
    print('| run | neurons | effective classes in top 16 / 100 / 500 (fires most) | same (fires least) | classes a neuron answers to (mean +1 sd), most / least |')
    print('|---|---:|---|---|---|')
    for name in a.runs:
        model, _ = load_model(checkpoints(name, a.data)[-1])
        r = readouts(model, x)[0][-1]
        H = r.shape[1]
        out = dict(run=name, neurons=H, most={}, least={})
        cm = np.stack([r[y == c].mean(0) for c in range(24)])  # 24 x H class means
        z = (cm - r.mean(0)) / r.std(0)
        out['most']['classes_answered'] = float((z > 1).sum(0).mean())
        out['least']['classes_answered'] = float((z < -1).sum(0).mean())
        o = np.argsort(r, axis=0, kind='stable')
        for k in KS:
            out['most'][f'top{k}'] = float(np.mean([eff(y[o[::-1][:k, j]]) for j in range(H)]))
            out['least'][f'top{k}'] = float(np.mean([eff(y[o[:k, j]]) for j in range(H)]))
        with open(os.path.join(run_dir(name, a.data), 'breadth.json'), 'w') as f:
            json.dump(out, f, indent=1)
        fm = ' / '.join(f"{out['most'][f'top{k}']:.1f}" for k in KS)
        fl = ' / '.join(f"{out['least'][f'top{k}']:.1f}" for k in KS)
        print(f"| {name} | {H} | {fm} | {fl} | {out['most']['classes_answered']:.1f} / {out['least']['classes_answered']:.1f} |", flush=True)


if __name__ == '__main__':
    main()
