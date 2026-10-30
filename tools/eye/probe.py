"""probe.py: find each neuron's poles in a trained Tiny Eye, with no API calls.

    python -I tools/eye/probe.py eye20k eye80k eye300k eye80k-sq8 eye80k-sqmlp8

For every run, from the final checkpoint and the held-out drawings:
  - accuracy and the most confused class pairs;
  - for every neuron in every stage, its top-16 and bottom-16 drawings in half A of the
    held-out set (the label set), and the top/bottom 200 in half B (the scoring set);
  - class purity of each pole: the share of its 16 drawings that come from its most common
    class. This is a rough, free stand-in for "one main job" (a neuron whose job is a shape,
    like "round things", spans classes and so scores low here even when it is clean).
Across all checkpoints, for the last stage: each neuron's top-100 and bottom-100 in half A,
for the growth replay.

Writes <run>/poles.npz, <run>/growth.npz and <run>/probe.json.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, checkpoints, halves, load_data, load_model, readouts, run_dir


def ranks(v, idx, k):
    """Indices (into the held-out set) of the k highest and k lowest values of v over idx."""
    o = np.argsort(v[idx], kind='stable')
    return idx[o[::-1][:k]], idx[o[:k]]


def purity(lab):
    c = np.bincount(lab, minlength=24)
    return c.max() / len(lab), int(c.argmax())


def probe(name, data):
    d = load_data(data)
    x, y = d['x_test'], d['y_test'].astype(int)
    A, B = halves(y)
    cks = checkpoints(name, data)
    model, step = load_model(cks[-1])
    reads, logits = readouts(model, x)
    pred = logits.argmax(1)
    acc = float((pred == y).mean())
    conf = np.zeros((24, 24), int)
    np.add.at(conf, (y, pred), 1)
    off = [(int(conf[i, j] + conf[j, i]), i, j) for i in range(24) for j in range(i + 1, 24)]
    off.sort(reverse=True)
    classes = [str(c) for c in d['classes']]
    out = dict(name=name, step=step, acc=acc,
               per_class_acc={classes[i]: float(conf[i, i] / conf[i].sum()) for i in range(24)},
               confused=[dict(pair=[classes[i], classes[j]], count=n) for n, i, j in off[:8]], stages=[])
    save = {}
    for s, r in enumerate(reads):
        H = r.shape[1]
        topA, botA, topB, botB, pur, cls = [], [], [], [], [], []
        for j in range(H):
            ta, ba = ranks(r[:, j], A, 16)
            tb, bb = ranks(r[:, j], B, 200)
            topA.append(ta); botA.append(ba); topB.append(tb); botB.append(bb)
            pt, ct = purity(y[ta]); pb, cb = purity(y[ba])
            pur.append([pt, pb]); cls.append([ct, cb])
        pur = np.array(pur)
        save.update({f's{s}_topA': np.array(topA), f's{s}_botA': np.array(botA),
                     f's{s}_topB': np.array(topB), f's{s}_botB': np.array(botB),
                     f's{s}_purity': pur, f's{s}_cls': np.array(cls), f's{s}_std': r.std(0)})
        out['stages'].append(dict(stage=s + 1, neurons=H, mean_purity=float(pur.mean()),
                                  poles_purity_ge_75=float((pur >= 0.75).mean()),
                                  poles_purity_ge_50=float((pur >= 0.5).mean()),
                                  quiet_neurons=int((r.std(0) < 0.05 * r.std(0).max()).sum())))
    np.savez(os.path.join(run_dir(name, data), 'poles.npz'), **save)

    steps, tops, bots, last_pur = [], [], [], []
    for ck in cks:
        m, st = load_model(ck)
        r = readouts(m, x[A])[0][-1]
        o = np.argsort(r, axis=0, kind='stable')
        top, bot = A[o[::-1][:100].T], A[o[:100].T]
        steps.append(st); tops.append(top); bots.append(bot)
        last_pur.append(np.mean([[purity(y[t[:16]])[0], purity(y[b[:16]])[0]] for t, b in zip(top, bot)]))
    np.savez(os.path.join(run_dir(name, data), 'growth.npz'), steps=np.array(steps), top=np.array(tops), bot=np.array(bots))
    out['growth_mean_purity'] = [dict(step=int(s), mean_purity=float(p)) for s, p in zip(steps, last_pur)]
    with open(os.path.join(run_dir(name, data), 'probe.json'), 'w') as f:
        json.dump(out, f, indent=1)
    st = out['stages'][-1]
    print(f'{name}: acc {acc:.4f}; last stage {st["neurons"]} neurons, mean pole purity {st["mean_purity"]:.2f}, '
          f'purity>=0.75 {st["poles_purity_ge_75"]:.0%}, quiet {st["quiet_neurons"]}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--data', default=DATA)
    a = ap.parse_args()
    for name in a.runs:
        probe(name, a.data)


if __name__ == '__main__':
    main()
