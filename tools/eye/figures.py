"""figures.py: neuron cards and growth strips for reports/eye/, plus growth-story candidates.

    python -I tools/eye/figures.py cards eye80k 3,17,40 [--out reports/eye/img]
    python -I tools/eye/figures.py growth eye80k [--show 12] [--strips 3,17]

cards: one PNG per neuron: its 16 top and 16 bottom held-out drawings (half A), each with
  its class name underneath.
growth: for every last-stage neuron, follow the classes of its top-100 drawings (half A)
  across the training snapshots and sort neurons into rough stories by class shares:
    kept       the final main class is the main class (half or more) from early on
    offshoots  the main class arrives early, and other classes join it later (10% or more
               at the end, under 3% when the main class arrived)
    changed    a different class led (half or more) at some earlier snapshot, once the net was
               already --settled (default 60%) right
    narrowed   the top-100 was spread over many classes and became one class later
  Class shares can't see shape jobs ("round things"), so these are only candidates to look
  at. Prints the candidates and, for --strips, writes each neuron's top-8 at several snapshots
  and its top-32 at the end (offshoots often sit at ranks 9-32).
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, load_data, panel, run_dir, run_meta, save_png, stacked, titled  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def cards(a, d):
    x, y = d['x_test'], d['y_test'].astype(int)
    classes = [str(c) for c in d['classes']]
    P = np.load(os.path.join(run_dir(a.run, a.data), 'poles.npz'))
    s = a.stage if a.stage else max(int(k[1]) for k in P.files if k.endswith('_topA')) + 1
    topA, botA = P[f's{s - 1}_topA'], P[f's{s - 1}_botA']
    os.makedirs(a.out, exist_ok=True)
    for j in [int(v) for v in a.neurons.split(',')]:
        cap = lambda idx: [classes[y[i]] for i in idx]
        im = titled([panel(x[topA[j]], captions=cap(topA[j])), panel(x[botA[j]], captions=cap(botA[j]))],
                    [f'{a.run} stage {s} neuron {j}: fires MOST', 'fires LEAST'])
        p = os.path.join(a.out, f'{a.run}-s{s}-n{j}.png')
        save_png(im, p)
        print('wrote', p)


def shares(lab):
    return np.bincount(lab, minlength=24) / len(lab)


def growth(a, d):
    x, y = d['x_test'], d['y_test'].astype(int)
    classes = [str(c) for c in d['classes']]
    G = np.load(os.path.join(run_dir(a.run, a.data), 'growth.npz'))
    steps, top = G['steps'], G['top']
    acc = {c['step']: c['test_acc'] for c in run_meta(a.run, a.data)['checkpoints']}
    T, H = top.shape[:2]
    first = int(np.argmax([acc[s] >= 0.25 for s in steps]))  # skip the blind first snapshots
    settled = int(np.argmax([acc[s] >= a.settled for s in steps]))  # brief early leads don't count as "changed"
    mid = int(np.argmin(np.abs(steps - steps[-1] / 4)))  # about a quarter of the way through training
    rows = []
    for j in range(H):
        S = np.array([shares(y[top[t, j]]) for t in range(T)])
        main = int(S[-1].argmax())
        onset = next((t for t in range(first, T) if S[t:, main].min() >= 0.5), None)
        story, extra = None, ''
        leaders = [(t, int(S[t].argmax())) for t in range(settled, T - 1) if S[t].max() >= 0.5 and S[t].argmax() != main]
        if leaders:
            story = 'changed'
            spans = {}
            for t, c in leaders:
                spans.setdefault(c, []).append(int(steps[t]))
            extra = 'led by ' + ', '.join(f'{classes[c]} (steps {v[0]}-{v[-1]})' for c, v in spans.items())
        elif onset is not None:
            off = [k for k in range(24) if k != main and S[-1, k] >= 0.10 and S[onset, k] < 0.03]
            if off and onset < T - 3:
                story = 'offshoots'
                extra = '+ ' + ', '.join(f'{classes[k]} {S[-1, k]:.0%}' for k in off)
            elif onset < T - 6:
                story = 'kept'
            if story is None and S[first:onset, :].max(1).min() < 0.25:
                story = 'narrowed'
        rows.append(dict(j=j, story=story, main=classes[main], final=S[-1, main], onset=None if onset is None else int(steps[onset]),
                         overlap=len(set(top[-1, j]) & set(top[mid, j])) / 100, extra=extra))
    curve = [dict(step=int(steps[t]), acc=acc[steps[t]],
                  overlap_with_final=float(np.mean([len(set(top[t, j]) & set(top[-1, j])) / 100 for j in range(H)])),
                  overlap_with_previous=float(np.mean([len(set(top[t, j]) & set(top[t - 1, j])) / 100 for j in range(H)])) if t else None,
                  top16_purity=float(np.mean([shares(y[top[t, j, :16]]).max() for j in range(H)])))
             for t in range(T)]
    print('snapshot: held-out acc, mean top-100 overlap with final / with previous snapshot, mean top-16 class purity')
    for c in curve:
        prev = '' if c['overlap_with_previous'] is None else f"{c['overlap_with_previous']:.0%}"
        print(f"  step {c['step']:5d}  {c['acc']:.0%}  {c['overlap_with_final']:.0%} / {prev:4s}  {c['top16_purity']:.2f}")
    with open(os.path.join(run_dir(a.run, a.data), 'growth-summary.json'), 'w') as f:
        json.dump(dict(stories={str(k): sum(r['story'] == k for r in rows) for k in ('offshoots', 'changed', 'kept', 'narrowed', None)},
                       curve=curve, neurons=[{**r, 'final': float(r['final'])} for r in rows]), f, indent=1)
    print(f'(per-neuron overlap below compares the end with step {steps[mid]})')
    for kind in ('offshoots', 'changed', 'kept', 'narrowed', None):
        rs = [r for r in rows if r['story'] == kind]
        print(f'\n{kind or "no story"}: {len(rs)} neurons')
        for r in rs[:a.show]:
            print(f"  n{r['j']:3d} main {r['main']:10s} {r['final']:.0%} at end, from step {r['onset']}; "
                  f"top-100 overlap mid vs end {r['overlap']:.0%} {r['extra']}")
    if a.strips:
        cks = [int(c) for c in a.ckpts.split(',')] if a.ckpts else np.linspace(first, T - 1, 5).round().astype(int).tolist()
        os.makedirs(a.out, exist_ok=True)
        for j in [int(v) for v in a.strips.split(',')]:
            im = stacked([panel(x[top[c, j][:8]], cols=8, captions=[classes[y[i]] for i in top[c, j][:8]]) for c in cks],
                         [f'step {steps[c]}\n{acc[steps[c]]:.0%} right' for c in cks], head_w=120)
            p = os.path.join(a.out, f'{a.run}-growth-n{j}.png')
            save_png(im, p)
            print('wrote', p, 'snapshots', [int(steps[c]) for c in cks])
            end = top[-1, j][:32]  # offshoots tend to sit just below the top 8
            im = titled([panel(x[end], cols=8, captions=[classes[y[i]] for i in end])],
                        [f'{a.run} neuron {j}: top-32 at the end of training'])
            p = os.path.join(a.out, f'{a.run}-end32-n{j}.png')
            save_png(im, p)
            print('wrote', p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['cards', 'growth'])
    ap.add_argument('run')
    ap.add_argument('neurons', nargs='?', default='')
    ap.add_argument('--stage', type=int, default=0)
    ap.add_argument('--show', type=int, default=12)
    ap.add_argument('--strips', default='')
    ap.add_argument('--ckpts', default='')
    ap.add_argument('--settled', type=float, default=0.6)
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=os.path.join(REPO, 'reports', 'eye', 'img'))
    a = ap.parse_args()
    d = load_data(a.data)
    (cards if a.mode == 'cards' else growth)(a, d)


if __name__ == '__main__':
    main()
