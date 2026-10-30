"""label.py: name Tiny Eye neurons with a vision model, then check each name with a distractor test.

    python -I tools/eye/label.py neurons eye80k [--sample 64] [--model gpt-6-luna]
    python -I tools/eye/label.py neurons eye80k --retest 32
    python -I tools/eye/label.py growth eye80k --neurons 12,40 --ckpts 6,9,12,15,19

neurons: for each sampled last-stage neuron, one call shows the vision model a picture of
  its 16 top and 16 bottom drawings (held-out half A) and asks for a short label per pole and
  a judgment ("one" main job, "two" related jobs, "mixed" or "unclear"). Then one call per
  pole shows only that label and 16 numbered drawings from half B: 4 drawn from the pole's
  top-16 in B and 12 random drawings from outside its top-200. The model must pick the 4.
  A pole passes with at least 3 of 4 right (chance of passing by guessing: 2.7%).
--retest N: score the first N sampled neurons' poles once more, on a fresh random draw, to
  see how often a single trial's verdict flips.
growth: one call per neuron shows its top-8 drawings at several checkpoints, one row each,
  and asks for a label per row and whether the late job grew out of the early one.

Model: --model, else gpt-6.1-luna if GET /v1/models/gpt-6.1-luna answers 200, else gpt-6-luna.
Results go to <data>/labels/<run>.json (and growth-<run>.json); calls are cached there, so a
rerun only asks for what is missing. Every call is counted in <data>/labels/calls.json and the
tool stops at --max-calls (default 1000) across all runs.

The key comes from OPENAI_API_KEY in the environment. It is only put in the Authorization
header of each request: never printed, logged or written to a file.
"""
import argparse
import base64
import io
import json
import os
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, halves, load_data, panel, run_dir, stacked, titled

API = 'https://api.openai.com/v1'
KEY = os.environ.get('OPENAI_API_KEY')
LOCK = threading.Lock()


def pick_model(asked):
    if asked:
        return asked
    req = urllib.request.Request(API + '/models/gpt-6.1-luna', headers={'Authorization': 'Bearer ' + KEY})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            if r.status == 200:
                return 'gpt-6.1-luna'
    except urllib.error.HTTPError:
        pass
    return 'gpt-6-luna'


class Ledger:
    def __init__(self, path, cap):
        self.path, self.cap = path, cap
        self.d = json.load(open(path)) if os.path.exists(path) else dict(calls=0, prompt_tokens=0, completion_tokens=0, reasoning_tokens=0)

    def take(self):
        with LOCK:
            if self.d['calls'] >= self.cap:
                raise RuntimeError(f'call cap {self.cap} reached')
            self.d['calls'] += 1
            self.save()

    def usage(self, u):
        with LOCK:
            self.d['prompt_tokens'] += u.get('prompt_tokens', 0)
            self.d['completion_tokens'] += u.get('completion_tokens', 0)
            self.d['reasoning_tokens'] += (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)
            self.save()

    def save(self):
        json.dump(self.d, open(self.path, 'w'), indent=1)


def png_url(im):
    buf = io.BytesIO()
    im.save(buf, 'PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def ask(model, ledger, text, im, max_tokens=8000):
    """One chat call with a text part and an image part; returns the parsed JSON answer."""
    err = None
    for attempt in range(4):
        ledger.take()
        body = dict(model=model, max_completion_tokens=max_tokens, response_format={'type': 'json_object'},
                    messages=[{'role': 'user', 'content': [
                        {'type': 'text', 'text': text},
                        {'type': 'image_url', 'image_url': {'url': png_url(im), 'detail': 'high'}}]}])
        req = urllib.request.Request(API + '/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Authorization': 'Bearer ' + KEY, 'Content-Type': 'application/json'})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                j = json.load(r)
        except urllib.error.HTTPError as e:
            err = f'HTTP {e.code}: {e.read()[:300]!r}'
            time.sleep(5 * (attempt + 1))
            continue
        except (urllib.error.URLError, TimeoutError) as e:
            err = str(e)
            time.sleep(5 * (attempt + 1))
            continue
        ledger.usage(j.get('usage', {}))
        ch = j['choices'][0]
        if ch.get('finish_reason') == 'length':
            max_tokens *= 2
            err = 'ran out of tokens'
            continue
        txt = ch['message'].get('content') or ''
        m = re.search(r'\{.*\}', txt, re.S)
        try:
            return json.loads(m.group(0) if m else txt)
        except json.JSONDecodeError:
            err = 'bad JSON: ' + txt[:200]
    raise RuntimeError(err)


def kinds(classes):
    return ', '.join(classes)


LABEL_PROMPT = """This picture shows one neuron of a tiny neural network that recognizes children's doodles: 28x28-pixel sketches from the Quick, Draw! game. The doodles come from 24 kinds of things: {kinds}.
Left: the 16 doodles that make this neuron fire MOST. Right: the 16 doodles that make it fire LEAST (its other pole).
For each side, say what its doodles have in common, in at most 8 simple words a 10-year-old understands (American spelling). It can be a kind of thing ("dragons"), a shape or stroke ("long flat lines", "a circle with spikes around it"), where the drawing sits, how big or busy it is, or a mix of these. Describe the one shared idea; do not just list things.
Also judge each side: "one" = one main job; "two" = two related jobs; "mixed" = several unrelated things; "unclear" = no pattern you can name.
Answer only with JSON: {{"most": {{"label": "...", "jobs": "one|two|mixed|unclear"}}, "least": {{"label": "...", "jobs": "one|two|mixed|unclear"}}}}"""

SCORE_PROMPT = """A neuron in a tiny network that recognizes children's doodles (28x28-pixel sketches from the Quick, Draw! game) was described like this:
"{label}"
The picture shows 16 numbered doodles. Exactly 4 of them are among the doodles that description was written for; the other 12 are random doodles. Pick the 4 that best fit the description.
Answer only with JSON: {{"picks": [n1, n2, n3, n4]}}"""

GROWTH_PROMPT = """Each row of this picture shows the 8 doodles that make the same neuron of a tiny doodle-recognizing network fire most, at one moment during its training, from early (top row) to late (bottom row). The doodles are 28x28-pixel sketches from the Quick, Draw! game, of 24 kinds of things: {kinds}.
For each row, say what its doodles have in common in at most 8 simple words (American spelling), or "no pattern".
Then say how the neuron's job changed over training: "kept" (the same idea throughout, once one appears), "offshoots" (an early idea is kept and related things are added later), "narrowed" (a broad early idea shrinks to a narrower one), "changed" (the early idea is replaced by an unrelated one), or "never clear". Add one short sentence explaining.
Answer only with JSON: {{"rows": ["...", ...], "story": "kept|offshoots|narrowed|changed|never clear", "why": "..."}}"""


def label_neurons(a, model, ledger):
    d = load_data(a.data)
    x, y = d['x_test'], d['y_test'].astype(int)
    classes = [str(c) for c in d['classes']]
    A, B = halves(y)
    P = np.load(os.path.join(run_dir(a.run, a.data), 'poles.npz'))
    s = max(int(k[1]) for k in P.files if k.endswith('_topA'))
    topA, botA, topB, botB = P[f's{s}_topA'], P[f's{s}_botA'], P[f's{s}_topB'], P[f's{s}_botB']
    H = len(topA)
    rng = np.random.default_rng(a.seed)
    chosen = sorted(rng.choice(H, min(a.sample, H), replace=False).tolist())
    out_dir = os.path.join(a.data, 'labels')
    img_dir = os.path.join(out_dir, a.run)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(out_dir, a.run + '.json')
    res = json.load(open(path)) if os.path.exists(path) else dict(run=a.run, stage=s + 1, neurons={})
    res['model'] = model
    res['sampled'] = chosen

    def save():
        with LOCK:
            json.dump(res, open(path, 'w'), indent=1)

    def one(j):
        key = str(j)
        rec = res['neurons'].get(key) or {}
        if 'most' not in rec:
            im = titled([panel(x[topA[j]]), panel(x[botA[j]])], ['fires MOST', 'fires LEAST'])
            im.save(os.path.join(img_dir, f'n{j}.png'))
            ans = ask(model, ledger, LABEL_PROMPT.format(kinds=kinds(classes)), im)
            rec = {pole: dict(label=(ans.get(pole) or {}).get('label'), jobs=(ans.get(pole) or {}).get('jobs'))
                   for pole in ('most', 'least')}
            for pole, top in (('most', topA[j]), ('least', botA[j])):
                c = np.bincount(y[top], minlength=24)
                rec[pole]['classes'] = {classes[i]: int(c[i]) for i in np.argsort(-c) if c[i]}
            res['neurons'][key] = rec
            save()
        for pole, near in (('most', topB[j]), ('least', botB[j])):
            r = rec[pole]
            if 'hits' in r:
                continue
            lab = r.get('label')
            if not lab or r.get('jobs') == 'unclear' or str(lab).strip().lower() in ('null', 'none', 'no pattern'):
                r.update(hits=0, passed=False, skipped='no label')
                save()
                continue
            r.update(test(j, pole, near, lab, 0))
            save()
        return j

    def test(j, pole, near, lab, salt):
        """The distractor test: 4 of the pole's top-16 in half B among 12 random others."""
        g = np.random.default_rng([a.seed + salt, j, int(pole == 'most')])
        truth = g.choice(near[:16], 4, replace=False)
        fakes = g.choice(np.setdiff1d(B, near), 12, replace=False)
        order = g.permutation(16)
        cands = np.concatenate([truth, fakes])[order]
        truth_pos = sorted(int(np.flatnonzero(order == k)[0]) + 1 for k in range(4))
        im = panel(x[cands], numbers=list(range(1, 17)))
        im.save(os.path.join(img_dir, f'n{j}-{pole}-test{salt or ""}.png'))
        ans = ask(model, ledger, SCORE_PROMPT.format(label=lab), im)
        picks = [int(p) for p in (ans.get('picks') or []) if str(p).strip().isdigit()][:4]
        hits = len(set(picks) & set(truth_pos))
        return dict(truth=truth_pos, picks=picks, hits=hits, passed=hits >= 3)

    def retest(j):
        """Score already-labeled poles again on a fresh draw of drawings, to see how noisy one trial is."""
        rec = res['neurons'][str(j)]
        for pole, near in (('most', topB[j]), ('least', botB[j])):
            r = rec[pole]
            if 'retest' not in r and not r.get('skipped'):
                r['retest'] = test(j, pole, near, r['label'], 1)
                save()
        return j

    if a.retest:
        with ThreadPoolExecutor(a.parallel) as ex:
            list(ex.map(retest, chosen[:a.retest]))
        rs = [res['neurons'][str(j)][p] for j in chosen[:a.retest] for p in ('most', 'least')]
        rs = [r for r in rs if 'retest' in r]
        res['retest'] = dict(poles=len(rs), same_verdict=sum(r['passed'] == r['retest']['passed'] for r in rs),
                             first_pass=sum(r['passed'] for r in rs), second_pass=sum(r['retest']['passed'] for r in rs))
        save()
        print(a.run, 'retest', json.dumps(res['retest']), 'ledger', json.dumps(ledger.d), flush=True)
        return

    todo = chosen[:a.limit] if a.limit else chosen
    with ThreadPoolExecutor(a.parallel) as ex:
        for j in ex.map(one, todo):
            rec = res['neurons'][str(j)]
            print(f"  n{j}: most '{rec['most']['label']}' ({rec['most'].get('hits')}/4) | "
                  f"least '{rec['least']['label']}' ({rec['least'].get('hits')}/4)", flush=True)
    if todo != chosen:
        return
    poles = [res['neurons'][str(j)][p] for j in chosen for p in ('most', 'least')]
    res['summary'] = dict(poles=len(poles), passed=sum(p['passed'] for p in poles),
                          pass_rate=sum(p['passed'] for p in poles) / len(poles),
                          mean_hits=float(np.mean([p['hits'] for p in poles])),
                          jobs={k: sum(p.get('jobs') == k for p in poles) for k in ('one', 'two', 'mixed', 'unclear')})
    save()
    print(a.run, json.dumps(res['summary']), 'ledger', json.dumps(ledger.d), flush=True)


def label_growth(a, model, ledger):
    d = load_data(a.data)
    x = d['x_test']
    classes = [str(c) for c in d['classes']]
    G = np.load(os.path.join(run_dir(a.run, a.data), 'growth.npz'))
    steps, top = G['steps'], G['top']
    cks = [int(c) for c in a.ckpts.split(',')]
    out_dir = os.path.join(a.data, 'labels')
    img_dir = os.path.join(out_dir, a.run)
    os.makedirs(img_dir, exist_ok=True)
    path = os.path.join(out_dir, f'growth-{a.run}.json')
    res = json.load(open(path)) if os.path.exists(path) else dict(run=a.run, neurons={})
    res['model'] = model

    def one(j):
        key = f'{j}@' + ','.join(map(str, cks))
        if key in res['neurons']:
            return j
        im = stacked([panel(x[top[c][j][:8]], cols=8) for c in cks], [f'step {steps[c]}' for c in cks])
        im.save(os.path.join(img_dir, f'growth-n{j}.png'))
        ans = ask(model, ledger, GROWTH_PROMPT.format(kinds=kinds(classes)), im)
        with LOCK:
            res['neurons'][key] = dict(neuron=j, steps=[int(steps[c]) for c in cks], **ans)
            json.dump(res, open(path, 'w'), indent=1)
        return j

    with ThreadPoolExecutor(a.parallel) as ex:
        for j in ex.map(one, [int(v) for v in a.neurons.split(',')]):
            print(f'  n{j}: done', flush=True)
    print('ledger', json.dumps(ledger.d), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['neurons', 'growth'])
    ap.add_argument('run')
    ap.add_argument('--model', default=None)
    ap.add_argument('--sample', type=int, default=64)
    ap.add_argument('--limit', type=int, default=0, help='only do the first N sampled neurons (a trial)')
    ap.add_argument('--retest', type=int, default=0, help='score the first N sampled neurons again on fresh drawings')
    ap.add_argument('--neurons', default='')
    ap.add_argument('--ckpts', default='')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--parallel', type=int, default=6)
    ap.add_argument('--max-calls', type=int, default=1000)
    ap.add_argument('--data', default=DATA)
    a = ap.parse_args()
    if not KEY:
        raise SystemExit('OPENAI_API_KEY is not set in the environment')
    model = pick_model(a.model)
    os.makedirs(os.path.join(a.data, 'labels'), exist_ok=True)
    ledger = Ledger(os.path.join(a.data, 'labels', 'calls.json'), a.max_calls)
    print('labeling model:', model, flush=True)
    (label_neurons if a.mode == 'neurons' else label_growth)(a, model, ledger)


if __name__ == '__main__':
    main()
