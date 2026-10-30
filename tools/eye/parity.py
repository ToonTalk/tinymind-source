"""parity.py: the PyTorch numbers src/eye.js must reproduce (tests/e19-eye.test.js).

    python -I tools/eye/parity.py [--run eye80k]

On the CPU in plain float32 (no TF32), for doodles from the shipped sample set
(eye/tiny-eye-sample.bin): logits and last-stage neurons for 100 doodles spread over every
category, and every stage's neurons for one doodle of each category; the same for one
training snapshot with its weights rounded to float16 (as eye/tiny-eye-growth.bin holds it,
--snapshot) and for the squeezed eye (--squeezed). A neuron's value is
its fc1 output (before the GELU) averaged over the grid, as in Phase 0. Numbers are stored
as base64 float32 so the fixture stays small and exact.
Writes tests/fixtures/eye-parity.json.
"""
import argparse
import base64
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, checkpoints, load_model, readouts  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def b64(a):
    return base64.b64encode(np.ascontiguousarray(a, dtype='<f4').tobytes()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='eye80k')
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--snapshot', type=int, default=1206, help='a training step whose snapshot is checked too')
    ap.add_argument('--squeezed', default='eye80k-sq8')
    a = ap.parse_args()
    man = json.load(open(os.path.join(ROOT, 'eye', 'tiny-eye.json'), encoding='utf-8'))
    x = np.fromfile(os.path.join(ROOT, 'eye', man['sample']['file']), dtype=np.uint8).reshape(-1, 28, 28)
    counts = man['sample']['classCounts']
    spread = [int(round(i * (len(x) - 1) / 99)) for i in range(100)]
    firsts = [int(v) for v in np.cumsum([0] + counts[:-1])]
    model, step = load_model(checkpoints(a.run, a.data)[-1], device='cpu')
    reads, logits = readouts(model, x[spread], device='cpu')
    early, _ = readouts(model, x[firsts], device='cpu')
    out = dict(run=a.run, step=step, note='PyTorch float32 on the CPU; base64 little-endian float32, row-major',
               spread_rows=spread, logits=b64(logits), last_stage=b64(reads[-1]),
               per_class_rows=firsts, stages=[b64(r) for r in early], hidden=[r.shape[1] for r in early])
    # a training snapshot as the room loads it (weights rounded to float16), and the squeezed eye
    snap_path = next(p for p in checkpoints(a.run, a.data) if p.endswith(f'{a.snapshot:06d}.pt'))
    snap, snap_step = load_model(snap_path, device='cpu')
    with torch.no_grad():
        for p in snap.parameters():
            p.copy_(p.half().float())
    s_reads, s_logits = readouts(snap, x[firsts], device='cpu')
    out['snapshot'] = dict(step=snap_step, rows=firsts, logits=b64(s_logits), last_stage=b64(s_reads[-1]),
                           note='weights rounded to float16, as in eye/tiny-eye-growth.bin')
    sq, sq_step = load_model(checkpoints(a.squeezed, a.data)[-1], device='cpu')
    q_reads, q_logits = readouts(sq, x[firsts], device='cpu')
    out['squeezed'] = dict(run=a.squeezed, step=sq_step, rows=firsts, logits=b64(q_logits),
                           stages=[b64(r) for r in q_reads], hidden=[r.shape[1] for r in q_reads])
    path = os.path.join(ROOT, 'tests', 'fixtures', 'eye-parity.json')
    with open(path, 'w') as f:
        json.dump(out, f)
    print(f'{path}: {os.path.getsize(path):,} bytes; step {step}')


if __name__ == '__main__':
    main()
