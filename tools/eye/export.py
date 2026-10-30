"""export.py: weights, training snapshots, a sample set and a parity fixture for the browser.

    python -I tools/eye/export.py eye80k [--out C:/Users/toont/tm-data/eye/export]

Writes to <out>/:
  tiny-eye.json        manifest: config, classes, tensor names/shapes/offsets, preprocessing
  tiny-eye.bin         final weights, float32 little-endian, in manifest order
  checkpoints/<step>.bin  every training snapshot, float16, same layout (for the growth scrubber)
  sample.bin           5,000 held-out doodles, uint8 28x28 (ink = 255), row after row
  sample.json          their classes, held-out indices and row numbers in Google's files
  sample.zip           sample.bin + sample.json zipped (to show the shipped size)
  parity.json          100 sample doodles with PyTorch logits and last-stage neuron values
"""
import argparse
import json
import os
import sys
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # -I leaves the script folder off the path
from eyelib import DATA, checkpoints, load_data, load_model, readouts, run_meta

ATTRIBUTION = ('Quick, Draw! drawings: made available by Google, Inc. under the Creative Commons '
               'Attribution 4.0 International license (https://github.com/googlecreativelab/quickdraw-dataset).')


def flat(model, dtype):
    names, parts, off = [], [], 0
    for k, v in model.state_dict().items():
        a = v.detach().cpu().numpy().astype(dtype)
        names.append(dict(name=k, shape=list(a.shape), offset=off, length=int(a.size)))
        parts.append(a.ravel())
        off += a.size
    return names, np.concatenate(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('run')
    ap.add_argument('--data', default=DATA)
    ap.add_argument('--out', default=None)
    ap.add_argument('--n', type=int, default=5000)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    out = a.out or os.path.join(a.data, 'export')
    os.makedirs(os.path.join(out, 'checkpoints'), exist_ok=True)
    meta = run_meta(a.run, a.data)
    d = load_data(a.data)
    classes = [str(c) for c in d['classes']]
    cks = checkpoints(a.run, a.data)

    model, step = load_model(cks[-1])
    tensors, w = flat(model, '<f4')
    w.tofile(os.path.join(out, 'tiny-eye.bin'))
    snaps = []
    for ck in cks:
        m, st = load_model(ck)
        flat(m, '<f2')[1].tofile(os.path.join(out, 'checkpoints', f'{st:06d}.bin'))
        acc = next(c['test_acc'] for c in meta['checkpoints'] if c['step'] == st)
        snaps.append(dict(step=st, epoch=round(st / meta['steps_per_epoch'], 3), file=f'checkpoints/{st:06d}.bin', held_out_acc=acc))
    manifest = dict(
        name='Tiny Eye', run=a.run, params=int(w.size), config=meta['config'], classes=classes,
        held_out_acc=meta['final_acc'], dtype='float32 little-endian', tensors=tensors, snapshots=snaps,
        snapshot_dtype='float16 little-endian',
        architecture=('stem: conv 2x2 stride 2 (1->w0) + LayerNorm over channels; stage i>0 starts with LayerNorm + '
                      'conv 2x2 stride 2 (padding 1 when the grid is odd: 14->7->4); each block: depthwise 7x7 conv '
                      '(padding 3) -> LayerNorm -> fc1 -> GELU (erf) -> fc2 -> times gamma -> add to the block input; '
                      'head: mean over the grid -> LayerNorm -> linear. LayerNorm eps 1e-5. Conv weights are '
                      '[out, in/groups, kh, kw]; Linear weights are [out, in].'),
        neurons='fc1 outputs (before GELU) of each block; a drawing\'s value for a neuron is its mean over the grid.',
        input=('28x28 grayscale, row-major, ink = 1.0 and paper = 0.0 (bitmap / 255). In the Quick, Draw! bitmaps the '
               'drawing is scaled so its longer side spans 26 pixels (a 1-pixel margin), centered on the shorter side, '
               'with soft anti-aliased strokes about 2 pixels wide (about 27% of pixels have ink).'),
        attribution=ATTRIBUTION)
    with open(os.path.join(out, 'tiny-eye.json'), 'w') as f:
        json.dump(manifest, f, indent=1)

    x, y = d['x_test'], d['y_test'].astype(int)
    rng = np.random.default_rng(a.seed)
    per = [a.n // 24 + (c < a.n % 24) for c in range(24)]
    idx = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c), per[c], replace=False) for c in range(24)]))
    x[idx].tofile(os.path.join(out, 'sample.bin'))
    sample = dict(count=int(len(idx)), shape=[28, 28], dtype='uint8, ink = 255', classes=classes,
                  labels=y[idx].tolist(), held_out_index=idx.tolist(),
                  source_row=d['src_test'][idx].tolist(), attribution=ATTRIBUTION)
    with open(os.path.join(out, 'sample.json'), 'w') as f:
        json.dump(sample, f)
    with zipfile.ZipFile(os.path.join(out, 'sample.zip'), 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.write(os.path.join(out, 'sample.bin'), 'sample.bin')
        z.write(os.path.join(out, 'sample.json'), 'sample.json')

    cpu, _ = load_model(cks[-1], device='cpu')  # plain float32, no TF32, for an exact fixture
    reads, logits = readouts(cpu, x[idx[:100]], device='cpu')
    with open(os.path.join(out, 'parity.json'), 'w') as f:
        json.dump(dict(run=a.run, step=step, sample_rows=list(range(100)),
                       logits=np.round(logits, 6).tolist(),
                       last_stage_neurons=np.round(reads[-1], 6).tolist()), f)
    sizes = {k: os.path.getsize(os.path.join(out, k)) for k in ('tiny-eye.bin', 'tiny-eye.json', 'sample.bin', 'sample.zip', 'parity.json')}
    print(json.dumps(dict(run=a.run, params=int(w.size), snapshots=len(snaps), sizes=sizes), indent=1))


if __name__ == '__main__':
    main()
