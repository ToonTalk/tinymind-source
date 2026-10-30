"""download.py: fetch 24 Quick, Draw! categories as 28x28 bitmaps (Phase 0 of plans/tiny-eye.md).

    python -I tools/eye/download.py [--out C:/Users/toont/tm-data/eye] [--train 10000] [--test 1000]

Google's numpy_bitmap files hold every drawing of a category (100-300 MB each). We only need
11,000 per category, so we read the .npy header and then fetch 11 evenly spaced chunks of
rows with HTTP range requests (about 9 MB per category). The rows are shuffled with a fixed
seed and split into train and held-out sets.

Data: Quick, Draw! by Google, Inc., under CC BY 4.0
(https://github.com/googlecreativelab/quickdraw-dataset).

Writes <out>/raw/<category>.npy (the fetched rows) and <out>/quickdraw24.npz with
x_train, y_train, x_test, y_test (uint8 28x28, ink = 255), the source row numbers, and the
class names.
"""
import argparse
import ast
import os
import struct
import urllib.parse
import urllib.request

import numpy as np

CLASSES = [
    'dragon', 'teapot', 'cloud', 'moon', 'castle',
    'cat', 'dog', 'fish', 'bird', 'rabbit', 'frog', 'owl', 'sheep',
    'tree', 'flower', 'house', 'sun', 'star', 'apple', 'rainbow', 'snowman', 'umbrella',
    'ambulance', 'police car',
]
BASE = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
ROW = 784


def fetch(url, start, end):
    req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req, timeout=120) as r:
        data = r.read()
    if len(data) != end - start + 1:
        raise IOError(f'{url}: asked for {end - start + 1} bytes, got {len(data)}')
    return data


def read_header(url):
    head = fetch(url, 0, 511)
    if head[:6] != b'\x93NUMPY':
        raise ValueError(f'{url}: not a .npy file')
    major = head[6]
    if major == 1:
        hlen, off = struct.unpack('<H', head[8:10])[0], 10
    else:
        hlen, off = struct.unpack('<I', head[8:12])[0], 12
    info = ast.literal_eval(head[off:off + hlen].decode('latin1'))
    if info['descr'] not in ('|u1', 'u1') or info['fortran_order'] or info['shape'][1] != ROW:
        raise ValueError(f'{url}: unexpected layout {info}')
    return off + hlen, info['shape'][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='C:/Users/toont/tm-data/eye')
    ap.add_argument('--train', type=int, default=10000)
    ap.add_argument('--test', type=int, default=1000)
    ap.add_argument('--chunks', type=int, default=11)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    raw_dir = os.path.join(a.out, 'raw')
    os.makedirs(raw_dir, exist_ok=True)
    need = a.train + a.test
    per = -(-need // a.chunks)
    rng = np.random.default_rng(a.seed)
    xs_tr, ys_tr, xs_te, ys_te, src_tr, src_te = [], [], [], [], [], []
    for ci, name in enumerate(CLASSES):
        path = os.path.join(raw_dir, name + '.npy')
        if os.path.exists(path):
            blob = np.load(path, allow_pickle=False)
            rows, idx = blob['x'], blob['row']
        else:
            url = BASE + urllib.parse.quote(name) + '.npy'
            data_off, n = read_header(url)
            starts = np.linspace(0, n - per, a.chunks).astype(int)
            parts, idx = [], []
            for s in starts:
                b = fetch(url, data_off + s * ROW, data_off + (s + per) * ROW - 1)
                parts.append(np.frombuffer(b, np.uint8).reshape(per, ROW))
                idx.append(np.arange(s, s + per))
            rows, idx = np.concatenate(parts), np.concatenate(idx)
            rows, keep = np.unique(rows, axis=0, return_index=True)  # drop overlap/duplicates
            idx = idx[keep]
            blob = np.zeros(len(rows), dtype=[('x', np.uint8, ROW), ('row', np.int64)])
            blob['x'], blob['row'] = rows, idx
            np.save(path, blob, allow_pickle=False)
            print(f'{name:12s} {n:7d} drawings in file, fetched {len(rows)}', flush=True)
        if len(rows) < need:
            raise ValueError(f'{name}: only {len(rows)} rows')
        order = rng.permutation(len(rows))[:need]
        tr, te = order[:a.train], order[a.train:]
        xs_tr.append(rows[tr]); ys_tr.append(np.full(len(tr), ci, np.uint8)); src_tr.append(idx[tr])
        xs_te.append(rows[te]); ys_te.append(np.full(len(te), ci, np.uint8)); src_te.append(idx[te])
    out = os.path.join(a.out, 'quickdraw24.npz')
    np.savez_compressed(
        out,
        x_train=np.concatenate(xs_tr).reshape(-1, 28, 28), y_train=np.concatenate(ys_tr),
        x_test=np.concatenate(xs_te).reshape(-1, 28, 28), y_test=np.concatenate(ys_te),
        src_train=np.concatenate(src_tr), src_test=np.concatenate(src_te),
        classes=np.array(CLASSES))
    print('wrote', out)


if __name__ == '__main__':
    main()
