"""eyelib.py: shared helpers for the Tiny Eye Phase 0 tools (loading, neuron readouts, image grids)."""
import glob
import json
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import TinyEye  # noqa: E402

DATA = 'C:/Users/toont/tm-data/eye'


def load_data(data=DATA):
    d = np.load(os.path.join(data, 'quickdraw24.npz'), allow_pickle=False)
    return {k: d[k] for k in d.files}


def halves(y):
    """Split held-out indices into A (examples for labels) and B (examples for scoring):
    alternate drawings within each class, so both halves have every class equally."""
    a, b = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        a.append(idx[0::2]); b.append(idx[1::2])
    return np.sort(np.concatenate(a)), np.sort(np.concatenate(b))


def run_dir(name, data=DATA):
    return os.path.join(data, 'runs', name)


def run_meta(name, data=DATA):
    with open(os.path.join(run_dir(name, data), 'meta.json')) as f:
        return json.load(f)


def checkpoints(name, data=DATA):
    return sorted(glob.glob(os.path.join(run_dir(name, data), 'ckpt-*.pt')))


def load_model(path, device='cuda'):
    ck = torch.load(path, map_location=device, weights_only=True)
    m = TinyEye(**ck['config']).to(device)
    m.load_state_dict(ck['state'])
    m.eval()
    return m, ck['step']


@torch.no_grad()
def readouts(model, x, bs=2048, device='cuda', reduce='mean'):
    """Per-drawing neuron values for every block: list of (n_drawings, hidden) arrays,
    plus logits. reduce='mean' averages each neuron over the grid; 'max' takes the peak."""
    outs, logits = None, []
    for i in range(0, len(x), bs):
        xb = torch.from_numpy(x[i:i + bs]).to(device).unsqueeze(1).float() / 255
        taps = []
        logits.append(model(xb, taps).cpu().numpy())
        red = [(t.mean((1, 2)) if reduce == 'mean' else t.amax((1, 2))).cpu().numpy() for t in taps]
        outs = [[r] for r in red] if outs is None else [o + [r] for o, r in zip(outs, red)]
    return [np.concatenate(o) for o in outs], np.concatenate(logits)


def save_png(im, path):
    """Save a figure as a 32-color palette PNG: the doodles look the same at about 40% of the size."""
    im.convert('RGB').convert('P', palette=Image.ADAPTIVE, colors=32).save(path, optimize=True)


def font(size):
    for name in ('arial.ttf', 'C:/Windows/Fonts/arial.ttf', 'DejaVuSans.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def doodle(img, scale=3):
    """28x28 uint8 (ink = 255) -> dark ink on white, enlarged."""
    return Image.fromarray(255 - img).resize((28 * scale, 28 * scale), Image.BILINEAR)


def panel(imgs, cols=4, scale=3, gap=4, numbers=None, captions=None):
    cell = 28 * scale
    top = 18 if numbers is not None else 0
    bot = 16 if captions is not None else 0
    rows = -(-len(imgs) // cols)
    W, H = cols * cell + (cols + 1) * gap, rows * (cell + top + bot) + (rows + 1) * gap
    im = Image.new('RGB', (W, H), (205, 205, 205))
    dr = ImageDraw.Draw(im)
    f, fs = font(15), font(11)
    for k, img in enumerate(imgs):
        r, c = divmod(k, cols)
        x0, y0 = gap + c * (cell + gap), gap + r * (cell + top + bot + gap)
        if numbers is not None:
            dr.rectangle([x0, y0, x0 + cell - 1, y0 + top - 1], fill=(255, 255, 255))
            dr.text((x0 + 3, y0 + 1), str(numbers[k]), fill=(200, 0, 0), font=f)
        im.paste(doodle(img, scale), (x0, y0 + top))
        if captions is not None:
            dr.rectangle([x0, y0 + top + cell, x0 + cell - 1, y0 + top + cell + bot - 1], fill=(245, 245, 245))
            dr.text((x0 + 2, y0 + top + cell + 1), captions[k][:14], fill=(90, 90, 90), font=fs)
    return im


def titled(parts, titles, gap=20, head=30, bg=(255, 255, 255)):
    """Place panels side by side, each with a title above it."""
    W = sum(p.width for p in parts) + gap * (len(parts) - 1)
    H = max(p.height for p in parts) + head
    im = Image.new('RGB', (W, H), bg)
    dr = ImageDraw.Draw(im)
    f = font(18)
    x = 0
    for p, t in zip(parts, titles):
        dr.text((x + 4, 5), t, fill=(0, 0, 0), font=f)
        im.paste(p, (x, head))
        x += p.width + gap
    return im


def stacked(rows, titles, head_w=110, gap=6, bg=(255, 255, 255)):
    """Place panels one above another, each with a title to its left."""
    W = head_w + max(p.width for p in rows)
    H = sum(p.height for p in rows) + gap * (len(rows) - 1)
    im = Image.new('RGB', (W, H), bg)
    dr = ImageDraw.Draw(im)
    f = font(16)
    y = 0
    for p, t in zip(rows, titles):
        dr.multiline_text((4, y + p.height // 2 - 18), t, fill=(0, 0, 0), font=f)
        im.paste(p, (head_w, y))
        y += p.height + gap
    return im
