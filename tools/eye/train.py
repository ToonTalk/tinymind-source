"""train.py: the Tiny Eye, a tiny ConvNeXt-style doodle classifier (Phase 0 of plans/tiny-eye.md).

    python -I tools/eye/train.py --name eye80k --widths 24,48,64
    python -I tools/eye/train.py --name eye80k-sq8 --widths 24,48,8 --hidden 96,192,8

Layout: stem (2x2 patchify) -> 3 stages of [7x7 depthwise conv + LayerNorm + pointwise MLP
+ residual] -> average pool -> LayerNorm -> linear. Each later stage starts with a 2x2
stride-2 downsample, so the stages see 14x14, 7x7 and 4x4 grids.

The "neurons" are the MLP hidden channels, read before the GELU (the summed votes, which
can be negative, so every neuron has two poles). One number per drawing per neuron is the
mean over the grid positions.

Reads <data>/quickdraw24.npz from download.py. Writes <data>/runs/<name>/ckpt-<step>.pt
(about 20, log-spaced so early growth is visible) and meta.json.

The GPU is shared, so this process caps itself at 15% of its memory.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelNorm(nn.Module):
    """LayerNorm over the channels of an N,C,H,W tensor."""

    def __init__(self, c):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class Block(nn.Module):
    def __init__(self, c, hidden):
        super().__init__()
        self.dw = nn.Conv2d(c, c, 7, padding=3, groups=c)
        self.ln = nn.LayerNorm(c)
        self.fc1 = nn.Linear(c, hidden)
        self.fc2 = nn.Linear(hidden, c)
        self.gamma = nn.Parameter(torch.full((c,), 0.1))

    def forward(self, x, taps=None, mask=None):
        h = self.fc1(self.ln(self.dw(x).permute(0, 2, 3, 1)))  # N,H,W,hidden: the neurons
        if taps is not None:
            taps.append(h)
        a = F.gelu(h)
        if mask is not None:
            a = a * mask
        return x + (self.fc2(a) * self.gamma).permute(0, 3, 1, 2)


class TinyEye(nn.Module):
    def __init__(self, widths=(24, 48, 64), hidden=None, depths=(1, 1, 1), n_classes=24):
        super().__init__()
        hidden = hidden or [4 * w for w in widths]
        self.config = dict(widths=list(widths), hidden=list(hidden), depths=list(depths), n_classes=n_classes)
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 2, stride=2), ChannelNorm(widths[0]))
        self.downs = nn.ModuleList()
        self.stages = nn.ModuleList()
        size = 14
        for i, (w, hd, d) in enumerate(zip(widths, hidden, depths)):
            if i:
                pad = size % 2  # 14 -> 7, then 7 -> 4
                self.downs.append(nn.Sequential(ChannelNorm(widths[i - 1]),
                                                nn.Conv2d(widths[i - 1], w, 2, stride=2, padding=pad)))
                size = (size + 2 * pad - 2) // 2 + 1
            self.stages.append(nn.ModuleList([Block(w, hd) for _ in range(d)]))
        self.head_ln = nn.LayerNorm(widths[-1])
        self.head = nn.Linear(widths[-1], n_classes)

    def forward(self, x, taps=None, masks=None):
        """x: N,1,28,28 in [0,1]. taps: list that collects each block's neurons.
        masks: {block index: 0/1 tensor over that block's hidden channels} to switch neurons off."""
        x = self.stem(x)
        b = 0
        for i, stage in enumerate(self.stages):
            if i:
                x = self.downs[i - 1](x)
            for blk in stage:
                x = blk(x, taps, None if masks is None else masks.get(b))
                b += 1
        return self.head(self.head_ln(x.mean((2, 3))))


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def load_data(path, device):
    d = np.load(path, allow_pickle=False)
    t = lambda k: torch.from_numpy(d[k]).to(device)
    return t('x_train'), t('y_train').long(), t('x_test'), t('y_test').long(), [str(c) for c in d['classes']]


def augment(x, gen):
    """Small random rotation, scale and shift, done on the GPU."""
    n = x.shape[0]
    ang = (torch.rand(n, device=x.device, generator=gen) * 2 - 1) * math.radians(12)
    sc = 1 / (0.85 + torch.rand(n, device=x.device, generator=gen) * 0.25)
    tx, ty = ((torch.rand(2, n, device=x.device, generator=gen) * 2 - 1) * 0.12)
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos, -sin, tx], 1), torch.stack([sin, cos, ty], 1)], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode='zeros')


@torch.no_grad()
def evaluate(model, x, y, bs=2048):
    model.eval()
    correct = 0
    for i in range(0, len(x), bs):
        logits = model(x[i:i + bs].unsqueeze(1).float() / 255)
        correct += (logits.argmax(1) == y[i:i + bs]).sum().item()
    model.train()
    return correct / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--widths', default='24,48,64')
    ap.add_argument('--hidden', default=None, help='MLP hidden channels per stage (default 4x width)')
    ap.add_argument('--depths', default='1,1,1')
    ap.add_argument('--data', default='C:/Users/toont/tm-data/eye')
    ap.add_argument('--epochs', type=float, default=20)
    ap.add_argument('--batch', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-3)
    ap.add_argument('--wd', type=float, default=0.05)
    ap.add_argument('--smoothing', type=float, default=0.1)
    ap.add_argument('--ckpts', type=int, default=20)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--mem-fraction', type=float, default=0.15)
    a = ap.parse_args()

    dev = 'cuda'
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    torch.manual_seed(a.seed)
    ints = lambda s: [int(v) for v in s.split(',')]
    model = TinyEye(ints(a.widths), ints(a.hidden) if a.hidden else None, ints(a.depths)).to(dev)
    xtr, ytr, xte, yte, classes = load_data(os.path.join(a.data, 'quickdraw24.npz'), dev)
    out = os.path.join(a.data, 'runs', a.name)
    os.makedirs(out, exist_ok=True)

    steps_per_epoch = len(xtr) // a.batch
    total = int(a.epochs * steps_per_epoch)
    warm = min(500, total // 10)
    save_at = sorted({0, total} | {int(round(s)) for s in np.geomspace(20, total, a.ckpts - 1)})
    decay = [p for n, p in model.named_parameters() if p.ndim > 1]
    rest = [p for n, p in model.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{'params': decay, 'weight_decay': a.wd}, {'params': rest, 'weight_decay': 0}], lr=a.lr)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    print(f'{a.name}: {n_params(model)} parameters, {total} steps, checkpoints at {save_at}', flush=True)

    meta = dict(name=a.name, config=model.config, params=n_params(model), classes=classes, args=vars(a),
                steps=total, steps_per_epoch=steps_per_epoch, checkpoints=[])
    step, t0, perm, pos = 0, time.time(), None, len(xtr)
    while True:
        if step in save_at:
            acc = evaluate(model, xte, yte)
            torch.save({'config': model.config, 'state': model.state_dict(), 'step': step}, os.path.join(out, f'ckpt-{step:06d}.pt'))
            meta['checkpoints'].append(dict(step=step, epoch=round(step / steps_per_epoch, 3), test_acc=acc))
            print(f'  step {step:6d} epoch {step / steps_per_epoch:6.2f} held-out acc {acc:.4f} ({time.time() - t0:.0f}s)', flush=True)
        if step >= total:
            break
        if pos + a.batch > len(xtr):
            perm, pos = torch.randperm(len(xtr), device=dev, generator=gen), 0
        idx = perm[pos:pos + a.batch]
        pos += a.batch
        x = augment(xtr[idx].unsqueeze(1).float() / 255, gen)
        loss = F.cross_entropy(model(x), ytr[idx], label_smoothing=a.smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        step += 1
    meta['final_acc'] = meta['checkpoints'][-1]['test_acc']
    meta['train_seconds'] = round(time.time() - t0)
    with open(os.path.join(out, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print(f'{a.name}: final held-out acc {meta["final_acc"]:.4f}', flush=True)


if __name__ == '__main__':
    main()
