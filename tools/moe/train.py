"""Train a TinyStories mind: a mixture of experts, or its dense twin.

Shape: llama2.c stories15M (dim 288, 6 layers, 6 heads, the 32,000-piece llama2
tokenizer, 256-word context). MoE: every feed-forward block is 8 experts of hidden 192
with a top-2 router, a Switch-style load-balancing loss and a small router z-loss.
Dense twin: one feed-forward of hidden 384 per layer, the same active size as top-2 x 192.

Both runs see the same data in the same order: one shuffled pass over 256-word windows
of TinyStories train (prepare.py), bf16 autocast, AdamW, cosine schedule.

Resumable: a checkpoint (model + optimizer + step) is written atomically every
--ckpt-every steps and on exit; rerunning the same command continues from it. A CUDA
out-of-memory error (another process on the GPU) retries the step after a pause.

  python tools/moe/train.py --arch moe   --out C:/Users/toont/tm-data/moe/ckpt/moe
  python tools/moe/train.py --arch dense --out C:/Users/toont/tm-data/moe/ckpt/dense
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from model import Config, Transformer

SEQ = 256


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["moe", "dense"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="C:/Users/toont/tm-data/moe/data")
    ap.add_argument("--n-experts", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=2)
    ap.add_argument("--renorm", choices=["auto", "yes", "no"], default="auto",
                    help="renormalize the top-k weights (auto: yes for top-2 and up, no for top-1, Switch style)")
    ap.add_argument("--expert-hidden", type=int, default=192)
    ap.add_argument("--hidden", type=int, default=384, help="dense FFN hidden")
    ap.add_argument("--micro", type=int, default=32, help="sequences per micro-batch")
    ap.add_argument("--accum", type=int, default=8, help="micro-batches per optimizer step")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--aux", type=float, default=0.01, help="load-balancing loss weight")
    ap.add_argument("--zloss", type=float, default=0.001, help="router z-loss weight")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--mem-frac", type=float, default=0.55)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--ckpt-every", type=int, default=100)
    ap.add_argument("--max-steps", type=int, default=0, help="stop early (testing)")
    return ap.parse_args()


def make_config(a) -> Config:
    if a.arch == "moe":
        renorm = a.top_k > 1 if a.renorm == "auto" else a.renorm == "yes"
        return Config(n_experts=a.n_experts, top_k=a.top_k, expert_hidden=a.expert_hidden, renorm=renorm,
                      max_seq_len=SEQ)
    return Config(hidden_dim=a.hidden, max_seq_len=SEQ)


def lr_at(step, total, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    r = min(1.0, (step - a.warmup) / max(1, total - a.warmup))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * r))


def batch_from(tokens: np.ndarray, chunk_ids: np.ndarray, dev):
    offs = chunk_ids[:, None] * SEQ + np.arange(SEQ + 1)[None, :]
    b = torch.from_numpy(tokens[offs].astype(np.int64)).pin_memory().to(dev, non_blocking=True)
    return b[:, :-1], b[:, 1:]


@torch.no_grad()
def evaluate(model, tokens, n_windows, micro, dev):
    """Mean loss over the first n_windows non-overlapping windows (all if 0). The count must not depend
    on micro, or runs with different micro-batches are scored on different text."""
    model.eval()
    n_chunks = (len(tokens) - 1) // SEQ
    ids = np.arange(n_chunks)
    if n_windows:
        ids = ids[:n_windows]
    tot, n = 0.0, 0
    for i in range(0, len(ids), micro):
        x, y = batch_from(tokens, ids[i:i + micro], dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y, reduction="sum")
        tot += loss.item()
        n += y.numel()
    model.train()
    return tot / n


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    a = get_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    # one trainer per folder: hold an OS lock on train.lock until this process exits
    lock = open(out / "train.lock", "a+")
    try:
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except ImportError:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"another trainer is already using {out}")
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data = Path(a.data)
    train_tok = np.fromfile(data / "train.bin", dtype=np.uint16)
    val_tok = np.fromfile(data / "val.bin", dtype=np.uint16)
    n_chunks = (len(train_tok) - 1) // SEQ
    perm = np.random.default_rng(a.seed).permutation(n_chunks)
    per_step = a.micro * a.accum
    total = int(n_chunks * a.epochs) // per_step
    if a.max_steps:
        total = min(total, a.max_steps)

    cfg = make_config(a)
    torch.manual_seed(a.seed)
    model = Transformer(cfg).to(dev)
    n_total, n_active = model.param_counts()
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)

    step = 0
    ck = out / "last.pt"
    if ck.exists():
        state = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        step = state["step"]
        print(f"resumed from {ck} at step {step}", flush=True)
    else:
        (out / "run.json").write_text(json.dumps(dict(args=vars(a), config=asdict(cfg), n_params=n_total,
                                                      n_active=n_active, n_chunks=n_chunks, total_steps=total,
                                                      tokens_per_step=per_step * SEQ), indent=2))
    print(f"{a.arch}: {n_total/1e6:.2f}M params, {n_active/1e6:.2f}M active per word; "
          f"{total} steps x {per_step * SEQ:,} tokens = {total * per_step * SEQ / 1e6:.0f}M tokens", flush=True)

    log = open(out / "log.jsonl", "a", encoding="utf-8")
    moe = model.moe_blocks()
    usage = torch.zeros(len(moe), cfg.n_experts or 1, device=dev)
    t_log, tok_log, oom_retries = time.time(), 0, 0
    model.train()

    def checkpoint():
        save_atomic(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=step,
                         config=asdict(cfg), args=vars(a)), ck)

    while step < total:
        lr = lr_at(step, total, a)
        for g in opt.param_groups:
            g["lr"] = lr
        try:
            # no host syncs inside the step: under GPU time-slicing each one waits out
            # the other processes' slices
            loss_acc = torch.zeros((), device=dev)
            aux_acc = torch.zeros((), device=dev)
            z_acc = torch.zeros((), device=dev)
            for m in range(a.accum):
                i0 = (step * a.accum + m) * a.micro
                x, y = batch_from(train_tok, perm[i0:i0 + a.micro], dev)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    _, loss = model(x, y)
                    aux, z = model.aux_losses()
                    total_loss = loss
                    if aux is not None:
                        total_loss = loss + a.aux * aux + a.zloss * z
                (total_loss / a.accum).backward()
                loss_acc += loss.detach() / a.accum
                if aux is not None:
                    aux_acc += aux.detach() / a.accum
                    z_acc += z.detach() / a.accum
                    for li, blk in enumerate(moe):
                        usage[li] += blk.stats[0]
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        except torch.OutOfMemoryError as err:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            oom_retries += 1
            print(f"OOM at step {step} (retry {oom_retries}); waiting 30s: {str(err)[:300]}", flush=True)
            time.sleep(30)
            continue
        step += 1
        tok_log += per_step * SEQ

        if step % a.log_every == 0 or step == total:
            dt = time.time() - t_log
            loss_acc, aux_acc, z_acc, gnorm = (float(v) for v in (loss_acc, aux_acc, z_acc, gnorm))
            rec = dict(step=step, loss=round(loss_acc, 4), lr=lr, gnorm=round(gnorm, 3),
                       tok_per_s=round(tok_log / dt), mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2))
            if moe:
                frac = (usage / usage.sum(1, keepdim=True)).cpu().numpy()
                rec.update(aux=round(aux_acc, 4), z=round(z_acc, 3),
                           usage=[[round(float(v), 3) for v in row] for row in frac],
                           usage_max=round(float(frac.max()), 3), usage_min=round(float(frac.min()), 3))
                if step > a.warmup and (frac.max() > 0.35 or frac.min() < 0.02):
                    print(f"WARNING step {step}: uneven expert usage (min {frac.min():.3f}, max {frac.max():.3f})",
                          flush=True)
                usage.zero_()
            print(json.dumps({k: v for k, v in rec.items() if k != "usage"}), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()
            t_log, tok_log = time.time(), 0

        if step % a.eval_every == 0 or step == total:
            vl = evaluate(model, val_tok, 640, a.micro, dev)
            rec = dict(step=step, val_loss=round(vl, 4))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + "\n")
            log.flush()

        if step % a.ckpt_every == 0 or step == total:
            checkpoint()

    torch.save(dict(model=model.state_dict(), config=asdict(cfg), args=vars(a), step=step), out / "final.pt")
    vl = evaluate(model, val_tok, 0, a.micro, dev)
    res = dict(arch=a.arch, steps=step, tokens=step * per_step * SEQ, val_loss_full=vl,
               n_params=n_total, n_active=n_active, oom_retries=oom_retries)
    (out / "final.json").write_text(json.dumps(res, indent=2))
    print("FINAL", json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
