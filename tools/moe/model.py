"""A llama2.c-shaped Transformer whose feed-forward blocks can be a mixture of experts.

Matches vendor/llama2.c/model.py and run.c conventions exactly, so a dense
checkpoint exports to a .bin that Tiny Mind's engine runs unchanged:
RMSNorm eps 1e-5, RoPE on adjacent pairs (theta 10000), SwiGLU w2(silu(w1 x) * w3 x),
no biases, token embedding tied to the classifier.

MoE block (per layer): a router Linear(dim, n_experts) scores every word; the top_k
experts by softmax probability run, and their outputs are mixed by the top-k
probabilities renormalized to sum to 1 (Mixtral style; renorm=True), or by the raw
probabilities (Switch style; renorm=False). Top-1 needs renorm=False: renormalizing a
single weight makes it exactly 1, so the router gets no gradient from the language-model
loss and learns only from the balancing losses. Dropless: every routed word is
processed, as it will be in the browser.

`off` / `silence` hooks (for characterization): `off` masks an expert's router logit
to -inf before the top-k choice, so its words go to the next-best expert (reroute);
`silence` keeps the choice but zeroes that expert's output (mute).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class Config:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    vocab_size: int = 32000
    max_seq_len: int = 256
    hidden_dim: int = 768        # dense FFN hidden (ignored when n_experts > 0)
    n_experts: int = 0           # 0 = dense
    top_k: int = 2
    expert_hidden: int = 192
    renorm: bool = True          # mix by top-k probabilities renormalized to 1 (False: raw, Switch style)
    norm_eps: float = 1e-5

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return xf.type_as(x) * self.weight


def precompute_freqs_cis(head_size: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_size, 2)[: head_size // 2].float() / head_size))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb(xq, xk, cos, sin):
    # adjacent pairs (x0,x1), (x2,x3), ... exactly as llama2.c model.py / run.c
    def rot(x):
        xr, xi = x.float().reshape(x.shape[:-1] + (-1, 2)).unbind(-1)
        c = cos.view(1, cos.shape[0], 1, cos.shape[1])
        s = sin.view(1, sin.shape[0], 1, sin.shape[1])
        out_r = xr * c - xi * s
        out_i = xr * s + xi * c
        return torch.stack([out_r, out_i], dim=-1).flatten(3).type_as(x)
    return rot(xq), rot(xk)


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_size = cfg.dim // cfg.n_heads
        self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.wo = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        q = self.wq(x).view(b, t, self.n_heads, self.head_size)
        k = self.wk(x).view(b, t, self.n_heads, self.head_size)
        v = self.wv(x).view(b, t, self.n_heads, self.head_size)
        q, k = apply_rotary_emb(q, k, cos, sin)
        q, k, v = (z.transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).contiguous().view(b, t, -1))


class DenseFFN(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden, bias=False)
        self.w2 = nn.Linear(hidden, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MoEFFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        e, h, d = cfg.n_experts, cfg.expert_hidden, cfg.dim
        self.n_experts, self.top_k, self.renorm = e, cfg.top_k, cfg.renorm
        self.router = nn.Linear(d, e, bias=False)
        # stacked expert weights in run.c (out, in) orientation: w1,w3 (E,h,d); w2 (E,d,h)
        self.w1 = nn.Parameter(torch.empty(e, h, d))
        self.w2 = nn.Parameter(torch.empty(e, d, h))
        self.w3 = nn.Parameter(torch.empty(e, h, d))
        # characterization hooks (sets of expert ids); empty during training
        self.off: set[int] = set()
        self.silence: set[int] = set()
        self.last = None   # filled with routing info when record=True

    def forward(self, x, record: bool = False):
        b, t, d = x.shape
        xf = x.reshape(-1, d)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.router(xf.float())                       # (N, E) fp32
        probs_all = logits.softmax(-1)                             # unmasked, for the aux loss
        sel_logits = logits
        if self.off:
            mask = torch.zeros(self.n_experts, dtype=torch.bool, device=x.device)
            mask[list(self.off)] = True
            sel_logits = logits.masked_fill(mask, float("-inf"))
        probs = sel_logits.softmax(-1)
        topw, topi = probs.topk(self.top_k, dim=-1)                # (N, k)
        if self.renorm:
            topw = topw / topw.sum(-1, keepdim=True)

        # Gate matrix: each word's top-k weights in its chosen experts' columns, 0 elsewhere.
        # All experts then run as one wide SwiGLU whose hidden units are scaled by their
        # expert's gate: sum_e g_e * w2_e(silu(w1_e x) * w3_e x). That is exactly the
        # dropless top-k mixture (unchosen experts get gate 0, so no output and no gradient),
        # computed with three big matmuls and no host syncs. The browser runs it sparsely.
        e_n, h = self.n_experts, self.w1.shape[1]
        gates = torch.zeros_like(probs).scatter(1, topi, topw)
        if self.silence:
            keep = torch.ones(e_n, device=x.device)
            keep[list(self.silence)] = 0
            gates = gates * keep
        a = F.silu(F.linear(xf, self.w1.reshape(e_n * h, d))) * F.linear(xf, self.w3.reshape(e_n * h, d))
        a = (a.view(-1, e_n, h) * gates.unsqueeze(-1).to(a.dtype)).view(-1, e_n * h)
        out = F.linear(a, self.w2.permute(1, 0, 2).reshape(d, e_n * h))

        # load-balancing loss (Switch/Mixtral): E * sum_e f_e * P_e, f over all k slots
        n = xf.shape[0]
        counts = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        counts.scatter_add_(0, topi.reshape(-1), torch.ones(n * self.top_k, device=x.device))
        frac = counts / (n * self.top_k)
        aux = self.n_experts * (frac * probs_all.mean(0)).sum()
        zloss = torch.logsumexp(logits, -1).pow(2).mean()
        self.stats = (counts.detach(), aux, zloss)
        if record:
            self.last = dict(logits=logits.view(b, t, -1), topi=topi.view(b, t, -1), topw=topw.view(b, t, -1))
        return out.view(b, t, d)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attention = Attention(cfg)
        self.attention_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn_norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.feed_forward = MoEFFN(cfg) if cfg.is_moe else DenseFFN(cfg.dim, cfg.hidden_dim)

    def forward(self, x, cos, sin, record=False):
        h = x + self.attention(self.attention_norm(x), cos, sin)
        ff = self.feed_forward
        y = ff(self.ffn_norm(h), record=record) if isinstance(ff, MoEFFN) else ff(self.ffn_norm(h))
        return h + y


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_embeddings = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.layers = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.output = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        self.output.weight = self.tok_embeddings.weight   # tied, as in stories15M
        cos, sin = precompute_freqs_cis(cfg.dim // cfg.n_heads, cfg.max_seq_len)
        self.register_buffer("freqs_cos", cos, persistent=False)
        self.register_buffer("freqs_sin", sin, persistent=False)
        self.apply(self._init)
        # llama2.c: residual projections get std 0.02/sqrt(2L)
        resid_std = 0.02 / math.sqrt(2 * cfg.n_layers)
        for name, p in self.named_parameters():
            if name.endswith("w3.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, 0.0, resid_std)
        for blk in self.layers:
            ff = blk.feed_forward
            if isinstance(ff, MoEFFN):
                nn.init.normal_(ff.w1, 0.0, 0.02)
                nn.init.normal_(ff.w2, 0.0, 0.02)
                nn.init.normal_(ff.w3, 0.0, resid_std)
                nn.init.normal_(ff.router.weight, 0.0, 0.02)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def moe_blocks(self):
        return [b.feed_forward for b in self.layers if isinstance(b.feed_forward, MoEFFN)]

    def forward(self, tokens, targets=None, record=False, reduction="mean"):
        t = tokens.shape[1]
        h = self.tok_embeddings(tokens)
        cos, sin = self.freqs_cos[:t], self.freqs_sin[:t]
        for blk in self.layers:
            h = blk(h, cos, sin, record=record)
        h = self.norm(h)
        logits = self.output(h)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                               ignore_index=-1, reduction=reduction)
        return logits, loss

    def aux_losses(self):
        blocks = self.moe_blocks()
        if not blocks:
            return None, None
        aux = sum(b.stats[1] for b in blocks) / len(blocks)
        z = sum(b.stats[2] for b in blocks) / len(blocks)
        return aux, z

    def param_counts(self):
        total = sum(p.numel() for p in self.parameters())
        c = self.cfg
        if c.is_moe:
            per_expert = 3 * c.dim * c.expert_hidden
            inactive = c.n_layers * (c.n_experts - c.top_k) * per_expert
        else:
            inactive = 0
        return total, total - inactive
