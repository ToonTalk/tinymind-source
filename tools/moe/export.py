"""Export a trained mind to a llama2.c .bin (plus an experts block for a MoE mind).

Dense mind: a plain llama2.c legacy (v0) checkpoint that src/bin-reader.js reads today.

MoE mind ("MoE1" layout; all little-endian, int32 / float32):

  1. llama2.c header, 7 x int32: dim, hidden_dim = 0, n_layers, n_heads, n_kv_heads,
     vocab_size (> 0: classifier tied to the embedding), seq_len.
  2. The usual v0 weights with hidden_dim = 0, so w1/w2/w3 take no space:
     token_embedding [vocab, dim], rms_att [L, dim], wq, wk, wv, wo [L, dim, dim],
     rms_ffn [L, dim], (w1, w2, w3 empty), rms_final [dim],
     freq_cis_real, freq_cis_imag [seq_len, head_size / 2] each (unused, as in run.c).
  3. Experts block, starting right after (2):
     magic b"MoE1", then 7 x int32: version = 1, n_experts, top_k, expert_hidden,
     flags (bit 0 = renormalize the top-k weights to sum to 1), 0, 0.
     router [L, n_experts, dim]            logits = router[l] @ rmsnorm(x)
     w1     [L, n_experts, expert_hidden, dim]
     w2     [L, n_experts, dim, expert_hidden]
     w3     [L, n_experts, expert_hidden, dim]
     expert e's output: w2[l, e] @ (silu(w1[l, e] @ x) * (w3[l, e] @ x)).
  The routing rule: p = softmax(logits); pick the top_k experts by p; weight each by
  p / sum of the chosen p (flags bit 0 set), or by p itself (bit 0 clear, used for top-1);
  the feed-forward output is the weighted sum.

Today's reader rejects a MoE file (its layout sum does not reach the end of the file),
so it can't be mistaken for a dense mind.

Also writes <name>.fixture.json: greedy text and the top logits at every step for a few
prompts, computed in float32 from the exported file, for Phase 1's parity test.

  python tools/moe/export.py C:/Users/toont/tm-data/moe/ckpt/moe/final.pt C:/Users/toont/tm-data/moe/out/stories-moe.bin
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np
import torch

from model import Config, MoEFFN, Transformer, precompute_freqs_cis
from tok import RunCTokenizer

MAGIC = b"MoE1"
FIXTURE_PROMPTS = ["Once upon a time", "Lily and Max went to the park. Max saw a big",
                   "The little dragon was sad because"]


def load_model(ckpt: str | Path, dev="cpu") -> Transformer:
    state = torch.load(ckpt, map_location=dev, weights_only=False)
    model = Transformer(Config(**state["config"])).to(dev)
    model.load_state_dict(state["model"])
    return model.eval()


def write_bin(model: Transformer, path: Path):
    c = model.cfg
    sd = {k: v.detach().float().cpu().numpy() for k, v in model.state_dict().items()}
    L = range(c.n_layers)
    hidden = 0 if c.is_moe else c.hidden_dim
    parts = [struct.pack("<7i", c.dim, hidden, c.n_layers, c.n_heads, c.n_heads, c.vocab_size, c.max_seq_len)]

    def put(arrs):
        for a in arrs:
            parts.append(np.ascontiguousarray(a, dtype="<f4").tobytes())

    put([sd["tok_embeddings.weight"]])
    put(sd[f"layers.{i}.attention_norm.weight"] for i in L)
    for w in ("wq", "wk", "wv", "wo"):
        put(sd[f"layers.{i}.attention.{w}.weight"] for i in L)
    put(sd[f"layers.{i}.ffn_norm.weight"] for i in L)
    if not c.is_moe:
        for w in ("w1", "w2", "w3"):
            put(sd[f"layers.{i}.feed_forward.{w}.weight"] for i in L)
    put([sd["norm.weight"]])
    cos, sin = precompute_freqs_cis(c.dim // c.n_heads, c.max_seq_len)
    put([cos.numpy(), sin.numpy()])
    if c.is_moe:
        parts.append(MAGIC + struct.pack("<7i", 1, c.n_experts, c.top_k, c.expert_hidden, int(c.renorm), 0, 0))
        put(sd[f"layers.{i}.feed_forward.router.weight"] for i in L)
        for w in ("w1", "w2", "w3"):
            put(sd[f"layers.{i}.feed_forward.{w}"] for i in L)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"".join(parts))


def read_bin(path: Path) -> Transformer:
    """Rebuild a Transformer from an exported .bin (the inverse of write_bin)."""
    buf = path.read_bytes()
    dim, hidden, n_layers, n_heads, n_kv, vocab, seq = struct.unpack_from("<7i", buf, 0)
    f = np.frombuffer(buf, dtype="<f4", offset=28)
    p = 0

    def take(*shape):
        nonlocal p
        n = int(np.prod(shape))
        a = f[p:p + n].reshape(shape)
        p += n
        return torch.from_numpy(a.copy())

    head = dim // n_heads
    emb = take(vocab, dim)
    rms_att = take(n_layers, dim)
    wq, wk, wv, wo = (take(n_layers, dim, dim) for _ in range(4))
    rms_ffn = take(n_layers, dim)
    dense = None
    if hidden:
        dense = (take(n_layers, hidden, dim), take(n_layers, dim, hidden), take(n_layers, hidden, dim))
    rms_final = take(dim)
    take(seq, head // 2)
    take(seq, head // 2)
    cfg = Config(dim=dim, n_layers=n_layers, n_heads=n_heads, vocab_size=abs(vocab), max_seq_len=seq,
                 hidden_dim=hidden or 768)
    moe = None
    off = 28 + p * 4
    if off < len(buf):
        assert buf[off:off + 4] == MAGIC, "unknown block after llama2.c weights"
        ver, ne, k, eh, flags, _, _ = struct.unpack_from("<7i", buf, off + 4)
        assert ver == 1
        cfg.n_experts, cfg.top_k, cfg.expert_hidden, cfg.renorm = ne, k, eh, bool(flags & 1)
        p += 8
        moe = (take(n_layers, ne, dim), take(n_layers, ne, eh, dim), take(n_layers, ne, dim, eh),
               take(n_layers, ne, eh, dim))
    assert 28 + p * 4 == len(buf), "file size does not match layout"
    model = Transformer(cfg)
    sd = {"tok_embeddings.weight": emb, "output.weight": emb, "norm.weight": rms_final}
    for i in range(n_layers):
        sd[f"layers.{i}.attention_norm.weight"] = rms_att[i]
        sd[f"layers.{i}.ffn_norm.weight"] = rms_ffn[i]
        for name, w in zip(("wq", "wk", "wv", "wo"), (wq, wk, wv, wo)):
            sd[f"layers.{i}.attention.{name}.weight"] = w[i]
        if dense:
            for name, w in zip(("w1", "w2", "w3"), dense):
                sd[f"layers.{i}.feed_forward.{name}.weight"] = w[i]
        else:
            sd[f"layers.{i}.feed_forward.router.weight"] = moe[0][i]
            for name, w in zip(("w1", "w2", "w3"), moe[1:]):
                sd[f"layers.{i}.feed_forward.{name}"] = w[i]
    model.load_state_dict(sd)
    return model.eval()


@torch.no_grad()
def greedy(model, toks, steps):
    toks = list(toks)
    out_logits = []
    for _ in range(steps):
        logits, _ = model(torch.tensor([toks[-model.cfg.max_seq_len:]]))
        last = logits[0, -1]
        out_logits.append(last)
        toks.append(int(last.argmax()))
    return toks, out_logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    model = load_model(args.ckpt)
    write_bin(model, out)
    back = read_bin(out)
    tok = RunCTokenizer()
    fixtures, max_diff = [], 0.0
    for prompt in FIXTURE_PROMPTS:
        ids = tok.encode(prompt)
        seq, logits = greedy(back, ids, args.steps)
        ref, _ = model(torch.tensor([seq[:-1]]))
        max_diff = max(max_diff, float((ref[0, len(ids) - 1:] - torch.stack(logits)).abs().max()))
        routes = None
        if back.cfg.is_moe:
            back(torch.tensor([seq]), record=True)
            routes = [[blk.feed_forward.last["topi"][0].tolist()] for blk in back.layers]
        fixtures.append(dict(prompt=prompt, prompt_tokens=ids, tokens=seq, text=tok.decode(seq),
                             top5=[[(int(i), round(float(v), 4)) for v, i in zip(*l.topk(5))] for l in logits],
                             routes=routes))
    meta = dict(file=out.name, bytes=out.stat().st_size, config=back.cfg.__dict__,
                note="greedy, float32, from the exported .bin; routes[layer][0][pos] = chosen experts (best first)",
                max_abs_logit_diff_vs_checkpoint=max_diff, fixtures=fixtures)
    out.with_suffix(".fixture.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out} ({out.stat().st_size:,} bytes); re-read max |logit diff| {max_diff:.2e}")
    for fx in fixtures:
        print(" ", fx["text"][:160].replace("\n", " "))


if __name__ == "__main__":
    main()
