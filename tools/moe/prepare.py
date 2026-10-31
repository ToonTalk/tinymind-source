"""Tokenize TinyStories (Hugging Face parquet) into flat uint16 token files.

Each story is stripped and written as [BOS] + pieces, back to back, exactly like
llama2.c's pretokenize step. Writes, under --data:
  train.bin / val.bin      uint16 tokens
  train_starts.npy / val_starts.npy   int64 offset of each story's BOS
and checks the fast tokenizer against the run.c port on the first validation stories.

  python tools/moe/prepare.py --data C:/Users/toont/tm-data/moe/data
"""

from __future__ import annotations

import argparse
import glob
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from tok import BOS, RunCTokenizer, fast_encoder


def encode_split(enc, files, out_bin: Path, out_starts: Path, batch=4096):
    starts, n_tok = [], 0
    with open(out_bin, "wb") as f:
        for fn in files:
            texts = pq.read_table(fn, columns=["text"]).column("text").to_pylist()
            for i in range(0, len(texts), batch):
                chunk = [t.strip() for t in texts[i:i + batch]]
                for e in enc.encode_batch(chunk, add_special_tokens=False):
                    ids = np.asarray([BOS] + e.ids, dtype=np.uint16)
                    starts.append(n_tok)
                    n_tok += len(ids)
                    f.write(ids.tobytes())
            print(f"  {Path(fn).name}: {len(starts):,} stories, {n_tok:,} tokens so far", flush=True)
    np.save(out_starts, np.asarray(starts, dtype=np.int64))
    return len(starts), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="C:/Users/toont/tm-data/moe/data")
    ap.add_argument("--check", type=int, default=300, help="validation stories to cross-check")
    args = ap.parse_args()
    data = Path(args.data)
    hf = data / "hf" / "data"
    enc = fast_encoder()

    # cross-check: HF fast tokenizer vs the run.c port the browser uses
    rc = RunCTokenizer()
    val_files = sorted(glob.glob(str(hf / "validation-*.parquet")))
    texts = pq.read_table(val_files[0], columns=["text"]).column("text").to_pylist()[:args.check]
    bad = 0
    for t in texts:
        t = t.strip()
        a = [BOS] + enc.encode(t, add_special_tokens=False).ids
        b = rc.encode(t)
        bad += a != b
    print(f"tokenizer check: {bad}/{len(texts)} stories differ between HF fast and run.c encode")

    t0 = time.time()
    for split, pat in [("val", "validation-*.parquet"), ("train", "train-*.parquet")]:
        files = sorted(glob.glob(str(hf / pat)))
        n_s, n_t = encode_split(enc, files, data / f"{split}.bin", data / f"{split}_starts.npy")
        print(f"{split}: {n_s:,} stories, {n_t:,} tokens ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
