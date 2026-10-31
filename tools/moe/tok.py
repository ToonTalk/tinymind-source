"""The llama2 32,000-piece tokenizer, as Tiny Mind's engine sees it.

`RunCTokenizer` reads models/tokenizer.bin and mirrors run.c encode()/decode() (the same
algorithm as src/tokenizer.js), so prompts here tokenize exactly as in the browser.
`fast_encoder()` returns the Hugging Face fast tokenizer for the same vocabulary; it is
only used to tokenize the 2M training stories quickly, and prepare.py checks that it
agrees with RunCTokenizer.
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOKENIZER_BIN = REPO / "models" / "tokenizer.bin"
BOS, EOS = 1, 2
HF_TOKENIZER_JSON = (Path.home() / ".cache/huggingface/hub/models--Xenova--llama2.c-stories15M/"
                     "snapshots/17c2f1eabe1e163acc15ad35e225794e7b907682/tokenizer.json")


class RunCTokenizer:
    def __init__(self, path: Path = TOKENIZER_BIN, vocab_size: int = 32000):
        data = path.read_bytes()
        self.max_token_length = struct.unpack_from("<i", data, 0)[0]
        self.pieces: list[bytes] = []
        self.scores: list[float] = []
        p = 4
        for _ in range(vocab_size):
            score, n = struct.unpack_from("<fi", data, p)
            p += 8
            self.pieces.append(data[p:p + n])
            self.scores.append(score)
            p += n
        assert p == len(data), "tokenizer.bin not fully consumed"
        self.lookup: dict[bytes, int] = {}
        for i, s in enumerate(self.pieces):
            self.lookup.setdefault(s, i)

    def encode(self, text: str, bos: bool = True, eos: bool = False) -> list[int]:
        toks = [BOS] if bos else []
        if text:
            toks.append(self.lookup[b" "])
        utf8 = text.encode("utf-8")
        chunk = b""
        for i, b in enumerate(utf8):
            if (b & 0xC0) != 0x80:
                chunk = b""
            chunk += bytes([b])
            nxt = utf8[i + 1] if i + 1 < len(utf8) else None
            if nxt is not None and (nxt & 0xC0) == 0x80 and len(chunk) < 4:
                continue
            tid = self.lookup.get(chunk)
            if tid is not None:
                toks.append(tid)
            else:
                toks.extend(c + 3 for c in chunk)
            chunk = b""
        start = 1 if bos else 0
        body = toks[start:]
        while True:
            best, best_id, best_i = -1e10, -1, -1
            for i in range(len(body) - 1):
                tid = self.lookup.get(self.pieces[body[i]] + self.pieces[body[i + 1]])
                if tid is not None and self.scores[tid] > best:
                    best, best_id, best_i = self.scores[tid], tid, i
            if best_i < 0:
                break
            body[best_i] = best_id
            del body[best_i + 1]
        toks = toks[:start] + body
        if eos:
            toks.append(EOS)
        return toks

    def piece_bytes(self, prev: int, tok: int) -> bytes:
        s = self.pieces[tok]
        if prev == BOS and s.startswith(b" "):
            s = s[1:]
        m = re.fullmatch(rb"<0x([0-9A-Fa-f]{2})>", s)
        if m:
            s = bytes([int(m.group(1), 16)])
        return s

    def decode(self, toks) -> str:
        out, prev = b"", BOS
        for t in toks:
            t = int(t)
            if t in (BOS, EOS):
                prev = t
                continue
            out += self.piece_bytes(prev, t)
            prev = t
        return out.decode("utf-8", errors="replace")

    def piece(self, tok: int) -> str:
        """Display form of one piece (leading space kept, byte pieces decoded)."""
        return self.piece_bytes(-1, int(tok)).decode("utf-8", errors="replace")


def fast_encoder():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(HF_TOKENIZER_JSON))
