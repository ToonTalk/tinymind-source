#!/bin/sh
# make-fixtures.sh — compile fixture-gen.c (no OpenMP: sequential matmul, deterministic)
# and emit golden greedy token sequences for both models. Run from the project root
# (works in WSL or any POSIX shell with gcc).
set -e
cd "$(dirname "$0")/.."

gcc -O2 -o tools/fixture-gen tools/fixture-gen.c -lm
echo "compiled tools/fixture-gen"

gen() { # gen <model> <tokenizer> <prompt> <steps> <outfile>
  ./tools/fixture-gen "models/$1" "models/$2" "$3" "$4" > "tests/fixtures/$5"
  echo "wrote tests/fixtures/$5 ($(wc -l < "tests/fixtures/$5") tokens)"
}

gen stories15M.bin  tokenizer.bin "Once upon a time"                    256 15M-p1.tokens.txt
gen stories15M.bin  tokenizer.bin "One day, Lily met a small dragon"    256 15M-p2.tokens.txt
gen stories260K.bin tok512.bin    "Once upon a time"                    300 260K-p1.tokens.txt
gen stories260K.bin tok512.bin    "One day, Lily met a small dragon"    300 260K-p2.tokens.txt

cat > tests/fixtures/manifest.json <<'EOF'
{
  "note": "golden greedy (temp 0) token ids from llama2.c run.c via tools/fixture-gen.c; fixed length, no BOS stop",
  "fixtures": [
    { "file": "15M-p1.tokens.txt",  "model": "stories15M.bin",  "tokenizer": "tokenizer.bin", "prompt": "Once upon a time",                 "steps": 256 },
    { "file": "15M-p2.tokens.txt",  "model": "stories15M.bin",  "tokenizer": "tokenizer.bin", "prompt": "One day, Lily met a small dragon", "steps": 256 },
    { "file": "260K-p1.tokens.txt", "model": "stories260K.bin", "tokenizer": "tok512.bin",    "prompt": "Once upon a time",                 "steps": 300 },
    { "file": "260K-p2.tokens.txt", "model": "stories260K.bin", "tokenizer": "tok512.bin",    "prompt": "One day, Lily met a small dragon", "steps": 300 }
  ]
}
EOF
echo "wrote tests/fixtures/manifest.json"
