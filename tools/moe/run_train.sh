#!/usr/bin/env bash
# Train one mind, restarting from the last checkpoint if the process dies (e.g. an
# out-of-memory crash outside the step). Stops when final.json appears.
#   bash tools/moe/run_train.sh moe   [extra train.py args...]
#   bash tools/moe/run_train.sh dense [extra train.py args...]
set -u
arch=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
py=${MOE_PY:-/c/Users/toont/Documents/GitHub/jlens-tinystories/.venv/Scripts/python.exe}
out=${MOE_CKPT:-C:/Users/toont/tm-data/moe/ckpt}/$arch
mkdir -p "$out"
for try in $(seq 1 20); do
  [ -f "$out/final.json" ] && break
  echo "=== $arch attempt $try $(date)" >> "$out/train.out"
  (cd "$here" && PYTHONUNBUFFERED=1 "$py" train.py --arch "$arch" --out "$out" "$@") >> "$out/train.out" 2>&1
  [ -f "$out/final.json" ] || sleep 60
done
