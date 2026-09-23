#!/usr/bin/env bash
# Generate the maze dataset (~2 min, ~200 MB on disk).
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mazelora.gen_dataset \
  --size 5 --min_len 6 \
  --train 20000 --eval 1000 \
  --out data/maze5 --workers 16 --seed 1234
