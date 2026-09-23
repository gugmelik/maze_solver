#!/usr/bin/env bash
# The untuned reference point. Run this ONCE per backend, before or during
# training: every later report shows its numbers as a delta against this one.
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-flux_kontext}"
python -m mazelora.evaluate \
  --backend "$BACKEND" --lora none --out "outputs/${BACKEND}_base" \
  --data data/maze5 --cache cache/maze5 --n 100 --steps 28
