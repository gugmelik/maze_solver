#!/usr/bin/env bash
# The untuned reference point. Run this ONCE, before or during training:
# every later report shows its numbers as a delta against this one.
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mazelora.evaluate \
  --lora none --out outputs/base_model \
  --data data/maze5 --cache cache/maze5 --n 100 --steps 28 --guidance 2.5
