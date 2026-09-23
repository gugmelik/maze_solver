#!/usr/bin/env bash
# Evaluate a checkpoint and write outputs/<ckpt>/eval/report.html
# Usage: scripts/03_eval.sh [checkpoint_dir] [extra flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/baseline/checkpoints/$(cat outputs/baseline/checkpoints/latest)}"
shift || true
python -m mazelora.evaluate \
  --lora "$CKPT" --data data/maze5 --cache cache/maze5 \
  --n 100 --steps 28 --guidance 2.5 \
  --compare_to outputs/base_model/metrics.json "$@"
echo
echo "open: $CKPT/eval/report.html"
