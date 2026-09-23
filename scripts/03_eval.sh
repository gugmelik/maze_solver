#!/usr/bin/env bash
# Evaluate a checkpoint and write <checkpoint>/eval/report.html
# Usage: [BACKEND=qwen_edit] scripts/03_eval.sh [checkpoint_dir] [extra flags...]
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-flux_kontext}"
CKPT="${1:-outputs/$BACKEND/checkpoints/$(cat "outputs/$BACKEND/checkpoints/latest")}"
shift || true
python -m mazelora.evaluate \
  --backend "$BACKEND" --lora "$CKPT" \
  --data data/maze5 --cache cache/maze5 \
  --n 100 --steps 28 \
  --compare_to "outputs/${BACKEND}_base/metrics.json" "$@"
echo
echo "open: $CKPT/eval/report.html"
