#!/usr/bin/env bash
# Inference-time scaling: sample N candidates per maze, keep the one a
# rule-based verifier accepts, and plot solved-rate against the sample budget.
#
# Usage: [BACKEND=qwen_edit] scripts/05_best_of_n.sh [checkpoint_dir] [extra flags...]
#
# Cost is n_mazes x n_samples generations. The defaults (40 x 8 = 320) take
# roughly an hour on Qwen at 20 steps. Trim with --n_mazes / --n_samples, or
# --guidance 1.0 to drop CFG and halve the per-image cost.
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-qwen_edit}"
CKPT="${1:-outputs/$BACKEND/checkpoints/$(cat "outputs/$BACKEND/checkpoints/latest")}"
shift || true
python -m mazelora.best_of_n \
  --backend "$BACKEND" --lora "$CKPT" \
  --data data/maze5 --cache cache/maze5 \
  --n_mazes 40 --n_samples 8 --steps 20 "$@"
echo
echo "open: $CKPT/best_of_n/report.html"
