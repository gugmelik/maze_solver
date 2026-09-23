#!/usr/bin/env bash
# Interactive solver at http://127.0.0.1:7860
# Usage: [BACKEND=qwen_edit] scripts/04_app.sh [checkpoint_dir]
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-flux_kontext}"
CKPT="${1:-outputs/$BACKEND/checkpoints/$(cat "outputs/$BACKEND/checkpoints/latest")}"
python app.py --backend "$BACKEND" --lora "$CKPT" --data data/maze5 --cache cache/maze5
