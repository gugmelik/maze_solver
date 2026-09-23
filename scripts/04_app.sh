#!/usr/bin/env bash
# Interactive solver at http://127.0.0.1:7860
set -euo pipefail
cd "$(dirname "$0")/.."
CKPT="${1:-outputs/baseline/checkpoints/$(cat outputs/baseline/checkpoints/latest)}"
python app.py --lora "$CKPT" --data data/maze5 --cache cache/maze5
