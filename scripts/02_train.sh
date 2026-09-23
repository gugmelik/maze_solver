#!/usr/bin/env bash
# LoRA training.  BACKEND=flux_kontext (default) or BACKEND=qwen_edit
# Resume with: scripts/02_train.sh --resume outputs/<backend>/checkpoints/step-003000
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-flux_kontext}"
python -m mazelora.train --config "configs/${BACKEND}.yaml" "$@"
