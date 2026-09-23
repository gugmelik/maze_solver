#!/usr/bin/env bash
# LoRA training. ~1.2 s/step at batch 1 -> roughly 8 h for 6000 optimiser steps.
# Resume with: scripts/02_train.sh --resume outputs/baseline/checkpoints/step-003000
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mazelora.train --config configs/baseline.yaml "$@"
