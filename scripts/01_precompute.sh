#!/usr/bin/env bash
# Cache VAE latents (and the prompt embedding, where the model allows it).
# First run downloads the base model -- accept its licence and log in first:
#   hf auth login
#
#   BACKEND=flux_kontext  ~34 GB download, ~5.5 GB of latents, ~15 min
#   BACKEND=qwen_edit     ~54 GB download, ~5.5 GB of latents, ~20 min
set -euo pipefail
cd "$(dirname "$0")/.."
BACKEND="${BACKEND:-flux_kontext}"
python -m mazelora.precompute \
  --backend "$BACKEND" \
  --data data/maze5 --cache cache/maze5 \
  --quantization none --batch_size 8
