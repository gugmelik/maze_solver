#!/usr/bin/env bash
# Cache the prompt embedding and all VAE latents (~15 min, ~5.5 GB).
# First run downloads FLUX.1-Kontext-dev (~34 GB) -- accept the licence and log
# in first:  hf auth login
set -euo pipefail
cd "$(dirname "$0")/.."
python -m mazelora.precompute \
  --data data/maze5 --cache cache/maze5 \
  --quantization none --batch_size 8
