#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ldm

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

CKPT_DIR="${CKPT_DIR:-saved/endocot_klein4b_lora_final}"
TEST_ROOT="${TEST_ROOT:-data/EndoCoT_Maze_test}"
GEN_ROOT="${GEN_ROOT:-${CKPT_DIR}/maze_test}"

python - << PY
from pathlib import Path
import subprocess
import sys

test_root = Path("${TEST_ROOT}")
needed = [test_root / f"{level}_test" for level in (8, 16, 32)]
if all((path / "path.json").is_file() for path in needed):
    print(f"Maze tests already present under {test_root}", flush=True)
    raise SystemExit(0)

test_root.mkdir(parents=True, exist_ok=True)
clone_dir = Path("third_party/EndoCoT")
if not (clone_dir / ".git").exists():
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(
        [
            "git",
            "clone",
            "--filter=blob:none",
            "--sparse",
            "https://github.com/InternLM/EndoCoT.git",
            str(clone_dir),
        ]
    )
subprocess.check_call(
    ["git", "-C", str(clone_dir), "sparse-checkout", "set", "Maze/8_test", "Maze/16_test", "Maze/32_test"]
)
for level in (8, 16, 32):
    src = clone_dir / "Maze" / f"{level}_test"
    dst = test_root / f"{level}_test"
    if dst.exists():
        continue
    dst.symlink_to(src.resolve(), target_is_directory=True)
    print(f"Linked {dst} -> {src}", flush=True)
PY

EPOCHS_JSON=$(python - << PY
from pathlib import Path
import json
import re

ckpt_dir = Path("${CKPT_DIR}") / "checkpoints"
epochs = []
for path in sorted(ckpt_dir.glob("checkpoint-epoch*.pth")):
    match = re.search(r"epoch(\d+)", path.name)
    if match:
        epochs.append(int(match.group(1)))
epochs = sorted(set(epochs))
if not epochs:
    raise SystemExit(f"No checkpoints found under {ckpt_dir}")
print(json.dumps(epochs))
PY
)
echo "Evaluating epochs: ${EPOCHS_JSON}"
echo "Checkpoints: ${CKPT_DIR}"
echo "Test root: ${TEST_ROOT}"
echo "Outputs: ${GEN_ROOT}"

"$CONDA_PREFIX/bin/python" -m accelerate.commands.launch \
    --config_file=src/configs/ddp/ddp.yaml \
    --mixed_precision=bf16 \
    --num_processes=1 \
    --num_machines=1 \
    inference.py --config-name=endocot_klein4b_lora_final_maze_test \
    "inferencer.ckpt_dir=${CKPT_DIR}" \
    "inferencer.exp_save_dir=${GEN_ROOT}" \
    "inferencer.epochs_to_infer=${EPOCHS_JSON}" \
    "datasets.val.maze_8_test.data_root=${TEST_ROOT}" \
    "datasets.val.maze_16_test.data_root=${TEST_ROOT}" \
    "datasets.val.maze_32_test.data_root=${TEST_ROOT}"

"$CONDA_PREFIX/bin/python" tools/endocot_maze_eval/score_checkpoints.py \
    --test-root "${TEST_ROOT}" \
    --gen-root "${GEN_ROOT}" \
    --levels 8 16 32

echo "Done. Summary: ${GEN_ROOT}/maze_test_summary.json"
