#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ldm

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

TEST_ROOT="${TEST_ROOT:-data/EndoCoT_Maze_test}"

TASKS_JSON=$(TEST_ROOT="${TEST_ROOT}" python - << 'PY'
import json
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

root = Path("data/EndoCoT")
root.mkdir(parents=True, exist_ok=True)

# Final-only Maze pairs: start image -> solution + prompt.
tasks = [
    ("Maze", "Maze/metadata_edit_final.csv", "Maze/Maze.tar", 23.0),
]
model_reserve_gb = 18.0

def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)

def free_gb() -> float:
    return shutil.disk_usage(root).free / 1e9

ready = []
for task, metadata_name, archive_name, archive_gb in tasks:
    image_marker = root / task / ".extracted"
    metadata_path = root / metadata_name
    archive_path = root / archive_name
    if not metadata_path.is_file():
        log(f"downloading {metadata_name}")
        hf_hub_download(
            repo_id="internlm/EndoCoT-Data",
            filename=metadata_name,
            repo_type="dataset",
            local_dir=str(root),
        )
    if image_marker.exists():
        ready.append(task)
        log(f"{task} already extracted, free_gb={free_gb():.1f}")
        continue
    if not archive_path.is_file() and free_gb() < archive_gb + model_reserve_gb:
        log(f"skip {task}: {free_gb():.1f} GB free")
        continue
    if not archive_path.is_file():
        log(f"downloading {archive_name}")
        hf_hub_download(
            repo_id="internlm/EndoCoT-Data",
            filename=archive_name,
            repo_type="dataset",
            local_dir=str(root),
        )
    log(f"extracting {archive_path}")
    with tarfile.open(archive_path) as archive:
        archive.extractall(root)
    archive_path.unlink()
    image_marker.write_text("ok\n")
    ready.append(task)
    log(f"{task} ready, free_gb={free_gb():.1f}")

if not ready:
    raise SystemExit("No EndoCoT tasks were extracted.")

# Official EndoCoT Maze test suites used for every-epoch validation.
test_root = Path(os.environ["TEST_ROOT"])
needed = [test_root / f"{level}_test" for level in (8, 16, 32)]
if not all((path / "path.json").is_file() for path in needed):
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
        [
            "git",
            "-C",
            str(clone_dir),
            "sparse-checkout",
            "set",
            "Maze/8_test",
            "Maze/16_test",
            "Maze/32_test",
        ]
    )
    for level in (8, 16, 32):
        src = clone_dir / "Maze" / f"{level}_test"
        dst = test_root / f"{level}_test"
        if dst.exists() or dst.is_symlink():
            continue
        dst.symlink_to(src.resolve(), target_is_directory=True)
        log(f"Linked {dst} -> {src}")
else:
    log(f"Maze tests ready under {test_root}")

print(json.dumps(ready))
PY
)
echo "Training tasks: ${TASKS_JSON}"
echo "Maze test root: ${TEST_ROOT}"
echo "Base model: black-forest-labs/FLUX.2-klein-base-4B (50-step validation)"

"$CONDA_PREFIX/bin/python" -m accelerate.commands.launch \
    --config_file=src/configs/ddp/ddp.yaml \
    --mixed_precision=bf16 \
    --num_processes=1 \
    --num_machines=1 \
    train.py --config-name=endocot_klein4b_base_lora \
    writer=wandb \
    writer.project_name=endocot \
    writer.run_name=endocot_klein4b_base_lora \
    writer.mode=online \
    dataloaders.train.batch_size=1 \
    dataloaders.train.num_workers=2 \
    "datasets.train.endocot_final.tasks=${TASKS_JSON}" \
    "datasets.val.maze_8_test.data_root=${TEST_ROOT}" \
    "datasets.val.maze_16_test.data_root=${TEST_ROOT}" \
    "datasets.val.maze_32_test.data_root=${TEST_ROOT}"
