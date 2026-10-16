# Maze solver

Fine-tune FLUX.2 [klein] 4B on EndoCoT Maze so the model draws the solution path from a start image. Training is image editing: start maze plus a text prompt in, solution image out. A sample counts as correct when the generated red path overlaps the ground-truth path with IoU of at least 0.5.

## Requirements

- Linux with an NVIDIA GPU
- Python 3.11
- Conda, or any other environment tool. The launch scripts activate a conda environment named `ldm`
- About 45 GB free on the disk that holds this repo. The Maze archive is about 23 GB, and the script will not download it unless at least 41 GB is free
- Extra space in the Hugging Face cache for the bf16 4B weights
- A Hugging Face account that has accepted the license for the Black Forest Labs Klein checkpoint you train
- A Weights & Biases account if you keep the default online logging

## Setup

Dependencies are listed in `requirements.txt`. Create the environment the launch scripts expect and install that file:

```bash
conda create -n ldm python=3.11 -y
conda activate ldm
python -m pip install -U pip
pip install -r requirements.txt
```

`scripts/*.sh` runs `conda activate ldm`, so use that environment name when you launch training from those scripts. A venv uses the same file:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

`pip install -r requirements.txt` installs PyTorch from PyPI. Confirm the build sees the GPU:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

If that prints `False`, install a CUDA build that matches your driver from [pytorch.org](https://pytorch.org/get-started/locally/), then run `pip install -r requirements.txt` again.

Log in before the first training run. The trainer downloads `black-forest-labs/FLUX.2-klein-4B` or `black-forest-labs/FLUX.2-klein-base-4B` in bf16 with no quantization. Weights land in `$HF_HOME` (`~/.cache/huggingface` if unset).

```bash
huggingface-cli login
wandb login
```

## Data

Each training script downloads and extracts what it needs:

- `data/EndoCoT/Maze/` from `internlm/EndoCoT-Data` (`Maze/Maze.tar` plus the Maze metadata CSV). The tar is deleted after extraction
- `data/EndoCoT_Maze_test/{8,16,32}_test/` from a sparse checkout of [InternLM/EndoCoT](https://github.com/InternLM/EndoCoT) under `third_party/EndoCoT`

The training script validates on the official 8, 16, and 32 Maze tests.

## Train

From the repo root, with the `ldm` environment available to `conda activate`:

```bash
bash scripts/endocot_klein4b_base_lora.sh
```

That run trains a rank-16 LoRA on undistilled `FLUX.2-klein-base-4B` for 150 epochs of 1000 steps (150000 optimizer steps, batch size 1). It uses start-image to final-solution pairs from `metadata_edit_final.csv`. After every epoch it validates on the official 8, 16, and 32 Maze tests with 50 denoising steps and guidance scale 4.0. Checkpoints are written every epoch.

Logs go to the Weights & Biases project `endocot`. To log locally, add `writer.mode=offline` to the `train.py` command inside the script.

Single-GPU launch is already set (`--num_processes=1`, bf16, batch size 1, gradient checkpointing). VAE and the text encoder stay frozen. LoRA learning rate is `1e-4`.

## Outputs

A run named `endocot_klein4b_base_lora` writes:

```text
saved/endocot_klein4b_base_lora/
  checkpoints/checkpoint-epoch{N}.pth
  metrics/validation.json
```

`validation.json` stores per-step accuracy. Maze accuracy is the fraction of images whose red path has IoU ≥ 0.5 with the ground-truth path.

Resume from a checkpoint by adding this Hydra override to the `train.py` command in the script:

```text
trainer.resume_from=saved/endocot_klein4b_base_lora/checkpoints/checkpoint-epoch10.pth
```

Keep the same `writer.run_name` when resuming.

## Score saved checkpoints

After `endocot_klein4b_lora_final` has checkpoints, score every saved epoch on the official tests:

```bash
bash scripts/endocot_klein4b_lora_final_maze_test.sh
```

Optional paths:

```bash
CKPT_DIR=saved/endocot_klein4b_lora_final \
TEST_ROOT=data/EndoCoT_Maze_test \
bash scripts/endocot_klein4b_lora_final_maze_test.sh
```

The summary is `saved/endocot_klein4b_lora_final/maze_test/maze_test_summary.json`.
