# Maze diffusion reasoning — LoRA baseline

Baseline for testing whether an **image-to-image diffusion model can learn to
reason**, using maze solving as the probe. The model gets a rendered maze with a
yellow start and a blue goal, and must output the same maze with the shortest
path drawn in red.

```
   input (puzzle)              target (solution)
   ┌───────────────┐           ┌───────────────┐
   │ ▓▓  ▓   ▓   ● │           │ ▓▓  ▓   ▓   ●─┐
   │  ▓  ▓▓▓▓▓   ▓ │    ───▶   │  ▓  ▓▓▓▓▓   ▓│ │
   │ ●   ▓       ▓ │           │ ●───┓ ▓ ┌────┘ │
   └───────────────┘           └───────────────┘
```

The point of a *visual* maze task is that the answer is verifiable. Every
generated image is decoded back into a symbolic move sequence and checked
against BFS ground truth, so "did it actually solve it" is a number, not a
vibe check.

- **Model:** `black-forest-labs/FLUX.1-Kontext-dev` (12B DiT), NF4-quantized, LoRA
- **Task:** 5×5 mazes, 20k train / 1k held out, 512×512
- **Hardware:** one RTX A5000 (24 GB) — everything here is sized for that

Maze generation and rendering are ported from
[DiffThinker](https://github.com/lcqysl/DiffThinker)'s `Maze/gen_image.py` so the
images match the reference task pixel for pixel.

---

## Setup

FLUX.1-Kontext-dev is a **gated** repo. Accept the licence at
<https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev>, then:

```bash
conda activate ldm            # this env already has every dependency
hf auth login                 # or: huggingface-cli login
pip install -r requirements.txt   # only if you are not using `ldm`
```

Check the offline pieces work before spending GPU hours:

```bash
python tests/test_offline.py      # maze gen, decoder, metrics, report — no GPU
python tests/test_train_step.py   # LoRA + flow-matching step on a toy DiT — ~50 MB VRAM
```

## Run it

```bash
scripts/00_data.sh          # ~2 min      →  data/maze5/      (~200 MB)
scripts/01_precompute.sh    # ~15 min     →  cache/maze5/     (~5.5 GB, + 34 GB model download)
scripts/03b_eval_base.sh    # ~25 min     →  outputs/base_model/report.html
scripts/02_train.sh         # ~8 h        →  outputs/baseline/checkpoints/
scripts/03_eval.sh          # ~25 min     →  .../step-006000/eval/report.html
scripts/04_app.sh           #             →  http://127.0.0.1:7860
```

Run `03b_eval_base.sh` (the **untuned** model) before or during training. It is
the reference point: every later report shows its numbers as a delta against it,
and it tells you what the base model does with this prompt on its own.

Timings are estimates from the memory/compute budget below; the first real run
will replace them.

---

## Weights & Biases

Needs a W&B account. If `~/.netrc` already holds `api.wandb.ai` credentials
there is nothing to do; otherwise run `wandb login`. Check which account you are
logged in as:

```bash
python -c "import wandb; print(wandb.Api().default_entity)"
```

Logging is **on by default** (`wandb: true` in `configs/baseline.yaml`), to
project `maze-diff-reasoning`. Training prints the run URL at startup.

| logged | when |
|---|---|
| `train/loss`, `lr`, `grad_norm`, `steps_per_sec`, `vram_gb` | every 10 steps |
| `val/solved`, `edge_f1`, `structure_acc`, `wall_violations`, … | every `validate_every` steps |
| `val/samples` — 8 generated mazes with solved/failed captions | every `validate_every` steps |
| `val/solved_by_path_len` — bar chart, accuracy vs. difficulty | every `validate_every` steps |

`val/vram_gb` is worth watching on the first run: if it sits well under 20 GB,
raise `batch_size` to 2 and halve `grad_accum`.

Turning it off or pointing it elsewhere:

```bash
scripts/02_train.sh --no_wandb                    # off, overrides the config file
scripts/02_train.sh --wandb_project my-ablations  # different project
scripts/02_train.sh --wandb_entity my-team        # team instead of personal
scripts/02_train.sh --wandb_run_name rank8-lr2e4  # name the run
WANDB_MODE=offline scripts/02_train.sh            # log locally; `wandb sync <dir>` later
```

Runs are written under `outputs/baseline/wandb/`, which `.gitignore` already
excludes. Note that `evaluate.py` does **not** log to W&B — it writes
`metrics.json` + `report.html` instead, so eval artefacts stay next to the
checkpoint they came from.

---

## How it is put together

```
mazelora/
  maze.py         maze generation, BFS, rendering, wall codec   (ported from DiffThinker)
  gen_dataset.py  train/eval splits, deduplicated, + manifest.jsonl
  precompute.py   VAE latents + the single prompt embedding → disk
  dataset.py      latent pair loader
  flux_utils.py   NF4 loading, latent packing, prompt cache
  train.py        LoRA training (rectified flow matching)
  infer.py        MazeSolver — pipeline without text encoders
  decode.py       generated image → symbolic path   ← the interesting part
  metrics.py      scoring
  evaluate.py     generate → score → report
  report.py       self-contained report.html
app.py            Gradio playground
```

### Reading a solution back out of an image

`decode.py` samples the **midpoint of every internal edge** between adjacent
cells. The renderer makes those three cases visually distinct:

| appearance at the edge midpoint | meaning |
|---|---|
| black | wall |
| white / grey hairline | open passage |
| red | passage, path drawn through it |

One sampler therefore recovers both the drawn path (as an edge set, which
converts straight to `U/D/L/R` moves) and the maze the model redrew. Sampling
cell interiors instead would collide with the yellow/blue endpoint dots.

This is validated in `tests/test_offline.py`: ground-truth images decode to a
perfect score, path-free and wrong-maze images score zero, and decoding survives
Gaussian blur 4px + σ=30 noise + 8px shift + JPEG q60 — far worse than anything
a diffusion model produces.

### Metrics

| metric | meaning |
|---|---|
| **`solved`** | **headline.** A connected red route runs start→goal and crosses no wall. Mazes here are *perfect* (exactly one simple route between any two cells), so a wall-legal route is necessarily the shortest one — `solved` implies optimality. |
| `exact_edges` | drawn edge set is *identical* to ground truth — no stray strokes |
| `edge_f1`, `edge_iou` | overlap with the true path; partial credit while `solved` is still 0 |
| `wall_violations` | red edges drawn straight through a wall |
| `structure_acc` | walls the model redrew correctly, on edges it did not paint over. Catches "solved the wrong maze" |
| `endpoints_ok` | yellow/blue markers still in their cells |
| `red_pixel_iou` | pixel-level agreement, for tracking early training before any maze is solved |
| `solved_by_path_len` | accuracy vs. difficulty — where it breaks down |

`edge_f1`, `structure_acc` and `red_pixel_iou` matter early: `solved` sits at 0
for a long while, and you need something that moves before it does.

---

## Why it fits in 24 GB

| | |
|---|---|
| DiT, NF4 4-bit | ~7 GB |
| LoRA adapters (fp32) + 8-bit AdamW state | <1 GB |
| activations, batch 1, seq 2048, grad checkpointing | ~4 GB |
| T5-XXL + CLIP + VAE | **0 GB — never loaded** |

The prompt is identical for every maze, so it is encoded **once** and cached;
VAE latents are precomputed too. Training is then pure transformer compute — no
text encoder, no VAE, no image decode in the loop. That is what buys the
headroom for a 12B model on a 24 GB card.

`nvidia-smi` should show roughly 12–14 GB in use. If there is headroom, raise
`batch_size` to 2 in `configs/baseline.yaml` and halve `grad_accum`.

### Conditioning

Kontext conditions by **sequence concatenation**, not channel concatenation:
the reference image's latents are appended to the noisy target along the token
axis and tagged by setting `img_ids[..., 0] = 1`. Only the target half of the
prediction is supervised. Training objective is rectified flow matching
(predict `noise - clean`) with logit-normal timestep sampling — the same recipe
as the diffusers FLUX reference scripts.

Note that 512×512 is *not* one of Kontext's "preferred resolutions", so the
pipeline would silently rescale a 512px input. Evaluation therefore feeds
precomputed latents (and `_auto_resize=False` for uploads), keeping the geometry
the decoder expects.

---

## Knobs worth turning

In `configs/baseline.yaml`:

- **`lora_rank`** — set to 16, not 8. Maze solving is far outside the base
  model's distribution and attention-only r=8 tends to underfit. Drop to 8 for a
  cheaper baseline; add `ff.net.0.proj` / `ff.net.2` to `lora_targets` for a
  stronger one (~2 GB more).
- **`guidance_scale`** — 1.0 during training (baked into the distilled guidance
  embedding), 2.5 at inference. Worth sweeping at eval: `--guidance 1.0 3.5`.
- **`max_steps`** — 6000 steps × batch 4 ≈ 1.2 epochs over 20k mazes.
- **`validate_every`** — mid-training validation generates 16 mazes (~2 min) and
  logs `val/solved` plus a sample grid to W&B.

Harder settings: regenerate with `--size 8 --min_len 10` and point
`data`/`cache` at the new directory. Nothing else changes.

## Where your research idea plugs in

- **A different conditioning or reasoning scheme** — `flow_matching_loss()` in
  `train.py` is self-contained and unit-tested. Iterative / multi-step
  refinement changes that function and `MazeSolver._generate`, nothing else.
- **A different base model** — swap `load_transformer` + `MazeSolver._build_pipe`
  in `flux_utils.py` / `infer.py`. Data, decoding, metrics and reporting are
  model-agnostic.
- **A new metric** — add it to `score_sample()`; `aggregate()`, the report tiles
  and W&B pick it up automatically.

Because `evaluate.py` writes `metrics.json` and takes `--compare_to`, any two
runs can be diffed in one report.

## Troubleshooting

- **`401 / gated repo`** — accept the licence, then `hf auth login`.
- **OOM at start of training** — `batch_size: 1`, confirm
  `gradient_checkpointing: true`, and check nothing else holds VRAM
  (`nvidia-smi`). The desktop session on this machine already uses ~1.4 GB.
- **`solved` stuck at 0 but `edge_f1` climbing** — normal early on; the model
  draws roughly-right paths before legal ones. Watch `wall_violations` fall.
- **`structure_acc` low** — the model is redrawing the maze instead of editing
  it. Lower the LoRA scale at eval, or train longer.
- **Disk** — model ~34 GB + latents ~5.5 GB. After `01_precompute.sh` the T5
  weights in `~/.cache/huggingface` are no longer needed for training or eval.
