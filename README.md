# Maze diffusion reasoning — LoRA baseline

Baseline for testing whether an **image-to-image diffusion model can learn to
reason**, using maze solving as the probe. The model gets a rendered maze with a
yellow start and a blue goal, and must output the same maze with the shortest
path drawn in red.

The point of a *visual* maze task is that the answer is verifiable. Every
generated image is decoded back into a symbolic move sequence and checked
against BFS ground truth, so "did it actually solve it" is a number, not a
vibe check.

- **Task:** 5×5 mazes, 20k train / 1k held out, 512×512
- **Models:** FLUX.1-Kontext-dev or Qwen-Image-Edit-2511, selected with `--backend`
- **Hardware:** one RTX A5000 (24 GB) — everything here is sized for that

Maze generation and rendering are ported from
[DiffThinker](https://github.com/lcqysl/DiffThinker)'s `Maze/gen_image.py` so the
images match the reference task pixel for pixel.

---

## Choosing a backend

Both are rectified-flow image-edit transformers that condition by concatenating
the reference image's latents onto the noisy target along the sequence axis, so
the training objective is genuinely shared and the two runs are directly
comparable. Everything downstream — dataset, decoder, metrics, reports — is
model-agnostic.

| | `flux_kontext` | `qwen_edit` |
|---|---|---|
| model | `black-forest-labs/FLUX.1-Kontext-dev` | `Qwen/Qwen-Image-Edit-2511` |
| DiT | 12B | 20B |
| download | **~34 GB** (gated licence) | **~54 GB** (open) |
| text encoder | T5-XXL, text only | Qwen2.5-VL, **text + image** |
| prompt embedding | cached **once**, encoder never loaded again | per-sample; VLM stays resident in 4-bit (~4 GB) |
| guidance | distilled guidance embedding | no embedding; true CFG at inference |
| VRAM in training | ~13 GB | ~18–21 GB |
| speed | faster | ~2× slower per step |

**Start with `flux_kontext`.** It is smaller, faster, and its cached prompt
embedding keeps the whole text tower out of VRAM, so you get more experiments
per day. Move to `qwen_edit` when you want the stronger editor, or to check that
a result is not an artefact of one base model.

> **Disk:** these do not both fit. With ~75 GB free, FLUX (34 GB + 5.5 GB of
> latents) and Qwen (54 GB + 5.5 GB) come to ~99 GB together. Pick one, or clear
> `~/.cache/huggingface` between them.

---

## Setup

FLUX.1-Kontext-dev is **gated**: accept the licence at
<https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev> first.
Qwen-Image-Edit-2511 is open, but logging in is still the easiest path.

```bash
conda activate ldm            # this env already has every dependency
hf auth login
pip install -r requirements.txt   # only if you are not using `ldm`
```

Check the offline pieces work before spending GPU hours or bandwidth:

```bash
python tests/test_offline.py      # maze gen, decoder, metrics, report — no GPU, no weights
python tests/test_train_step.py   # both backends' training step on toy models
```

`test_train_step.py` builds a tiny randomly-initialised transformer with each
real model's interface and pushes a genuine batch through `backend.loss`. It
catches wrong sequence concatenation, a missing reference-image tag, LoRA
attached to nothing, or gradients leaking into the frozen base — without
downloading anything. It falls back to CPU if the GPU is busy.

## Run it

```bash
scripts/00_data.sh                        # ~35 s   → data/maze5/   (171 MB)
scripts/01_precompute.sh                  # + model download → cache/maze5/<backend>/
scripts/03b_eval_base.sh                  # untuned reference point
scripts/02_train.sh                       # → outputs/<backend>/checkpoints/
scripts/03_eval.sh                        # → .../eval/report.html
scripts/04_app.sh                         # → http://127.0.0.1:7860
```

Every script after `00_data.sh` takes the backend from a `BACKEND` environment
variable, defaulting to `flux_kontext`:

```bash
BACKEND=qwen_edit scripts/01_precompute.sh
BACKEND=qwen_edit scripts/02_train.sh
BACKEND=qwen_edit scripts/03_eval.sh
```

Caches and outputs are namespaced by backend (`cache/maze5/qwen_edit/`,
`outputs/qwen_edit/`), so switching does not clobber anything.

Run `03b_eval_base.sh` for a backend before or during its training. It is the
reference point: every later report shows its numbers as a delta against it, and
it tells you what the base model does with this prompt on its own.

---

## How it is put together

```
mazelora/
  maze.py         maze generation, BFS, rendering, wall codec  (ported from DiffThinker)
  gen_dataset.py  train/eval splits, deduplicated, + manifest.jsonl
  precompute.py   VAE latents (+ prompt embedding where cacheable)
  dataset.py      latent pair loader
  backends/
    base.py         the interface + shared rectified-flow maths
    flux_kontext.py FLUX.1-Kontext-dev
    qwen_edit.py    Qwen-Image-Edit-2511
  train.py        LoRA training loop
  infer.py        MazeSolver — backend-agnostic generation
  decode.py       generated image → symbolic path   ← the interesting part
  metrics.py      scoring
  evaluate.py     generate → score → report
  report.py       self-contained report.html
app.py            Gradio playground
```

Only `backends/` knows which model is in use. Adding a third model means adding
one file there; nothing else changes.

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

## Weights & Biases

Needs a W&B account. If `~/.netrc` already holds `api.wandb.ai` credentials
there is nothing to do; otherwise run `wandb login`. Check which account you are
logged in as:

```bash
python -c "import wandb; print(wandb.Api().default_entity)"
```

Logging is **on by default** in both configs, to project `maze-diff-reasoning`.
Training prints the run URL at startup.

| logged | when |
|---|---|
| `train/loss`, `lr`, `grad_norm`, `steps_per_sec`, `vram_gb` | every 10 steps |
| `val/solved`, `edge_f1`, `structure_acc`, `wall_violations`, … | every `validate_every` steps |
| `val/samples` — 8 generated mazes with solved/failed captions | every `validate_every` steps |
| `val/solved_by_path_len` — bar chart, accuracy vs. difficulty | every `validate_every` steps |

`train/vram_gb` is worth watching on the first run: if it sits well under 20 GB
on FLUX, raise `batch_size` to 2 and halve `grad_accum`. On Qwen it will be
close to the ceiling — leave `batch_size` at 1.

```bash
scripts/02_train.sh --no_wandb                    # off, overrides the config file
scripts/02_train.sh --wandb_project my-ablations  # different project
scripts/02_train.sh --wandb_entity my-team        # team instead of personal
scripts/02_train.sh --wandb_run_name rank8-lr2e4  # name the run
WANDB_MODE=offline scripts/02_train.sh            # log locally; `wandb sync <dir>` later
```

`evaluate.py` deliberately does **not** log to W&B — it writes `metrics.json` +
`report.html` next to the checkpoint they came from.

---

## Why it fits in 24 GB

| | FLUX | Qwen |
|---|---|---|
| DiT, NF4 4-bit | ~7 GB | ~12 GB |
| LoRA adapters (fp32) + 8-bit AdamW state | <1 GB | <1 GB |
| activations, batch 1, grad checkpointing | ~4 GB | ~5 GB |
| text encoder | **0 GB — cached** | ~4 GB — sees each maze |
| VAE | **0 GB — latents precomputed** | **0 GB** |

FLUX's prompt is identical for every maze, so it is encoded **once** and cached
and T5-XXL never loads again. Qwen's text encoder is a vision-language model
that reads the maze itself, so its embedding is per-sample; caching it for 20k
mazes would cost ~37 GB of disk, so the VLM stays resident in 4-bit instead.

VAE latents are precomputed for both, so no training step decodes a PNG or runs
the VAE.

### Conditioning

Both models append the reference image's latents to the noisy target along the
**sequence** axis, and only the target half of the prediction is supervised.
They differ in how positions and text arrive:

- **FLUX** tags the reference with `img_ids[..., 0] = 1`, and takes a pooled CLIP
  projection plus a distilled guidance embedding.
- **Qwen** passes `img_shapes` — a list of `(frames, h/2, w/2)` per image — has no
  pooled projection and no guidance embedding, and uses `zero_cond_t` so the
  reference tokens are modulated at timestep 0.

Training objective for both is rectified flow matching (predict `noise - clean`)
with logit-normal timestep sampling, following the diffusers reference scripts.

Two resolution traps, both handled: 512×512 is not one of Kontext's "preferred
resolutions", so the FLUX pipeline would silently rescale a 512px input;
`QwenImageEditPlusPipeline` hard-codes its reference image to 1024×1024 and
refuses batch sizes above 1. Evaluation therefore feeds precomputed latents on
FLUX, and uses a small custom Euler sampler on Qwen, keeping evaluation geometry
identical to training in both cases.

---

## Knobs worth turning

In `configs/flux_kontext.yaml` / `configs/qwen_edit.yaml`:

- **`lora_rank`** — 16 by default. Maze solving is far outside either base
  model's distribution and attention-only r=8 tends to underfit. Drop to 8 for a
  cheaper baseline; add feed-forward modules to `lora_targets` for a stronger one.
- **`lora_targets`** — leave unset to get the backend's default. FLUX targets
  `to_q/to_k/to_v/to_out.0`; Qwen also targets `add_*_proj`/`to_add_out`, because
  its blocks are dual-stream and the `to_*` set alone would leave the instruction
  pathway frozen.
- **`guidance_scale`** — FLUX only: 1.0 during training, 2.5 at inference. Qwen
  ignores it and uses true CFG (`validate_guidance`, default 4.0) instead.
- **`max_steps`** — 6000 steps × batch 4 ≈ 1.2 epochs over 20k mazes.

Harder settings: regenerate with `--size 8 --min_len 10` and point `data`/`cache`
at the new directory. Nothing else changes.

## Where your research idea plugs in

- **A different conditioning or reasoning scheme** — `Backend.loss()` is
  self-contained and unit-tested per backend. Iterative or multi-step refinement
  changes that method and the backend's solver, nothing else.
- **A third base model** — add one file under `mazelora/backends/`, implement the
  interface in `base.py`, register it in `backends/__init__.py`. Data, decoding,
  metrics and reporting are already model-agnostic.
- **A new metric** — add it to `score_sample()`; `aggregate()`, the report tiles
  and W&B pick it up automatically.

Because `evaluate.py` writes `metrics.json` and takes `--compare_to`, any two
runs can be diffed in one report — including across backends.

## Troubleshooting

- **`401 / gated repo`** — accept the FLUX licence, then `hf auth login`.
- **OOM at start of training** — `batch_size: 1`, confirm
  `gradient_checkpointing: true`, and check nothing else holds VRAM
  (`nvidia-smi`). A desktop session or a stray Jupyter kernel can easily hold
  20 GB.
- **`no cache for backend ...`** — run `01_precompute.sh` with the same `BACKEND`.
- **`solved` stuck at 0 but `edge_f1` climbing** — normal early on; the model
  draws roughly-right paths before legal ones. Watch `wall_violations` fall.
- **`structure_acc` low** — the model is redrawing the maze instead of editing
  it. Lower the LoRA scale at eval, or train longer.
