"""Cache VAE latents (and, where the model allows it, the prompt embedding).

Why precompute:
  * with latents on disk, training never runs the VAE or decodes a PNG
  * for FLUX the instruction never changes, so T5-XXL runs once instead of 20k
    times and its 9.5 GB never touches VRAM again

Qwen-Image-Edit conditions its text encoder on the image itself, so its prompt
embedding is per-sample and cannot be cached this way -- the VLM is loaded
during training instead. See `mazelora/backends/qwen_edit.py`.

Latents land in one float16 memmap per split/kind for O(1) random access, under
`<cache>/<backend>/` so several backends can coexist.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .backends import backend_names, get_backend

DEFAULT_PROMPT = (
    "Solve the maze: draw a red line along the corridors from the yellow dot "
    "to the blue dot, without crossing any black wall. Keep the maze unchanged."
)


def read_manifest(split_dir: Path) -> list[dict]:
    with open(split_dir / "manifest.jsonl") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def encode_split(backend, vae, split_dir: Path, kind: str, records: list[dict],
                 out_dir: Path, size_px: int, batch_size: int, device: str):
    h = w = size_px // backend.vae_scale
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}.npy"
    if path.exists():
        print(f"  {kind}: exists, skipping")
        return
    tmp = path.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                   shape=(len(records), backend.latent_channels, h, w))
    for i in tqdm(range(0, len(records), batch_size), desc=f"  vae {kind}"):
        chunk = records[i:i + batch_size]
        arr = np.stack([
            np.asarray(Image.open(split_dir / kind / f"{r['id']}.png").convert("RGB"),
                       dtype=np.float32) for r in chunk])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device) / 127.5 - 1.0
        lat = backend.encode_images(vae, t)
        mm[i:i + len(chunk)] = lat.float().cpu().numpy().astype(np.float16)
    mm.flush()
    del mm
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", type=str, default="flux_kontext", choices=backend_names())
    ap.add_argument("--data", type=str, default="data/maze5")
    ap.add_argument("--cache", type=str, default="cache/maze5")
    ap.add_argument("--model_id", type=str, default=None, help="override the backend default")
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--quantization", type=str, default="none",
                    choices=["nf4", "int8", "none"],
                    help="quantization for the one-shot text-encoder load; bf16 "
                         "('none') fits in 24 GB and gives exact embeddings")
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    backend = get_backend(args.backend)
    model_id = args.model_id or backend.default_model_id
    data = Path(args.data)
    cache = Path(args.cache) / backend.name

    meta = json.loads((data / "meta.json").read_text())
    size_px = meta["size_px"]
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "meta.json").write_text(json.dumps(
        {**meta, "prompt": args.prompt, "backend": backend.name, "model_id": model_id},
        indent=2))

    print(f"backend: {backend.name}  ({model_id})")
    backend.precompute_extra(cache, model_id, args.quantization, args.prompt, args.device)

    print("loading VAE...")
    vae = backend.load_vae(model_id, torch.bfloat16, args.device)
    vae.eval()
    for split in args.splits:
        sdir = data / split
        recs = read_manifest(sdir)
        print(f"{split}: {len(recs)} samples")
        for kind in ("puzzle", "solution"):
            encode_split(backend, vae, sdir, kind, recs, cache / split, size_px,
                         args.batch_size, args.device)
    print(f"\ncache -> {cache.resolve()}")


if __name__ == "__main__":
    main()
