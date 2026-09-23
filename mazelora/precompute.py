"""Cache VAE latents and the (single, fixed) prompt embedding.

Why precompute:
  * the maze instruction never changes, so T5-XXL runs once instead of 20k times
  * with latents on disk, training never loads the VAE or text encoders, which
    leaves the whole 24 GB card for the 4-bit DiT and its activations
  * a training step becomes pure transformer compute -- no JPEG decode, no VAE

Latents land in one float16 memmap per split/kind for O(1) random access.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .flux_utils import (LATENT_CHANNELS, MODEL_ID, VAE_SCALE, bnb_config,
                         load_vae, save_prompt_cache, vae_encode)

DEFAULT_PROMPT = (
    "Solve the maze: draw a red line along the corridors from the yellow dot "
    "to the blue dot, without crossing any black wall. Keep the maze unchanged."
)


def read_manifest(split_dir: Path) -> list[dict]:
    with open(split_dir / "manifest.jsonl") as f:
        return [json.loads(line) for line in f]


@torch.no_grad()
def encode_prompt_once(out_path: Path, prompt: str, model_id: str, quantization: str,
                       device: str = "cuda"):
    """Run CLIP-L + T5-XXL exactly once and cache the result to disk.

    The transformer and VAE are explicitly passed as None so `from_pretrained`
    skips them -- we only want the text towers here.
    """
    if out_path.exists():
        print(f"prompt cache exists, skipping: {out_path}")
        return
    from diffusers import FluxKontextPipeline
    from transformers import T5EncoderModel

    _, tcfg = bnb_config(quantization)
    print(f"loading text encoders (one-shot, t5 quantization={quantization})...")
    te2_kwargs = dict(subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
    if tcfg is not None:
        te2_kwargs.update(quantization_config=tcfg, device_map=device)
    text_encoder_2 = T5EncoderModel.from_pretrained(model_id, **te2_kwargs)

    pipe = FluxKontextPipeline.from_pretrained(
        model_id, transformer=None, vae=None,
        text_encoder_2=text_encoder_2, torch_dtype=torch.bfloat16,
    )
    if tcfg is None:
        pipe.to(device)
    else:
        # bitsandbytes modules are already placed; only move the small CLIP tower
        pipe.text_encoder.to(device)

    embeds, pooled, text_ids = pipe.encode_prompt(
        prompt=prompt, prompt_2=prompt, device=device, num_images_per_prompt=1,
        max_sequence_length=512,
    )
    save_prompt_cache(out_path, prompt, embeds, pooled, text_ids)
    print(f"prompt embeds {tuple(embeds.shape)} pooled {tuple(pooled.shape)} -> {out_path}")
    del pipe, text_encoder_2
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def encode_split(vae, split_dir: Path, kind: str, records: list[dict], out_dir: Path,
                 size_px: int, batch_size: int, device: str):
    h = w = size_px // VAE_SCALE
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{kind}.npy"
    if path.exists():
        print(f"  {kind}: exists, skipping")
        return
    tmp = path.with_suffix(".tmp.npy")
    mm = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.float16,
                                   shape=(len(records), LATENT_CHANNELS, h, w))
    for i in tqdm(range(0, len(records), batch_size), desc=f"  vae {kind}"):
        chunk = records[i:i + batch_size]
        arr = np.stack([
            np.asarray(Image.open(split_dir / kind / f"{r['id']}.png").convert("RGB"),
                       dtype=np.float32) for r in chunk])
        t = torch.from_numpy(arr).permute(0, 3, 1, 2).to(device) / 127.5 - 1.0
        lat = vae_encode(vae, t)
        mm[i:i + len(chunk)] = lat.float().cpu().numpy().astype(np.float16)
    mm.flush()
    del mm
    tmp.rename(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=str, default="data/maze5")
    ap.add_argument("--cache", type=str, default="cache/maze5")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--prompt", type=str, default=DEFAULT_PROMPT)
    ap.add_argument("--quantization", type=str, default="none",
                    choices=["nf4", "int8", "none"],
                    help="quantization for the one-shot T5 load; bf16 ('none') "
                         "fits in 24 GB and gives exact embeddings")
    ap.add_argument("--splits", nargs="+", default=["train", "eval"])
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    data, cache = Path(args.data), Path(args.cache)
    meta = json.loads((data / "meta.json").read_text())
    size_px = meta["size_px"]
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "meta.json").write_text(json.dumps({**meta, "prompt": args.prompt}, indent=2))

    encode_prompt_once(cache / "prompt.safetensors", args.prompt, args.model_id,
                       args.quantization, args.device)

    print("loading VAE...")
    vae = load_vae(args.model_id, device=args.device)
    vae.eval()
    for split in args.splits:
        sdir = data / split
        recs = read_manifest(sdir)
        print(f"{split}: {len(recs)} samples")
        for kind in ("puzzle", "solution"):
            encode_split(vae, sdir, kind, recs, cache / split, size_px,
                         args.batch_size, args.device)
    print(f"\ncache -> {cache.resolve()}")


if __name__ == "__main__":
    main()
