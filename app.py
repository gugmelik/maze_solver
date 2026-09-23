"""Interactive maze solver: poke at a trained LoRA one puzzle at a time.

    python app.py --lora outputs/baseline/checkpoints/step-006000

Works on eval-set mazes *and* on arbitrary uploads -- an uploaded puzzle is
decoded back into a maze (walls + endpoints), solved with BFS for a reference,
and the model's output is scored against that. So every prediction gets a real
verdict, not just a picture.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import gradio as gr

from mazelora.dataset import load_records
from mazelora.decode import record_from_image
from mazelora.flux_utils import MODEL_ID
from mazelora.maze import render_record
from mazelora.metrics import score_sample

CSS = """
.verdict-ok   {color:#0ca30c; font-weight:600;}
.verdict-bad  {color:#d03b3b; font-weight:600;}
"""


def build(args):
    from mazelora.infer import MazeSolver

    lora = None if args.lora in (None, "none", "None", "") else args.lora
    print(f"loading model ({'LoRA: ' + str(lora) if lora else 'base model'})...")
    solver = MazeSolver.from_checkpoint(lora, Path(args.cache), args.model_id,
                                        args.quantization)
    meta = json.loads((Path(args.cache) / "meta.json").read_text())
    default_size = meta.get("size", 5)

    records: list = []
    try:
        records = load_records(args.data, args.split)
    except FileNotFoundError:
        print(f"note: no {args.split} manifest under {args.data}; upload-only mode")

    def sample_random():
        if not records:
            raise gr.Error(f"No dataset found at {args.data}/{args.split}.")
        rec = random.choice(records)
        return render_record(rec, with_path=False), rec.id, default_size

    def solve(image, size, steps, guidance, seed, scale, use_seed):
        if image is None:
            raise gr.Error("Give me a maze first - sample one or upload an image.")
        size = int(size)
        rec = record_from_image(image, size)
        if rec is None:
            raise gr.Error(
                f"Could not read a {size}x{size} maze from that image. Check the grid "
                "size, and that the start dot is yellow and the goal dot is blue.")

        if lora:
            solver.pipe.set_adapters(["maze"], adapter_weights=[float(scale)])
        pred = solver.solve_images([image], num_steps=int(steps),
                                   guidance_scale=float(guidance),
                                   seed=int(seed) if use_seed else None)[0]

        gt = render_record(rec, with_path=True)
        s = score_sample(pred, rec, gt)
        verdict = ('<span class="verdict-ok">&#10003; Solved</span>' if s["solved"]
                   else '<span class="verdict-bad">&#10007; Not solved</span>')
        notes = []
        if not s["route_found"]:
            notes.append("no connected red route from start to goal")
        if s["wall_violations"]:
            notes.append(f'{s["wall_violations"]} edge(s) drawn through a wall')
        if not s["endpoints_ok"]:
            notes.append("start/goal marker moved")
        if s["solved"] and not s["exact_edges"]:
            notes.append("legal route, plus some stray red")

        md = f"""### {verdict}

| metric | value |
|---|---|
| model moves | `{s['pred_udrl'] or '—'}` |
| true moves | `{s['gt_udrl']}` |
| edge F1 vs true path | {s['edge_f1']:.3f} |
| wall crossings | {s['wall_violations']} |
| maze walls preserved | {100*s['structure_acc']:.1f}% |
| endpoints preserved | {'yes' if s['endpoints_ok'] else 'no'} |
| red-pixel IoU | {s.get('red_pixel_iou', float('nan')):.3f} |

{('**' + '; '.join(notes) + '**') if notes else '**Exactly matches the reference solution.**'}
"""
        return pred, gt, md

    with gr.Blocks(title="Maze LoRA", css=CSS) as demo:
        gr.Markdown(
            f"# Maze solving &mdash; FLUX.1-Kontext LoRA\n"
            f"**Checkpoint:** `{lora or 'base model (no LoRA)'}` &nbsp;&middot;&nbsp; "
            f"**Prompt:** _{meta.get('prompt', '')[:110]}…_")
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Maze puzzle", height=360)
                with gr.Row():
                    btn_rand = gr.Button("🎲 Random eval maze")
                    btn_run = gr.Button("Solve", variant="primary")
                rec_id = gr.Textbox(label="sample id", interactive=False)
                size = gr.Number(value=default_size, label="grid size (n × n)",
                                 precision=0, minimum=2, maximum=16)
                with gr.Accordion("Sampling", open=True):
                    steps = gr.Slider(4, 50, value=28, step=1, label="denoising steps")
                    guidance = gr.Slider(1.0, 7.0, value=2.5, step=0.1, label="guidance scale")
                    scale = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA scale",
                                      interactive=bool(lora))
                    with gr.Row():
                        use_seed = gr.Checkbox(value=True, label="fixed seed")
                        seed = gr.Number(value=0, label="seed", precision=0)
            with gr.Column(scale=1):
                with gr.Row():
                    out = gr.Image(type="pil", label="Model prediction", height=360)
                    ref = gr.Image(type="pil", label="Reference (BFS shortest path)", height=360)
                report = gr.Markdown()

        btn_rand.click(sample_random, outputs=[inp, rec_id, size])
        btn_run.click(solve, [inp, size, steps, guidance, seed, scale, use_seed],
                      [out, ref, report])
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lora", type=str, default=None,
                    help="checkpoint dir, or 'none' for the untuned base model")
    ap.add_argument("--data", type=str, default="data/maze5")
    ap.add_argument("--cache", type=str, default="cache/maze5")
    ap.add_argument("--split", type=str, default="eval")
    ap.add_argument("--model_id", type=str, default=MODEL_ID)
    ap.add_argument("--quantization", type=str, default="nf4", choices=["nf4", "int8", "none"])
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    build(a).launch(server_name=a.host, server_port=a.port, share=a.share)
