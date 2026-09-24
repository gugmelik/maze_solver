"""HTML report for the Best-of-N inference-time-scaling experiment."""

from __future__ import annotations

import html
from pathlib import Path

from .report import TOKENS, TOKENS_DARK, _b64


def _tiles(agg: dict) -> str:
    n = agg["n_samples"]
    base = agg["solved_at_1"]
    spec = [
        (f"Best-of-{n}", agg["solved_best_of_n"], True,
         "picked by the verifier, using the puzzle image alone"),
        ("Single sample", base, True, "the same model, one generation"),
        (f"pass@{n}", agg["pass_at_n"], True,
         "any of the N was correct — a perfect selector's ceiling"),
        ("Verifier precision", agg.get("verifier_precision"), True,
         "accepted candidates that were genuinely solved"),
        ("Cost per maze", float(n), False, f"generations, at {agg['seconds_per_image']:.1f}s each"),
    ]
    out = []
    for i, (label, v, is_pct, sub) in enumerate(spec):
        if v is None:
            continue
        val = f"{100*v:.1f}%" if is_pct else f"{v:.0f}×"
        delta = ""
        if i == 0 and base is not None:
            d = v - base
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "■")
            cls = "up" if d > 0 else ("flat" if d == 0 else "down")
            delta = (f'<div class="delta {cls}"><span aria-hidden="true">{arrow}</span>'
                     f'{"+" if d > 0 else ""}{100*d:.1f} pp '
                     f'<span class="vs">vs a single sample</span></div>')
        out.append(f'''<div class="tile{' hero' if i == 0 else ''}">
      <div class="tile-label">{html.escape(label)}</div>
      <div class="tile-value">{val}</div>
      {delta}
      <div class="tile-sub">{html.escape(sub)}</div>
    </div>''')
    return "\n".join(out)


def _curve(agg: dict) -> str:
    """Solved rate vs sample budget. Two series, so a legend is mandatory."""
    c = agg["curve"]
    xs, W, H = c["n"], 760, 300
    PAD_L, PAD_R, PAD_T, PAD_B = 48, 18, 20, 52
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    # snap the axis to a top that divides into four whole-percent ticks,
    # so the scale never reads "99% / 74% / 49%"
    peak = max(c["pass_at_n"]) * 1.1
    top = next((t for t in (0.2, 0.4, 0.8, 1.0) if peak <= t), 1.0)

    def X(i):
        return PAD_L + (pw * i / max(len(xs) - 1, 1))

    def Y(v):
        return PAD_T + ph * (1 - v / top)

    p = [f'<svg viewBox="0 0 {W} {H}" role="img" '
         f'aria-label="Solved rate versus number of samples" class="chart">']
    for g in range(5):
        v = top * g / 4
        y = Y(v)
        p.append(f'<line x1="{PAD_L}" x2="{W-PAD_R}" y1="{y:.1f}" y2="{y:.1f}" '
                 f'stroke="var(--grid)" stroke-width="1"/>')
        p.append(f'<text x="{PAD_L-8}" y="{y+4:.1f}" text-anchor="end" class="tick">'
                 f'{round(v*100)}%</text>')

    # A perfect verifier makes the two series coincide exactly. Drawing the
    # oracle as a wide translucent band underneath means the overlap reads as a
    # halo around the line rather than silently hiding a whole series.
    bon, oracle = c["best_of_n"], c["pass_at_n"]
    band = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(oracle))
    p.append(f'<polyline points="{band}" fill="none" stroke="var(--series-2)" '
             f'stroke-width="6" stroke-opacity="0.35" stroke-linejoin="round" '
             f'stroke-linecap="round"/>')
    line = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(bon))
    p.append(f'<polyline points="{line}" fill="none" stroke="var(--series-1)" '
             f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>')
    for i, v in enumerate(bon):
        # 2px surface ring keeps overlapping markers legible
        p.append(f'<circle cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5" fill="var(--series-1)" '
                 f'stroke="var(--surface-1)" stroke-width="2"/>')

    gap = oracle[-1] - bon[-1]
    if gap > 0.02:
        p.append(f'<text x="{X(len(oracle)-1):.1f}" y="{Y(oracle[-1])-14:.1f}" '
                 f'text-anchor="end" class="endlabel" fill="var(--series-2)">'
                 f'{oracle[-1]*100:.0f}%</text>')
        p.append(f'<text x="{X(len(bon)-1):.1f}" y="{Y(bon[-1])+20:.1f}" '
                 f'text-anchor="end" class="endlabel" fill="var(--series-1)">'
                 f'{bon[-1]*100:.0f}%</text>')
    else:
        p.append(f'<text x="{X(len(bon)-1):.1f}" y="{Y(bon[-1])-14:.1f}" '
                 f'text-anchor="end" class="endlabel" fill="var(--series-1)">'
                 f'{bon[-1]*100:.0f}%</text>')

    for i, n in enumerate(xs):
        tip = (f"N = {n}&nbsp;&middot;&nbsp; Best-of-N {c['best_of_n'][i]*100:.0f}% "
               f"&middot; pass@N {c['pass_at_n'][i]*100:.0f}%")
        p.append(f'<rect class="hit" x="{X(i)-pw/(2*max(len(xs)-1,1))-1:.1f}" y="{PAD_T}" '
                 f'width="{pw/max(len(xs)-1,1)+2:.1f}" height="{ph}" fill="transparent" '
                 f'data-tip="{tip}"/>')
        p.append(f'<text x="{X(i):.1f}" y="{H-PAD_B+20}" text-anchor="middle" '
                 f'class="tick">{n}</text>')
    p.append(f'<line x1="{PAD_L}" x2="{W-PAD_R}" y1="{Y(0)}" y2="{Y(0)}" '
             f'stroke="var(--axis)" stroke-width="1"/>')
    p.append(f'<text x="{W/2:.0f}" y="{H-10}" text-anchor="middle" class="axtitle">'
             f'samples generated per maze (N)</text>')
    p.append("</svg>")

    legend = ('<div class="legend">'
              '<span><i style="background:var(--series-1)"></i>Best-of-N (verified)</span>'
              '<span><i class="band" style="background:var(--series-2)"></i>'
              'pass@N (oracle ceiling)</span></div>')
    identical = all(abs(a - b) < 1e-9 for a, b in zip(bon, oracle))
    note = ('<p class="note">The two curves coincide at every N: the verifier never '
            'picked a wrong candidate when a right one was available, so Best-of-N '
            'sits exactly on the oracle ceiling.</p>' if identical else
            f'<p class="note">At N={xs[-1]} the selector leaves '
            f'{(oracle[-1]-bon[-1])*100:.1f} pp on the table &mdash; that many mazes had a '
            f'correct candidate the verifier did not pick.</p>')
    rows = "".join(
        f"<tr><td>{n}</td><td>{c['best_of_n'][i]*100:.1f}%</td>"
        f"<td>{c['pass_at_n'][i]*100:.1f}%</td>"
        f"<td>{c['accepted_rate'][i]*100:.1f}%</td></tr>"
        for i, n in enumerate(xs))
    table = ('<details class="tableview"><summary>View as table</summary><table>'
             '<tr><th>N</th><th>Best-of-N</th><th>pass@N</th>'
             '<th>verifier accepted something</th></tr>' + rows + "</table></details>")
    return legend + "".join(p) + note + table


def _gallery(rows: list[dict], images: dict, n: int) -> str:
    cards = []
    for m in rows[:n]:
        trio = images.get(m["id"])
        if not trio:
            continue
        puzzle, cands, gt = trio
        thumbs = []
        for k, c in enumerate(cands):
            v = m["verdicts"][k]
            picked = k == m["pick"]
            truly = m["solved"][k]
            cls = "ok" if v["accepted"] else "no"
            mark = "✓" if v["accepted"] else "✕"
            why = ("accepted by the verifier" if v["accepted"]
                   else ("no connected route" if not v["route_found"]
                         else f'{v["wall_violations"]} wall crossing(s)'))
            flag = "" if v["accepted"] == truly else " ⚠ verifier disagrees with truth"
            thumbs.append(
                f'<div class="cand {"picked" if picked else ""}" '
                f'title="candidate {k} — {why}{flag}">'
                f'<img src="data:image/png;base64,{_b64(c, 132)}" alt="candidate {k}">'
                f'<span class="cmark {cls}">{mark}</span>'
                f'{"<span class=pick>picked</span>" if picked else ""}</div>')
        outcome = ("rescued by sampling" if m["solved_best_of_n"] and not m["solved_at_1"]
                   else "solved on the first try" if m["solved_at_1"]
                   else "unsolved at any N" if not m["pass_at_n"]
                   else "a correct candidate existed but was not picked")
        cards.append(f'''<figure class="card">
      <div class="head">
        <img class="ref" src="data:image/png;base64,{_b64(puzzle, 132)}" alt="puzzle"><span>input</span>
        <img class="ref" src="data:image/png;base64,{_b64(gt, 132)}" alt="ground truth"><span>truth</span>
        <div class="meta"><code>#{m['id']}</code>
          <div class="outcome">{html.escape(outcome)}</div></div>
      </div>
      <div class="cands">{"".join(thumbs)}</div>
    </figure>''')
    return "\n".join(cards)


def build_bon_report(path: Path, agg: dict, rows: list[dict], images: dict,
                     gallery_n: int = 12) -> Path:
    n = agg["n_samples"]
    rescued = sum(1 for m in rows if m["solved_best_of_n"] and not m["solved_at_1"])
    missed = sum(1 for m in rows if m["pass_at_n"] and not m["solved_best_of_n"])
    meta_rows = "".join(
        f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(str(v))}</dd></div>"
        for k, v in [("backend", agg["backend"]), ("checkpoint", agg["checkpoint"]),
                     ("mazes", agg["n_mazes"]), ("candidates each", n),
                     ("denoise steps", agg["steps"]), ("guidance", agg["guidance"]),
                     ("sec / image", f"{agg['seconds_per_image']:.1f}")])

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Best-of-N maze scaling</title>
<style>
  :root {{ color-scheme: light; {TOKENS} }}
  @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ color-scheme: dark; {TOKENS_DARK} }} }}
  :root[data-theme="dark"] {{ color-scheme: dark; {TOKENS_DARK} }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:32px 16px 64px; background:var(--plane); color:var(--text-primary);
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }}
  .wrap {{ max-width:1080px; margin:0 auto; }}
  h1 {{ font-size:26px; margin:0 0 4px; letter-spacing:-0.01em; }}
  h2 {{ font-size:17px; margin:40px 0 14px; }}
  .lede {{ color:var(--text-secondary); margin:0 0 24px; max-width:70ch; }}
  .panel {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  dl.meta {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(130px,1fr)); gap:10px 20px; margin:0; }}
  dl.meta dt {{ color:var(--text-muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  dl.meta dd {{ margin:2px 0 0; font-size:13px; word-break:break-all; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }}
  .tile {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:16px; }}
  .tile.hero {{ grid-column:span 2; }}
  .tile-label {{ font-size:12px; color:var(--text-muted); text-transform:uppercase; letter-spacing:.04em; }}
  .tile-value {{ font-size:32px; font-weight:600; margin-top:6px; letter-spacing:-0.02em; }}
  .tile.hero .tile-value {{ font-size:52px; }}
  .tile-sub {{ font-size:12px; color:var(--text-secondary); margin-top:6px; }}
  .delta {{ font-size:12px; margin-top:4px; font-weight:600; }}
  .delta.up {{ color:var(--good); }} .delta.down {{ color:var(--critical); }} .delta.flat {{ color:var(--text-muted); }}
  .delta .vs {{ color:var(--text-muted); font-weight:400; }}
  .chart {{ width:100%; height:auto; display:block; }}
  .tick {{ fill:var(--text-muted); font-size:11px; font-variant-numeric:tabular-nums; }}
  .endlabel {{ font-size:12px; font-weight:600; font-variant-numeric:tabular-nums; }}
  .axtitle {{ fill:var(--text-muted); font-size:11px; }}
  .legend {{ display:flex; gap:16px; font-size:12px; color:var(--text-secondary); margin-bottom:8px; }}
  .legend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:middle; }}
  .legend i.band {{ opacity:0.45; }}
  .note {{ font-size:12px; color:var(--text-secondary); margin:10px 0 0; max-width:66ch; }}
  .tableview {{ margin-top:12px; font-size:13px; }} .tableview summary {{ cursor:pointer; color:var(--text-secondary); }}
  .tableview table {{ border-collapse:collapse; margin-top:10px; width:100%; }}
  .tableview th, .tableview td {{ text-align:left; padding:5px 10px; border-bottom:1px solid var(--grid);
                                  font-variant-numeric:tabular-nums; }}
  .tableview th {{ color:var(--text-muted); font-weight:500; }}
  .card {{ margin:0 0 14px; background:var(--surface-1); border:1px solid var(--border);
           border-radius:12px; padding:14px; }}
  .head {{ display:flex; align-items:center; gap:8px; margin-bottom:10px; }}
  .head img.ref {{ width:64px; height:64px; border-radius:6px; background:#fff; }}
  .head span {{ font-size:10px; color:var(--text-muted); text-transform:uppercase; letter-spacing:.04em; }}
  .head .meta {{ margin-left:auto; text-align:right; }}
  .head code {{ font-size:12px; color:var(--text-muted); }}
  .outcome {{ font-size:12px; color:var(--text-secondary); }}
  .cands {{ display:flex; flex-wrap:wrap; gap:8px; }}
  .cand {{ position:relative; width:78px; }}
  .cand img {{ width:78px; height:78px; border-radius:6px; display:block; background:#fff;
               border:2px solid transparent; }}
  .cand.picked img {{ border-color:var(--series-1); }}
  .cmark {{ position:absolute; top:3px; left:3px; font-size:11px; font-weight:700;
            border-radius:4px; padding:0 4px; background:var(--surface-1); }}
  .cmark.ok {{ color:var(--good); }} .cmark.no {{ color:var(--critical); }}
  .pick {{ display:block; text-align:center; font-size:10px; color:var(--series-1);
           font-weight:600; margin-top:2px; }}
  #tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .1s; z-index:9;
          background:var(--text-primary); color:var(--plane); font-size:12px;
          padding:6px 10px; border-radius:6px; max-width:280px; }}
  @media (max-width:560px) {{ .tile.hero {{ grid-column:span 1; }} }}
</style></head>
<body><div class="wrap">
  <h1>Inference-time scaling &mdash; Best-of-N</h1>
  <p class="lede">Sample {n} candidate solutions per maze and keep the one a
     rule-based verifier accepts. The verifier reads the walls and endpoints
     from the <em>puzzle image only</em> and checks that the drawn red route
     connects start to goal without crossing a wall &mdash; it never consults the
     ground-truth solution. Sampling rescued <strong>{rescued}</strong> of
     {agg['n_mazes']} mazes that a single sample got wrong;
     <strong>{missed}</strong> had a correct candidate the selector failed to pick.</p>

  <div class="panel"><dl class="meta">{meta_rows}</dl></div>

  <h2>Results</h2>
  <div class="tiles">{_tiles(agg)}</div>

  <h2>Solved rate vs sample budget</h2>
  <div class="panel">{_curve(agg)}</div>

  <h2>Candidates</h2>
  <p class="lede">Hardest mazes first. A green tick means the verifier accepted
     that candidate; the blue border is the one it picked.</p>
  {_gallery(rows, images, gallery_n)}
</div>
<div id="tip" role="status"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.hit').forEach(el => {{
  el.addEventListener('mousemove', e => {{
    tip.innerHTML = el.dataset.tip; tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 290) + 'px';
    tip.style.top = (e.clientY + 16) + 'px';
  }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
