"""Self-contained HTML evaluation report (no server, no assets, no network)."""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path

from PIL import Image

# palette roles (validated default instance from the dataviz reference palette)
TOKENS = """
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --text-muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834;
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219;
"""
TOKENS_DARK = """
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#ffffff; --text-secondary:#c3c2b7; --text-muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926;
  --good:#0ca30c; --critical:#d03b3b; --warning:#fab219;
"""


def _b64(img: Image.Image, px: int = 256) -> str:
    im = img.convert("RGB")
    if im.size[0] != px:
        im = im.resize((px, px), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _tiles(agg: dict, base: dict | None) -> str:
    """Stat tiles. The headline number is a hero figure, not a chart."""
    spec = [
        ("Solved", "solved", True, "legal red route from start to goal"),
        ("Exact edge set", "exact_edges", True, "drawn path identical to ground truth"),
        ("Edge F1", "edge_f1", True, "overlap with the true path"),
        ("Maze preserved", "structure_acc", True, "walls redrawn correctly"),
        ("Endpoints kept", "endpoints_ok", True, "yellow + blue dots in place"),
        ("Wall crossings", "wall_violations", False, "illegal moves per maze"),
    ]
    out = []
    for i, (label, key, is_pct, sub) in enumerate(spec):
        v = agg.get(key)
        if v is None:
            continue
        val = _pct(v) if is_pct else f"{v:.2f}"
        delta = ""
        if base and base.get(key) is not None:
            d = v - base[key]
            better = (d > 0) if is_pct else (d < 0)
            sign = "+" if d > 0 else ""
            txt = f"{sign}{100*d:.1f} pp" if is_pct else f"{sign}{d:.2f}"
            cls = "up" if better else ("flat" if abs(d) < 1e-9 else "down")
            arrow = "▲" if d > 0 else ("▼" if d < 0 else "■")
            delta = (f'<div class="delta {cls}"><span aria-hidden="true">{arrow}</span>'
                     f'{txt} <span class="vs">vs base model</span></div>')
        hero = " hero" if i == 0 else ""
        out.append(f'''<div class="tile{hero}">
      <div class="tile-label">{html.escape(label)}</div>
      <div class="tile-value">{val}</div>
      {delta}
      <div class="tile-sub">{html.escape(sub)}</div>
    </div>''')
    return "\n".join(out)


def _bar_chart(agg: dict, base: dict | None) -> str:
    """Solved rate by solution-path length. One series (two when comparing)."""
    d = agg.get("solved_by_path_len") or {}
    if not d:
        return "<p class='empty'>No per-length breakdown available.</p>"
    keys = sorted(d, key=int)
    bd = (base or {}).get("solved_by_path_len") or {}
    two = bool(bd)

    W, H = 760, 260
    PAD_L, PAD_R, PAD_T, PAD_B = 46, 14, 18, 46
    pw, ph = W - PAD_L - PAD_R, H - PAD_T - PAD_B
    slot = pw / len(keys)
    bw = min(46, slot - 10) / (2 if two else 1)

    def y(v):  # value in [0,1] -> pixel
        return PAD_T + ph * (1 - v)

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Solved rate by path length" class="chart">']
    for g in (0, 0.25, 0.5, 0.75, 1.0):
        yy = y(g)
        parts.append(f'<line x1="{PAD_L}" x2="{W-PAD_R}" y1="{yy:.1f}" y2="{yy:.1f}" '
                     f'stroke="var(--grid)" stroke-width="1"/>')
        parts.append(f'<text x="{PAD_L-8}" y="{yy+4:.1f}" text-anchor="end" '
                     f'class="tick">{int(g*100)}%</text>')

    def bars(src, color, offset, name):
        for i, k in enumerate(keys):
            rec = src.get(k)
            if not rec:
                continue
            v = rec["solved"]
            x = PAD_L + slot * i + (slot - bw * (2 if two else 1)) / 2 + offset
            top, h = y(v), ph * v
            r = min(4, h)  # 4px rounded data-end, square at the baseline
            if h <= 0.5:
                parts.append(f'<rect x="{x:.1f}" y="{y(0)-1.5:.1f}" width="{bw:.1f}" '
                             f'height="1.5" fill="{color}" opacity="0.45"/>')
            else:
                parts.append(
                    f'<path d="M{x:.1f},{y(0):.1f} V{top+r:.1f} Q{x:.1f},{top:.1f} '
                    f'{x+r:.1f},{top:.1f} H{x+bw-r:.1f} Q{x+bw:.1f},{top:.1f} '
                    f'{x+bw:.1f},{top+r:.1f} V{y(0):.1f} Z" fill="{color}"/>')
            parts.append(
                f'<rect class="hit" x="{x-3:.1f}" y="{PAD_T}" width="{bw+6:.1f}" '
                f'height="{ph}" fill="transparent" data-tip="{name} &middot; '
                f'path length {k}: {v*100:.0f}% solved ({rec["n"]} mazes)"/>')
            if not two:
                parts.append(f'<text x="{x+bw/2:.1f}" y="{top-6:.1f}" text-anchor="middle" '
                             f'class="barlabel">{v*100:.0f}</text>')

    if two:
        bars(bd, "var(--series-2)", 0, "Base model")
        bars(d, "var(--series-1)", bw + 2, "LoRA")   # 2px surface gap between fills
    else:
        bars(d, "var(--series-1)", 0, "LoRA")

    parts.append(f'<line x1="{PAD_L}" x2="{W-PAD_R}" y1="{y(0)}" y2="{y(0)}" '
                 f'stroke="var(--axis)" stroke-width="1"/>')
    for i, k in enumerate(keys):
        parts.append(f'<text x="{PAD_L+slot*i+slot/2:.1f}" y="{H-PAD_B+18}" '
                     f'text-anchor="middle" class="tick">{k}</text>')
    parts.append(f'<text x="{W/2:.0f}" y="{H-8}" text-anchor="middle" class="axtitle">'
                 f'cells on the shortest path</text>')
    parts.append("</svg>")

    legend = ""
    if two:
        legend = ('<div class="legend">'
                  '<span><i style="background:var(--series-1)"></i>LoRA</span>'
                  '<span><i style="background:var(--series-2)"></i>Base model</span></div>')

    rows = "".join(
        f"<tr><td>{k}</td><td>{d[k]['n']}</td><td>{d[k]['solved']*100:.1f}%</td>"
        + (f"<td>{bd[k]['solved']*100:.1f}%</td>" if two and k in bd else ("<td>-</td>" if two else ""))
        + "</tr>" for k in keys)
    head = "<tr><th>Path length</th><th>Mazes</th><th>Solved (LoRA)</th>" + \
           ("<th>Solved (base)</th>" if two else "") + "</tr>"
    table = (f'<details class="tableview"><summary>View as table</summary>'
             f'<table>{head}{rows}</table></details>')
    return legend + "".join(parts) + table


def _gallery(rows: list[dict], images: dict, n: int) -> str:
    shown = rows[:n]
    cards = []
    for r in shown:
        trio = images.get(r["id"])
        if not trio:
            continue
        puzzle, pred, gt = trio
        ok = r["solved"]
        badge = ('<span class="badge good"><span aria-hidden="true">✓</span> Solved</span>'
                 if ok else
                 '<span class="badge bad"><span aria-hidden="true">✕</span> Failed</span>')
        notes = []
        if not r["route_found"]:
            notes.append("no connected route drawn")
        if r["wall_violations"]:
            notes.append(f'{r["wall_violations"]} wall crossing(s)')
        if not r["endpoints_ok"]:
            notes.append("endpoint marker moved")
        if ok and not r["exact_edges"]:
            notes.append("legal route plus stray strokes")
        note = " &middot; ".join(notes) or "matches ground truth"
        cards.append(f'''<figure class="card" data-solved="{int(ok)}">
      <div class="trio">
        <div><img src="data:image/png;base64,{_b64(puzzle)}" alt="maze puzzle {r['id']}"><span>input</span></div>
        <div><img src="data:image/png;base64,{_b64(pred)}" alt="model prediction {r['id']}"><span>prediction</span></div>
        <div><img src="data:image/png;base64,{_b64(gt)}" alt="ground truth {r['id']}"><span>ground truth</span></div>
      </div>
      <figcaption>
        <div class="cap-top">{badge}<code>#{r['id']}</code></div>
        <div class="cap-note">{note}</div>
        <div class="moves"><span>pred</span><code>{html.escape(str(r['pred_udrl'] or '—'))}</code></div>
        <div class="moves"><span>true</span><code>{html.escape(r['gt_udrl'])}</code></div>
      </figcaption>
    </figure>''')
    return "\n".join(cards)


def build_report(path: Path, agg: dict, rows: list[dict], images: dict, meta: dict,
                 baseline_agg: dict | None = None, gallery_n: int = 48) -> Path:
    solved_n = sum(1 for r in rows if r["solved"])
    meta_rows = "".join(
        f"<div><dt>{html.escape(str(k))}</dt><dd>{html.escape(str(v))}</dd></div>"
        for k, v in meta.items())

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Maze LoRA evaluation</title>
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
  .lede {{ color:var(--text-secondary); margin:0 0 24px; }}
  .panel {{ background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:20px; }}
  dl.meta {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px 20px; margin:0; }}
  dl.meta dt {{ color:var(--text-muted); font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  dl.meta dd {{ margin:2px 0 0; font-size:13px; word-break:break-all; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(136px,1fr)); gap:12px; }}
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
  .barlabel {{ fill:var(--text-secondary); font-size:11px; font-variant-numeric:tabular-nums; }}
  .axtitle {{ fill:var(--text-muted); font-size:11px; }}
  .legend {{ display:flex; gap:16px; font-size:12px; color:var(--text-secondary); margin-bottom:8px; }}
  .legend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:middle; }}
  .tableview {{ margin-top:12px; font-size:13px; }} .tableview summary {{ cursor:pointer; color:var(--text-secondary); }}
  .tableview table {{ border-collapse:collapse; margin-top:10px; width:100%; }}
  .tableview th, .tableview td {{ text-align:left; padding:5px 10px; border-bottom:1px solid var(--grid);
                                  font-variant-numeric:tabular-nums; }}
  .tableview th {{ color:var(--text-muted); font-weight:500; }}
  .filters {{ display:flex; gap:8px; margin:0 0 14px; flex-wrap:wrap; }}
  .filters button {{ font:inherit; font-size:13px; padding:6px 14px; border-radius:999px; cursor:pointer;
                     border:1px solid var(--border); background:var(--surface-1); color:var(--text-secondary); }}
  .filters button[aria-pressed="true"] {{ background:var(--text-primary); color:var(--plane); border-color:transparent; }}
  .gallery {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(330px,1fr)); gap:14px; }}
  .card {{ margin:0; background:var(--surface-1); border:1px solid var(--border); border-radius:12px; padding:12px; }}
  .trio {{ display:grid; grid-template-columns:repeat(3,1fr); gap:6px; }}
  .trio img {{ width:100%; aspect-ratio:1; border-radius:6px; display:block; background:#fff; }}
  .trio span {{ display:block; text-align:center; font-size:10px; color:var(--text-muted);
                text-transform:uppercase; letter-spacing:.04em; margin-top:4px; }}
  figcaption {{ margin-top:10px; }}
  .cap-top {{ display:flex; align-items:center; gap:8px; }}
  .cap-top code {{ color:var(--text-muted); font-size:12px; margin-left:auto; }}
  .badge {{ font-size:12px; font-weight:600; padding:2px 8px; border-radius:999px; }}
  .badge.good {{ color:var(--good); background:color-mix(in srgb, var(--good) 12%, transparent); }}
  .badge.bad {{ color:var(--critical); background:color-mix(in srgb, var(--critical) 12%, transparent); }}
  .cap-note {{ font-size:12px; color:var(--text-secondary); margin-top:6px; }}
  .moves {{ display:flex; gap:8px; align-items:baseline; margin-top:4px; font-size:12px; }}
  .moves span {{ color:var(--text-muted); width:30px; flex:none; }}
  .moves code {{ word-break:break-all; font-size:11px; }}
  #tip {{ position:fixed; pointer-events:none; opacity:0; transition:opacity .1s; z-index:9;
          background:var(--text-primary); color:var(--plane); font-size:12px;
          padding:6px 10px; border-radius:6px; max-width:260px; }}
  .empty {{ color:var(--text-muted); }}
  @media (max-width:560px) {{ .tile.hero {{ grid-column:span 1; }} }}
</style></head>
<body><div class="wrap">
  <h1>Maze solving &mdash; FLUX.1-Kontext LoRA</h1>
  <p class="lede">{solved_n} of {len(rows)} held-out mazes solved legally
     ({_pct(agg.get('solved'))}). A maze counts as solved only when the decoded red
     route connects the yellow start to the blue goal without crossing a wall.</p>

  <div class="panel"><dl class="meta">{meta_rows}</dl></div>

  <h2>Results</h2>
  <div class="tiles">{_tiles(agg, baseline_agg)}</div>

  <h2>Solved rate by maze difficulty</h2>
  <div class="panel">{_bar_chart(agg, baseline_agg)}</div>

  <h2>Samples</h2>
  <div class="filters" role="group" aria-label="Filter samples">
    <button aria-pressed="true" data-f="all">All</button>
    <button aria-pressed="false" data-f="1">Solved</button>
    <button aria-pressed="false" data-f="0">Failed</button>
  </div>
  <div class="gallery">{_gallery(rows, images, gallery_n)}</div>
</div>
<div id="tip" role="status"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('.hit').forEach(el => {{
  el.addEventListener('mousemove', e => {{
    tip.innerHTML = el.dataset.tip; tip.style.opacity = 1;
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 270) + 'px';
    tip.style.top = (e.clientY + 16) + 'px';
  }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
document.querySelectorAll('.filters button').forEach(b => b.onclick = () => {{
  document.querySelectorAll('.filters button').forEach(x =>
    x.setAttribute('aria-pressed', String(x === b)));
  const f = b.dataset.f;
  document.querySelectorAll('.card').forEach(c =>
    c.style.display = (f === 'all' || c.dataset.solved === f) ? '' : 'none');
}});
</script></body></html>"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(doc, encoding="utf-8")
    return path
