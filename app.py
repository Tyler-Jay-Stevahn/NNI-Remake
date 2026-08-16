#!/usr/bin/env python3
"""
NNI-Test Dashboard — zero-dependency web UI (Python standard library only).

Serves the NNI-Test repo data (proposals, verification, smoke tests, MNIST
results) as a read-only dashboard. Binds to 0.0.0.0:6123 so any machine on the
same LAN can open it at http://<this-host-ip>:6123.

Requires only the Python standard library. Run:

    cd ~/NNI-Test
    python3 webui/app.py

(or use the systemd user unit in this folder for boot auto-start).

Data is read from the jsonl files on every request, so a browser refresh
always shows the current state on disk.
"""
import html
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOST = "0.0.0.0"
PORT = 6123

# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------
def load_jsonl(name):
    path = os.path.join(ROOT, name)
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def index_by(records, key):
    return {r[key]: r for r in records if key in r}


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def esc(s):
    if s is None:
        return ""
    return html.escape(str(s))


def status_color(status):
    if isinstance(status, bool):
        return "ok" if status else "bad"
    s = (status or "").lower()
    if s in ("ok", "tested", "pass", "above_chance", "true"):
        return "ok"
    if s in ("fail", "error", "false", "untested", "below_chance"):
        return "bad"
    if s in ("warn", "proposed", "approved"):
        return "warn"
    return "neutral"


BASE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{title}} | NNI-Test</title>
<style>
  :root { --bg:#0f1115; --card:#171b22; --ink:#e6e9ef; --muted:#8b94a7;
          --line:#262c38; --ok:#3fb950; --warn:#d29922; --bad:#f85149;
          --accent:#58a6ff; }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:14px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; }
  header { padding:14px 22px; border-bottom:1px solid var(--line);
           background:var(--card); display:flex; align-items:baseline; gap:16px;
           flex-wrap:wrap; position:sticky; top:0; z-index:10; }
  header h1 { font-size:16px; margin:0; letter-spacing:.5px; }
  header a.brand { color:var(--ink); text-decoration:none; }
  nav { display:flex; gap:14px; flex-wrap:wrap; }
  nav a { color:var(--muted); text-decoration:none; font-size:13px; }
  nav a:hover { color:var(--accent); }
  main { padding:22px; max-width:1180px; margin:0 auto; }
  h2 { font-size:15px; margin:26px 0 12px; border-left:3px solid var(--accent);
       padding-left:8px; }
  .cards { display:flex; gap:12px; flex-wrap:wrap; margin:10px 0 4px; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:14px 16px; min-width:120px; }
  .card .n { font-size:24px; font-weight:700; }
  .card .l { color:var(--muted); font-size:12px; margin-top:2px; }
  table { width:100%; border-collapse:collapse; margin-top:10px; background:var(--card);
          border:1px solid var(--line); border-radius:10px; overflow:hidden; }
  th,td { text-align:left; padding:9px 12px; border-bottom:1px solid var(--line);
          vertical-align:top; }
  th { color:var(--muted); font-weight:600; font-size:12px; text-transform:uppercase;
       letter-spacing:.4px; }
  tr:last-child td { border-bottom:none; }
  tr:hover td { background:#1b212b; }
  a { color:var(--accent); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .badge { display:inline-block; padding:2px 8px; border-radius:999px; font-size:11px;
           border:1px solid var(--line); }
  .ok { color:var(--ok); border-color:var(--ok); }
  .bad { color:var(--bad); border-color:var(--bad); }
  .warn { color:var(--warn); border-color:var(--warn); }
  .neutral { color:var(--muted); }
  .muted { color:var(--muted); }
  .desc { color:var(--muted); font-size:13px; line-height:1.6; max-width:760px;
          margin:6px 0 18px; padding:12px 14px; background:var(--card);
          border:1px solid var(--line); border-left:3px solid var(--accent);
          border-radius:8px; }
  pre { background:#0b0e13; border:1px solid var(--line); border-radius:8px;
        padding:12px; overflow:auto; font-size:12px; }
  .detail-grid { display:grid; grid-template-columns:140px 1fr; gap:4px 14px; }
  .detail-grid .k { color:var(--muted); }
  ul.checks { margin:6px 0; padding-left:18px; }
  ul.checks li { margin:3px 0; }
  .foot { color:var(--muted); font-size:12px; margin-top:30px; text-align:center; }
  code { background:#0b0e13; padding:1px 5px; border-radius:4px; }
  .legend { display:flex; gap:18px; align-items:center; flex-wrap:wrap;
           font-size:12px; color:var(--muted); margin:8px 0 14px; }
  .legend .sw { display:inline-block; width:11px; height:11px; border-radius:3px;
               margin-right:5px; vertical-align:middle; }
  .chart { overflow-x:auto; padding:10px 4px; border:1px solid var(--line);
           border-radius:10px; background:var(--card); }
  .pcoord { display:block; }
  .pcoord text { font-family:ui-monospace,Menlo,Consolas,monospace; }
  .vlegend { display:flex; gap:16px; flex-wrap:wrap; margin:6px 0 4px; font-size:12px; }
  .vlegend .sw { display:inline-block; width:14px; height:3px; margin-right:5px;
                 vertical-align:middle; }
  .catlegend { margin:10px 0 4px; font-size:12px; color:var(--muted);
               max-width:520px; }
  .catlegend .ci { padding:1px 0; }
  .catlegend .sw { display:inline-block; width:10px; height:10px; border-radius:3px;
                   margin-right:6px; vertical-align:middle; }
  .dropdown { position:relative; display:inline-block; }
  .dropdown > a { color:var(--muted); text-decoration:none; font-size:13px; cursor:pointer; }
  .dropdown > a:hover { color:var(--accent); }
  .dropdown-menu { display:none; position:absolute; top:100%; left:0; margin-top:10px;
                   background:var(--card); border:1px solid var(--line); border-radius:8px;
                   padding:6px; min-width:210px; z-index:50; flex-direction:column;
                   box-shadow:0 8px 24px rgba(0,0,0,.4); }
  .dropdown:hover .dropdown-menu, .dropdown:focus-within .dropdown-menu { display:flex; }
  .dropdown-menu a { padding:6px 9px; border-radius:5px; color:var(--ink); font-size:13px; }
  .dropdown-menu a:hover { background:#1b212b; text-decoration:none; color:var(--accent); }
  .filterbar { margin:8px 0 4px; font-size:13px; color:var(--ink); }
  .filterbar select { background:var(--card); color:var(--ink); border:1px solid var(--line);
                      border-radius:6px; padding:4px 8px; font-size:13px; }
</style>
</head>
<body>
<header>
  <h1><a class="brand" href="/">NNI-Test</a></h1>
  <nav>
{{nav}}
  </nav>
</header>
<main>
{{body}}
<div class="foot">NNI-Test Dashboard &middot; data read live from <code>*.jsonl</code></div>
</main>
</body>
</html>"""


def build_nav():
    families = sorted({p.get("task_family") for p in load_jsonl("proposals.jsonl")})
    base_links = [
        ("/proposals", "Proposals"),
        ("/verification", "Verification"),
        ("/smoke", "Smoke Tests"),
        ("/models", "All models"),
    ]
    items = "".join(f"<a href='{href}'>{esc(label)}</a>"
                    for href, label in base_links)
    fam_links = ("<a href='/mnist'>mnist</a>"
                 + "".join(
                     f"<a href='/family/{esc(f)}'>{esc(f)}</a>" for f in families
                     if f != "hpo-mnist"
                 ))
    dropdown = (
        "<span class='dropdown'><a href='#'>Task &#9662;</a>"
        f"<span class='dropdown-menu'>{fam_links}</span></span>"
    )
    return items + dropdown + "<a href='/api/proposals'>JSON API</a>"


def render(title, body):
    return (BASE.replace("{{title}}", title)
              .replace("{{body}}", body)
              .replace("{{nav}}", build_nav()).encode("utf-8"))


def badge(status):
    return f'<span class="badge {status_color(status)}">{esc(status)}</span>'


def _test_status(p, real):
    """Effective display status: the real-data test result status if a test
    exists (tests/results.jsonl), else the proposal lifecycle status
    (proposals.jsonl). This is what the Status column should show after a
    proposal has been trained/validated, so the column updates on test."""
    rl = (real or {}).get(p.get("id")) if real else None
    if rl and rl.get("status"):
        return rl["status"]
    return p.get("status")


def proposals_table(records, real=None):
    rows = []
    for p in records:
        pid = esc(p.get("id"))
        spec = p.get("spec", {}) or {}
        model = spec.get("model") if isinstance(spec, dict) else ""
        dataset = spec.get("dataset") if isinstance(spec, dict) else ""
        rows.append(
            f"<tr>"
            f"<td><a href='/proposal/{esc(pid)}'>{pid}</a></td>"
            f"<td>{esc(p.get('task_family'))}</td>"
            f"<td>{esc(model)}</td>"
            f"<td>{esc(dataset)}</td>"
            f"<td>{badge(_test_status(p, real))}</td>"
            f"<td>{badge(p.get('compile_status'))}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>ID</th><th>Family</th><th>Model</th>"
        "<th>Dataset</th><th>Status</th><th>Compile</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table>"
    )


def pcoord_chart(rows, dims, fixed_max=None, palette=None, col_w=90,
                 note=None, color_by=None, group_by=None):
    """Parallel-coordinates SVG: one line per row, one vertical column per dim.

    rows:    list of dicts (each must have 'id'); plotted in order.
    dims:    list of (key, label, getter) where getter(row) -> value.
    fixed_max: optional {key: max} to pin a column's scale (others auto-max).
    palette: line colours cycled when color_by/group_by is None.
    color_by: optional fn(row) -> css colour (e.g. colour by family).
    group_by: optional fn(row) -> str group (e.g. model). When set, lines are
              coloured per distinct group and tagged with data-group so a
              client-side dropdown can filter them.
    Returns (chart_html, vlegend_html, catlegend_html).
    """
    dims = sorted(dims, key=lambda d: d[1].lower())
    fixed_max = fixed_max or {}
    palette = palette or ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f778ba"]
    dim_max = dict(fixed_max)
    for key, label, fn in dims:
        if key not in dim_max:
            vals = [fn(r) for r in rows]
            vals = [v for v in vals
                    if isinstance(v, (int, float)) and not isinstance(v, bool)]
            dim_max[key] = max(vals) if vals else 1

    # group colours (for group_by)
    groups = []
    if group_by:
        seen = set()
        for r in rows:
            g = str(group_by(r))
            if g not in seen:
                seen.add(g)
                groups.append(g)
    def group_color(r):
        g = str(group_by(r))
        return palette[groups.index(g) % len(palette)]

    PLOT_H, TOP, PADX = 320, 56, 40
    n = len(dims)
    W = PADX * 2 + col_w * (n - 1)
    H = TOP + PLOT_H + 70
    col_x = [PADX + i * col_w for i in range(n)]

    def y_of(key, raw):
        mx = dim_max.get(key, 1) or 1
        if isinstance(raw, (int, float)) and not isinstance(raw, bool):
            frac = max(0.0, min(1.0, raw / mx))
        else:
            frac = 1.0  # categorical / missing -> top
        return TOP + (1.0 - frac) * PLOT_H

    paths = []
    for vi, r in enumerate(rows):
        rid = r.get("id")
        if color_by:
            color = color_by(r)
        elif group_by:
            color = group_color(r)
        else:
            color = palette[vi % len(palette)]
        gtag = f" data-group='{esc(str(group_by(r)))}'" if group_by else ""
        gcls = " pc-line" if group_by else ""
        pts = []
        for j, (key, label, fn) in enumerate(dims):
            raw = fn(r)
            pts.append((col_x[j], y_of(key, raw), raw, label))
        d_attr = " ".join(f"L{x:.1f},{y:.1f}" for x, y, _, _ in pts)
        d_attr = "M" + d_attr[1:] if d_attr.startswith("L") else d_attr
        circ = "".join(
            f"<circle class='pc-dot{gcls}'{gtag} cx='{x:.1f}' cy='{y:.1f}' r='4' fill='{color}'>"
            f"<title>{esc(rid)} — {esc(label)}: {esc(raw)}</title></circle>"
            for x, y, raw, label in pts
        )
        paths.append(
            f"<path class='pc-line{gcls}'{gtag} d='{d_attr}' fill='none' stroke='{color}' "
            f"stroke-width='2.5' stroke-linejoin='round'>"
            f"<title>{esc(rid)}</title></path>{circ}"
        )

    axes = []
    for j, (key, label, fn) in enumerate(dims):
        x = col_x[j]
        axes.append(
            f"<line x1='{x:.1f}' y1='{TOP:.1f}' x2='{x:.1f}' "
            f"y2='{TOP+PLOT_H:.1f}' stroke='#3a4252' stroke-width='1'/>"
        )
        words = label.split()
        ty = TOP - 30
        for w in words:
            axes.append(
                f"<text x='{x:.1f}' y='{ty:.1f}' fill='#8b94a7' "
                f"font-size='11' text-anchor='middle'>{esc(w)}</text>"
            )
            ty += 14
        mx = dim_max.get(key, 1)
        axes.append(
            f"<text x='{x:.1f}' y='{TOP+PLOT_H+18:.1f}' fill='#6b7280' "
            f"font-size='10' text-anchor='middle'>{esc(mx)}</text>"
        )

    svg = (
        f"<svg class='pcoord' viewBox='0 0 {W:.0f} {H:.0f}' "
        f"width='{W:.0f}' height='{H:.0f}' "
        f"role='img' aria-label='parallel-coordinates chart'>"
        + "".join(axes) + "".join(paths) + "</svg>"
    )
    chart = f"<div class='chart'>{svg}</div>"

    # variant / group legend
    if color_by:
        seen = {}
        for r in rows:
            c = color_by(r)
            seen.setdefault(c, r.get("task_family") or r.get("id"))
        vlegend = ("<div class='vlegend'>"
                   + "".join(f"<span><span class='sw' style='background:{c}'></span>"
                             f"{esc(lbl)}</span>" for c, lbl in seen.items())
                   + "</div>")
    elif group_by:
        vlegend = ("<div class='vlegend'>"
                   + "".join(f"<span><span class='sw' style='background:"
                             f"{palette[i % len(palette)]}'></span>{esc(g)}</span>"
                             for i, g in enumerate(groups)) + "</div>")
    else:
        if len(rows) <= 12:
            vlegend = ("<div class='vlegend'>"
                       + "".join(f"<span><span class='sw' style='background:"
                                 f"{palette[i % len(palette)]}'></span>{esc(r.get('id'))}</span>"
                                 for i, r in enumerate(rows)) + "</div>")
        else:
            vlegend = ""

    catlegend = (
        "<div class='catlegend'>"
        "<div class='ci'><span class='sw' style='background:#3a4252'></span>"
        "columns left&rarr;right (alphabetical): "
        + " → ".join(f"{esc(label)}" for _, label, _ in dims)
        + "</div>"
        + (f"<div class='muted' style='margin-top:4px'>{esc(note)}</div>" if note else "")
        + "</div>"
    )
    return chart, vlegend, catlegend


# ---------------------------------------------------------------------------
# Page builders
# ---------------------------------------------------------------------------
def page_index():
    proposals = load_jsonl("proposals.jsonl")
    verify = load_jsonl("proposals_verification.jsonl")
    smoke = load_jsonl("pipeline_smoke_results.jsonl")
    mnist = [r for r in load_jsonl("tests/results.jsonl") if r.get("declared_dataset") == "mnist"]

    fams = sorted({p.get("task_family") for p in proposals})
    real = index_by(load_jsonl("tests/results.jsonl"), "id")
    v_ok = sum(1 for v in verify if (v.get("status") or "").lower() == "ok")
    v_warn = sum(1 for v in verify if (v.get("status") or "").lower() == "warn")
    v_fail = sum(1 for v in verify if (v.get("status") or "").lower() == "fail")
    s_ok = sum(1 for s in smoke if (s.get("status") or "").lower() == "ok")

    fam_rows = []
    for f in fams:
        if f == "hpo-mnist":
            continue  # merged into the MNIST page
        fps = [p for p in proposals if p.get("task_family") == f]
        # Lead status = first proposal's effective test status (ok if tested)
        fam_rows.append(
            f"<tr><td><a href='/family/{esc(f)}'>{esc(f)}</a></td>"
            f"<td>{len(fps)}</td>"
            f"<td>{badge(_test_status(fps[0], real))}</td></tr>"
        )

    body = f"""
    <h2>Overview</h2>
    <div class="cards">
      <div class="card"><div class="n">{len(proposals)}</div><div class="l">Proposals</div></div>
      <div class="card"><div class="n">{len(fams)}</div><div class="l">Task families</div></div>
      <div class="card"><div class="n" style="color:var(--ok)">{v_ok}</div><div class="l">Verify OK</div></div>
      <div class="card"><div class="n" style="color:var(--warn)">{v_warn}</div><div class="l">Verify Warn</div></div>
      <div class="card"><div class="n" style="color:var(--bad)">{v_fail}</div><div class="l">Verify Fail</div></div>
      <div class="card"><div class="n" style="color:var(--ok)">{s_ok}</div><div class="l">Smoke OK</div></div>
      <div class="card"><div class="n">{len(mnist)}</div><div class="l">MNIST tested</div></div>
    </div>
    <h2>Task families</h2>
    <table><thead><tr><th>Family</th><th>Proposals</th><th>Lead status</th></tr></thead>
    <tbody>{''.join(fam_rows)}</tbody></table>
    """
    return "Dashboard", body


def page_proposals(family=None):
    proposals = load_jsonl("proposals.jsonl")
    real = index_by(load_jsonl("tests/results.jsonl"), "id")
    if family:
        proposals = [p for p in proposals if p.get("task_family") == family]
    title = f"Proposals{f' — {family}' if family else ''}"
    body = f"<h2>{esc(title)}</h2>" + proposals_table(proposals, real)
    return title, body


def page_proposal(pid):
    proposals = load_jsonl("proposals.jsonl")
    rec = next((p for p in proposals if p.get("id") == pid), None)
    if rec is None:
        return "Not found", f"<h2>404</h2><p class='muted'>proposal {esc(pid)} not found</p>"

    verify = index_by(load_jsonl("proposals_verification.jsonl"), "id")
    smoke = index_by(load_jsonl("pipeline_smoke_results.jsonl"), "id")
    real = index_by(load_jsonl("tests/results.jsonl"), "id")
    rl = real.get(rec.get("id")) or {}

    spec = rec.get("spec", {}) or {}
    spec_lines = "".join(
        f"<div class='k'>{esc(k)}</div><div>{esc(v)}</div>"
        for k, v in spec.items() if k != "blocks"
    )
    blocks = spec.get("blocks", [])
    block_txt = "<pre>" + esc(json.dumps(blocks, indent=2)) + "</pre>" if blocks else ""

    cites = rec.get("citations", [])
    cite_html = "<ul>" + "".join(
        f"<li><a href='{esc(c.get('url'))}' target='_blank' rel='noopener'>"
        f"{esc(c.get('title'))}</a> <span class='muted'>— {esc(c.get('why'))}</span></li>"
        for c in cites
    ) + "</ul>"

    v = verify.get(pid)
    v_html = ""
    if v:
        checks = v.get("checks", [])
        # verify_proposals.py writes `checks` as a LIST of
        # {field, name, result, reason, refs}; render each.
        if isinstance(checks, dict):
            checks = [{"field": f, "name": f, "result": r, "reason": v2}
                      for f, r, v2 in checks.items()]
        items = "".join(
            f"<li>{badge(c.get('result'))} "
            f"<code>{esc(c.get('field',''))}/{esc(c.get('name',''))}</code> "
            f"— {esc(c.get('reason',''))}</li>"
            for c in checks
        )
        if not items:
            items = "<li class='muted'>no checks recorded</li>"
        v_html = (f"<h2>Verification</h2>{badge(v.get('status'))}"
                  f"<ul class='checks'>{items}</ul>")

    s = smoke.get(pid)
    s_html = ""
    if s:
        s_html = (
            "<h2>Smoke test</h2>"
            f"<div class='detail-grid'>"
            f"<div class='k'>status</div><div>{badge(s.get('status'))}</div>"
            f"<div class='k'>params</div><div>{esc(s.get('params'))}</div>"
            f"<div class='k'>out_shape</div><div>{esc(s.get('out_shape'))}</div>"
            f"<div class='k'>grads_finite</div><div>{esc(s.get('grads_finite'))}</div>"
            f"<div class='k'>test_out_shape</div><div>{esc(s.get('test_out_shape'))}</div>"
            f"<div class='k'>predict</div><div>{esc(s.get('predict'))}</div>"
            f"<div class='k'>message</div><div>{esc(s.get('message'))}</div>"
            f"</div>"
        )

    body = f"""
    <h2>{esc(rec.get('id'))}</h2>
    <div class="detail-grid">
      <div class="k">family</div><div>{esc(rec.get('task_family'))}</div>
      <div class="k">lifecycle</div><div>{badge(rec.get('status'))}</div>
      <div class="k">test status</div><div>{badge(rl.get('status') or 'untested')}</div>
      <div class="k">compile</div><div>{badge(rec.get('compile_status'))}</div>
      <div class="k">created</div><div>{esc(rec.get('created'))}</div>
    </div>
    <h2>Spec</h2>
    <div class="detail-grid">{spec_lines}</div>
    {block_txt}
    <h2>Rationale</h2>
    <p>{esc(rec.get('rationale'))}</p>
    <h2>Citations</h2>
    {cite_html}
    {v_html}
    {s_html}
    """
    return rec.get("id", "Proposal"), body


def page_verification():
    verify = load_jsonl("proposals_verification.jsonl")
    rows = []
    for v in verify:
        checks = v.get("checks", [])
        # verify_proposals.py writes `checks` as a LIST of
        # {field, name, result, reason, refs}; render each.
        if isinstance(checks, dict):
            checks = [{"field": f, "name": f, "result": r, "reason": v2}
                      for f, r, v2 in checks.items()]
        chk = ("<ul class='checks'>" + "".join(
            f"<li>{badge(c.get('result'))} "
            f"<code>{esc(c.get('field',''))}/{esc(c.get('name',''))}</code> "
            f"— {esc(c.get('reason',''))}</li>"
            for c in checks
        ) + "</ul>") if checks else "<span class='muted'>—</span>"
        pid = esc(v.get("id"))
        rows.append(
            f"<tr><td><a href='/proposal/{pid}'>{pid}</a></td>"
            f"<td>{esc(v.get('task_family'))}</td>"
            f"<td>{badge(v.get('status'))}</td><td>{chk}</td></tr>"
        )
    body = ("<h2>Verification results</h2><table><thead><tr><th>ID</th><th>Family</th>"
            "<th>Status</th><th>Checks</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")
    return "Verification", body


def page_smoke():
    smoke = load_jsonl("pipeline_smoke_results.jsonl")

    def _loss(r):
        m = re.search(r"loss=([0-9.]+)", r.get("message", "") or "")
        return float(m.group(1)) if m else None

    def _out_classes(r):
        s = r.get("out_shape") or []
        return s[1] if isinstance(s, list) and len(s) > 1 else None

    dims = [
        ("grads finite", "grads finite", lambda r: r.get("grads_finite")),
        ("out classes", "out classes", _out_classes),
        ("loss", "loss", _loss),
        ("params", "params", lambda r: r.get("params")),
        ("status", "status", lambda r: r.get("status")),
        ("task family", "task family", lambda r: r.get("task_family")),
    ]
    families = sorted({r.get("task_family") for r in smoke})
    fam_palette = ["#58a6ff", "#3fb950", "#d29922", "#bc8cff", "#f778ba",
                   "#56d4dd", "#ff7b72", "#e3b341", "#a371f7"]
    fam_color = {f: fam_palette[i % len(fam_palette)] for i, f in enumerate(families)}

    chart, vlegend, catlegend = pcoord_chart(
        smoke, dims, col_w=90,
        color_by=lambda r: fam_color.get(r.get("task_family"), "#8b949e"),
        note="One line per proposal; colour = task family. Columns alphabetical. "
             "'params' spans 42 to 4.8M (most lines sit low), 'loss' is parsed "
             "from the smoke message. Source: pipeline/smoke_test.py.",
    )

    rows = []
    for s in smoke:
        pid = esc(s.get("id"))
        rows.append(
            f"<tr><td><a href='/proposal/{pid}'>{pid}</a></td>"
            f"<td>{esc(s.get('task_family'))}</td>"
            f"<td>{esc(s.get('params'))}</td>"
            f"<td>{esc(s.get('out_shape'))}</td>"
            f"<td>{esc(s.get('grads_finite'))}</td>"
            f"<td>{esc(s.get('test_out_shape'))}</td>"
            f"<td>{esc(s.get('predict'))}</td>"
            f"<td>{badge(s.get('status'))}</td>"
            f"<td class='muted'>{esc(s.get('message'))}</td></tr>"
        )
    body = ("<h2>Smoke-test results</h2>" + vlegend + chart + catlegend
            + "<table><thead><tr><th>ID</th><th>Family</th>"
            "<th>Params</th><th>Out</th><th>Grads</th><th>Test out</th><th>Predict</th>"
            "<th>Status</th><th>Message</th></tr></thead><tbody>"
            + "".join(rows) + "</tbody></table>")
    return "Smoke Tests", body


def page_mnist():
    allreal = load_jsonl("tests/results.jsonl")
    mnist = [r for r in allreal if r.get("declared_dataset") == "mnist"]
    proposals = index_by(load_jsonl("proposals.jsonl"), "id")

    # Parallel-coordinates line chart: one vertical column per category
    # (architecture choice + training parameter + measured accuracy), sorted
    # alphabetically top-to-bottom. Each MNIST variant is a coloured line that
    # flows through every column, positioned by its raw value (normalised per
    # column). Training parameters are the same for every variant, so those
    # lines run horizontally.
    def first_units(spec):
        blocks = spec.get("blocks", []) if isinstance(spec, dict) else []
        for b in blocks:
            if isinstance(b, dict) and "units" in b:
                return b["units"]
        return None

    # Recognised standard block types. Anything else is treated as a custom
    # layer and bucketed into the "custom" catch-all for the chart.
    STANDARD_BLOCK_TYPES = {
        "linear", "conv", "bottleneck", "coordconv", "dilated",
        "inverted_residual", "basicblock", "boosted_trees", "choice",
        "resonant_spectral_mix",
    }

    def layer_type(spec):
        blocks = spec.get("blocks", []) if isinstance(spec, dict) else []
        if not blocks:
            return "custom"
        b0 = blocks[0]
        t = b0.get("type") if isinstance(b0, dict) else None
        if t in STANDARD_BLOCK_TYPES:
            return t
        return "custom"

    # (key, label, getter) — one column per category. Getters take the MNIST
    # result row and pull architecture choices from the linked proposal spec.
    def rec_units(m):
        p = proposals.get(m.get("id"))
        spec = p.get("spec") if p else None
        return first_units(spec) if isinstance(spec, dict) else None

    def rec_layer(m):
        p = proposals.get(m.get("id"))
        spec = p.get("spec") if p else None
        return layer_type(spec) if isinstance(spec, dict) else "custom"

    def rec_param(m):
        pc = m.get("param_count")
        if pc is not None:
            return pc
        u = rec_units(m)
        return (794 * u + 10) if u else None

    dims = [
        ("epochs", "epochs", lambda m: 3),
        ("first linear units", "first linear units", rec_units),
        ("layer type", "layer type", rec_layer),
        ("learning rate", "learning rate", lambda m: 0.01),
        ("optimizer", "optimizer", lambda m: "SGD"),
        ("train samples", "train samples", lambda m: 2000),
        ("val accuracy", "val accuracy", lambda m: m.get("val_acc")),
        ("val samples", "val samples", lambda m: 1000),
        # measured by tests/test_mnist.py (Option A + B)
        ("parameter count", "parameter count", rec_param),
        ("train loss", "train loss", lambda m: m.get("train_loss")),
        ("val loss", "val loss", lambda m: m.get("val_loss")),
        ("inference time (ms)", "inference time (ms)", lambda m: m.get("inference_ms")),
    ]
    fixed_max = {
        "epochs": 3, "first linear units": 512, "learning rate": 0.01,
        "optimizer": 1, "train samples": 2000, "val accuracy": 1.0,
        "val samples": 1000,
    }
    chart, vlegend, catlegend = pcoord_chart(
        mnist, dims, fixed_max=fixed_max, col_w=90,
        note="Vertical position = raw value (top = nominal max under each "
             "column). Training parameters are identical for every variant, "
             "so those lines run flat. Source: tests/test_mnist.py.",
    )

    # table (full data)
    rows = []
    for m in mnist:
        pid = esc(m.get("id"))
        p = proposals.get(m.get("id")) or {}
        fam = esc(p.get("task_family") or "hpo-mnist")
        rows.append(
            f"<tr><td><a href='/proposal/{pid}'>{pid}</a></td>"
            f"<td>{fam}</td>"
            f"<td>{esc(m.get('declared_dataset'))}</td>"
            f"<td>{badge(m.get('status'))}</td>"
            f"<td>{esc(m.get('val_acc'))}</td>"
            f"<td>{esc(m.get('param_count'))}</td>"
            f"<td>{esc(m.get('train_loss'))}</td>"
            f"<td>{esc(m.get('val_loss'))}</td>"
            f"<td>{esc(m.get('inference_ms'))}</td>"
            f"<td>{badge(m.get('above_chance'))}</td></tr>"
        )
    table = ("<table><thead><tr><th>ID</th>"
             "<th>Family</th><th>Dataset</th><th>Status</th><th>Val acc</th><th>Params</th>"
             "<th>Train loss</th><th>Val loss</th><th>Infer ms</th>"
             "<th>Above chance</th>"
             "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")

    body = (
        "<h2>MNIST real-data results</h2>"
        "<p class='desc'>Handwritten digits (0-9) in 28x28 pixel grayscale images, sorted "
        "into 10 categories. A small image-classification task.</p>"
        + vlegend + chart + catlegend
        + "<h2>Raw data</h2>" + table
    )
    return "MNIST", body


def page_models():
    """All proposals on one parallel-coordinates chart, coloured by model.
    A dropdown filters the chart to a single model (client-side)."""
    proposals = load_jsonl("proposals.jsonl")
    smoke = index_by(load_jsonl("pipeline_smoke_results.jsonl"), "id")
    real = index_by(load_jsonl("tests/results.jsonl"), "id")

    def model_of(p):
        return (p.get("spec") or {}).get("model") or "unknown"

    def dataset_of(p):
        return (p.get("spec") or {}).get("dataset") or "-"

    def task_of(p):
        return (p.get("spec") or {}).get("task_type") or "-"

    def params_of(p):
        sm = smoke.get(p.get("id")) or {}
        rl = real.get(p.get("id")) or {}
        return rl.get("param_count") if rl.get("param_count") is not None else sm.get("params")

    def valacc_of(p):
        rl = real.get(p.get("id")) or {}
        return rl.get("val_acc")

    rows = [{
        "id": p.get("id"),
        "model": model_of(p),
        "dataset": dataset_of(p),
        "task": task_of(p),
        "param_count": params_of(p),
        "val_acc": valacc_of(p),
        "task_family": p.get("task_family"),
    } for p in proposals]

    dims = [
        ("model", "model", lambda r: r["model"]),
        ("dataset", "dataset", lambda r: r["dataset"]),
        ("task", "task", lambda r: r["task"]),
        ("parameter count", "parameter count", lambda r: r["param_count"]),
        ("val accuracy", "val accuracy", lambda r: r["val_acc"]),
    ]
    chart, vlegend, catlegend = pcoord_chart(
        rows, dims, col_w=90, group_by=lambda r: r["model"],
        note="One line per proposal, coloured by model. Use the dropdown to "
             "show a single model. Architecture choices and measured metrics "
             "(where a real-data or MNIST run exists) are the columns. "
             "Source: proposals.jsonl + smoke / real / MNIST results.",
    )

    models = sorted({r["model"] for r in rows})
    opts = "".join(f"<option value='{esc(m)}'>{esc(m)}</option>" for m in models)
    dropdown = (
        "<div class='filterbar'>"
        "<label for='modelFilter'>Filter by model: </label>"
        "<select id='modelFilter' onchange='filterModels()'>"
        f"<option value='__all__'>All models ({len(models)})</option>{opts}"
        "</select></div>"
        "<script>function filterModels(){"
        "var v=document.getElementById('modelFilter').value;"
        "document.querySelectorAll('.pc-line,.pc-dot').forEach(function(e){"
        "if(v==='__all__'){e.style.display='';}"
        "else{e.style.display=(e.getAttribute('data-group')===v)?'':'none';}});}"
        "</script>"
    )

    body = (
        "<h2>All models</h2>"
        "<p class='desc'>Every proposal across all task families on one chart, "
        "coloured and filterable by model architecture.</p>"
        + dropdown + vlegend + chart + catlegend
    )
    return "All models", body


def page_family(fam):
    # hpo-mnist is merged into the MNIST page (/mnist); it has no standalone page.
    if fam == "hpo-mnist":
        return (404, "Not found",
                "<h2>404</h2><p class='muted'>hpo-mnist is shown on the "
                "<a href='/mnist'>MNIST</a> page.</p>")
    proposals = load_jsonl("proposals.jsonl")
    fps = sorted([p for p in proposals if p.get("task_family") == fam],
                 key=lambda p: p.get("id", ""))
    if not fps:
        return "Not found", f"<h2>404</h2><p class='muted'>family {esc(fam)} not found</p>"
    smoke = index_by(load_jsonl("pipeline_smoke_results.jsonl"), "id")
    real = index_by(load_jsonl("tests/results.jsonl"), "id")

    desc = TASK_DESC.get(fam, "")

    STANDARD_BLOCK_TYPES = {
        "linear", "conv", "bottleneck", "coordconv", "dilated",
        "inverted_residual", "basicblock", "boosted_trees", "choice",
        "resonant_spectral_mix",
    }

    def first_block(spec):
        blocks = (spec or {}).get("blocks", []) if isinstance(spec, dict) else []
        return blocks[0] if blocks else {}

    def first_btype(spec):
        t = first_block(spec).get("type")
        return t if t in STANDARD_BLOCK_TYPES else "custom"

    def first_units(spec):
        for k in ("units", "channels", "estimators", "modes"):
            if k in first_block(spec):
                return first_block(spec)[k]
        return None

    def n_blocks(spec):
        blocks = (spec or {}).get("blocks", []) if isinstance(spec, dict) else []
        return len(blocks)

    def tuning(spec):
        for k in ("prune", "quant", "feature_search", "novel"):
            v = (spec or {}).get(k) if isinstance(spec, dict) else None
            if v:
                if isinstance(v, dict):
                    return ", ".join(f"{kk}={vv}" for kk, vv in v.items())
                return str(v)
        return None

    # one enriched row per proposal: proposal fields + its real-data result
    # joined in. All families run through the same runner (tests/
    # run_proposal_tests.py / test_real.py), so run constants are identical
    # across families: SGD lr=1e-2, 3 epochs, 600 train / 200 val samples.
    rows = []
    for p in fps:
        row = dict(p)
        row["_real"] = real.get(p.get("id")) or {}
        row["_smoke"] = smoke.get(p.get("id")) or {}
        rows.append(row)

    def val_acc(p):
        return p.get("_real", {}).get("val_acc")
    def train_loss(p):
        return p.get("_real", {}).get("train_loss")
    def val_loss(p):
        return p.get("_real", {}).get("val_loss")
    def infer_ms(p):
        return p.get("_real", {}).get("inference_ms")
    def param_count(p):
        pc = p.get("_real", {}).get("param_count")
        if pc is not None:
            return pc
        return p.get("_smoke", {}).get("params")

    # Same column shape as page_mnist so every family page reads the same way:
    # run-constants (fixed by the shared runner) then measured metrics, plus
    # architecture-choice columns pulled from each proposal's spec.
    dims = [
        ("epochs",       "epochs",       lambda p: 3),
        ("first units",  "first units",  lambda p: first_units(p.get("spec"))),
        ("layer type",   "layer type",   lambda p: first_btype(p.get("spec"))),
        ("layers",       "layers",       lambda p: n_blocks(p.get("spec"))),
        ("learning rate","learning rate",lambda p: 0.01),
        ("optimizer",    "optimizer",    lambda p: "SGD"),
        ("train samples","train samples", lambda p: 600),
        ("val accuracy", "val accuracy", val_acc),
        ("val samples",  "val samples",  lambda p: 200),
        ("parameter count","parameter count", param_count),
        ("train loss",   "train loss",   train_loss),
        ("val loss",     "val loss",     val_loss),
        ("inference time (ms)", "inference time (ms)", infer_ms),
        ("tuning",       "tuning",       lambda p: tuning(p.get("spec"))),
    ]
    fixed_max = {"epochs": 3, "learning rate": 0.01, "train samples": 600,
                 "val accuracy": 1.0, "val samples": 200}
    chart, vlegend, catlegend = pcoord_chart(
        rows, dims, fixed_max=fixed_max, col_w=90, color_by=lambda p: "#3fb950",
        note="One line per proposal in this family; columns are the run "
             "constants (SGD, 3 epochs, 600/200 split — shared by the real-data "
             "runner) plus measured metrics and architecture choices pulled "
             "from each proposal's spec. Source: proposals.jsonl + tests/results.jsonl.",
    )

    body = (
        f"<h2>{esc(fam)} — {len(fps)} proposals</h2>"
        + (f"<p class='desc'>{esc(desc)}</p>" if desc else "")
        + vlegend + chart + catlegend
        + "<h2>Raw data</h2>" + family_table(fps, smoke, real)
    )
    return fam, body


def family_table(fps, smoke, real):
    """Replicates the MNIST raw-data table layout exactly:
    ID, Family, Dataset, Status, Val acc, Params, Train loss, Val loss,
    Infer ms, Above chance. For proposals that were real-data tested, the
    metrics come from tests/results.jsonl (val_acc, train_loss, val_loss,
    inference_ms, param_count, above_chance). Untested (proposed) rows show
    only what the smoke compile-gate produced (params) and leave the rest
    blank."""
    rows = []
    for p in fps:
        pid = esc(p.get("id"))
        sm = smoke.get(p.get("id")) or {}
        rl = real.get(p.get("id")) or {}
        # actual dataset name: prefer the real-data result's declared_dataset
        # (matches /mnist), fall back to the proposal's spec.dataset, then the
        # smoke modality. Previously read sm.get("modality") which is the data
        # MODALITY (image/text/timeseries), not the dataset NAME — so every
        # family table showed "image" instead of "cifar10" etc.
        ds = (rl.get("declared_dataset")
              or (p.get("spec") or {}).get("dataset")
              or sm.get("modality") or "")
        # prefer real results, fall back to smoke
        val_acc = rl.get("val_acc")
        params = rl.get("param_count") if rl.get("param_count") is not None else sm.get("params")
        train_loss = rl.get("train_loss")
        val_loss = rl.get("val_loss")
        infer_ms = rl.get("inference_ms")
        above = rl.get("above_chance")
        rows.append(
            f"<tr><td><a href='/proposal/{pid}'>{pid}</a></td>"
            f"<td>{esc(p.get('task_family'))}</td>"
            f"<td>{esc(ds)}</td>"
            f"<td>{badge(_test_status(p, real))}</td>"
            f"<td>{esc(val_acc)}</td>"
            f"<td>{esc(params)}</td>"
            f"<td>{esc(train_loss)}</td>"
            f"<td>{esc(val_loss)}</td>"
            f"<td>{esc(infer_ms)}</td>"
            f"<td>{badge(above) if above is not None else ''}</td></tr>"
        )
    return ("<table><thead><tr><th>ID</th><th>Family</th><th>Dataset</th>"
            "<th>Status</th><th>Val acc</th><th>Params</th>"
            "<th>Train loss</th><th>Val loss</th><th>Infer ms</th>"
            "<th>Above chance</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>")


# Task-only descriptions: what the data/task is, with no mention of models,
# architectures, or what varies. Keyed by task_family id in proposals.jsonl.
TASK_DESC = {
    "hpo-mnist": "Handwritten digits (0-9) in 28x28 pixel grayscale images, sorted into 10 "
                 "categories. A small image-classification task.",
    "nas-cifar": "32x32 color photographs of everyday objects (animals, vehicles, and other "
                 "classes) sorted into 10 categories. A small image-classification task.",
    "compression-imagenet": "Natural images from a large set of object categories, sorted by "
                            "class. A large-scale image-classification task.",
    "adv-robust": "Natural images sorted into object categories, where the inputs may be "
                  "visually altered to mislead the classifier. A robustness test on image "
                  "classification.",
    "quant-int8": "Natural images sorted into object categories, represented with reduced "
                  "numeric precision. A low-precision image-classification task.",
    "prune-structured": "Natural images sorted into object categories, with parts of the "
                       "network removed. An image-classification task under sparsity.",
    "feature-eng-tabular": "Rows of mixed-type tabular measurements with a numeric target to "
                           "predict. A tabular regression task.",
    "nas-retiarii": "32x32 color photographs of everyday objects sorted into 10 categories. "
                    "A small image-classification task.",
    "novel-spectral": "Signal and image data processed by a spectral-mixing operation that "
                      "combines frequency-domain and wavelet information. A research task for "
                      "signal and image data.",
    "audio-keyword-mel": "Short spoken commands rendered as log-mel spectrogram images, each "
                         "labeled with a spoken word. A small audio-classification task.",
    "audio-waveform-tcn": "Raw audio waveforms of environmental sounds, each labeled with a "
                          "sound class. An audio-classification task on waveform input.",
    "cluster-image-dec": "Unlabeled 28x28 images grouped into clusters by similarity, with no "
                         "category labels provided. An unsupervised image-clustering task.",
    "cluster-tabular-dec": "Unlabeled rows of numeric feature vectors grouped into clusters by "
                           "similarity, with no category labels provided. An unsupervised "
                           "tabular-clustering task.",
    "text-attention-imdb": "Movie reviews written as text, each labeled positive or negative. "
                           "A binary text-sentiment task using attention over words.",
    "text-cnn-news": "News snippets written as text, each labeled with a topic. A multi-class "
                     "text-topic task over word sequences.",
    "timeseries-har": "Multichannel sensor recordings of body motion over time, each labeled "
                     "with an activity. A human-activity-recognition time-series task.",
    "lenet-mnist": "Handwritten digits (0-9) in 28x28 pixel grayscale images, sorted into 10 "
                   "categories. A small image-classification task.",
}


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    def _send(self, status, body_bytes, content_type="text/html; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body_bytes)))
        self.end_headers()
        self.wfile.write(body_bytes)

    def _json(self, obj):
        self._send(200, json.dumps(obj, indent=2).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        q = parse_qs(parsed.query)
        try:
            if path == "/" or path == "":
                title, body = page_index()
            elif path == "/proposals":
                family = q.get("family", [None])[0]
                title, body = page_proposals(family)
            elif path.startswith("/proposal/"):
                pid = path.split("/proposal/", 1)[1].strip("/")
                title, body = page_proposal(pid)
            elif path == "/verification":
                title, body = page_verification()
            elif path == "/smoke":
                title, body = page_smoke()
            elif path == "/models":
                title, body = page_models()
            elif path == "/mnist":
                title, body = page_mnist()
            elif path.startswith("/family/"):
                fam = path.split("/family/", 1)[1].strip("/")
                res = page_family(fam)
                if isinstance(res, tuple) and len(res) == 3:
                    status, title, body = res
                else:
                    status, title, body = 200, res[0], res[1]
                self._send(status, render(title, body))
                return
            elif path == "/api/proposals":
                self._json(load_jsonl("proposals.jsonl"))
                return
            elif path == "/api/summary":
                proposals = load_jsonl("proposals.jsonl")
                verify = load_jsonl("proposals_verification.jsonl")
                smoke = load_jsonl("pipeline_smoke_results.jsonl")
                self._json({
                    "proposals": len(proposals),
                    "families": sorted({p.get("task_family") for p in proposals}),
                    "verify_status": {v.get("id"): v.get("status") for v in verify},
                    "smoke_status": {s.get("id"): s.get("status") for s in smoke},
                })
                return
            else:
                self._send(404, render("Not found",
                         "<h2>404</h2><p class='muted'>no such page</p>"))
                return
            self._send(200, render(title, body))
        except Exception as e:  # noqa: BLE001
            self._send(500, render("Error",
                     f"<h2>500</h2><pre>{esc(repr(e))}</pre>"))

    def log_message(self, fmt, *args):
        # Quiet access log; errors still go to stderr via BaseHTTPRequestHandler.
        pass


def main():
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"NNI-Test dashboard on http://{HOST}:{PORT}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
