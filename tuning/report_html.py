"""Self-contained HTML report for the tuner — one file you open in a browser.

This is **pure assembly** over artifacts the tuner already produced: it reads
the ``result`` dict from :func:`tuning.tuner.run_tuning`, the ``fig_*.png`` that
:func:`tuning.report.write_report` wrote into the run folder, and (optionally)
the full-recording validation output, and emits a single ``report.html`` with
every figure base64-inlined, a click-to-sort candidate table, and a side-by-side
recommended-vs-lowthr comparison. No matplotlib, no figure regeneration, stdlib
only — the Markdown report (``report.md``) stays as the terminal/GitHub view.
"""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path

from tuning.report import METRICS_BLURB, SYMPTOM_CAUSE_KNOB

# Figures written by ``tuning.report.write_report``, grouped + captioned in the
# order they should appear. Only those present on disk are embedded.
_FIG_GROUPS = [
    ("Stage 1 — Motion correction", [
        ("fig_mc_gsig.png", "Neuron radius (temporal-std + blob_log) → mc_gSig_filt / sigma"),
        ("fig_max_shift.png", "Per-frame shift histogram → max_shift / border_px"),
        ("fig_downsample.png", "Downsample candidates (ssub: FWHM≥4 px, tsub: dt≤decay/2)"),
    ]),
    ("Stage 3 — Initialisation", [
        ("fig_sigma.png", "CORR / PNR / CORR·PNR + top blobs → extraction sigma"),
        ("fig_corr_pnr.png", "Seed-count surface over (min_corr, min_pnr) → thresholds"),
        ("fig_min_pixel.png", "Footprint-area distribution → min_pixel"),
    ]),
    ("Stage 4 — Temporal / merge / evaluation", [
        ("fig_decay.png", "AR(1) g + per-component decay τ"),
        ("fig_g_prior.png", "Yule-Walker g vs physical target → g_prior_weight"),
        ("fig_merge.png", "Pairwise trace correlations → merge_thr_corr"),
        ("fig_snr_eval.png", "SNR distribution + ghost-vs-real footprint thumbnails"),
    ]),
    ("Sweep", [
        ("fig_sweep_scatter.png", "Density↔purity: K vs corr(C, C+YrA), size=accepted, ★=best"),
        ("fig_sweep_footprints.png", "Best-candidate footprints over the correlation image"),
        ("fig_sweep_traces.png", "Best candidate: C (blue) vs C+YrA (grey)"),
    ]),
]

_SWEEP_COLS = ["idx", "sigma", "min_corr", "min_pnr", "merge_thr_corr",
               "global_bg_rank", "init_stride", "K", "K_accepted",
               "cprojcorr_median", "npix_median", "multipeak_frac", "npix_oversize",
               "snr_median", "score", "wall_s"]

_CSS = """
:root { color-scheme: light dark; }
body { font: 15px/1.5 -apple-system, Segoe UI, Roboto, sans-serif; margin: 0 auto;
       max-width: 1100px; padding: 24px; }
h1 { margin-top: 0; } h2 { border-bottom: 1px solid #8884; padding-bottom: 4px; margin-top: 2em; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #8884; padding: 4px 8px; text-align: right; }
th:first-child, td:first-child { text-align: left; }
thead th { cursor: pointer; user-select: none; background: #8881; position: sticky; top: 0; }
thead th:hover { background: #8883; }
tr.best td { background: #ffd70033; font-weight: 600; }
code { background: #8882; padding: 1px 4px; border-radius: 3px; }
img { max-width: 100%; height: auto; border: 1px solid #8883; border-radius: 4px; }
.fig { margin: 14px 0; } .fig .cap { font-size: 13px; color: #888; margin-bottom: 4px; }
.row { display: flex; flex-wrap: wrap; gap: 16px; } .row > div { flex: 1 1 380px; }
.muted { color: #888; } .tag { font-size: 12px; color: #888; }
"""

_SORT_JS = """
function sortTable(th){
  var table=th.closest('table'), tb=table.tBodies[0];
  var idx=Array.prototype.indexOf.call(th.parentNode.children, th);
  var dir=th.dataset.dir==='asc'?'desc':'asc'; th.dataset.dir=dir;
  var rows=Array.prototype.slice.call(tb.rows);
  rows.sort(function(a,b){
    var x=a.cells[idx].textContent.trim(), y=b.cells[idx].textContent.trim();
    var nx=parseFloat(x), ny=parseFloat(y);
    var both=!isNaN(nx)&&!isNaN(ny);
    var c = both ? nx-ny : x.localeCompare(y);
    return dir==='asc'?c:-c;
  });
  rows.forEach(function(r){ tb.appendChild(r); });
}
"""


def _png_b64(path) -> "str | None":
    """Read a PNG and return a ``data:image/png;base64,...`` URI, or None."""
    p = Path(path)
    if not p.exists():
        return None
    return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _img(path, caption: str = "") -> str:
    uri = _png_b64(path)
    if uri is None:
        return ""
    cap = f'<div class="cap">{html.escape(caption)}</div>' if caption else ""
    return f'<div class="fig">{cap}<img src="{uri}" alt="{html.escape(caption)}"></div>'


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}" if abs(v) < 1e6 else f"{v:.1e}"
    return "—" if v is None else str(v)


def _sortable_table(rows, cols, *, highlight_idx: int = 0) -> str:
    """An HTML ``<table>`` with click-to-sort headers; row ``highlight_idx`` is
    tagged ``class="best"``."""
    head = "".join(f'<th onclick="sortTable(this)">{html.escape(c)}</th>' for c in cols)
    body = []
    for i, r in enumerate(rows):
        cls = ' class="best"' if i == highlight_idx else ""
        cells = "".join(f"<td>{html.escape(_fmt(r.get(c)))}</td>" for c in cols)
        body.append(f"<tr{cls}>{cells}</tr>")
    return (f'<table><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def _md_table_to_html(md: str) -> str:
    """Render a GitHub pipe-table (with `code` / **bold** inline) to an HTML
    table. Non-table lines become paragraphs. Minimal by design."""
    out, rows = [], []

    def flush():
        if not rows:
            return
        head, *rest = rows
        rest = [r for r in rest if not set(r) <= set("-: |")]  # drop separator row
        th = "".join(f"<th>{_inline(c)}</th>" for c in _cells(head))
        trs = "".join("<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in _cells(r)) + "</tr>"
                      for r in rest)
        out.append(f"<table><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table>")
        rows.clear()

    for line in md.splitlines():
        if line.strip().startswith("|"):
            rows.append(line)
        else:
            flush()
            if line.strip():
                out.append(f"<p>{_inline(line)}</p>")
    flush()
    return "\n".join(out)


def _cells(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def _inline(text: str) -> str:
    """Escape, then apply `code` and **bold** inline markdown."""
    import re
    s = html.escape(text)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def _section(title: str, body: str) -> str:
    return f"<h2>{html.escape(title)}</h2>\n{body}" if body else ""


def write_html_report(run_dir, result: dict, validation: "dict | None" = None,
                      *, title: "str | None" = None) -> Path:
    """Assemble ``run_dir/report.html`` from ``result`` + on-disk figures.

    Args:
        run_dir: the tuner run folder (already holds ``fig_*.png`` + JSONs).
        result: the dict returned by :func:`tuning.tuner.run_tuning`.
        validation: optional dict from :func:`tuning.validate.validate_session`
            (``rows`` + ``out_dir``); embeds the per-run figures + comparison.
        title: report ``<h1>`` (defaults to the session name).

    Returns the ``report.html`` path.
    """
    run_dir = Path(run_dir)
    cfg = result.get("config", {})
    name = title or cfg.get("name", "session")
    parts: list[str] = []

    # -- Run configuration --
    cfg_rows = "".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(str(cfg[k]))}</td></tr>"
        for k in ("input", "input_kind", "mode", "region", "ssub", "tsub",
                  "frame_rate_hz", "decay_time_ms", "n_jobs", "timestamp", "cli")
        if k in cfg)
    parts.append(_section("Run configuration",
                          f"<table><tbody>{cfg_rows}</tbody></table>"))

    # -- Recommended parameters --
    rec = result.get("recommended", {})
    src = result.get("sources", {})
    rat = result.get("rationale", {})
    rec_rows = "".join(
        f"<tr><td><code>{html.escape(k)}</code></td><td>{html.escape(_fmt(rec[k]))}</td>"
        f"<td>{html.escape(str(src.get(k, 'default')))}</td>"
        f"<td>{_inline(str(rat.get(k, '')))}</td></tr>"
        for k in sorted(rec))
    rec_tbl = ("<table><thead><tr><th>param</th><th>value</th><th>source</th>"
               f"<th>rationale</th></tr></thead><tbody>{rec_rows}</tbody></table>")
    rec_tbl += (f'<p class="tag">ssub={result.get("ssub", 1)} tsub={result.get("tsub", 1)} '
                "(written to downsample.json; not CNMFeParams fields)</p>")
    parts.append(_section("Recommended parameters", rec_tbl))

    # -- Sweep candidate table (sortable) --
    sweep = result.get("sweep")
    if sweep and sweep.get("rows"):
        rows = sweep["rows"]
        region = sweep.get("region", "")
        crop = sweep.get("region_crop")
        cap = (f'<p class="tag">region: <strong>{html.escape(str(region))}</strong>'
               + (f" — crop {html.escape(str(crop))}" if crop else "") + "</p>")
        cap += '<p class="muted">Click a column header to sort. Best (★) row highlighted.</p>'
        parts.append(_section("Sweep candidates",
                              cap + _sortable_table(rows, _SWEEP_COLS, highlight_idx=0)))

    # -- Figures (base64-inlined, grouped) --
    seen = set()
    for group_title, figs in _FIG_GROUPS:
        imgs = []
        for fname, caption in figs:
            img = _img(run_dir / fname, caption)
            if img:
                imgs.append(img)
                seen.add(fname)
        if imgs:
            parts.append(_section(group_title, "\n".join(imgs)))
    # any extra fig_*.png not in the known groups
    extra = sorted(p for p in run_dir.glob("fig_*.png") if p.name not in seen)
    if extra:
        parts.append(_section("Other figures",
                              "\n".join(_img(p, p.stem) for p in extra)))

    # -- Full-recording validation --
    if validation:
        parts.append(_validation_section(validation))

    # -- How to read / cheat sheet --
    parts.append(_section("How to read these metrics", f"<p>{_inline(METRICS_BLURB)}</p>"))
    parts.append(_section("Symptom → cause → knob (by-eye troubleshooting)",
                          _md_table_to_html(SYMPTOM_CAUSE_KNOB)))

    doc = (f"<!doctype html><html><head><meta charset='utf-8'>"
           f"<title>CNMFe tuning — {html.escape(str(name))}</title>"
           f"<style>{_CSS}</style></head><body>"
           f"<h1>CNMFe parameter-tuning report — {html.escape(str(name))}</h1>"
           + "\n".join(p for p in parts if p)
           + f"<script>{_SORT_JS}</script></body></html>")
    out = run_dir / "report.html"
    out.write_text(doc, encoding="utf-8")
    return out


def _validation_section(validation: dict) -> str:
    """Side-by-side per-run footprints + each run's figures + comparison table."""
    out_dir = Path(validation.get("out_dir", ""))
    rows = validation.get("rows", [])
    labels = [r["label"] for r in rows] if rows else \
        sorted(p.name[4:] for p in out_dir.glob("run_*") if p.is_dir())

    body = []
    # comparison.md as an HTML table
    comp = out_dir / "comparison.md"
    if comp.exists():
        body.append(_md_table_to_html(comp.read_text()))

    # side-by-side footprints_on_corr for each run
    cols = []
    for lab in labels:
        img = _img(out_dir / f"run_{lab}" / "figs" / "footprints_on_corr.png",
                   f"run_{lab}: footprints on correlation image")
        if img:
            cols.append(f"<div>{img}</div>")
    if cols:
        body.append('<div class="row">' + "".join(cols) + "</div>")

    # the remaining per-run diagnostic figures
    for lab in labels:
        figs_dir = out_dir / f"run_{lab}" / "figs"
        imgs = [_img(figs_dir / n, f"run_{lab}: {n[:-4]}")
                for n in ("traces.png", "npix_dist.png", "snr_eval.png",
                          "mc_shifts.png", "footprint_grid.png", "eccentricity.png",
                          "jaccard_merge.png", "centroid_drift.png")]
        imgs = [i for i in imgs if i]
        if imgs:
            body.append(f"<h3>run_{html.escape(lab)}</h3>" + "\n".join(imgs))

    return _section("Full-recording validation", "\n".join(body))
