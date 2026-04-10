"""Build the static archive index.

Walks the runs/ directory, regenerates each run's view.html using the
shared renderer, then produces a top-level runs/index.html with a grid
of run cards (run id, summary stats, preview thumbnail, link to the full
view).

Per-run metrics are embedded in each view.html as a collapsed "stats"
section.  The charts are static SVG produced by a small helper below -
no matplotlib, no runtime dependency beyond stdlib and numpy.

Usage:

    python scripts/build_index.py
    python scripts/build_index.py --runs-root /path/to/runs
    python scripts/build_index.py --run showcase-1-baseline   # single run
    python scripts/build_index.py --no-regen   # index only, skip html regen
"""

import argparse
import contextlib
import json
import math
import pickle
import sys
from pathlib import Path
from typing import Any

from biota.search.archive import Archive
from biota.viz.render import render_archive_page

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = ROOT / "runs"


# ---------------------------------------------------------------------------
# Events parsing
# ---------------------------------------------------------------------------


def _load_events(run_dir: Path) -> list[dict[str, Any]]:
    """Read events.jsonl; return list of event dicts sorted by created_at."""
    path = run_dir / "events.jsonl"
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            with contextlib.suppress(json.JSONDecodeError):
                events.append(json.loads(line))
    events.sort(key=lambda e: e.get("created_at", 0))
    return events


def _load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# SVG chart helpers
# No matplotlib.  We produce SVG strings directly.
# ---------------------------------------------------------------------------

_SVG_W = 320
_SVG_H = 160
_PAD_L = 36
_PAD_R = 12
_PAD_T = 12
_PAD_B = 28
_PLOT_W = _SVG_W - _PAD_L - _PAD_R
_PLOT_H = _SVG_H - _PAD_T - _PAD_B

# Color palette matching the dark theme
_C_INSERTED = "#fbbf77"
_C_REPLACED = "#c084fc"
_C_REJ_QUAL = "#f87171"
_C_REJ_SIM = "#60a5fa"
_C_REJ_FILT = "#6b7280"
_C_GRID = "rgba(255,255,255,0.06)"
_C_AXIS = "#3f3f46"
_C_LABEL = "#71717a"


def _svg_open(title: str = "") -> str:
    title_el = f"<title>{title}</title>" if title else ""
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{_SVG_W}" height="{_SVG_H}" '
        f'viewBox="0 0 {_SVG_W} {_SVG_H}">'
        f"{title_el}"
    )


def _svg_close() -> str:
    return "</svg>"


def _px(v: float, data_min: float, data_max: float, axis: str) -> float:
    """Map a data value to a pixel coordinate within the plot area."""
    span = data_max - data_min if data_max != data_min else 1.0
    frac = (v - data_min) / span
    if axis == "x":
        return _PAD_L + frac * _PLOT_W
    else:
        # y axis: data max maps to top (small pixel y)
        return _PAD_T + (1.0 - frac) * _PLOT_H


def _fmt_label(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    if v == int(v):
        return str(int(v))
    return f"{v:.2f}"


def _gridlines_y(y_max: float) -> list[float]:
    """Pick 3-4 round y tick values up to y_max."""
    if y_max <= 0:
        return [0.0]
    magnitude = 10 ** math.floor(math.log10(y_max))
    step = magnitude
    for candidate in (magnitude * 0.25, magnitude * 0.5, magnitude, magnitude * 2):
        if y_max / candidate <= 5:
            step = candidate
            break
    ticks: list[float] = []
    v = 0.0
    while v <= y_max * 1.01:
        ticks.append(v)
        v += step
    return ticks[:6]


def _axis_box() -> str:
    """Draw the plot-area border lines."""
    x0, y0 = _PAD_L, _PAD_T
    x1 = _PAD_L + _PLOT_W
    y1 = _PAD_T + _PLOT_H
    return (
        f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" '
        f'stroke="{_C_AXIS}" stroke-width="1"/>'
        f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" '
        f'stroke="{_C_AXIS}" stroke-width="1"/>'
    )


def _y_gridlines(y_max: float) -> str:
    ticks = _gridlines_y(y_max)
    parts: list[str] = []
    for t in ticks:
        y = _px(t, 0, y_max, "y")
        x0, x1 = _PAD_L, _PAD_L + _PLOT_W
        parts.append(
            f'<line x1="{x0}" y1="{y:.1f}" x2="{x1}" y2="{y:.1f}" '
            f'stroke="{_C_GRID}" stroke-width="1"/>'
        )
        label = _fmt_label(t)
        parts.append(
            f'<text x="{_PAD_L - 4}" y="{y + 4:.1f}" '
            f'text-anchor="end" font-size="9" fill="{_C_LABEL}">{label}</text>'
        )
    return "".join(parts)


def _svg_line_chart(
    series: dict[str, list[float]],
    x_values: list[float],
    y_max: float | None = None,
    title: str = "",
) -> str:
    """Multi-line chart.  series maps label -> list of y values aligned to x_values."""
    colors = {
        "inserted": _C_INSERTED,
        "replaced": _C_REPLACED,
        "rej:qual": _C_REJ_QUAL,
        "rej:sim": _C_REJ_SIM,
        "rej:filt": _C_REJ_FILT,
        "cells/min": _C_INSERTED,
    }
    if not x_values or not series:
        return _svg_open(title) + _svg_close()

    x_min, x_max = min(x_values), max(x_values)
    all_y = [v for vals in series.values() for v in vals]
    computed_y_max = max(all_y) if all_y else 1.0
    if y_max is None:
        y_max = computed_y_max if computed_y_max > 0 else 1.0
    # y_max is float from here on; assertion satisfies pyright's narrowing.
    assert isinstance(y_max, float)

    parts: list[str] = [_svg_open(title), _y_gridlines(y_max), _axis_box()]

    for label, y_vals in series.items():
        color = colors.get(label, "#a1a1aa")
        points: list[str] = []
        for xv, yv in zip(x_values, y_vals, strict=False):
            px = _px(xv, x_min, x_max, "x")
            py = _px(min(yv, y_max), 0, y_max, "y")
            points.append(f"{px:.1f},{py:.1f}")
        if points:
            parts.append(
                f'<polyline points="{" ".join(points)}" '
                f'fill="none" stroke="{color}" stroke-width="1.5" '
                f'stroke-linejoin="round" stroke-linecap="round"/>'
            )

    # X-axis label: first and last rollout index
    x_label_y = _PAD_T + _PLOT_H + 18
    parts.append(f'<text x="{_PAD_L}" y="{x_label_y}" font-size="9" fill="{_C_LABEL}">0</text>')
    parts.append(
        f'<text x="{_PAD_L + _PLOT_W}" y="{x_label_y}" '
        f'text-anchor="end" font-size="9" fill="{_C_LABEL}">{int(x_max)}</text>'
    )
    if title:
        parts.append(
            f'<text x="{_PAD_L + _PLOT_W // 2}" y="{_PAD_T - 2}" '
            f'text-anchor="middle" font-size="9" fill="{_C_LABEL}">{title}</text>'
        )

    parts.append(_svg_close())
    return "".join(parts)


def _svg_histogram(
    values: list[float],
    n_bins: int = 20,
    title: str = "",
    color: str = _C_INSERTED,
) -> str:
    """Simple histogram of quality values in [0, 1]."""
    if not values:
        return _svg_open(title) + _svg_close()

    lo, hi = 0.0, 1.0
    bin_w = (hi - lo) / n_bins
    counts = [0] * n_bins
    for v in values:
        idx = min(int((v - lo) / bin_w), n_bins - 1)
        counts[idx] += 1
    count_max = max(counts) if counts else 1

    parts: list[str] = [_svg_open(title), _y_gridlines(float(count_max)), _axis_box()]

    bar_w = _PLOT_W / n_bins
    for i, c in enumerate(counts):
        x = _PAD_L + i * bar_w
        bar_h = (c / count_max) * _PLOT_H if count_max > 0 else 0
        y = _PAD_T + _PLOT_H - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" '
            f'width="{bar_w - 1:.1f}" height="{bar_h:.1f}" '
            f'fill="{color}" opacity="0.8"/>'
        )

    x_label_y = _PAD_T + _PLOT_H + 18
    parts.append(f'<text x="{_PAD_L}" y="{x_label_y}" font-size="9" fill="{_C_LABEL}">0</text>')
    parts.append(
        f'<text x="{_PAD_L + _PLOT_W}" y="{x_label_y}" '
        f'text-anchor="end" font-size="9" fill="{_C_LABEL}">1</text>'
    )
    if title:
        parts.append(
            f'<text x="{_PAD_L + _PLOT_W // 2}" y="{_PAD_T - 2}" '
            f'text-anchor="middle" font-size="9" fill="{_C_LABEL}">{title}</text>'
        )
    parts.append(_svg_close())
    return "".join(parts)


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

_WINDOW = 50  # rolling window size for rate charts


def _compute_metrics(events: list[dict[str, Any]], started_at: float) -> dict[str, Any]:
    """Derive per-run metrics from the event stream."""
    if not events:
        return {}

    statuses = [e.get("insertion_status", "") for e in events]
    n = len(events)

    totals = {
        "inserted": statuses.count("inserted"),
        "replaced": statuses.count("replaced"),
        "rejected_filter": statuses.count("rejected_filter"),
        "rejected_quality": statuses.count("rejected_quality"),
        "rejected_sim": statuses.count("rejected_sim"),
    }

    # Elapsed time from run start
    t_end = events[-1].get("created_at", started_at)
    wall_seconds = t_end - started_at if started_at else 0.0

    # Cells per minute over rolling windows
    cells_per_min: list[float] = []
    rollout_indices: list[float] = []
    for i in range(0, n, max(1, _WINDOW // 4)):
        lo = max(0, i - _WINDOW // 2)
        hi = min(n, i + _WINDOW // 2)
        window_events = events[lo:hi]
        if len(window_events) < 2:
            continue
        t0 = window_events[0].get("created_at", 0)
        t1 = window_events[-1].get("created_at", 0)
        dt = t1 - t0
        ins = sum(1 for e in window_events if e.get("insertion_status") in ("inserted", "replaced"))
        rate = (ins / dt * 60) if dt > 0 else 0.0
        cells_per_min.append(rate)
        rollout_indices.append(float(i))

    # Rolling rejection breakdown (fraction of window)
    rej_series: dict[str, list[float]] = {
        "inserted": [],
        "replaced": [],
        "rej:qual": [],
        "rej:sim": [],
        "rej:filt": [],
    }
    rej_x: list[float] = []
    for i in range(0, n, max(1, _WINDOW // 4)):
        lo = max(0, i - _WINDOW // 2)
        hi = min(n, i + _WINDOW // 2)
        window = events[lo:hi]
        wn = len(window)
        if wn == 0:
            continue
        rej_x.append(float(i))
        rej_series["inserted"].append(
            sum(1 for e in window if e.get("insertion_status") == "inserted") / wn
        )
        rej_series["replaced"].append(
            sum(1 for e in window if e.get("insertion_status") == "replaced") / wn
        )
        rej_series["rej:qual"].append(
            sum(1 for e in window if e.get("insertion_status") == "rejected_quality") / wn
        )
        rej_series["rej:sim"].append(
            sum(1 for e in window if e.get("insertion_status") == "rejected_sim") / wn
        )
        rej_series["rej:filt"].append(
            sum(1 for e in window if e.get("insertion_status") == "rejected_filter") / wn
        )

    # Quality distribution of inserted/replaced cells
    quality_values = [
        e["quality"]
        for e in events
        if e.get("insertion_status") in ("inserted", "replaced") and e.get("quality") is not None
    ]

    return {
        "n_rollouts": n,
        "totals": totals,
        "wall_seconds": wall_seconds,
        "cells_per_min": cells_per_min,
        "cells_per_min_x": rollout_indices,
        "rej_series": rej_series,
        "rej_x": rej_x,
        "quality_values": quality_values,
    }


def _build_metrics_html(metrics: dict[str, Any]) -> str:
    """Return the HTML for the collapsed stats section to embed in a view.html."""
    if not metrics:
        return ""

    totals = metrics["totals"]
    n = metrics["n_rollouts"]
    wall = metrics["wall_seconds"]
    wall_str = f"{wall / 60:.1f} min" if wall >= 60 else f"{wall:.0f} s"

    total_inserted = totals["inserted"] + totals["replaced"]
    insert_pct = f"{100 * total_inserted / n:.1f}%" if n else "-"
    rej_qual_pct = f"{100 * totals['rejected_quality'] / n:.1f}%" if n else "-"
    rej_sim_pct = f"{100 * totals['rejected_sim'] / n:.1f}%" if n else "-"
    rej_filt_pct = f"{100 * totals['rejected_filter'] / n:.1f}%" if n else "-"

    # Build SVGs
    rej_svg = (
        _svg_line_chart(
            metrics["rej_series"],
            metrics["rej_x"],
            y_max=1.0,
            title="rejection breakdown (rolling)",
        )
        if metrics["rej_x"]
        else ""
    )

    cpm_svg = (
        _svg_line_chart(
            {"cells/min": metrics["cells_per_min"]},
            metrics["cells_per_min_x"],
            title="archive growth rate",
        )
        if metrics["cells_per_min_x"]
        else ""
    )

    qual_svg = (
        _svg_histogram(
            metrics["quality_values"],
            title="quality distribution",
            color=_C_INSERTED,
        )
        if metrics["quality_values"]
        else ""
    )

    # Legend for rejection chart
    legend_items = [
        (_C_INSERTED, "inserted"),
        (_C_REPLACED, "replaced"),
        (_C_REJ_QUAL, "rej:qual"),
        (_C_REJ_SIM, "rej:sim"),
        (_C_REJ_FILT, "rej:filt"),
    ]
    legend_html = "".join(
        f'<span class="stats-legend-item">'
        f'<span class="stats-legend-dot" style="background:{c}"></span>{label}'
        f"</span>"
        for c, label in legend_items
    )

    return f"""
<div class="stats-section" id="stats-section">
    <button class="stats-toggle" type="button"
            onclick="document.getElementById('stats-section').classList.toggle('open')"
            aria-expanded="false">
        <span class="stats-toggle-label">run stats</span>
        <span class="stats-toggle-arrow">&#9660;</span>
    </button>
    <div class="stats-body">
        <div class="stats-summary">
            <div class="stats-kv"><span class="stats-k">rollouts</span>
                <span class="stats-v">{n}</span></div>
            <div class="stats-kv"><span class="stats-k">wall time</span>
                <span class="stats-v">{wall_str}</span></div>
            <div class="stats-kv"><span class="stats-k">insertion rate</span>
                <span class="stats-v">{insert_pct}</span></div>
            <div class="stats-kv"><span class="stats-k">rej:qual</span>
                <span class="stats-v">{rej_qual_pct}</span></div>
            <div class="stats-kv"><span class="stats-k">rej:sim</span>
                <span class="stats-v">{rej_sim_pct}</span></div>
            <div class="stats-kv"><span class="stats-k">rej:filt</span>
                <span class="stats-v">{rej_filt_pct}</span></div>
        </div>
        <div class="stats-charts">
            <div class="stats-chart">{rej_svg}</div>
            <div class="stats-chart">{cpm_svg}</div>
            <div class="stats-chart">{qual_svg}</div>
        </div>
        <div class="stats-legend">{legend_html}</div>
    </div>
</div>
"""


_STATS_CSS = """
/* === stats section === */
.stats-section {
    border-top: 1px solid rgba(255,255,255,0.05);
    margin-top: 16px;
}
.stats-toggle {
    width: 100%;
    background: none;
    border: none;
    color: #71717a;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
    padding: 14px 48px;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 8px;
    font-family: inherit;
    text-align: left;
}
.stats-toggle:hover { color: #a1a1aa; }
.stats-toggle-arrow { transition: transform 0.2s ease; }
.stats-section.open .stats-toggle-arrow { transform: rotate(180deg); }
.stats-body {
    display: none;
    padding: 0 48px 32px 48px;
}
.stats-section.open .stats-body { display: block; }
.stats-summary {
    display: flex;
    flex-wrap: wrap;
    gap: 12px 32px;
    margin-bottom: 20px;
}
.stats-kv {
    display: flex;
    flex-direction: column;
    gap: 2px;
}
.stats-k {
    font-size: 9px;
    color: #52525b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.stats-v {
    font-size: 13px;
    color: #e4e4e7;
    font-variant-numeric: tabular-nums;
}
.stats-charts {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-bottom: 12px;
}
.stats-chart svg {
    display: block;
    background: rgba(0,0,0,0.2);
    border-radius: 4px;
    border: 1px solid rgba(255,255,255,0.04);
}
.stats-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 8px 16px;
}
.stats-legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 9px;
    color: #71717a;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.stats-legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}
"""


def _inject_stats(html: str, metrics_html: str, stats_css: str) -> str:
    """Inject the stats section and CSS into a rendered archive page.

    Inserts the CSS before </style> and the stats HTML before </body>.
    Uses simple string replacement - the renderer produces known marker
    strings we can anchor on.
    """
    if not metrics_html:
        return html
    html = html.replace("</style>", stats_css + "\n    </style>", 1)
    html = html.replace("</body>", metrics_html + "\n</body>", 1)
    return html


# ---------------------------------------------------------------------------
# Per-run rendering
# ---------------------------------------------------------------------------


def _render_run(run_dir: Path, archive: Archive) -> str:
    """Render a full view.html for a run, including the stats section."""
    run_id = run_dir.name
    events = _load_events(run_dir)
    manifest = _load_manifest(run_dir)
    started_at = manifest.get("started_at", 0.0)

    html = render_archive_page(archive, run_id, run_dir)

    if events:
        metrics = _compute_metrics(events, started_at)
        metrics_html = _build_metrics_html(metrics)
        html = _inject_stats(html, metrics_html, _STATS_CSS)

    return html


def _load_archive(run_dir: Path) -> Archive | None:
    pkl = run_dir / "archive.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Archive):
        return None
    return obj


# ---------------------------------------------------------------------------
# Index page
# ---------------------------------------------------------------------------


def _first_thumbnail_data_url(archive: Archive) -> str | None:
    """Return the data URL of the first occupied cell's thumbnail, or None."""
    import base64
    import io

    import imageio.v3 as iio

    from biota.viz.colormap import apply_magma

    for _coord, result in archive.iter_occupied():
        colored = apply_magma(result.thumbnail)
        # Use the middle frame
        frame = colored[len(colored) // 2]
        buf = io.BytesIO()
        iio.imwrite(buf, frame, extension=".png")
        enc = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{enc}"
    return None


def _build_run_card(run_dir: Path, archive: Archive, events: list[dict[str, Any]]) -> str:
    """Build one run card for the index page."""
    run_id = run_dir.name
    n_cells = len(archive)
    fill_pct = f"{archive.fill_fraction * 100:.1f}%"
    n_rollouts = len(events)

    thumb = _first_thumbnail_data_url(archive)
    thumb_html = (
        f'<img class="card-thumb" src="{thumb}" alt="preview" />'
        if thumb
        else '<div class="card-thumb card-thumb-empty"></div>'
    )

    insertion_rate = ""
    if n_rollouts:
        n_inserted = sum(1 for e in events if e.get("insertion_status") in ("inserted", "replaced"))
        insertion_rate = f"{100 * n_inserted / n_rollouts:.1f}%"

    return f"""
<a class="run-card" href="{run_id}/view.html">
    {thumb_html}
    <div class="card-body">
        <div class="card-id">{run_id}</div>
        <div class="card-stats">
            <span class="card-stat"><span class="card-stat-v">{n_cells}</span> cells</span>
            <span class="card-sep">&middot;</span>
            <span class="card-stat"><span class="card-stat-v">{fill_pct}</span> fill</span>
            {f'<span class="card-sep">&middot;</span><span class="card-stat"><span class="card-stat-v">{insertion_rate}</span> insertion</span>' if insertion_rate else ""}
        </div>
        <div class="card-rollouts">{n_rollouts} rollouts</div>
    </div>
</a>"""


def _build_index_html(run_cards: list[str]) -> str:
    cards_html = "\n".join(run_cards)
    n = len(run_cards)
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>biota runs</title>
    <style>
        *, *::before, *::after {{ box-sizing: border-box; }}
        html, body {{
            margin: 0; padding: 0;
            background: #08090b;
            color: #e4e4e7;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            min-height: 100vh;
            -webkit-font-smoothing: antialiased;
        }}
        body {{
            background: radial-gradient(
                ellipse at 50% 25%,
                #141820 0%, #0b0d10 45%, #08090b 100%
            );
            background-attachment: fixed;
            padding: 64px 48px 96px 48px;
        }}
        .page-header {{ margin-bottom: 48px; }}
        .page-title {{
            font-size: 32px;
            font-weight: 300;
            letter-spacing: -0.01em;
            color: #f4f4f5;
            margin: 0 0 8px 0;
        }}
        .page-title .title-prefix {{
            color: #52525b;
            font-weight: 300;
            margin-right: 10px;
        }}
        .page-meta {{
            font-size: 12px;
            color: #71717a;
        }}
        .run-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
            gap: 20px;
        }}
        .run-card {{
            background: rgba(20, 22, 28, 0.6);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 10px;
            overflow: hidden;
            text-decoration: none;
            color: inherit;
            display: flex;
            flex-direction: column;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }}
        .run-card:hover {{
            border-color: rgba(251,191,119,0.3);
            box-shadow:
                0 0 0 1px rgba(251,191,119,0.15),
                0 8px 24px rgba(0,0,0,0.4);
        }}
        .card-thumb {{
            width: 100%;
            aspect-ratio: 1;
            object-fit: cover;
            image-rendering: pixelated;
            display: block;
            background: #0b0d10;
        }}
        .card-thumb-empty {{
            background: #0b0d10;
        }}
        .card-body {{
            padding: 14px 16px 16px 16px;
        }}
        .card-id {{
            font-size: 12px;
            color: #f4f4f5;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
            margin-bottom: 8px;
            word-break: break-all;
        }}
        .card-stats {{
            font-size: 11px;
            color: #71717a;
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 0 4px;
            margin-bottom: 4px;
        }}
        .card-stat-v {{
            color: #a1a1aa;
            font-weight: 500;
            font-variant-numeric: tabular-nums;
        }}
        .card-sep {{ color: #3f3f46; }}
        .card-rollouts {{
            font-size: 10px;
            color: #52525b;
            font-variant-numeric: tabular-nums;
        }}
        @media (max-width: 600px) {{
            body {{ padding: 32px 20px 64px 20px; }}
            .page-title {{ font-size: 24px; }}
        }}
    </style>
</head>
<body>
    <header class="page-header">
        <h1 class="page-title">
            <span class="title-prefix">biota</span>runs
        </h1>
        <div class="page-meta">{n} run{"s" if n != 1 else ""}</div>
    </header>
    <main>
        <div class="run-grid">
            {cards_html}
        </div>
    </main>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _discover_runs(runs_root: Path) -> list[Path]:
    """Return run dirs that have an archive.pkl, sorted newest-first."""
    if not runs_root.exists():
        return []
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "archive.pkl").exists()]
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Regenerate only this single run (by directory name).",
    )
    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="Skip regenerating per-run view.html; only rebuild index.html.",
    )
    args = parser.parse_args()

    if not args.runs_root.exists():
        print(f"error: runs directory not found: {args.runs_root}", file=sys.stderr)
        sys.exit(1)

    if args.run:
        run_dirs = [args.runs_root / args.run]
        if not run_dirs[0].exists():
            print(f"error: run not found: {run_dirs[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        run_dirs = _discover_runs(args.runs_root)

    if not run_dirs:
        print("no runs with archive.pkl found", file=sys.stderr)
        sys.exit(1)

    # Regenerate per-run view.html files
    good_runs: list[tuple[Path, Archive]] = []
    for run_dir in run_dirs:
        archive = _load_archive(run_dir)
        if archive is None:
            print(f"  skip {run_dir.name}: no archive.pkl or wrong type")
            continue

        if not args.no_regen:
            try:
                html = _render_run(run_dir, archive)
                out = run_dir / "view.html"
                out.write_text(html)
                size_kb = out.stat().st_size / 1024
                print(f"  {run_dir.name}: {len(archive)} cells, {size_kb:.0f} KB")
            except Exception as exc:
                print(f"  {run_dir.name}: FAILED - {exc}", file=sys.stderr)
                continue

        good_runs.append((run_dir, archive))

    if args.run:
        print(f"regenerated {args.run}")
        return

    # Build the index
    cards: list[str] = []
    for run_dir, archive in good_runs:
        events = _load_events(run_dir)
        cards.append(_build_run_card(run_dir, archive, events))

    index_html = _build_index_html(cards)
    index_out = args.runs_root / "index.html"
    index_out.write_text(index_html)
    size_kb = index_out.stat().st_size / 1024
    print(f"index: {index_out} ({size_kb:.0f} KB, {len(cards)} runs)")
    print(f"open with: open {index_out}")


if __name__ == "__main__":
    main()
