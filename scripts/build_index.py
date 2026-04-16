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
    python scripts/build_index.py --output-dir archive
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

from jinja2 import Environment, FileSystemLoader

from biota.search.archive import Archive
from biota.viz.colormap import apply_magma
from biota.viz.render import render_archive_page

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = ROOT / "archive"
DEFAULT_ECO_ROOT = ROOT / "ecosystem"

_TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "src" / "biota" / "viz" / "templates"
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=False,
    keep_trailing_newline=True,
)
_STATS_CSS = (_TEMPLATES_DIR / "stats_section.css").read_text()


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


def _load_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}


# SVG chart helpers - no matplotlib, strings produced directly.
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
    """Render the collapsed stats section HTML from the Jinja2 template."""
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

    legend_items = [
        (_C_INSERTED, "inserted"),
        (_C_REPLACED, "replaced"),
        (_C_REJ_QUAL, "rej:qual"),
        (_C_REJ_SIM, "rej:sim"),
        (_C_REJ_FILT, "rej:filt"),
    ]

    template = _ENV.get_template("stats_section.html")
    return template.render(
        n_rollouts=n,
        wall_str=wall_str,
        insert_pct=insert_pct,
        rej_qual_pct=rej_qual_pct,
        rej_sim_pct=rej_sim_pct,
        rej_filt_pct=rej_filt_pct,
        rej_svg=rej_svg,
        cpm_svg=cpm_svg,
        qual_svg=qual_svg,
        legend_items=legend_items,
    )


def _render_run(run_dir: Path, archive: Archive, publish: bool = False) -> str:
    """Render a full view.html for a run, including the stats section.

    publish=True writes GIF thumbnails as separate files under run_dir/thumbs/
    and references them by relative path. The HTML goes from ~250 MB to ~2 MB.
    publish=False (default) embeds thumbnails as base64 data URLs for a fully
    self-contained file suitable for local viewing without a server.
    """
    run_id = run_dir.name
    events = _load_events(run_dir)
    manifest = _load_manifest(run_dir)
    started_at = manifest.get("started_at", 0.0)

    stats_html = ""
    if events:
        metrics = _compute_metrics(events, started_at)
        stats_html = _build_metrics_html(metrics)

    thumbs_dir = run_dir / "thumbs" if publish else None

    cfg = _load_config(run_dir)
    border: str = cfg.get("rollout", {}).get("sim", {}).get("border", "wall")
    return render_archive_page(
        archive,
        run_id,
        run_dir,
        stats_html=stats_html,
        stats_css=_STATS_CSS if stats_html else "",
        thumbs_dir=thumbs_dir,
        border=border,
    )


def _load_archive(run_dir: Path) -> Archive | None:
    pkl = run_dir / "archive.pkl"
    if not pkl.exists():
        return None
    with open(pkl, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, Archive):
        return None
    return obj


def _first_thumbnail_data_url(archive: Archive) -> str | None:
    """Return the data URL of the first occupied cell's thumbnail, or None."""
    import base64
    import io

    import imageio.v3 as iio

    for _coord, result in archive.iter_occupied():
        colored = apply_magma(result.thumbnail)
        frame = colored[len(colored) // 2]
        buf = io.BytesIO()
        iio.imwrite(buf, frame, extension=".png")
        enc = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{enc}"
    return None


def _build_card_context(
    run_dir: Path, archive: Archive, events: list[dict[str, Any]]
) -> dict[str, Any]:
    """Build the context dict for one run card in the index template."""
    n_rollouts = len(events)
    insertion_rate = ""
    if n_rollouts:
        n_inserted = sum(1 for e in events if e.get("insertion_status") in ("inserted", "replaced"))
        insertion_rate = f"{100 * n_inserted / n_rollouts:.1f}%"

    cfg = _load_config(run_dir)
    sim_cfg = cfg.get("rollout", {}).get("sim", {})
    # Handle both old (grid) and new (grid_h/grid_w) config field names
    grid = sim_cfg.get("grid_h") or sim_cfg.get("grid")
    if grid == 96:
        preset = "dev"
    elif grid == 192:
        preset = "standard"
    elif grid == 384:
        preset = "pretty"
    elif grid is not None:
        preset = f"{grid}px"
    else:
        preset = ""

    budget = cfg.get("budget", "")
    device = cfg.get("device", "")
    descriptor_names: list[str] = cfg.get("descriptor_names", [])
    border: str = cfg.get("rollout", {}).get("sim", {}).get("border", "wall")

    return {
        "run_id": run_dir.name,
        "n_cells": len(archive),
        "fill_pct": f"{archive.fill_fraction * 100:.1f}%",
        "n_rollouts": n_rollouts,
        "thumb_url": _first_thumbnail_data_url(archive),
        "insertion_rate": insertion_rate,
        "preset": preset,
        "budget": budget,
        "device": device,
        "descriptor_names": descriptor_names,
        "border": border,
    }


def _build_index_html(
    cards: list[dict[str, Any]], eco_cards: list[dict[str, Any]] | None = None
) -> str:
    template = _ENV.get_template("index.html")
    # Inline the architecture diagrams so the deployed site is self-contained
    # (no assumption about whether docs/ gets shipped alongside index.html).
    docs_dir = Path(__file__).resolve().parent.parent / "docs"
    svgs: dict[str, str] = {}
    for name in ("archive-grid", "search-loop", "ecosystem-dispatch"):
        path = docs_dir / f"{name}.svg"
        svgs[name.replace("-", "_")] = path.read_text(encoding="utf-8") if path.exists() else ""
    return template.render(
        n_runs=len(cards),
        cards=cards,
        eco_cards=eco_cards or [],
        n_eco_runs=len(eco_cards or []),
        svg_archive_grid=svgs["archive_grid"],
        svg_search_loop=svgs["search_loop"],
        svg_ecosystem_dispatch=svgs["ecosystem_dispatch"],
    )


def _needs_rebuild(run_dir: Path, publish: bool) -> bool:
    """Return True if view.html needs to be regenerated.

    Skips rebuild when view.html exists and is newer than both archive.pkl
    and the index template (so template changes still trigger a rebuild).
    In publish mode, also checks that the thumbs/ directory exists - if GIFs
    were never written the view.html references broken paths.
    """
    view = run_dir / "view.html"
    if not view.exists():
        return True

    view_mtime = view.stat().st_mtime

    # Rebuild if archive was updated after the last render
    archive_pkl = run_dir / "archive.pkl"
    if archive_pkl.stat().st_mtime > view_mtime:
        return True

    # Rebuild if the archive template itself changed
    template_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "biota"
        / "viz"
        / "templates"
        / "archive.html"
    )
    if template_path.exists() and template_path.stat().st_mtime > view_mtime:
        return True

    # In publish mode, rebuild if thumbs directory is missing
    return publish and not (run_dir / "thumbs").exists()


def _discover_runs(runs_root: Path) -> list[Path]:
    """Return run dirs that have an archive.pkl, sorted newest-first."""
    if not runs_root.exists():
        return []
    candidates = [p for p in runs_root.iterdir() if p.is_dir() and (p / "archive.pkl").exists()]
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates


# ============================================================
# === Ecosystem run support ==================================
# ============================================================


def _discover_ecosystem_runs(eco_root: Path) -> list[Path]:
    """Return ecosystem run dirs that have a summary.json, sorted newest-first."""
    if not eco_root.exists():
        return []
    candidates = [p for p in eco_root.iterdir() if p.is_dir() and (p / "summary.json").exists()]
    candidates.sort(key=lambda p: p.name, reverse=True)
    return candidates


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)  # type: ignore[no-any-return]


def _mass_svg_paths(
    mass_history: list[float], width: int = 800, height: int = 80
) -> tuple[str, str]:
    """Build SVG path strings for the mass-over-time chart.

    Returns (area_path, line_path) as SVG 'd' attribute strings.
    """
    if not mass_history or len(mass_history) < 2:
        return "", ""
    n = len(mass_history)
    lo = min(mass_history)
    hi = max(mass_history)
    span = hi - lo if hi > lo else max(abs(lo) * 0.01, 1.0)  # show flat line if constant
    margin = 4

    def px(i: int, v: float) -> tuple[float, float]:
        x = i / (n - 1) * width
        y = height - margin - (v - lo) / span * (height - 2 * margin)
        return x, y

    pts = [px(i, v) for i, v in enumerate(mass_history)]
    line = "M " + " L ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = line + f" L {pts[-1][0]:.1f},{height} L {pts[0][0]:.1f},{height} Z"
    return area, line


def _extract_primary_source(summary: dict[str, Any]) -> tuple[str, list[int], int]:
    """Pull the first source's run_id, coords, and n from a summary dict.

    Handles both the v2.2.0 sources-list shape and the legacy v2.0.x shape
    where source_run_id and source_coords were top-level scalars and n lived
    under spawn. Returns (run_id, coords, n_for_first_source).
    """
    sources = summary.get("sources")
    if isinstance(sources, list) and sources:
        first = sources[0]
        if isinstance(first, dict):
            return (
                str(first.get("run", "")),
                list(first.get("cell", [])),
                int(first.get("n", 0)),
            )
    # Legacy v2.0.x shape
    legacy_n = summary.get("spawn", {}).get("n_creatures", summary.get("spawn", {}).get("n", 0))
    return (
        str(summary.get("source_run_id", "")),
        list(summary.get("source_coords", [])),
        int(legacy_n) if legacy_n != "" else 0,
    )


def _build_sources_context(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Build the per-source list passed to ecosystem.html.

    Each entry has run_id, coords_str ("(y, x, z)" parenthesized for display),
    coords_hash (joined with '-' for archive viewer deep links), and n.
    Falls back to a single-source list reconstructed from legacy fields when
    a v2.0.x summary doesn't have a sources list.
    """
    sources_raw = summary.get("sources")
    if isinstance(sources_raw, list) and sources_raw:
        out: list[dict[str, Any]] = []
        for s in sources_raw:
            if not isinstance(s, dict):
                continue
            cell = s.get("cell", [])
            out.append(
                {
                    "run_id": s.get("run", ""),
                    "coords_str": f"({', '.join(str(c) for c in cell)})",
                    "coords_hash": "-".join(str(c) for c in cell),
                    "n": s.get("n", ""),
                }
            )
        return out
    # Legacy fallback: synthesize a single entry from top-level scalars.
    legacy_run, legacy_coords, legacy_n = _extract_primary_source(summary)
    if not legacy_run:
        return []
    return [
        {
            "run_id": legacy_run,
            "coords_str": f"({', '.join(str(c) for c in legacy_coords)})",
            "coords_hash": "-".join(str(c) for c in legacy_coords),
            "n": legacy_n,
        }
    ]


def _resolve_mode(summary: dict[str, Any]) -> str:
    """Read or infer the ecosystem run mode.

    v2.2.0 summaries carry an explicit 'mode' field. Legacy summaries are
    by definition single-source homogeneous.
    """
    mode = summary.get("mode")
    if isinstance(mode, str) and mode in ("homogeneous", "heterogeneous"):
        return mode
    return "homogeneous"


def _render_ecosystem_run(run_dir: Path, publish: bool = False) -> str:
    """Render an ecosystem run as a self-contained HTML page."""
    import base64
    import io as _io

    import imageio.v3 as iio

    summary = _load_summary(run_dir)
    if not summary:
        raise ValueError(f"no summary.json in {run_dir}")

    run_id = summary.get("run_id", run_dir.name)
    source_run_id, source_coords, n_creatures = _extract_primary_source(summary)
    grid_h = summary.get("grid_h", summary.get("grid", ""))
    grid_w = summary.get("grid_w", summary.get("grid", ""))
    steps = summary.get("steps", "")
    border = summary.get("border", "wall")
    spawn = summary.get("spawn", {})
    min_dist = spawn.get("min_dist", "")
    measures = summary.get("measures", {})
    mass_history: list[float] = measures.get("mass_history", [])
    snapshot_steps: list[int] = measures.get("snapshot_steps", [])
    initial_mass = measures.get("initial_mass", 0.0)
    final_mass = measures.get("final_mass", 0.0)
    peak_mass = measures.get("peak_mass", 0.0)
    mass_turnover = measures.get("mass_turnover", 0.0)
    elapsed_s = summary.get("elapsed_seconds", 0.0)

    # Format display values
    elapsed = f"{elapsed_s:.1f}s"
    initial_mass_fmt = f"{initial_mass:.1f}"
    final_mass_fmt = f"{final_mass:.1f}"
    peak_mass_fmt = f"{peak_mass:.1f}"
    mass_turnover_pct = f"{mass_turnover * 100:.3f}"

    # Build mass chart paths
    mass_area_path, mass_line_path = _mass_svg_paths(mass_history)

    # Build frame list
    THUMB_PX = 192
    frames_dir = run_dir / "frames"
    frames: list[dict[str, Any]] = []

    thumbs_dir = run_dir / "thumbs_eco" if publish else None
    if thumbs_dir is not None:
        thumbs_dir.mkdir(parents=True, exist_ok=True)

    for step in snapshot_steps:
        png_path = frames_dir / f"step_{step:06d}.png"
        if not png_path.exists():
            continue

        if publish and thumbs_dir is not None:
            # Copy frame to thumbs_eco/ and reference by relative path
            import shutil as _shutil

            thumb_path = thumbs_dir / f"step_{step:06d}.png"
            if not thumb_path.exists():
                _shutil.copy2(png_path, thumb_path)
            frames.append({"step": step, "src": f"thumbs_eco/step_{step:06d}.png"})
        else:
            # Embed as base64
            raw = iio.imread(png_path)
            buf = _io.BytesIO()
            iio.imwrite(buf, raw, extension=".png")
            b64 = base64.b64encode(buf.getvalue()).decode("ascii")
            frames.append({"step": step, "src": f"data:image/png;base64,{b64}"})

    output_format = summary.get("output_format", "frames")
    gif_path = run_dir / "ecosystem.gif"

    # For GIF output: embed the GIF directly instead of individual frames
    gif_src: str = ""
    if output_format == "gif" and gif_path.exists():
        if publish:
            # Reference by relative path
            gif_src = "ecosystem.gif"
        else:
            # Embed as base64
            import base64 as _b64

            gif_data = gif_path.read_bytes()
            gif_src = "data:image/gif;base64," + _b64.b64encode(gif_data).decode("ascii")

    # Use snapshot count from summary.json as ground truth - frames may not
    # exist on disk (gif mode only writes ecosystem.gif, not individual frames)
    n_snapshots = len(snapshot_steps) if snapshot_steps else len(frames)
    mode = _resolve_mode(summary)
    sources_ctx = _build_sources_context(summary)
    # n_creatures is the total across all sources for the header summary line.
    total_creatures = sum(int(s.get("n", 0)) for s in sources_ctx) if sources_ctx else n_creatures

    template = _ENV.get_template("ecosystem.html")
    return template.render(
        run_id=run_id,
        mode=mode,
        sources=sources_ctx,
        # Legacy single-source vars kept for the template's else-branch fallback
        # in case a future code path renders without a sources list.
        source_run_id=source_run_id,
        source_coords=", ".join(str(c) for c in source_coords),
        source_coords_hash="-".join(str(c) for c in source_coords),
        grid=f"{grid_h}x{grid_w}",
        steps=steps,
        border=border,
        n_creatures=total_creatures,
        min_dist=min_dist,
        n_snapshots=n_snapshots,
        elapsed=elapsed,
        initial_mass=initial_mass_fmt,
        final_mass=final_mass_fmt,
        peak_mass=peak_mass_fmt,
        mass_turnover_pct=mass_turnover_pct,
        mass_area_path=mass_area_path,
        mass_line_path=mass_line_path,
        frames=frames,
        thumb_px=THUMB_PX,
        output_format=output_format,
        gif_src=gif_src,
    )


def _needs_eco_rebuild(run_dir: Path, publish: bool) -> bool:
    """Return True if ecosystem view.html needs to be regenerated."""
    view = run_dir / "view.html"
    summary = run_dir / "summary.json"
    if not view.exists():
        return True
    view_mtime = view.stat().st_mtime
    if summary.stat().st_mtime > view_mtime:
        return True
    template_path = _TEMPLATES_DIR / "ecosystem.html"
    if template_path.exists() and template_path.stat().st_mtime > view_mtime:
        return True
    return publish and not (run_dir / "thumbs_eco").exists()


def _build_eco_card_context(run_dir: Path) -> dict[str, Any]:
    """Build the context dict for one ecosystem run card."""
    summary = _load_summary(run_dir)
    spawn = summary.get("spawn", {})
    measures = summary.get("measures", {})
    source_run_id, source_coords, n_creatures = _extract_primary_source(summary)
    sources = summary.get("sources", []) if isinstance(summary.get("sources"), list) else []
    mode = summary.get("mode")
    if mode is None:
        # Legacy summaries without 'mode' are by definition single-source homogeneous.
        mode = "homogeneous"
    return {
        "run_id": run_dir.name,
        "name": summary.get("name", ""),
        "mode": mode,
        "source_run_id": source_run_id,
        "source_coords": source_coords,
        "n_sources": len(sources) if sources else 1,
        "grid_h": summary.get("grid_h", summary.get("grid", "")),
        "grid_w": summary.get("grid_w", summary.get("grid", "")),
        "steps": summary.get("steps", ""),
        "border": summary.get("border", "wall"),
        "n_creatures": n_creatures,
        "min_dist": spawn.get("min_dist", ""),
        "initial_mass": f"{measures.get('initial_mass', 0.0):.1f}",
        "mass_turnover": f"{measures.get('mass_turnover', 0.0) * 100:.3f}",
        "elapsed": f"{summary.get('elapsed_seconds', 0.0):.1f}s",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
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
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "Write GIF thumbnails as separate files under runs/<run_id>/thumbs/ "
            "and reference them by relative path. The view.html drops from ~250 MB "
            "to ~2 MB per run. Use this when building for web hosting. "
            "Without this flag, thumbnails are embedded as base64 (self-contained, "
            "works offline, suitable for local viewing)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Rebuild all view.html files even if they are up to date. "
            "By default, runs are skipped when view.html is newer than "
            "archive.pkl and the archive template."
        ),
    )
    parser.add_argument(
        "--ecosystem-dir",
        type=Path,
        default=DEFAULT_ECO_ROOT,
        help="Directory containing ecosystem run subdirectories.",
    )
    args = parser.parse_args()

    if not args.output_dir.exists():
        print(f"error: runs directory not found: {args.output_dir}", file=sys.stderr)
        sys.exit(1)

    if args.run:
        run_dirs = [args.output_dir / args.run]
        if not run_dirs[0].exists():
            print(f"error: run not found: {run_dirs[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        run_dirs = _discover_runs(args.output_dir)

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
            if not args.force and not _needs_rebuild(run_dir, publish=args.publish):
                size_kb = (run_dir / "view.html").stat().st_size / 1024
                print(f"  {run_dir.name}: up to date ({len(archive)} cells, {size_kb:.0f} KB)")
            else:
                try:
                    html = _render_run(run_dir, archive, publish=args.publish)
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
    card_contexts: list[dict[str, Any]] = []
    for run_dir, archive in good_runs:
        events = _load_events(run_dir)
        card_contexts.append(_build_card_context(run_dir, archive, events))

    index_html = _build_index_html(card_contexts)
    index_out = args.output_dir.parent / "index.html"
    index_out.write_text(index_html)
    size_kb = index_out.stat().st_size / 1024
    print(f"index: {index_out} ({size_kb:.0f} KB, {len(card_contexts)} runs)")
    print(f"open with: open {index_out}")

    # === Ecosystem runs ===
    eco_dirs = _discover_ecosystem_runs(args.ecosystem_dir)
    eco_cards: list[dict[str, Any]] = []

    for eco_dir in eco_dirs:
        if not args.no_regen:
            if not args.force and not _needs_eco_rebuild(eco_dir, publish=args.publish):
                size_kb = (eco_dir / "view.html").stat().st_size / 1024
                print(f"  eco {eco_dir.name}: up to date ({size_kb:.0f} KB)")
            else:
                try:
                    html = _render_ecosystem_run(eco_dir, publish=args.publish)
                    out = eco_dir / "view.html"
                    out.write_text(html)
                    size_kb = out.stat().st_size / 1024
                    print(f"  eco {eco_dir.name}: {size_kb:.0f} KB")
                except Exception as exc:
                    print(f"  eco {eco_dir.name}: FAILED - {exc}", file=sys.stderr)
                    continue
        eco_cards.append(_build_eco_card_context(eco_dir))

    if eco_cards:
        # Rebuild index with ecosystem cards populated
        index_html = _build_index_html(card_contexts, eco_cards=eco_cards)
        index_out = args.output_dir.parent / "index.html"
        index_out.write_text(index_html)
        print(f"index: updated with {len(eco_cards)} ecosystem run(s)")


if __name__ == "__main__":
    main()
