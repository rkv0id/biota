"""Render an archive.pkl as a browsable HTML page.

Loads a biota search run directory, extracts the archive's occupied cells,
encodes each one's thumbnail as a base64 data URL animated GIF, and writes a
self-contained HTML file you can open in any browser.

Usage:

    # Most common: view the most recent run
    uv run python scripts/view_archive.py

    # View a specific run by id
    uv run python scripts/view_archive.py --run 20260407-155010-crisp-thistle

    # Override output location
    uv run python scripts/view_archive.py --latest --output /tmp/archive.html

The HTML is self-contained (all data is inline as base64 data URLs) so you
can `scp` it off the cluster, email it, or open it offline. No external
dependencies on the viewing side.

Cells are laid out by their (speed, size) coordinate. Structure is shown as
a per-cell metadata field. Click any cell to expand it. Hover for the full
descriptor triple, quality, seed, and parent cell.
"""

import argparse
import base64
import io
import pickle
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from biota.search.archive import Archive
from biota.search.result import RolloutResult

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = ROOT / "runs"

# Size of each thumbnail in the grid (CSS pixels). The underlying GIF stays
# 32x32; the browser scales it up with nearest-neighbor rendering, which
# preserves the pixel-art crispness.
CELL_RENDER_PX = 72

GIF_FRAME_DURATION_MS = 100


def _find_latest_run(runs_root: Path) -> Path:
    """Return the most recently modified run directory under runs_root."""
    if not runs_root.exists():
        raise FileNotFoundError(f"no runs directory at {runs_root}")
    subdirs = [p for p in runs_root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"{runs_root} contains no run subdirectories")
    return max(subdirs, key=lambda p: p.stat().st_mtime)


def _load_archive(run_dir: Path) -> Archive:
    pkl = run_dir / "archive.pkl"
    if not pkl.exists():
        raise FileNotFoundError(f"no archive.pkl in {run_dir}")
    with open(pkl, "rb") as f:
        archive = pickle.load(f)
    if not isinstance(archive, Archive):
        raise TypeError(f"pickle at {pkl} is not a biota Archive")
    return archive


def _thumbnail_to_data_url(thumbnail: np.ndarray) -> str:
    """Encode a (N, H, W) uint8 thumbnail as a base64 GIF data URL."""
    buf = io.BytesIO()
    iio.imwrite(
        buf,
        thumbnail,
        extension=".gif",
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
    )
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{encoded}"


def _format_descriptors(result: RolloutResult) -> str:
    if result.descriptors is None:
        return "None"
    s, sz, st = result.descriptors
    return f"speed={s:.3f}, size={sz:.3f}, struct={st:.3f}"


def _format_parent(result: RolloutResult) -> str:
    if result.parent_cell is None:
        return "(random phase)"
    y, x, z = result.parent_cell
    return f"({y}, {x}, {z})"


def _render_cell_html(coord: tuple[int, int, int], result: RolloutResult) -> str:
    y, x, z = coord
    data_url = _thumbnail_to_data_url(result.thumbnail)
    descriptors = _format_descriptors(result)
    parent = _format_parent(result)
    quality_str = f"{result.quality:.3f}" if result.quality is not None else "—"
    return f"""
    <div class="cell" style="grid-column: {x + 1}; grid-row: {y + 1};">
        <img src="{data_url}" alt="cell {y},{x},{z}" loading="lazy" />
        <div class="cell-meta">
            <div class="cell-coord">cell ({y}, {x}, {z})</div>
            <div class="cell-quality">q={quality_str}</div>
            <div class="cell-desc">{descriptors}</div>
            <div class="cell-parent">parent: {parent}</div>
            <div class="cell-seed">seed={result.seed}</div>
        </div>
    </div>
    """


def _render_html(archive: Archive, run_id: str, run_dir: Path) -> str:
    occupied = list(archive.iter_occupied())
    occupied.sort(key=lambda pair: pair[0])  # sort by coord for stable output

    if not occupied:
        body = "<p>Archive is empty. No occupied cells to display.</p>"
    else:
        cells_html = "\n".join(_render_cell_html(coord, r) for coord, r in occupied)
        grid_style = f"grid-template-columns: repeat({archive.bins_size}, {CELL_RENDER_PX}px);"
        body = f'<div class="grid" style="{grid_style}">{cells_html}</div>'

    grid_desc = f"{archive.bins_speed}x{archive.bins_size}x{archive.bins_structure}"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>biota archive: {run_id}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", monospace;
            background: #0e0e10;
            color: #d4d4d8;
            margin: 0;
            padding: 24px;
        }}
        h1 {{
            font-size: 18px;
            font-weight: 600;
            margin: 0 0 4px 0;
        }}
        .subtitle {{
            color: #71717a;
            font-size: 12px;
            margin-bottom: 24px;
        }}
        .stats {{
            display: flex;
            gap: 24px;
            margin-bottom: 24px;
            font-size: 12px;
            color: #a1a1aa;
        }}
        .stats span {{
            color: #e4e4e7;
            font-weight: 600;
        }}
        .grid {{
            display: grid;
            gap: 4px;
            background: #18181b;
            padding: 8px;
            border-radius: 4px;
            width: fit-content;
        }}
        .cell {{
            position: relative;
            width: {CELL_RENDER_PX}px;
            height: {CELL_RENDER_PX}px;
            background: #000;
            border-radius: 2px;
            overflow: hidden;
            cursor: pointer;
        }}
        .cell img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            image-rendering: pixelated;
            display: block;
        }}
        .cell-meta {{
            position: absolute;
            bottom: 0;
            left: 0;
            right: 0;
            background: rgba(0, 0, 0, 0.92);
            color: #e4e4e7;
            font-size: 10px;
            padding: 6px 8px;
            line-height: 1.4;
            opacity: 0;
            transition: opacity 0.12s ease-out;
            pointer-events: none;
        }}
        .cell:hover .cell-meta {{
            opacity: 1;
        }}
        .cell-coord {{ font-weight: 600; color: #fff; }}
        .cell-quality {{ color: #22d3ee; }}
        .cell-desc {{ color: #a1a1aa; font-size: 9px; }}
        .cell-parent {{ color: #71717a; font-size: 9px; }}
        .cell-seed {{ color: #71717a; font-size: 9px; }}
        .axes {{
            margin-top: 16px;
            font-size: 11px;
            color: #71717a;
        }}
        .legend {{
            margin-top: 8px;
            font-size: 11px;
            color: #71717a;
            max-width: 600px;
        }}
    </style>
</head>
<body>
    <h1>biota archive: {run_id}</h1>
    <div class="subtitle">{run_dir}</div>
    <div class="stats">
        <div>cells occupied: <span>{len(occupied)}</span></div>
        <div>grid: <span>{grid_desc}</span></div>
        <div>total capacity: <span>{archive.total_cells}</span></div>
        <div>fill: <span>{archive.fill_fraction * 100:.1f}%</span></div>
    </div>
    {body}
    <div class="axes">
        rows: speed (down) &nbsp;|&nbsp; columns: size (right)
        &nbsp;|&nbsp; structure: see hover tooltip
    </div>
    <div class="legend">
        Each cell shows the 16-frame thumbnail stored in the archive. Hover
        any cell for descriptors, quality, seed, and parent. The grid is a 2D
        projection; the archive is 3D and multiple structure layers collapse
        into one display position. Cells are shown at {CELL_RENDER_PX}px for
        visibility; the underlying pixel data is 32x32.
    </div>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Run directory name (relative to runs/). Defaults to --latest.",
    )
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the most recently modified run directory (default if --run not given).",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=DEFAULT_RUNS_ROOT,
        help="Directory containing run subdirectories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path. Defaults to <run_dir>/view.html",
    )
    args = parser.parse_args()

    if args.run:
        run_dir = args.runs_root / args.run
        if not run_dir.exists():
            print(f"error: run directory not found: {run_dir}", file=sys.stderr)
            sys.exit(1)
    else:
        try:
            run_dir = _find_latest_run(args.runs_root)
        except FileNotFoundError as e:
            print(f"error: {e}", file=sys.stderr)
            sys.exit(1)

    archive = _load_archive(run_dir)
    run_id = run_dir.name
    output = args.output if args.output else run_dir / "view.html"

    print(f"loaded archive: {len(archive)} occupied cells from {run_dir}")
    html = _render_html(archive, run_id, run_dir)
    output.write_text(html)

    size_kb = output.stat().st_size / 1024
    print(f"wrote {output} ({size_kb:.1f} KB)")
    print(f"open with: open {output}")


if __name__ == "__main__":
    main()
