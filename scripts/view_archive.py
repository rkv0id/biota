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

Cells are laid out by their (speed, size) coordinate. Structure is shown in
the per-cell metadata. Hover any cell for a compact tooltip with the
descriptor triple and quality. Click any cell to open a detail modal with
the full parameters, the larger animation, and the lineage info.
"""

import argparse
import base64
import io
import json
import pickle
import sys
from pathlib import Path
from typing import Any

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

# Detail-modal thumbnail render size. Same 32x32 source, larger on screen
# so the creature is actually inspectable.
DETAIL_RENDER_PX = 320

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


def _serialize_params(result: RolloutResult) -> dict[str, Any]:
    """Convert a result's params dict (with nested lists) into a JSON-safe
    dict that the modal can render. Trims long lists to a compact preview
    so the modal doesn't drown in numbers."""
    params = result.params
    out: dict[str, Any] = {
        "R": float(params["R"]),
    }
    # 1D vectors of length k - just round and include all k values
    for key in ("r", "m", "s", "h"):
        out[key] = [round(float(v), 4) for v in params[key]]
    # 2D (k, 3) lists - include all
    for key in ("a", "b", "w"):
        out[key] = [[round(float(v), 4) for v in row] for row in params[key]]
    return out


def _result_to_card_data(coord: tuple[int, int, int], result: RolloutResult) -> dict[str, Any]:
    """Build the per-cell payload that gets embedded as a JSON blob in the
    page. The frontend reads this on hover/click instead of digging through
    individual data attributes."""
    y, x, z = coord
    descriptors_obj: dict[str, float] | None = None
    if result.descriptors is not None:
        descriptors_obj = {
            "speed": round(float(result.descriptors[0]), 4),
            "size": round(float(result.descriptors[1]), 4),
            "structure": round(float(result.descriptors[2]), 4),
        }
    parent: list[int] | None
    if result.parent_cell is not None:
        py, px, pz = result.parent_cell
        parent = [py, px, pz]
    else:
        parent = None
    return {
        "coord": [y, x, z],
        "quality": round(float(result.quality), 4) if result.quality is not None else None,
        "descriptors": descriptors_obj,
        "seed": int(result.seed),
        "parent": parent,
        "rejection_reason": result.rejection_reason,
        "compute_seconds": round(float(result.compute_seconds), 3),
        "thumbnail": _thumbnail_to_data_url(result.thumbnail),
        "params": _serialize_params(result),
    }


def _render_html(archive: Archive, run_id: str, run_dir: Path) -> str:
    occupied = list(archive.iter_occupied())
    occupied.sort(key=lambda pair: pair[0])

    # Build the per-cell data payload as a JSON object indexed by cell key.
    # The frontend reads from this on hover/click; the cell DOM elements
    # only carry the cell key, not their data.
    cards: dict[str, dict[str, Any]] = {}
    for coord, result in occupied:
        y, x, z = coord
        key = f"{y}_{x}_{z}"
        cards[key] = _result_to_card_data(coord, result)

    cards_json = json.dumps(cards)

    if not occupied:
        body = "<p>Archive is empty. No occupied cells to display.</p>"
    else:
        cells_html_parts: list[str] = []
        for coord, _result in occupied:
            y, x, z = coord
            key = f"{y}_{x}_{z}"
            data_url = cards[key]["thumbnail"]
            cells_html_parts.append(
                f'<div class="cell" data-key="{key}" '
                f'style="grid-column: {x + 1}; grid-row: {y + 1};">'
                f'<img src="{data_url}" alt="cell {y},{x},{z}" loading="lazy" />'
                f"</div>"
            )
        cells_html = "\n".join(cells_html_parts)
        grid_style = f"grid-template-columns: repeat({archive.bins_size}, {CELL_RENDER_PX}px);"
        body = f'<div class="grid" style="{grid_style}">{cells_html}</div>'

    grid_desc = f"{archive.bins_speed}x{archive.bins_size}x{archive.bins_structure}"

    # CSS and JS are inline. Doubled braces are f-string escapes for literal
    # CSS/JS curlies. The JS is small enough to read inline; if it grows past
    # ~100 lines we should consider externalizing it.
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
            transition: outline 0.08s ease-out;
            outline: 1px solid transparent;
        }}
        .cell:hover {{
            outline: 1px solid #22d3ee;
        }}
        .cell img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            image-rendering: pixelated;
            display: block;
        }}

        /* Floating tooltip - positioned absolute on the page, follows the
           cursor on hover. Stays small and tight; never overlaps adjacent
           cells because it's on a separate layer above the grid. */
        #tooltip {{
            position: fixed;
            pointer-events: none;
            background: rgba(0, 0, 0, 0.95);
            border: 1px solid #3f3f46;
            color: #e4e4e7;
            font-size: 11px;
            padding: 6px 8px;
            border-radius: 3px;
            line-height: 1.4;
            white-space: nowrap;
            z-index: 1000;
            opacity: 0;
            transition: opacity 0.08s ease-out;
        }}
        #tooltip.visible {{
            opacity: 1;
        }}
        #tooltip .tt-coord {{ color: #fff; font-weight: 600; }}
        #tooltip .tt-quality {{ color: #22d3ee; }}
        #tooltip .tt-desc {{ color: #a1a1aa; }}

        /* Detail modal - fullscreen overlay with the larger animation,
           the descriptor block, and the (long) parameter dump. */
        #modal-bg {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
        }}
        #modal-bg.visible {{
            display: flex;
        }}
        #modal {{
            background: #0e0e10;
            border: 1px solid #3f3f46;
            border-radius: 4px;
            max-width: 720px;
            max-height: 88vh;
            overflow-y: auto;
            padding: 24px;
            color: #d4d4d8;
            font-size: 12px;
        }}
        #modal .modal-header {{
            display: flex;
            align-items: flex-start;
            gap: 24px;
            margin-bottom: 20px;
        }}
        #modal .modal-thumb {{
            width: {DETAIL_RENDER_PX}px;
            height: {DETAIL_RENDER_PX}px;
            background: #000;
            border-radius: 3px;
            flex-shrink: 0;
        }}
        #modal .modal-thumb img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            image-rendering: pixelated;
            display: block;
            border-radius: 3px;
        }}
        #modal .modal-info {{
            flex-grow: 1;
            min-width: 0;
        }}
        #modal h2 {{
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #fff;
            font-weight: 600;
        }}
        #modal .info-row {{
            margin-bottom: 6px;
            color: #a1a1aa;
        }}
        #modal .info-row .label {{
            color: #71717a;
            display: inline-block;
            min-width: 90px;
        }}
        #modal .info-row .value {{
            color: #e4e4e7;
        }}
        #modal .info-row .value.accent {{
            color: #22d3ee;
        }}
        #modal .params-section {{
            border-top: 1px solid #27272a;
            padding-top: 16px;
            margin-top: 16px;
        }}
        #modal .params-section h3 {{
            margin: 0 0 12px 0;
            font-size: 12px;
            color: #71717a;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        #modal pre {{
            background: #18181b;
            padding: 12px;
            border-radius: 3px;
            font-size: 10px;
            line-height: 1.5;
            overflow-x: auto;
            color: #d4d4d8;
            margin: 0;
        }}
        #modal .modal-close {{
            position: absolute;
            top: 12px;
            right: 16px;
            background: none;
            border: none;
            color: #71717a;
            font-size: 20px;
            cursor: pointer;
            padding: 4px 8px;
        }}
        #modal .modal-close:hover {{
            color: #e4e4e7;
        }}
        #modal {{
            position: relative;
        }}

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
        &nbsp;|&nbsp; structure: see hover or click
    </div>
    <div class="legend">
        Each cell shows the 16-frame thumbnail stored in the archive. Hover
        any cell for a tooltip with descriptors and quality. Click any cell
        to open the full detail with parameters and the lineage. The grid
        is a 2D projection; the archive is 3D and multiple structure layers
        collapse into one display position.
    </div>

    <div id="tooltip"></div>

    <div id="modal-bg">
        <div id="modal">
            <button class="modal-close" type="button" aria-label="Close">&times;</button>
            <div class="modal-header">
                <div class="modal-thumb"><img id="modal-thumb-img" alt="cell" /></div>
                <div class="modal-info">
                    <h2 id="modal-title">cell</h2>
                    <div class="info-row"><span class="label">quality</span>
                        <span class="value accent" id="modal-quality"></span></div>
                    <div class="info-row"><span class="label">descriptors</span>
                        <span class="value" id="modal-descriptors"></span></div>
                    <div class="info-row"><span class="label">seed</span>
                        <span class="value" id="modal-seed"></span></div>
                    <div class="info-row"><span class="label">parent cell</span>
                        <span class="value" id="modal-parent"></span></div>
                    <div class="info-row"><span class="label">compute time</span>
                        <span class="value" id="modal-compute"></span></div>
                </div>
            </div>
            <div class="params-section">
                <h3>parameters</h3>
                <pre id="modal-params"></pre>
            </div>
        </div>
    </div>

    <script>
const CARDS = {cards_json};

const tooltip = document.getElementById('tooltip');
const modalBg = document.getElementById('modal-bg');
const modalCloseBtn = modalBg.querySelector('.modal-close');

function fmtCoord(c) {{
    return '(' + c[0] + ', ' + c[1] + ', ' + c[2] + ')';
}}

function fmtDescriptors(d) {{
    if (d === null) return 'none';
    return 'speed=' + d.speed.toFixed(3)
        + '  size=' + d.size.toFixed(3)
        + '  struct=' + d.structure.toFixed(3);
}}

function fmtQuality(q) {{
    return q === null ? '—' : q.toFixed(3);
}}

function showTooltip(card, evt) {{
    const q = fmtQuality(card.quality);
    const d = card.descriptors;
    const dStr = d === null
        ? 'no descriptors'
        : 'spd ' + d.speed.toFixed(2)
            + '  sz ' + d.size.toFixed(2)
            + '  str ' + d.structure.toFixed(2);
    tooltip.innerHTML =
        '<div class="tt-coord">cell ' + fmtCoord(card.coord) + '</div>'
        + '<div class="tt-quality">q=' + q + '</div>'
        + '<div class="tt-desc">' + dStr + '</div>';
    tooltip.classList.add('visible');
    moveTooltip(evt);
}}

function moveTooltip(evt) {{
    // Position the tooltip near the cursor but not under it. Offset 14px
    // right and below; flip to left/above if we'd run off the viewport.
    let x = evt.clientX + 14;
    let y = evt.clientY + 14;
    const ttRect = tooltip.getBoundingClientRect();
    if (x + ttRect.width > window.innerWidth - 8) {{
        x = evt.clientX - ttRect.width - 14;
    }}
    if (y + ttRect.height > window.innerHeight - 8) {{
        y = evt.clientY - ttRect.height - 14;
    }}
    tooltip.style.left = x + 'px';
    tooltip.style.top = y + 'px';
}}

function hideTooltip() {{
    tooltip.classList.remove('visible');
}}

function openModal(card) {{
    document.getElementById('modal-thumb-img').src = card.thumbnail;
    document.getElementById('modal-title').textContent = 'cell ' + fmtCoord(card.coord);
    document.getElementById('modal-quality').textContent = fmtQuality(card.quality);
    document.getElementById('modal-descriptors').textContent = fmtDescriptors(card.descriptors);
    document.getElementById('modal-seed').textContent = card.seed;
    document.getElementById('modal-parent').textContent =
        card.parent === null ? '(random phase)' : fmtCoord(card.parent);
    document.getElementById('modal-compute').textContent = card.compute_seconds + ' s';
    document.getElementById('modal-params').textContent = JSON.stringify(card.params, null, 2);
    modalBg.classList.add('visible');
    hideTooltip();
}}

function closeModal() {{
    modalBg.classList.remove('visible');
}}

// Wire it all up. Cells delegate via the parent grid container so we don't
// attach individual listeners; faster and DOM-cheap.
document.querySelectorAll('.cell').forEach(function(cell) {{
    cell.addEventListener('mouseenter', function(evt) {{
        const card = CARDS[cell.dataset.key];
        if (card) showTooltip(card, evt);
    }});
    cell.addEventListener('mousemove', moveTooltip);
    cell.addEventListener('mouseleave', hideTooltip);
    cell.addEventListener('click', function() {{
        const card = CARDS[cell.dataset.key];
        if (card) openModal(card);
    }});
}});

modalCloseBtn.addEventListener('click', closeModal);
modalBg.addEventListener('click', function(evt) {{
    if (evt.target === modalBg) closeModal();
}});
document.addEventListener('keydown', function(evt) {{
    if (evt.key === 'Escape') closeModal();
}});
    </script>
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
