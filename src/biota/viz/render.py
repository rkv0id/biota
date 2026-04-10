"""Archive page renderer.

Converts a biota Archive into a self-contained HTML string that can be
written to disk and opened in any browser.  This module is the single
source of truth for the museum-gallery visual style.  Callers:

- scripts/view_archive.py   -- per-run CLI tool
- scripts/build_index.py    -- batch index builder (regenerates each run)
- (future) ecosystem renderer

The public API is one function:

    render_archive_page(archive, run_id, run_dir, stats_html, stats_css) -> str

The HTML/CSS/JS lives in src/biota/viz/templates/archive.html.
The optional stats section (collapsible metrics charts) is injected via the
stats_html and stats_css parameters, supplied by build_index.py.

The HTML output is fully self-contained.  All thumbnails are embedded as
base64 data URLs so the file can be scped off the cluster, emailed, or
hosted on GitHub Pages without any accompanying assets.
"""

import base64
import io
import json
from pathlib import Path
from typing import Any

import imageio.v3 as iio
import numpy as np
from jinja2 import Environment, FileSystemLoader

from biota.search.archive import Archive
from biota.search.result import RolloutResult
from biota.viz.colormap import apply_magma

# Cell render size in the gallery grid.  128 px gives the 192-pixel
# thumbnails room to breathe without dominating the page.
CELL_RENDER_PX = 128

# Detail-modal thumbnail render size.  Full native resolution of the
# 192-pixel thumbnails, upscaled 2x for screen legibility.
DETAIL_RENDER_PX = 384

# GIF frame duration in milliseconds.  50 ms = 20 fps, smoother than
# the old 100 ms for the 32-frame animations.
GIF_FRAME_DURATION_MS = 50

# Jinja2 environment pointed at the templates directory.
_TEMPLATES_DIR = Path(__file__).parent / "templates"
_ENV = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    autoescape=False,  # output is HTML we construct, not user input
    keep_trailing_newline=True,
)


def _thumbnail_to_data_url(thumbnail: np.ndarray) -> str:
    """Encode a (N, H, W) uint8 grayscale thumbnail as a base64 GIF data URL,
    applying the magma colormap so the GIF comes out colored.

    The input is the (frames, H, W) uint8 mass field as stored by the
    rollout worker.  We map each grayscale pixel through the magma lookup
    table to produce a (frames, H, W, 3) RGB stack, then write that as an
    animated GIF.  The GIF encoder auto-palettizes the RGB frames.
    """
    colored = apply_magma(thumbnail)  # (N, H, W, 3) uint8
    buf = io.BytesIO()
    iio.imwrite(
        buf,
        colored,
        extension=".gif",
        duration=GIF_FRAME_DURATION_MS,
        loop=0,
    )
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/gif;base64,{encoded}"


def _serialize_params(result: RolloutResult) -> dict[str, Any]:
    """Convert a result's params dict (with nested lists) into a JSON-safe
    dict that the modal can render."""
    params = result.params
    out: dict[str, Any] = {
        "R": float(params["R"]),
    }
    for key in ("r", "m", "s", "h"):
        out[key] = [round(float(v), 4) for v in params[key]]
    for key in ("a", "b", "w"):
        out[key] = [[round(float(v), 4) for v in row] for row in params[key]]
    return out


def _result_to_card_data(coord: tuple[int, int, int], result: RolloutResult) -> dict[str, Any]:
    """Build the per-cell payload embedded as a JSON blob in the page.

    Note: the field names in this JSON (speed/size/structure) are historical
    from the M1 descriptor design.  The underlying semantics are now
    (velocity, gyradius, spectral_entropy) but the field names are kept so
    existing archives continue to render.  The frontend maps positions to
    display labels via the DESC_LABELS JS constant.
    """
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


def _compute_grid_bbox(
    occupied: list[tuple[tuple[int, int, int], Any]],
    n_rows: int,
    n_cols: int,
) -> tuple[int, int, int, int]:
    """Return the bounding box (row_min, row_max, col_min, col_max) inclusive."""
    if not occupied:
        return 0, n_rows - 1, 0, n_cols - 1
    rows = [coord[0] for coord, _ in occupied]
    cols = [coord[1] for coord, _ in occupied]
    return min(rows), max(rows), min(cols), max(cols)


def _build_body_inner(
    occupied: list[tuple[tuple[int, int, int], Any]],
    cards: dict[str, dict[str, Any]],
    archive: Archive,
) -> str:
    """Build the HTML fragment that goes inside <main>."""
    if not occupied:
        return '<div class="empty">Archive is empty. No occupied cells to display.</div>'

    row_min, row_max, col_min, col_max = _compute_grid_bbox(
        occupied, archive.bins_speed, archive.bins_size
    )
    vis_rows = row_max - row_min + 1
    vis_cols = col_max - col_min + 1

    vel_tick_interval = max(1, vis_rows // 4)
    gyr_tick_interval = max(1, vis_cols // 4)

    vel_ticks: list[tuple[int, str]] = []
    for r in range(vis_rows):
        abs_row = r + row_min
        if abs_row % vel_tick_interval == 0 or abs_row == row_max:
            val = (abs_row + 0.5) / archive.bins_speed
            vel_ticks.append((r + 1, f"{val:.2f}"))

    gyr_ticks: list[tuple[int, str]] = []
    for c in range(vis_cols):
        abs_col = c + col_min
        if abs_col % gyr_tick_interval == 0 or abs_col == col_max:
            val = (abs_col + 0.5) / archive.bins_size
            gyr_ticks.append((c + 1, f"{val:.2f}"))

    cells_html_parts: list[str] = []
    for coord, _result in occupied:
        y, x, z = coord
        key = f"{y}_{x}_{z}"
        data_url = cards[key]["thumbnail"]
        grid_col = x - col_min + 1
        grid_row = y - row_min + 1
        cells_html_parts.append(
            f'<div class="cell" data-key="{key}" '
            f'style="grid-column: {grid_col}; grid-row: {grid_row};">'
            f'<img src="{data_url}" alt="cell {y},{x},{z}" loading="lazy" />'
            f"</div>"
        )

    grid_style = (
        f"grid-template-columns: repeat({vis_cols}, {CELL_RENDER_PX}px); "
        f"grid-template-rows: repeat({vis_rows}, {CELL_RENDER_PX}px);"
    )
    vel_label_style = f"grid-template-rows: repeat({vis_rows}, {CELL_RENDER_PX}px); gap: 14px;"
    gyr_label_style = f"grid-template-columns: repeat({vis_cols}, {CELL_RENDER_PX}px); gap: 14px;"
    vel_label_items = "".join(
        f'<div class="axis-tick" style="grid-row:{r};">{v}</div>' for r, v in vel_ticks
    )
    gyr_label_items = "".join(
        f'<div class="axis-tick" style="grid-column:{c};">{v}</div>' for c, v in gyr_ticks
    )
    cells_html = "\n".join(cells_html_parts)

    return (
        f'<div class="grid-with-axes">'
        f'<div class="axis-vel-label">'
        f'<div class="axis-arrow">&#8595; faster</div>'
        f'<div class="axis-vel-ticks" style="{vel_label_style}">{vel_label_items}</div>'
        f"</div>"
        f'<div class="grid-column-wrap">'
        f'<div class="grid" style="{grid_style}">{cells_html}</div>'
        f'<div class="axis-gyr-ticks" style="{gyr_label_style}">{gyr_label_items}</div>'
        f'<div class="axis-gyr-label">&#8594; larger (gyradius)</div>'
        f"</div>"
        f"</div>"
    )


def render_archive_page(
    archive: Archive,
    run_id: str,
    run_dir: Path,
    stats_html: str = "",
    stats_css: str = "",
) -> str:
    """Render an archive as a self-contained HTML page string.

    Args:
        archive:    The loaded biota Archive object.
        run_id:     The run directory name used in the page title.
        run_dir:    The run directory path shown in the page footer.
        stats_html: Optional HTML for the collapsible stats section,
                    produced by build_index.py.  Empty string omits it.
        stats_css:  Optional CSS for the stats section.  Empty string
                    omits it.  Should be provided alongside stats_html.

    Returns:
        A complete HTML document as a string.  Write it directly to disk.
    """
    occupied = list(archive.iter_occupied())
    occupied.sort(key=lambda pair: pair[0])

    cards: dict[str, dict[str, Any]] = {}
    for coord, result in occupied:
        y, x, z = coord
        key = f"{y}_{x}_{z}"
        cards[key] = _result_to_card_data(coord, result)

    swatch_stops = ", ".join(
        f"rgb({r},{g},{b})"
        for r, g, b in apply_magma(np.linspace(0, 255, 16, dtype=np.uint8).reshape(1, 16))[0]
    )

    body_inner = _build_body_inner(occupied, cards, archive)
    grid_desc = f"{archive.bins_speed} x {archive.bins_size} x {archive.bins_structure}"

    template = _ENV.get_template("archive.html")
    return template.render(
        run_id=run_id,
        run_dir=str(run_dir),
        cell_render_px=CELL_RENDER_PX,
        detail_render_px=DETAIL_RENDER_PX,
        n_cells=len(occupied),
        fill_pct=f"{archive.fill_fraction * 100:.1f}%",
        grid_desc=grid_desc,
        total_cells=archive.total_cells,
        swatch_stops=swatch_stops,
        body_inner=body_inner,
        cards_json=json.dumps(cards),
        stats_html=stats_html,
        stats_css=stats_css,
    )
