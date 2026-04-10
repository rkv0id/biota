"""Archive page renderer.

Converts a biota Archive into a self-contained HTML string that can be
written to disk and opened in any browser.  This module is the single
source of truth for the museum-gallery visual style.  Callers:

- scripts/view_archive.py   -- per-run CLI tool
- scripts/build_index.py    -- batch index builder (regenerates each run)
- (future) ecosystem renderer

The public API is one function:

    render_archive_page(archive, run_id, run_dir) -> str

Everything else in this module is a private implementation detail.

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


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


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
    # 1D vectors of length k - round and include all k values
    for key in ("r", "m", "s", "h"):
        out[key] = [round(float(v), 4) for v in params[key]]
    # 2D (k, 3) lists
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
    """Return the bounding box (row_min, row_max, col_min, col_max) of the
    occupied cells, inclusive.  Falls back to the full grid when empty."""
    if not occupied:
        return 0, n_rows - 1, 0, n_cols - 1
    rows = [coord[0] for coord, _ in occupied]
    cols = [coord[1] for coord, _ in occupied]
    return min(rows), max(rows), min(cols), max(cols)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_archive_page(archive: Archive, run_id: str, run_dir: Path) -> str:
    """Render an archive as a self-contained HTML page string.

    Args:
        archive:  The loaded biota Archive object.
        run_id:   The run directory name used in the page title.
        run_dir:  The run directory path shown in the page footer.

    Returns:
        A complete HTML document as a string.  Write it directly to disk.
    """
    occupied = list(archive.iter_occupied())
    occupied.sort(key=lambda pair: pair[0])

    # Build the per-cell data payload indexed by cell key.
    cards: dict[str, dict[str, Any]] = {}
    for coord, result in occupied:
        y, x, z = coord
        key = f"{y}_{x}_{z}"
        cards[key] = _result_to_card_data(coord, result)

    cards_json = json.dumps(cards)

    # Grid compaction: crop the rendered grid to the bounding box of
    # occupied cells so outer empty margins disappear.  Empty cells
    # *inside* the bbox stay visible - they represent genuine search
    # gaps and removing them would distort descriptor-space adjacency.
    row_min, row_max, col_min, col_max = _compute_grid_bbox(
        occupied, archive.bins_speed, archive.bins_size
    )
    # Compute visible grid dimensions after bbox crop.
    vis_rows = (row_max - row_min + 1) if occupied else archive.bins_speed
    vis_cols = (col_max - col_min + 1) if occupied else archive.bins_size

    # Axis tick labels for velocity (rows) and gyradius (columns).
    # We show one label per bin edge at a reasonable tick interval.
    # The displayed value is the normalized bin centre: (bin + 0.5) / n_bins.
    # row 0 = lowest velocity, higher rows = faster.
    # col 0 = smallest gyradius, higher cols = larger.
    vel_tick_interval = max(1, vis_rows // 4)
    gyr_tick_interval = max(1, vis_cols // 4)

    # Build velocity tick labels (one per visible row that hits the interval).
    vel_ticks: list[tuple[int, str]] = []  # (grid-row-index-1-based, label)
    for r in range(vis_rows):
        abs_row = r + row_min
        if abs_row % vel_tick_interval == 0 or abs_row == row_max:
            val = (abs_row + 0.5) / archive.bins_speed
            vel_ticks.append((r + 1, f"{val:.2f}"))

    # Build gyradius tick labels (one per visible col that hits the interval).
    gyr_ticks: list[tuple[int, str]] = []  # (grid-col-index-1-based, label)
    for c in range(vis_cols):
        abs_col = c + col_min
        if abs_col % gyr_tick_interval == 0 or abs_col == col_max:
            val = (abs_col + 0.5) / archive.bins_size
            gyr_ticks.append((c + 1, f"{val:.2f}"))

    if not occupied:
        body_inner = '<div class="empty">Archive is empty. No occupied cells to display.</div>'
    else:
        cells_html_parts: list[str] = []
        for coord, _result in occupied:
            y, x, z = coord
            key = f"{y}_{x}_{z}"
            data_url = cards[key]["thumbnail"]
            # Shift grid-column/row to the cropped coordinate system.
            grid_col = x - col_min + 1
            grid_row = y - row_min + 1
            cells_html_parts.append(
                f'<div class="cell" data-key="{key}" '
                f'style="grid-column: {grid_col}; grid-row: {grid_row};">'
                f'<img src="{data_url}" alt="cell {y},{x},{z}" loading="lazy" />'
                f"</div>"
            )
        cells_html = "\n".join(cells_html_parts)
        grid_style = (
            f"grid-template-columns: repeat({vis_cols}, {CELL_RENDER_PX}px); "
            f"grid-template-rows: repeat({vis_rows}, {CELL_RENDER_PX}px);"
        )

        # Velocity tick labels rendered as a left-side column.
        vel_label_items = "".join(
            f'<div class="axis-tick" style="grid-row:{r};">{v}</div>' for r, v in vel_ticks
        )
        vel_label_style = f"grid-template-rows: repeat({vis_rows}, {CELL_RENDER_PX}px); gap: 14px;"

        # Gyradius tick labels rendered as a bottom row.
        gyr_label_items = "".join(
            f'<div class="axis-tick" style="grid-column:{c};">{v}</div>' for c, v in gyr_ticks
        )
        gyr_label_style = (
            f"grid-template-columns: repeat({vis_cols}, {CELL_RENDER_PX}px); gap: 14px;"
        )

        body_inner = f"""
<div class="grid-with-axes">
  <div class="axis-vel-label">
    <div class="axis-arrow">&#8595; faster</div>
    <div class="axis-vel-ticks" style="{vel_label_style}">{vel_label_items}</div>
  </div>
  <div class="grid-column-wrap">
    <div class="grid" style="{grid_style}">{cells_html}</div>
    <div class="axis-gyr-ticks" style="{gyr_label_style}">{gyr_label_items}</div>
    <div class="axis-gyr-label">&#8594; larger (gyradius)</div>
  </div>
</div>"""

    grid_desc = f"{archive.bins_speed} x {archive.bins_size} x {archive.bins_structure}"
    fill_pct = f"{archive.fill_fraction * 100:.1f}%"

    # Magma gradient swatch: show a horizontal bar sampling the colormap
    # over 16 stops from dark (low spectral entropy) to bright (high).
    swatch_stops = ", ".join(
        f"rgb({r},{g},{b})"
        for r, g, b in apply_magma(np.linspace(0, 255, 16, dtype=np.uint8).reshape(1, 16))[0]
    )

    # CSS and JS are inline.  Doubled braces are f-string escapes for
    # literal CSS/JS curlies.
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>biota archive: {run_id}</title>
    <style>
        /* === reset + base === */
        *, *::before, *::after {{
            box-sizing: border-box;
        }}
        html, body {{
            margin: 0;
            padding: 0;
            height: 100%;
            background: #08090b;
            color: #e4e4e7;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                Helvetica, Arial, sans-serif;
            font-size: 14px;
            line-height: 1.5;
            -webkit-font-smoothing: antialiased;
        }}

        /* === fixed flex shell === */
        /* The page is divided into three rows: fixed header, scrolling
           main, fixed footer.  Only the main area scrolls so the title
           and legend stay visible at all times regardless of archive size. */
        body {{
            display: flex;
            flex-direction: column;
            background:
                radial-gradient(
                    ellipse at 50% 35%,
                    #141820 0%,
                    #0b0d10 45%,
                    #08090b 100%
                );
            background-attachment: fixed;
        }}

        /* === header === */
        .page-header {{
            flex: 0 0 auto;
            padding: 32px 48px 24px 48px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
        }}
        .page-title {{
            font-size: 32px;
            font-weight: 300;
            letter-spacing: -0.01em;
            color: #f4f4f5;
            margin: 0 0 10px 0;
            line-height: 1.15;
        }}
        .page-title .title-prefix {{
            color: #52525b;
            font-weight: 300;
            margin-right: 10px;
        }}
        .page-meta {{
            font-size: 12px;
            color: #71717a;
            letter-spacing: 0.02em;
            font-variant-numeric: tabular-nums;
            display: flex;
            align-items: center;
            gap: 4px;
            flex-wrap: wrap;
        }}
        .page-meta .meta-value {{
            color: #a1a1aa;
            font-weight: 500;
        }}
        .page-meta .meta-sep {{
            color: #3f3f46;
            margin: 0 6px;
        }}
        .page-path {{
            font-size: 11px;
            color: #3f3f46;
            margin-top: 6px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }}

        /* Magma swatch: gradient bar showing spectral entropy range */
        .swatch-wrap {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-top: 10px;
            font-size: 11px;
            color: #52525b;
        }}
        .swatch {{
            width: 120px;
            height: 6px;
            border-radius: 3px;
            background: linear-gradient(90deg, {swatch_stops});
            flex-shrink: 0;
        }}

        /* Quality filter slider */
        .filter-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            margin-top: 12px;
            font-size: 11px;
            color: #71717a;
        }}
        .filter-row label {{
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .filter-row input[type=range] {{
            -webkit-appearance: none;
            appearance: none;
            width: 140px;
            height: 3px;
            background: rgba(255,255,255,0.12);
            border-radius: 2px;
            outline: none;
            cursor: pointer;
        }}
        .filter-row input[type=range]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #fbbf77;
            cursor: pointer;
        }}
        .filter-row input[type=range]::-moz-range-thumb {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #fbbf77;
            border: none;
            cursor: pointer;
        }}
        #filter-val {{
            color: #fbbf77;
            font-variant-numeric: tabular-nums;
            min-width: 36px;
        }}
        #filter-count {{
            color: #52525b;
            font-variant-numeric: tabular-nums;
        }}

        /* === main scroll area === */
        main {{
            flex: 1 1 0;
            overflow: auto;
            padding: 32px 48px;
            /* Custom scrollbars to match the dark theme */
            scrollbar-width: thin;
            scrollbar-color: #3f3f46 transparent;
        }}
        main::-webkit-scrollbar {{
            width: 6px;
            height: 6px;
        }}
        main::-webkit-scrollbar-track {{
            background: transparent;
        }}
        main::-webkit-scrollbar-thumb {{
            background: #3f3f46;
            border-radius: 3px;
        }}
        main::-webkit-scrollbar-thumb:hover {{
            background: #52525b;
        }}

        /* === grid with axis indicators === */
        .grid-with-axes {{
            display: flex;
            gap: 8px;
            align-items: flex-start;
        }}

        /* Left-side velocity axis */
        .axis-vel-label {{
            display: flex;
            flex-direction: column;
            align-items: flex-end;
            gap: 6px;
            flex-shrink: 0;
            padding-top: 8px;
        }}
        .axis-arrow {{
            font-size: 10px;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            writing-mode: vertical-lr;
            transform: rotate(180deg);
            margin-bottom: 4px;
        }}
        .axis-vel-ticks {{
            display: grid;
            gap: 14px;
        }}

        /* Column wrapper for grid + bottom axis */
        .grid-column-wrap {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}

        /* Bottom gyradius axis */
        .axis-gyr-ticks {{
            display: grid;
            gap: 14px;
        }}
        .axis-gyr-label {{
            font-size: 10px;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            text-align: right;
            padding-right: 4px;
        }}

        /* Shared axis tick style */
        .axis-tick {{
            font-size: 9px;
            color: #3f3f46;
            font-variant-numeric: tabular-nums;
            display: flex;
            align-items: center;
            justify-content: flex-end;
            height: {CELL_RENDER_PX}px;
            padding-right: 4px;
        }}
        .axis-gyr-ticks .axis-tick {{
            height: auto;
            width: {CELL_RENDER_PX}px;
            justify-content: center;
            align-items: flex-start;
            padding-right: 0;
            padding-top: 4px;
        }}

        /* === main grid === */
        .grid {{
            display: grid;
            gap: 14px;
            padding: 8px;
        }}
        .cell {{
            width: {CELL_RENDER_PX}px;
            height: {CELL_RENDER_PX}px;
            background: transparent;
            border-radius: 4px;
            overflow: hidden;
            cursor: pointer;
            position: relative;
            transition:
                filter 0.18s ease-out,
                box-shadow 0.22s ease-out;
        }}
        .cell img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            image-rendering: pixelated;
            display: block;
            border-radius: 4px;
        }}
        .cell.hidden {{
            visibility: hidden;
            pointer-events: none;
        }}
        .cell:hover {{
            filter: brightness(1.2) saturate(1.1);
            box-shadow:
                0 0 0 1px rgba(252, 180, 100, 0.35),
                0 0 24px rgba(252, 120, 60, 0.28),
                0 0 48px rgba(220, 60, 40, 0.15);
            z-index: 2;
        }}
        .empty {{
            text-align: center;
            color: #52525b;
            font-style: italic;
            padding: 64px 0;
        }}

        /* === tooltip === */
        #tooltip {{
            position: fixed;
            pointer-events: none;
            background: rgba(14, 16, 20, 0.92);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #e4e4e7;
            font-size: 11px;
            padding: 12px 14px;
            border-radius: 8px;
            line-height: 1.5;
            z-index: 1000;
            opacity: 0;
            transform: translateY(4px);
            transition:
                opacity 0.12s ease-out,
                transform 0.12s ease-out;
            min-width: 200px;
            box-shadow:
                0 4px 24px rgba(0, 0, 0, 0.4),
                0 1px 3px rgba(0, 0, 0, 0.3);
        }}
        #tooltip.visible {{
            opacity: 1;
            transform: translateY(0);
        }}
        #tooltip .tt-coord {{
            color: #f4f4f5;
            font-weight: 500;
            font-size: 10px;
            letter-spacing: 0.05em;
            text-transform: uppercase;
            margin-bottom: 4px;
            font-variant-numeric: tabular-nums;
        }}
        #tooltip .tt-quality {{
            color: #fbbf77;
            font-size: 14px;
            font-weight: 500;
            margin-bottom: 10px;
            font-variant-numeric: tabular-nums;
        }}
        #tooltip .tt-descriptors {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        #tooltip .tt-desc-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 10px;
        }}
        #tooltip .tt-desc-label {{
            color: #71717a;
            min-width: 58px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        #tooltip .tt-desc-bar {{
            flex: 1;
            height: 3px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 2px;
            overflow: hidden;
            min-width: 60px;
        }}
        #tooltip .tt-desc-fill {{
            height: 100%;
            background: linear-gradient(90deg, #7c3aed, #ec4899, #fbbf77);
            border-radius: 2px;
        }}
        #tooltip .tt-desc-value {{
            color: #a1a1aa;
            font-variant-numeric: tabular-nums;
            min-width: 36px;
            text-align: right;
        }}

        /* === detail modal === */
        #modal-bg {{
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(6, 7, 10, 0.72);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
            padding: 32px;
            opacity: 0;
            transition: opacity 0.18s ease-out;
        }}
        #modal-bg.visible {{
            display: flex;
            opacity: 1;
        }}
        #modal {{
            background: rgba(14, 16, 20, 0.95);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            max-width: 900px;
            width: 100%;
            max-height: 90vh;
            overflow-y: auto;
            color: #e4e4e7;
            font-size: 13px;
            position: relative;
            box-shadow:
                0 20px 60px rgba(0, 0, 0, 0.6),
                0 4px 16px rgba(0, 0, 0, 0.4);
            transform: scale(0.96) translateY(8px);
            transition: transform 0.22s cubic-bezier(0.2, 0.6, 0.2, 1);
            /* Modal scrollbar */
            scrollbar-width: thin;
            scrollbar-color: #3f3f46 transparent;
        }}
        #modal::-webkit-scrollbar {{
            width: 6px;
        }}
        #modal::-webkit-scrollbar-track {{
            background: transparent;
        }}
        #modal::-webkit-scrollbar-thumb {{
            background: #3f3f46;
            border-radius: 3px;
        }}
        #modal-bg.visible #modal {{
            transform: scale(1) translateY(0);
        }}
        .modal-body {{
            display: grid;
            grid-template-columns: {DETAIL_RENDER_PX}px 1fr;
            gap: 32px;
            padding: 32px;
        }}
        .modal-thumb-wrap {{
            width: {DETAIL_RENDER_PX}px;
            flex-shrink: 0;
        }}
        .modal-thumb {{
            width: {DETAIL_RENDER_PX}px;
            height: {DETAIL_RENDER_PX}px;
            border-radius: 8px;
            overflow: hidden;
            background: #000;
            box-shadow:
                0 0 0 1px rgba(255, 255, 255, 0.06),
                0 8px 24px rgba(0, 0, 0, 0.4);
        }}
        .modal-thumb img {{
            width: 100%;
            height: 100%;
            object-fit: cover;
            image-rendering: pixelated;
            display: block;
        }}
        .modal-thumb-caption {{
            margin-top: 12px;
            font-size: 10px;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            text-align: center;
            font-variant-numeric: tabular-nums;
        }}
        .modal-info {{
            min-width: 0;
        }}
        .modal-info h2 {{
            font-size: 20px;
            font-weight: 400;
            color: #f4f4f5;
            margin: 0 0 4px 0;
            letter-spacing: -0.01em;
            font-variant-numeric: tabular-nums;
        }}
        .modal-subtitle {{
            font-size: 11px;
            color: #52525b;
            margin-bottom: 24px;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .modal-stat-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px 32px;
            margin-bottom: 24px;
        }}
        .modal-stat {{
            display: flex;
            flex-direction: column;
            gap: 4px;
        }}
        .modal-stat .stat-label {{
            font-size: 10px;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .modal-stat .stat-value {{
            font-size: 14px;
            color: #e4e4e7;
            font-variant-numeric: tabular-nums;
        }}
        .modal-stat .stat-value.accent {{
            color: #fbbf77;
            font-weight: 500;
        }}
        .modal-descriptors {{
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin-bottom: 24px;
        }}
        .modal-desc-row {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 11px;
        }}
        .modal-desc-label {{
            color: #71717a;
            min-width: 72px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .modal-desc-bar {{
            flex: 1;
            height: 4px;
            background: rgba(255, 255, 255, 0.06);
            border-radius: 2px;
            overflow: hidden;
        }}
        .modal-desc-fill {{
            height: 100%;
            background: linear-gradient(90deg, #7c3aed, #ec4899, #fbbf77);
            border-radius: 2px;
            transition: width 0.4s cubic-bezier(0.2, 0.6, 0.2, 1);
        }}
        .modal-desc-value {{
            color: #d4d4d8;
            font-variant-numeric: tabular-nums;
            min-width: 44px;
            text-align: right;
        }}
        .modal-params-section {{
            padding: 24px 32px 32px 32px;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
        }}
        .modal-params-section h3 {{
            margin: 0 0 14px 0;
            font-size: 10px;
            color: #52525b;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 600;
        }}
        .modal-params-section pre {{
            background: rgba(0, 0, 0, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.04);
            padding: 16px;
            border-radius: 6px;
            font-size: 11px;
            line-height: 1.6;
            overflow-x: auto;
            color: #a1a1aa;
            margin: 0;
            font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
        }}
        .modal-close {{
            position: absolute;
            top: 16px;
            right: 20px;
            background: rgba(255, 255, 255, 0.04);
            border: 1px solid rgba(255, 255, 255, 0.06);
            color: #71717a;
            font-size: 18px;
            line-height: 1;
            cursor: pointer;
            padding: 6px 12px;
            border-radius: 6px;
            transition: all 0.12s ease-out;
            font-family: inherit;
        }}
        .modal-close:hover {{
            background: rgba(255, 255, 255, 0.08);
            color: #e4e4e7;
        }}

        /* === footer === */
        .page-footer {{
            flex: 0 0 auto;
            padding: 16px 48px 20px 48px;
            border-top: 1px solid rgba(255, 255, 255, 0.04);
            font-size: 11px;
            color: #52525b;
            line-height: 1.6;
        }}
        .page-footer .axis-legend {{
            color: #71717a;
            margin-bottom: 4px;
        }}

        /* Narrow-viewport graceful degradation */
        @media (max-width: 900px) {{
            .page-header {{ padding: 20px 20px 16px 20px; }}
            main {{ padding: 20px; }}
            .page-footer {{ padding: 12px 20px; }}
            .page-title {{ font-size: 24px; }}
            .modal-body {{
                grid-template-columns: 1fr;
                padding: 20px;
            }}
            .modal-thumb-wrap, .modal-thumb {{
                width: 100%;
                max-width: {DETAIL_RENDER_PX}px;
            }}
        }}
    </style>
</head>
<body>
    <header class="page-header">
        <h1 class="page-title">
            <span class="title-prefix">biota</span>{run_id}
        </h1>
        <div class="page-meta">
            <span class="meta-value">{len(occupied)}</span> cells
            <span class="meta-sep">&middot;</span>
            <span class="meta-value">{fill_pct}</span> fill
            <span class="meta-sep">&middot;</span>
            <span class="meta-value">{grid_desc}</span> grid
            <span class="meta-sep">&middot;</span>
            <span class="meta-value">{archive.total_cells}</span> capacity
        </div>
        <div class="swatch-wrap">
            <span>spectral entropy</span>
            <div class="swatch"></div>
            <span>high</span>
        </div>
        <div class="filter-row">
            <label for="quality-filter">min quality</label>
            <input type="range" id="quality-filter" min="0" max="1" step="0.01" value="0" />
            <span id="filter-val">0.00</span>
            <span id="filter-count"></span>
        </div>
        <div class="page-path">{run_dir}</div>
    </header>

    <main>
        {body_inner}
    </main>

    <footer class="page-footer">
        <div class="axis-legend">
            rows velocity &nbsp;&middot;&nbsp; columns gyradius &nbsp;&middot;&nbsp;
            structure (spectral entropy) collapsed into cell position
        </div>
        <div>
            Hover any cell for descriptor values. Click any cell to open the
            full detail view with parameters and lineage. Escape or click
            outside to close.
        </div>
    </footer>

    <div id="tooltip"></div>

    <div id="modal-bg" aria-hidden="true">
        <div id="modal" role="dialog" aria-modal="true">
            <button class="modal-close" type="button" aria-label="Close">&times;</button>
            <div class="modal-body">
                <div class="modal-thumb-wrap">
                    <div class="modal-thumb">
                        <img id="modal-thumb-img" alt="cell" />
                    </div>
                    <div class="modal-thumb-caption" id="modal-thumb-caption"></div>
                </div>
                <div class="modal-info">
                    <h2 id="modal-title">cell</h2>
                    <div class="modal-subtitle" id="modal-subtitle">quality</div>

                    <div class="modal-descriptors" id="modal-desc-block"></div>

                    <div class="modal-stat-grid">
                        <div class="modal-stat">
                            <div class="stat-label">quality</div>
                            <div class="stat-value accent" id="modal-quality"></div>
                        </div>
                        <div class="modal-stat">
                            <div class="stat-label">seed</div>
                            <div class="stat-value" id="modal-seed"></div>
                        </div>
                        <div class="modal-stat">
                            <div class="stat-label">parent cell</div>
                            <div class="stat-value" id="modal-parent"></div>
                        </div>
                        <div class="modal-stat">
                            <div class="stat-label">compute time</div>
                            <div class="stat-value" id="modal-compute"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="modal-params-section">
                <h3>parameters</h3>
                <pre id="modal-params"></pre>
            </div>
        </div>
    </div>

    <script>
const CARDS = {cards_json};

// Descriptor labels for display.  Position index -> short label.  The
// archive's JSON field names are still (speed, size, structure) for
// historical reasons; DESC_LABELS maps position to the current display
// label without touching the data shape.
const DESC_LABELS = ['velocity', 'gyradius', 'spectral'];

const tooltip = document.getElementById('tooltip');
const modalBg = document.getElementById('modal-bg');
const modalCloseBtn = modalBg.querySelector('.modal-close');

function fmtCoord(c) {{
    return '(' + c[0] + ', ' + c[1] + ', ' + c[2] + ')';
}}

function fmtQuality(q) {{
    return q === null ? '-' : q.toFixed(4);
}}

// Turn a descriptors object into an ordered array of (label, value) pairs.
function descArray(d) {{
    if (d === null) return [];
    const raw = [d.speed, d.size, d.structure];
    return DESC_LABELS.map(function(label, i) {{
        return {{ label: label, value: raw[i] }};
    }});
}}

function showTooltip(card, evt) {{
    const descs = descArray(card.descriptors);
    let descHtml = '';
    if (descs.length === 0) {{
        descHtml = '<div class="tt-desc-row"><span class="tt-desc-label">no descriptors</span></div>';
    }} else {{
        descHtml = descs.map(function(d) {{
            const pct = Math.round(d.value * 100);
            return '<div class="tt-desc-row">'
                + '<span class="tt-desc-label">' + d.label + '</span>'
                + '<span class="tt-desc-bar"><span class="tt-desc-fill" style="width:' + pct + '%"></span></span>'
                + '<span class="tt-desc-value">' + d.value.toFixed(3) + '</span>'
                + '</div>';
        }}).join('');
    }}
    tooltip.innerHTML =
        '<div class="tt-coord">cell ' + fmtCoord(card.coord) + '</div>'
        + '<div class="tt-quality">q ' + fmtQuality(card.quality) + '</div>'
        + '<div class="tt-descriptors">' + descHtml + '</div>';
    tooltip.classList.add('visible');
    moveTooltip(evt);
}}

function moveTooltip(evt) {{
    let x = evt.clientX + 18;
    let y = evt.clientY + 18;
    const ttRect = tooltip.getBoundingClientRect();
    if (x + ttRect.width > window.innerWidth - 12) {{
        x = evt.clientX - ttRect.width - 18;
    }}
    if (y + ttRect.height > window.innerHeight - 12) {{
        y = evt.clientY - ttRect.height - 18;
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
    document.getElementById('modal-subtitle').textContent =
        card.parent === null ? 'random phase' : 'mutated from ' + fmtCoord(card.parent);
    document.getElementById('modal-thumb-caption').textContent = 'seed ' + card.seed;

    document.getElementById('modal-quality').textContent = fmtQuality(card.quality);
    document.getElementById('modal-seed').textContent = card.seed;
    document.getElementById('modal-parent').textContent =
        card.parent === null ? '(random phase)' : fmtCoord(card.parent);
    document.getElementById('modal-compute').textContent = card.compute_seconds.toFixed(2) + ' s';

    const descs = descArray(card.descriptors);
    const descBlock = document.getElementById('modal-desc-block');
    if (descs.length === 0) {{
        descBlock.innerHTML = '<div class="modal-desc-row">no descriptors</div>';
    }} else {{
        descBlock.innerHTML = descs.map(function(d) {{
            const pct = Math.round(d.value * 100);
            return '<div class="modal-desc-row">'
                + '<span class="modal-desc-label">' + d.label + '</span>'
                + '<span class="modal-desc-bar"><span class="modal-desc-fill" style="width:' + pct + '%"></span></span>'
                + '<span class="modal-desc-value">' + d.value.toFixed(4) + '</span>'
                + '</div>';
        }}).join('');
    }}

    document.getElementById('modal-params').textContent = JSON.stringify(card.params, null, 2);
    modalBg.classList.add('visible');
    modalBg.setAttribute('aria-hidden', 'false');
    hideTooltip();
}}

function closeModal() {{
    modalBg.classList.remove('visible');
    modalBg.setAttribute('aria-hidden', 'true');
}}

// Quality filter slider
const filterSlider = document.getElementById('quality-filter');
const filterVal = document.getElementById('filter-val');
const filterCount = document.getElementById('filter-count');

function applyFilter() {{
    const threshold = parseFloat(filterSlider.value);
    filterVal.textContent = threshold.toFixed(2);
    let visible = 0;
    document.querySelectorAll('.cell').forEach(function(cell) {{
        const card = CARDS[cell.dataset.key];
        if (!card) return;
        const q = card.quality;
        const show = q === null || q >= threshold;
        cell.classList.toggle('hidden', !show);
        if (show) visible++;
    }});
    filterCount.textContent = '(' + visible + ' shown)';
}}

filterSlider.addEventListener('input', applyFilter);
// Initialize count on load
applyFilter();

// Per-cell interaction listeners
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
    if (evt.key === 'Escape' && modalBg.classList.contains('visible')) closeModal();
}});
    </script>
</body>
</html>
"""
