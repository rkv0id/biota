"""Render an archive.pkl as a browsable HTML page.

Loads a biota search run directory, extracts the archive's occupied cells,
encodes each one's thumbnail as a base64 data URL animated GIF with the
magma colormap applied, and writes a self-contained HTML file you can open
in any browser.

Usage:

    # Most common: view the most recent run
    uv run python scripts/view_archive.py

    # View a specific run by id
    uv run python scripts/view_archive.py --run 20260407-155010-crisp-thistle

    # Override output location
    uv run python scripts/view_archive.py --latest --output /tmp/archive.html

The HTML is self-contained (all data is inline as base64 data URLs) so you
can scp it off the cluster, email it, or open it offline. No external
dependencies on the viewing side, no webfonts, no CDN JS.

Cells are laid out by their (velocity, gyradius) coordinate. The third
descriptor (spectral entropy) is visible in the tooltip and detail modal.
Hover any cell for a compact tooltip with the descriptor triple and
quality. Click any cell to open a detail modal with the full parameters,
the larger animation, and the lineage info.
"""

import argparse
import pickle
import sys
from pathlib import Path

from biota.search.archive import Archive
from biota.viz.render import render_archive_page

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = ROOT / "archive-runs"


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
    html = render_archive_page(archive, run_id, run_dir)
    output.write_text(html)

    size_kb = output.stat().st_size / 1024
    print(f"wrote {output} ({size_kb:.1f} KB)")
    print(f"open with: open {output}")


if __name__ == "__main__":
    main()
