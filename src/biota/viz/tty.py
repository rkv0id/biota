"""TTY-aware terminal display for the search loop.

Two phases:

  Calibration: single overwriting line with ASCII progress bar and survivor count.
  MAP-Elites:  7-line overwriting block with progress, archive stats, insertion
               summary, and per-descriptor coverage bars.

The display class checks sys.stdout.isatty() once at construction. In non-TTY
mode it falls back to plain append-only lines matching the pre-v4.1.0 format,
plus a periodic summary every NONTTTY_SUMMARY_EVERY rollouts.

All output goes to sys.stderr so stdout stays clean for scripting (archive_size=N).
ANSI cursor control is restricted to TTY mode; non-TTY output contains no escape codes.
"""

import sys
import time
from collections import deque
from dataclasses import dataclass, field

BAR_WIDTH = 28
NONTTTY_SUMMARY_EVERY = 50
_ESC = "\033"
_UP = f"{_ESC}[{{n}}A"  # move cursor up n lines
_COL0 = "\r"  # carriage return to column 0
_CLEAR = f"{_ESC}[K"  # clear to end of line


def _bar(filled: int, total: int, width: int = BAR_WIDTH) -> str:
    """ASCII progress bar. Uses block chars for a smooth look."""
    if total <= 0:
        return "░" * width
    frac = min(filled / total, 1.0)
    n_full = int(frac * width)
    n_empty = width - n_full
    return "█" * n_full + "░" * n_empty


def _desc_bar(values: list[float], width: int = BAR_WIDTH) -> tuple[str, float, float]:
    """Coverage bar for one descriptor axis across archive creatures.

    Bins values into width buckets, renders filled buckets as block chars
    with density shading. Returns (bar_str, p5, p95).
    """
    if not values:
        return "░" * width, 0.0, 0.0

    import numpy as np

    arr = sorted(values)
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    span = p95 - p5
    if span <= 0:
        # All values identical -- show a single filled column at center
        bar = "░" * (width // 2) + "█" + "░" * (width - width // 2 - 1)
        return bar, p5, p95

    # Bucket counts
    buckets = [0] * width
    for v in arr:
        idx = int((v - p5) / span * (width - 1))
        idx = max(0, min(width - 1, idx))
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    shades = " ░▒▓█"
    bar = ""
    for count in buckets:
        level = int(count / max_count * (len(shades) - 1))
        bar += shades[level]
    return bar, p5, p95


@dataclass
class _InsertionWindow:
    """Rolling window of last N insertion outcomes for the last-10 summary."""

    window: deque[str] = field(default_factory=lambda: deque(maxlen=10))

    def add(self, status: str) -> None:
        self.window.append(status)

    def summary(self) -> str:
        counts: dict[str, int] = {}
        for s in self.window:
            counts[s] = counts.get(s, 0) + 1
        parts = []
        for label, key in [
            ("ins", "inserted"),
            ("rep", "replaced"),
            ("rq", "rejected_quality"),
            ("rs", "rejected_similarity"),
            ("rf", "rejected_filter"),
        ]:
            if key in counts:
                parts.append(f"{counts[key]} {label}")
        return "  ".join(parts) if parts else "-"


class SearchDisplay:
    """Manages terminal output for a biota search run.

    Construct once before search starts, pass on_calibration_progress and
    on_event to the search() call. Call finish() after search completes.
    """

    def __init__(
        self,
        budget: int,
        calibration: int,
        descriptor_names: tuple[str, str, str],
        device: str,
        workers: int,
    ) -> None:
        self._tty = sys.stderr.isatty()
        self._run_id = ""  # set from SearchStarted event
        self._budget = budget
        self._calibration = calibration
        self._descriptor_names = descriptor_names
        self._device = device
        self._workers = workers

        # State updated by callbacks
        self._cal_completed = 0
        self._cal_survivors = 0
        self._cal_start = time.monotonic()
        self._cal_done = False

        self._completed = 0
        self._archive_size = 0
        self._fill_pct = 0.0
        self._best_quality = 0.0
        self._last_insert_ago = 0
        self._window = _InsertionWindow()
        # descriptor values across archive: list per axis
        self._desc_values: list[list[float]] = [[], [], []]
        self._last_summary = 0
        self._block_lines = 0  # how many lines the current TTY block occupies

    # === calibration callbacks ===

    def on_calibration_progress(self, completed: int, total: int, n_survivors: int) -> None:
        self._cal_completed = completed
        self._cal_survivors = n_survivors
        if self._tty:
            self._render_cal_line()
        else:
            # Non-TTY: one line per batch drain (not per rollout -- too noisy)
            pct = int(completed / total * 100) if total else 0
            print(
                f"[calibrating] {completed}/{total} ({pct}%)  survivors={n_survivors}",
                file=sys.stderr,
            )

    def on_calibration_done(
        self,
        n_survivors: int,
        descriptor_names: tuple[str, str, str],
        axis_ranges: list[tuple[float, float]] | None = None,
    ) -> None:
        """Call once after calibration completes to print the summary line."""
        self._cal_done = True
        elapsed = time.monotonic() - self._cal_start
        range_str = ""
        if axis_ranges:
            parts = []
            for (lo, hi), name in zip(axis_ranges, descriptor_names, strict=False):
                short = name[:6]
                parts.append(f"{short}=[{lo:.3g},{hi:.3g}]")
            range_str = "  " + "  ".join(parts)
        msg = (
            f"[calibrated]   {self._calibration} rollouts  "
            f"{n_survivors} survivors  "
            f"{elapsed:.1f}s{range_str}"
        )
        if self._tty:
            sys.stderr.write(f"{_COL0}{_CLEAR}{msg}\n")
            sys.stderr.flush()
        else:
            print(msg, file=sys.stderr)

    # === search event callbacks ===

    def on_search_started(self, run_id: str, config: object) -> None:
        self._run_id = run_id
        # Non-TTY only: print a single start line. TTY rendering begins on the
        # first RolloutCompleted so Ray/CUDA init noise lands before the block.
        if not self._tty:
            print(
                f"[search] starting run {run_id}  budget={self._budget}  device={self._device}",
                file=sys.stderr,
            )

    def on_rollout_completed(
        self,
        completed: int,
        status: str,
        quality: float | None,
        rejection_reason: str | None,
        seed: int,
        descriptors: tuple[float, float, float] | None,
    ) -> None:
        self._completed = completed
        self._last_insert_ago += 1
        self._window.add(status)

        if status in ("inserted", "replaced"):
            self._last_insert_ago = 0
            if quality is not None and quality > self._best_quality:
                self._best_quality = quality

        if self._tty:
            self._render_search_block()
        else:
            # Append-only: one line per rollout
            tag = {
                "inserted": "inserted ",
                "replaced": "replaced ",
                "rejected_filter": "rej:filt ",
                "rejected_quality": "rej:qual ",
                "rejected_similarity": "rej:sim  ",
            }.get(status, status[:9].ljust(9))
            q_str = f"q={quality:.3f}" if quality is not None else "q=  -  "
            reason = f"  {rejection_reason}" if rejection_reason else ""
            print(
                f"[{completed:4d}/{self._budget:4d}] seed={seed:<5d} {tag} {q_str}{reason}",
                file=sys.stderr,
            )
            if completed % NONTTTY_SUMMARY_EVERY == 0:
                self._print_nonttty_summary()

    def on_archive_snapshot(
        self,
        archive_size: int,
        fill_pct: float,
        desc_values: list[list[float]],
    ) -> None:
        """Update archive state for the coverage bars. Call after each insertion."""
        self._archive_size = archive_size
        self._fill_pct = fill_pct
        self._desc_values = desc_values

    def on_checkpoint(self, path: str, archive_size: int) -> None:
        if not self._tty:
            print(f"[checkpoint] {path} ({archive_size} cells)", file=sys.stderr)

    def on_search_finished(self, completed: int, archive_size: int, elapsed: float) -> None:
        if self._tty:
            # One final full render, then a newline to leave the block on screen
            self._render_search_block()
            sys.stderr.write("\n")
            sys.stderr.flush()
        print(
            f"[done] {completed} rollouts, {archive_size} archive cells, {elapsed:.1f}s elapsed",
            file=sys.stderr,
        )

    # === private rendering ===

    def _render_cal_line(self) -> None:
        completed = self._cal_completed
        total = self._calibration
        elapsed = time.monotonic() - self._cal_start
        bar = _bar(completed, total)
        eta_str = ""
        if completed > 0:
            eta = elapsed / completed * (total - completed)
            eta_str = f"  eta {eta:.0f}s"
        survivors_str = f"  surv={self._cal_survivors}"
        line = f"[calibrating]  {bar}  {completed}/{total}{survivors_str}{eta_str}"
        sys.stderr.write(f"{_COL0}{_CLEAR}{line}")
        sys.stderr.flush()

    def _render_search_block(self) -> None:
        completed = self._completed
        budget = self._budget
        d_names = self._descriptor_names

        # Move cursor back up to overwrite previous block
        if self._block_lines > 0:
            sys.stderr.write(_UP.format(n=self._block_lines))

        lines: list[str] = []

        # Line 1: header
        lines.append(
            f"[search]   {self._run_id}  "
            f"budget={budget}  device={self._device}  workers={self._workers}"
        )

        # Line 2: progress bar
        bar = _bar(completed, budget)
        pct = int(completed / budget * 100) if budget else 0
        lines.append(f"progress   {bar}  {completed}/{budget}  {pct}%")

        # Line 3: archive stats
        lines.append(
            f"archive    {self._archive_size} cells  "
            f"{self._fill_pct:.0%} fill  "
            f"best q={self._best_quality:.3f}  "
            f"last ins {self._last_insert_ago} ago"
        )

        # Line 4: last-10 insertion summary
        lines.append(f"last 10    {self._window.summary()}")

        # Lines 5-7: descriptor coverage bars
        for i, name in enumerate(d_names):
            vals = self._desc_values[i] if i < len(self._desc_values) else []
            bar_str, p5, p95 = _desc_bar(vals)
            short = name[:10].ljust(10)
            lines.append(f"{short}   {bar_str}  [{p5:.3g} .. {p95:.3g}]")

        self._block_lines = len(lines)
        for line in lines:
            sys.stderr.write(f"{_COL0}{_CLEAR}{line}\n")
        sys.stderr.flush()

    def _print_nonttty_summary(self) -> None:
        print(
            f"[summary] {self._completed}/{self._budget}  "
            f"archive={self._archive_size}  "
            f"fill={self._fill_pct:.0%}  "
            f"best_q={self._best_quality:.3f}",
            file=sys.stderr,
        )
