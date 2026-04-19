#!/usr/bin/env python3
"""suggest_signal_pairs.py -- Find interesting creature pairs from a signal-enabled archive.

Loads an archive pickle, computes pairwise emission-reception compatibility
between all occupied creatures, and ranks pairs by interaction type. Outputs
a ranked table and copy-ready ecosystem YAML snippets for the top pairs.

Only meaningful for signal-enabled archives. Non-signal archives are rejected
with a clear error.

Usage:
    python scripts/suggest_signal_pairs.py --archive archive/my-run
    python scripts/suggest_signal_pairs.py --archive archive/my-run/archive.pkl
    python scripts/suggest_signal_pairs.py --archive archive/my-run --mode repulsion --top 5
    python scripts/suggest_signal_pairs.py --archive archive/my-run --mode all --min-quality 0.8
    python scripts/suggest_signal_pairs.py --archive archive/my-run --yaml-only

Interaction modes:
    repulsion    Mutual chemorepulsion: both dots negative. Best for territorial
                 coexistence -- species actively partition space.
    pursuit      Asymmetric: one attracts, one repels. Produces chase dynamics.
    competition  Mutual chemotaxis: both dots positive. Species attract each
                 other and compete for the same space.
    blind        Near-zero compatibility: species ignore each other's signals.
                 Pure spatial competition, no chemical mediation.
    all          Show top pairs from all modes (default).
"""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────


def _load_archive(path: Path):
    """Load archive from a run directory or direct pickle path."""
    if path.is_dir():
        pkl = path / "archive.pkl"
        if not pkl.exists():
            # Try one level deeper (e.g. archive/run-id/archive.pkl)
            candidates = sorted(path.glob("*/archive.pkl"))
            if not candidates:
                sys.exit(f"error: no archive.pkl found under {path}")
            if len(candidates) > 1:
                sys.exit(
                    f"error: multiple archive.pkl files found under {path}. "
                    f"Pass a specific run directory.\n" + "\n".join(f"  {c}" for c in candidates)
                )
            pkl = candidates[0]
    elif path.suffix == ".pkl":
        pkl = path
    else:
        sys.exit(f"error: {path} is neither a directory nor a .pkl file")

    if not pkl.exists():
        sys.exit(f"error: archive.pkl not found at {pkl}")

    with open(pkl, "rb") as f:
        archive = pickle.load(f)

    return archive, pkl.parent


def _run_id_from_dir(run_dir: Path) -> str:
    return run_dir.name


def _dot(a: list[float], b: list[float]) -> float:
    return float(np.dot(a, b))


def _classify(ab: float, ba: float, threshold: float = 0.1) -> str:
    """Classify a pair by their mutual compatibility scores."""
    pos_ab = ab > threshold
    neg_ab = ab < -threshold
    pos_ba = ba > threshold
    neg_ba = ba < -threshold

    if neg_ab and neg_ba:
        return "repulsion"
    if pos_ab and pos_ba:
        return "competition"
    if (pos_ab and neg_ba) or (neg_ab and pos_ba):
        return "pursuit"
    return "blind"


def _asymmetry(ab: float, ba: float) -> float:
    return abs(ab - ba)


def _descriptor_distance(d1: tuple, d2: tuple) -> float:
    return float(np.linalg.norm(np.array(d1) - np.array(d2)))


MODE_LABEL = {
    "repulsion": "mutual chemorepulsion -- territorial coexistence candidates",
    "pursuit": "asymmetric -- one attracts, one repels -- chase dynamics",
    "competition": "mutual chemotaxis -- both attract, compete for same space",
    "blind": "chemically blind -- pure spatial competition, no signal coupling",
}

MODE_COLORS = {
    "repulsion": "\033[36m",  # cyan
    "pursuit": "\033[33m",  # yellow
    "competition": "\033[35m",  # magenta
    "blind": "\033[90m",  # dark gray
}
RESET = "\033[0m"
BOLD = "\033[1m"


def _color(text: str, code: str, use_color: bool) -> str:
    return f"{code}{text}{RESET}" if use_color else text


# ── YAML generation ───────────────────────────────────────────────────────────


def _yaml_snippet(run_id: str, cid_a: str, cid_b: str, n: int = 3) -> str:
    return (
        f"- name: {cid_a.split('-')[-1]}-vs-{cid_b.split('-')[-1]}\n"
        f"  grid: [192, 192]\n"
        f"  steps: 1000\n"
        f"  snapshot_every: 100\n"
        f"  border: wall\n"
        f"  output_format: gif\n"
        f"  sources:\n"
        f"    - run: {run_id}\n"
        f"      creature_id: {cid_a}\n"
        f"      n: {n}\n"
        f"    - run: {run_id}\n"
        f"      creature_id: {cid_b}\n"
        f"      n: {n}\n"
        f"  spawn:\n"
        f"    patch: 24\n"
        f"    min_dist: 40\n"
        f"    seed: 42"
    )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--archive",
        "-a",
        type=Path,
        required=True,
        help="Path to a run directory (containing archive.pkl) or directly to archive.pkl.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["repulsion", "pursuit", "competition", "blind", "all"],
        default="all",
        help="Interaction type to surface (default: all).",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=5,
        help="Number of top pairs to show per mode (default: 5).",
    )
    parser.add_argument(
        "--min-quality",
        "-q",
        type=float,
        default=0.0,
        help="Minimum quality threshold for both creatures (default: 0.0).",
    )
    parser.add_argument(
        "--n-creatures",
        type=int,
        default=3,
        help="n per source in generated YAML snippets (default: 3).",
    )
    parser.add_argument(
        "--yaml-only",
        action="store_true",
        help="Print only the YAML snippets for top pairs, no table.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI color output.",
    )
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    # ── load archive ──────────────────────────────────────────────────────────
    archive, run_dir = _load_archive(args.archive)
    run_id = _run_id_from_dir(run_dir)

    # ── check signal ──────────────────────────────────────────────────────────
    occupied = list(archive.iter_occupied())
    if not occupied:
        sys.exit("error: archive is empty -- no creatures to compare")

    sample_params = occupied[0][1].params
    if "emission_vector" not in sample_params:
        sys.exit(
            "error: this archive is not signal-enabled.\n"
            "suggest_signal_pairs.py only works with archives produced with --signal-field.\n"
            "Run biota search --signal-field ... first."
        )

    # ── filter by quality ─────────────────────────────────────────────────────
    creatures = [
        (cid, result)
        for _, result in occupied
        for cid in [result.creature_id or str(_)]
        if result.quality is not None and result.quality >= args.min_quality
    ]

    # Rebuild with correct creature_id from result
    creatures = []
    for _, result in occupied:
        if result.quality is None or result.quality < args.min_quality:
            continue
        if "emission_vector" not in result.params:
            continue
        creatures.append(result)

    if len(creatures) < 2:
        sys.exit(
            f"error: fewer than 2 creatures pass --min-quality {args.min_quality}. "
            f"Lower the threshold or use a larger archive."
        )

    # ── compute all pairs ─────────────────────────────────────────────────────
    pairs = []
    for i in range(len(creatures)):
        for j in range(i + 1, len(creatures)):
            a = creatures[i]
            b = creatures[j]

            ev_a = a.params["emission_vector"]
            rp_a = a.params["receptor_profile"]
            ev_b = b.params["emission_vector"]
            rp_b = b.params["receptor_profile"]

            ab = _dot(ev_a, rp_b)  # A's signal effect on B
            ba = _dot(ev_b, rp_a)  # B's signal effect on A

            mode = _classify(ab, ba)
            asym = _asymmetry(ab, ba)
            desc_dist = (
                _descriptor_distance(a.descriptors, b.descriptors)
                if a.descriptors and b.descriptors
                else 0.0
            )
            avg_quality = ((a.quality or 0) + (b.quality or 0)) / 2

            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "ab": ab,
                    "ba": ba,
                    "mode": mode,
                    "asym": asym,
                    "desc_dist": desc_dist,
                    "avg_quality": avg_quality,
                    # Score: strong signal + behavioral diversity + quality
                    "score": abs(ab) + abs(ba) + desc_dist * 0.5 + avg_quality * 0.3,
                }
            )

    if not pairs:
        sys.exit("error: no pairs to compare")

    # ── select modes to show ──────────────────────────────────────────────────
    modes_to_show = (
        ["repulsion", "pursuit", "competition", "blind"] if args.mode == "all" else [args.mode]
    )

    # ── print results ─────────────────────────────────────────────────────────
    all_yaml: list[str] = []

    for mode in modes_to_show:
        mode_pairs = sorted(
            [p for p in pairs if p["mode"] == mode],
            key=lambda p: -p["score"],
        )[: args.top]

        if not mode_pairs:
            continue

        if not args.yaml_only:
            color = MODE_COLORS.get(mode, "")
            header = f"\n{_color(mode.upper(), BOLD + color, use_color)}"
            print(header)
            print(_color(MODE_LABEL[mode], color, use_color))
            print()

            col_w = 42
            print(
                f"  {'creature A':<{col_w}} {'creature B':<{col_w}} "
                f"{'A→B':>6} {'B→A':>6} {'asym':>6} {'d_desc':>7} {'avg_q':>6}"
            )
            print("  " + "-" * (col_w * 2 + 40))

            for p in mode_pairs:
                cid_a = p["a"].creature_id or "?"
                cid_b = p["b"].creature_id or "?"
                print(
                    f"  {cid_a:<{col_w}} {cid_b:<{col_w}} "
                    f"{p['ab']:>+6.2f} {p['ba']:>+6.2f} "
                    f"{p['asym']:>6.2f} {p['desc_dist']:>7.3f} "
                    f"{p['avg_quality']:>6.3f}"
                )

        for p in mode_pairs:
            cid_a = p["a"].creature_id or "?"
            cid_b = p["b"].creature_id or "?"
            all_yaml.append(
                f"# {mode}: A→B={p['ab']:+.2f}  B→A={p['ba']:+.2f}  "
                f"desc_dist={p['desc_dist']:.3f}  avg_q={p['avg_quality']:.3f}\n"
                + _yaml_snippet(run_id, cid_a, cid_b, n=args.n_creatures)
            )

    # ── print YAML ────────────────────────────────────────────────────────────
    if not args.yaml_only:
        print(f"\n{'─' * 80}")
        print(f"{BOLD}experiments.yaml snippets{RESET if use_color else ''}")
        print(f"{'─' * 80}")
        print("experiments:")

    for snippet in all_yaml:
        print()
        for line in snippet.splitlines():
            print("  " + line if not line.startswith("#") else line)

    if not args.yaml_only:
        print()
        print(
            f"  # {len(all_yaml)} pair(s) from archive: {run_id}\n"
            f"  # run: biota ecosystem --config experiments.yaml --device cuda"
        )


if __name__ == "__main__":
    main()
