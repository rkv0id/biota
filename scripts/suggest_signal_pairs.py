#!/usr/bin/env python3
"""suggest_signal_pairs.py -- Find interesting creature pairs from signal-enabled archives.

Loads one or more signal archive pickles, computes pairwise emission-reception
compatibility between all occupied creatures (within and across archives), and
ranks pairs by interaction type. Outputs a ranked table and a ready-to-paste
experiments.yaml block.

Only meaningful for signal-enabled archives. Non-signal archives are skipped
with a warning.

Usage:
    # Single archive, all modes, top 3 per mode
    python scripts/suggest_signal_pairs.py \\
        --archive /mnt/scratch/biota/archive/04-wall-signal-specialist

    # Multiple archives -- cross-archive pairs included
    python scripts/suggest_signal_pairs.py \\
        --archive /mnt/scratch/biota/archive/04-wall-signal-specialist \\
        --archive /mnt/scratch/biota/archive/08-torus-signal-specialist \\
        --mode repulsion --top 4

    # Save experiments.yaml directly
    python scripts/suggest_signal_pairs.py \\
        --archive /mnt/scratch/biota/archive/04-wall-signal-specialist \\
        --top 3 --yaml-only > experiments.yaml

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
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ── types ─────────────────────────────────────────────────────────────────────


@dataclass
class Creature:
    creature_id: str
    run_id: str
    archive_dir: str
    emission_vector: list
    receptor_profile: list
    quality: float
    descriptors: tuple | None


# ── archive loading ───────────────────────────────────────────────────────────


def _find_pkl(path: Path) -> Path:
    if path.suffix == ".pkl":
        if not path.exists():
            sys.exit(f"error: not found: {path}")
        return path
    if not path.exists():
        sys.exit(f"error: not found: {path}")
    direct = path / "archive.pkl"
    if direct.exists():
        return direct
    candidates = sorted(path.glob("*/archive.pkl"))
    if not candidates:
        sys.exit(f"error: no archive.pkl found under {path}")
    if len(candidates) > 1:
        lines = "\n".join(f"  {c.parent}" for c in candidates)
        sys.exit(
            f"error: multiple runs found under {path} -- pass a specific run directory:\n{lines}"
        )
    return candidates[0]


def _load_creatures(archive_path: Path, min_quality: float) -> list[Creature]:
    pkl = _find_pkl(archive_path)
    run_id = pkl.parent.name
    archive_dir = str(pkl.parent.parent)

    with open(pkl, "rb") as f:
        archive = pickle.load(f)

    occupied = list(archive.iter_occupied())
    if not occupied:
        print(f"warning: {archive_path} is empty, skipping", file=sys.stderr)
        return []

    if "emission_vector" not in occupied[0][1].params:
        print(f"warning: {archive_path} is not signal-enabled, skipping", file=sys.stderr)
        return []

    creatures = []
    for _, result in occupied:
        if result.quality is None or result.quality < min_quality:
            continue
        if "emission_vector" not in result.params:
            continue
        cid = result.creature_id or f"{run_id}-{result.seed}"
        creatures.append(
            Creature(
                creature_id=cid,
                run_id=run_id,
                archive_dir=archive_dir,
                emission_vector=result.params["emission_vector"],
                receptor_profile=result.params["receptor_profile"],
                quality=result.quality,
                descriptors=tuple(result.descriptors) if result.descriptors else None,
            )
        )

    print(f"  {len(creatures)} creatures from {run_id} (q >= {min_quality})", file=sys.stderr)
    return creatures


# ── pair scoring ──────────────────────────────────────────────────────────────


def _dot(a: list, b: list) -> float:
    return float(np.dot(a, b))


def _classify(ab: float, ba: float, t: float = 0.1) -> str:
    if ab < -t and ba < -t:
        return "repulsion"
    if ab > t and ba > t:
        return "competition"
    if (ab > t and ba < -t) or (ab < -t and ba > t):
        return "pursuit"
    return "blind"


def _build_pairs(creatures: list[Creature]) -> list[dict]:
    pairs = []
    for i in range(len(creatures)):
        for j in range(i + 1, len(creatures)):
            a, b = creatures[i], creatures[j]
            ab = _dot(a.emission_vector, b.receptor_profile)
            ba = _dot(b.emission_vector, a.receptor_profile)
            dd = (
                float(np.linalg.norm(np.array(a.descriptors) - np.array(b.descriptors)))
                if a.descriptors and b.descriptors
                else 0.0
            )
            avg_q = (a.quality + b.quality) / 2
            pairs.append(
                {
                    "a": a,
                    "b": b,
                    "ab": ab,
                    "ba": ba,
                    "mode": _classify(ab, ba),
                    "asym": abs(ab - ba),
                    "desc_dist": dd,
                    "avg_quality": avg_q,
                    "cross": a.archive_dir != b.archive_dir,
                    # score: signal strength + behavioral diversity + quality
                    "score": abs(ab) + abs(ba) + dd * 0.3 + avg_q * 0.5,
                }
            )
    return pairs


# ── YAML ──────────────────────────────────────────────────────────────────────


def _yaml_snippet(p: dict, n: int, steps: int, grid: int) -> str:
    a, b = p["a"], p["b"]
    name = f"{a.creature_id.split('-')[-1]}-vs-{b.creature_id.split('-')[-1]}"
    cross_tag = " [cross-archive]" if p["cross"] else ""
    comment = (
        f"# {p['mode']}{cross_tag}: "
        f"A\u2192B={p['ab']:+.2f}  B\u2192A={p['ba']:+.2f}  "
        f"asym={p['asym']:.2f}  dist={p['desc_dist']:.2f}  avg_q={p['avg_quality']:.3f}"
    )
    snap = max(1, steps // 100)
    return "\n".join(
        [
            comment,
            f"- name: {name}",
            f"  grid: [{grid}, {grid}]",
            f"  steps: {steps}",
            f"  snapshot_every: {snap}",
            "  border: wall",
            "  output_format: gif",
            "  sources:",
            f"    - archive_dir: {a.archive_dir}",
            f"      run: {a.run_id}",
            f"      creature_id: {a.creature_id}",
            f"      n: {n}",
            f"    - archive_dir: {b.archive_dir}",
            f"      run: {b.run_id}",
            f"      creature_id: {b.creature_id}",
            f"      n: {n}",
            "  spawn:",
            "    patch: 24",
            "    min_dist: 40",
            "    seed: 42",
        ]
    )


# ── display ───────────────────────────────────────────────────────────────────

MODE_LABEL = {
    "repulsion": "mutual chemorepulsion -- territorial coexistence candidates",
    "pursuit": "asymmetric -- one attracts, one repels -- chase dynamics",
    "competition": "mutual chemotaxis -- both attract, compete for same space",
    "blind": "chemically blind -- pure spatial competition, no signal mediation",
}
MODE_COLOR = {
    "repulsion": "\033[36m",
    "pursuit": "\033[33m",
    "competition": "\033[35m",
    "blind": "\033[90m",
}
RESET = "\033[0m"
BOLD = "\033[1m"


def _c(text: str, code: str, on: bool) -> str:
    return f"{code}{text}{RESET}" if on else text


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
        action="append",
        dest="archives",
        metavar="PATH",
        required=True,
        help="Archive run directory or archive.pkl. Repeat for multiple archives.",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["repulsion", "pursuit", "competition", "blind", "all"],
        default="all",
    )
    parser.add_argument(
        "--top",
        "-n",
        type=int,
        default=3,
        help="Top pairs per mode (default: 3).",
    )
    parser.add_argument(
        "--min-quality",
        "-q",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--n-creatures",
        type=int,
        default=3,
        help="n per source in YAML (default: 3).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=2000,
        help="steps in YAML snippets (default: 2000).",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=192,
        help="Grid size in YAML snippets (default: 192).",
    )
    parser.add_argument(
        "--yaml-only",
        action="store_true",
        help="Print only YAML, no table.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
    )
    args = parser.parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    # load
    print("loading archives...", file=sys.stderr)
    all_creatures: list[Creature] = []
    for path in args.archives:
        all_creatures.extend(_load_creatures(path, args.min_quality))

    if len(all_creatures) < 2:
        sys.exit("error: need at least 2 signal creatures")

    # build pairs
    pairs = _build_pairs(all_creatures)
    n_cross = sum(1 for p in pairs if p["cross"])
    print(
        f"  {len(pairs)} pairs ({n_cross} cross-archive, {len(pairs) - n_cross} same-archive)",
        file=sys.stderr,
    )

    modes = ["repulsion", "pursuit", "competition", "blind"] if args.mode == "all" else [args.mode]

    all_yaml: list[str] = []

    for mode in modes:
        mode_pairs = sorted(
            [p for p in pairs if p["mode"] == mode],
            key=lambda p: -p["score"],
        )[: args.top]

        if not mode_pairs:
            continue

        if not args.yaml_only:
            col = MODE_COLOR.get(mode, "")
            print(f"\n{_c(mode.upper(), BOLD + col, use_color)}")
            print(_c(MODE_LABEL[mode], col, use_color))
            print()
            w = 44
            print(
                f"  {'creature A':<{w}} {'creature B':<{w}} "
                f"{'A\u2192B':>6} {'B\u2192A':>6} {'asym':>6} {'dist':>6} {'avg_q':>6} {'cross':>5}"
            )
            print("  " + "-" * (w * 2 + 42))
            for p in mode_pairs:
                print(
                    f"  {p['a'].creature_id:<{w}} {p['b'].creature_id:<{w}} "
                    f"{p['ab']:>+6.2f} {p['ba']:>+6.2f} "
                    f"{p['asym']:>6.2f} {p['desc_dist']:>6.2f} "
                    f"{p['avg_quality']:>6.3f} "
                    f"{'yes' if p['cross'] else '':>5}"
                )

        for p in mode_pairs:
            all_yaml.append(_yaml_snippet(p, n=args.n_creatures, steps=args.steps, grid=args.grid))

    if not args.yaml_only:
        print(f"\n{'─' * 80}")
        print(f"{BOLD if use_color else ''}experiments.yaml{RESET if use_color else ''}")
        print(f"{'─' * 80}")

    print("experiments:")
    for snippet in all_yaml:
        print()
        for line in snippet.splitlines():
            if line.startswith("#"):
                print(line)
            else:
                print("  " + line)

    if not args.yaml_only:
        print(
            f"\n  # {len(all_yaml)} experiment(s) from {len(args.archives)} archive(s)"
            f"\n  # run: biota ecosystem --config experiments.yaml --device cuda"
        )


if __name__ == "__main__":
    main()
