# biota

Distributed quality-diversity search over Flow-Lenia, with a live dashboard.

## What this is

Biota runs MAP-Elites over Flow-Lenia parameter space on a Ray cluster (or your laptop) and produces an interactive atlas of artificial creatures organized by their behavior — how they move, how big they are, how internally structured they are. Both the search and the atlas viewer are in a single CLI tool with a vanilla-JS dashboard.

The compute fabric is Ray. The artifact is the atlas.

## Quickstart

```bash
git clone https://github.com/rkv0id/biota
cd biota
uv sync
uv run biota search --preset dev --budget 200
```

Open http://localhost:8000 in a browser. The dashboard fills in as creatures are discovered.

To browse a previously-completed run as a static gallery:

```bash
uv run biota view runs/<run_id>
```

To export a self-contained static bundle for hosting on GitHub Pages or similar:

```bash
uv run biota export <run_id>
```

To run on a homelab Ray cluster, SSH into the head node and run biota in tmux there. The dashboard is reachable from your laptop browser at the cluster's IP.

## What it produces

(Hero screenshot lands here once we have one.)

The atlas is a 3D MAP-Elites archive over (speed, size, structure), rendered as three pairwise 2D projections that share the same cell-rendering code. Click any cell to see the creature in detail with its full parameters and lineage.

## Architecture

A single Python process is the driver. It owns the in-memory archive, runs the search loop, hosts the FastAPI dashboard server, and writes checkpoints to local disk. Ray workers are stateless: they take parameters, run a Flow-Lenia rollout, compute behavior descriptors, and return a result. A FrameRelay Ray actor lets workers push live preview frames back to the dashboard's Live tab without breaking the stateless contract.

## Benchmarks

Empty until M4 ships. The structure is fixed so the file looks the same once numbers land.

| Configuration | Rollouts/sec | Creatures/hour | Notes |
|---|---|---|---|
| Laptop M4 (MPS) | TBD | TBD | dev preset |
| Cluster CPU-only (3 nodes) | TBD | TBD | standard preset |
| Cluster GPU (3x RTX 5060 Ti) | TBD | TBD | standard preset |

Random search vs MAP-Elites at the same compute budget, same seed:

| Mode | Cells filled | Distinct creatures |
|---|---|---|
| `--mode random` | TBD | TBD |
| `--mode qd` | TBD | TBD |

## References

- Plantec et al. 2022, Flow-Lenia (best paper, ALIFE 2023): https://arxiv.org/abs/2212.07906
- Plantec et al. 2025, Flow-Lenia journal version: https://arxiv.org/abs/2506.08569
- Reference JAX implementation: https://github.com/erwanplantec/FlowLenia
- Mouret and Clune 2015, MAP-Elites: https://arxiv.org/abs/1504.04909
- Moroz, Reintegration tracking: https://michaelmoroz.github.io/Reintegration-Tracking/

## Status

Pre-alpha. The repo exists, the design is locked, M0 (FlowLenia port and mass-conservation verification) is the active milestone.
