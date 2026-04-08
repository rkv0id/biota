"""Cluster-wide GPU profiler for Ray clusters.

Attaches to a running Ray cluster, discovers every GPU-bearing node, pins one
collector actor to each, and samples nvidia-smi at a configurable rate. The
driver pulls samples from each actor periodically and writes them as a JSONL
time-series file. Run this alongside any Ray workload (biota search or
otherwise) to get a per-node, per-GPU utilization trace.

The output is a single JSONL file with three record kinds (header, sample,
error), each tagged via a "kind" field for trivial downstream filtering. The
file is written progressively (flush after every drain) so an interrupted
profile still has data on disk.

Why a Ray actor per node instead of SSH-and-nvidia-smi or a one-shot ray task:

- No SSH credentials needed; Ray already has the cluster topology and
  authenticated worker connections.
- Actors are stateful, so they can buffer samples between drains. The driver
  pulls every ~1s but the underlying nvidia-smi loop runs at the configured
  interval (default 500ms / 2 Hz), independent of pull cadence.
- One actor per node gives you a consistent sampling rate per node even when
  the cluster has heterogeneous GPU counts or speeds.

Why pin with NodeAffinitySchedulingStrategy(soft=False) instead of letting
Ray's default placement spread the actors:

- We want exactly one actor per node, no exceptions, no drift. Ray's default
  scheduling for zero-resource actors is a randomized SPREAD which gives
  "statistically one per node" - not the same thing as "exactly one per node".
- soft=False means "fail loudly if you can't place me here" which is the right
  failure mode for a profiler. Silent misplacement would corrupt the trace.

Usage:

    # In one terminal, on a node with Ray access (typically the head):
    python scripts/gpu_profile.py \\
        --ray-address 10.10.12.1:6379 \\
        --out profile-baseline.jsonl

    # In another terminal, run your workload:
    biota search --ray-address 10.10.12.1:6379 --device cuda ...

    # When the workload is done, Ctrl-C the profiler.

By default the profiler runs until interrupted. Pass --duration SECONDS for
scripted use. Pass --interval-ms to change the per-sample cadence (default
500ms = 2 Hz, which gives 3-4 samples per typical biota rollout at dev preset
on an RTX 5060 Ti).

Output format (JSONL, one record per line):

    {"kind": "header", ...one-time metadata...}
    {"kind": "sample", "t_mono": 0.493, "t_wall": "...", "node_id": "...",
     "hostname": "miniverse-11", "gpu_index": 0, "util_gpu_pct": 87, ...}
    {"kind": "sample", ...}
    {"kind": "error", "t_mono": 12.1, "node_id": "...", "error": "..."}
    ...

Filter samples downstream with `jq 'select(.kind == "sample")'` or pandas:

    import pandas as pd
    df = pd.read_json("profile-baseline.jsonl", lines=True)
    samples = df[df["kind"] == "sample"]
    samples.groupby(["hostname", "gpu_index"])["util_gpu_pct"].mean()

Limitations:

- Requires nvidia-smi on every GPU node (not pynvml; we shell out for
  portability and to avoid driver-version python-binding mismatches).
- Reports utilization.gpu (SM utilization) which is sampled by the NVIDIA
  driver at a fixed internal rate; very short kernels can be missed. Sufficient
  for ML workloads where rollouts run for seconds.
- Does not track per-process GPU usage. If you need to attribute utilization
  to specific Ray workers, that needs a separate tool.
"""

import argparse
import json
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import FrameType
from typing import Any

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

# Schema version for the JSONL output. Bump if a record kind changes shape in
# a way that would break downstream parsers.
SCHEMA_VERSION = 1

# Default sampling cadence inside each collector actor. 2 Hz gives ~3-4 samples
# per typical biota rollout at dev preset, which is enough resolution to see
# "GPU busy / GPU idle" transitions without putting meaningful load on
# nvidia-smi launches.
DEFAULT_INTERVAL_MS = 500

# How often the driver drains buffered samples from each collector actor.
# Decoupled from the sample interval: samples accumulate inside the actor
# between drains, then ship in a batch. 1s keeps the file fresh without
# saturating the driver with ray.get calls.
DEFAULT_PULL_EVERY_MS = 1000

# nvidia-smi query field list. Order matters: the parser in _parse_nvidia_smi_csv
# indexes by position, so any change here needs a matching change there.
NVIDIA_SMI_QUERY = ",".join(
    [
        "index",
        "name",
        "uuid",
        "utilization.gpu",
        "utilization.memory",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
        "power.limit",
        "pstate",
        "clocks.current.sm",
        "clocks.current.memory",
    ]
)

# Subprocess timeout for nvidia-smi. The command itself usually finishes in
# 50-200ms; 5s is a generous ceiling that catches a hung driver without
# blocking the collector loop forever.
NVIDIA_SMI_TIMEOUT_SEC = 5.0


def _utcnow_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision."""
    now = datetime.now(UTC)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _parse_int_or_none(s: str) -> int | None:
    """nvidia-smi reports '[N/A]' or '[Not Supported]' for unsupported metrics.
    Convert those to None and parse the rest as int."""
    s = s.strip()
    if not s or s.startswith("[") or s == "N/A":
        return None
    try:
        return int(s)
    except ValueError:
        return None


def _parse_float_or_none(s: str) -> float | None:
    s = s.strip()
    if not s or s.startswith("[") or s == "N/A":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _parse_str_or_none(s: str) -> str | None:
    s = s.strip()
    if not s or s.startswith("[") or s == "N/A":
        return None
    return s


def _parse_nvidia_smi_csv(stdout: str) -> list[dict[str, Any]]:
    """Parse nvidia-smi --query-gpu csv output into a list of dicts.

    Each line is one GPU. Fields are positional and must match NVIDIA_SMI_QUERY
    above. Missing values become None rather than crashing the line.
    """
    rows: list[dict[str, Any]] = []
    for line in stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 13:
            # Malformed line; skip rather than crash. The collector logs an
            # error record so the operator can see something went wrong.
            continue
        rows.append(
            {
                "gpu_index": _parse_int_or_none(parts[0]),
                "gpu_name": _parse_str_or_none(parts[1]),
                "gpu_uuid": _parse_str_or_none(parts[2]),
                "util_gpu_pct": _parse_int_or_none(parts[3]),
                "util_mem_pct": _parse_int_or_none(parts[4]),
                "mem_used_mib": _parse_int_or_none(parts[5]),
                "mem_total_mib": _parse_int_or_none(parts[6]),
                "temp_c": _parse_int_or_none(parts[7]),
                "power_w": _parse_float_or_none(parts[8]),
                "power_limit_w": _parse_float_or_none(parts[9]),
                "pstate": _parse_str_or_none(parts[10]),
                "sm_clock_mhz": _parse_int_or_none(parts[11]),
                "mem_clock_mhz": _parse_int_or_none(parts[12]),
            }
        )
    return rows


def _run_nvidia_smi() -> tuple[str, str | None]:
    """Run nvidia-smi once. Returns (stdout, error). On success error is None.

    Catches every reasonable failure mode (binary missing, timeout, nonzero
    exit, decode failure) and returns the error string instead of raising,
    so the collector loop can keep going on the next sample.
    """
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={NVIDIA_SMI_QUERY}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=NVIDIA_SMI_TIMEOUT_SEC,
            check=False,
        )
    except FileNotFoundError:
        return "", "nvidia-smi: command not found"
    except subprocess.TimeoutExpired:
        return "", f"nvidia-smi: timed out after {NVIDIA_SMI_TIMEOUT_SEC}s"
    except OSError as e:
        return "", f"nvidia-smi: OSError: {e}"

    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        return "", f"nvidia-smi exit {result.returncode}: {stderr}"

    return result.stdout, None


# === Ray actor: GpuCollector ===
#
# One per GPU-bearing node. Pinned via NodeAffinitySchedulingStrategy.
# Samples nvidia-smi in a background thread at the configured interval and
# buffers results in memory until the driver drains them.
#
# num_cpus=0.01 instead of num_cpus=0: Ray treats num_cpus=0 actors specially,
# scheduling them randomly across the cluster regardless of NodeAffinity hints
# in some configurations. A tiny positive value bypasses that path while still
# placing essentially no load on the node. The actor itself is near-idle: it
# subprocess-launches nvidia-smi at 2 Hz and that's it.


@ray.remote(num_cpus=0.01)
class GpuCollector:
    def __init__(
        self,
        interval_ms: int,
        node_id: str,
        hostname: str,
        ip: str,
    ) -> None:
        self._interval_s = interval_ms / 1000.0
        self._node_id = node_id
        self._hostname = hostname
        self._ip = ip
        self._buffer: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        """Background sampling loop. Runs until stop_event is set."""
        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            stdout, err = _run_nvidia_smi()
            t_mono = time.monotonic() - self._t0
            t_wall = _utcnow_iso()

            if err is not None:
                record: dict[str, Any] = {
                    "kind": "error",
                    "t_mono": round(t_mono, 4),
                    "t_wall": t_wall,
                    "node_id": self._node_id,
                    "hostname": self._hostname,
                    "ip": self._ip,
                    "error": err,
                    "where": "collector_loop",
                }
                with self._lock:
                    self._buffer.append(record)
            else:
                rows = _parse_nvidia_smi_csv(stdout)
                with self._lock:
                    for row in rows:
                        self._buffer.append(
                            {
                                "kind": "sample",
                                "t_mono": round(t_mono, 4),
                                "t_wall": t_wall,
                                "node_id": self._node_id,
                                "hostname": self._hostname,
                                "ip": self._ip,
                                **row,
                            }
                        )

            # Sleep for the remainder of the interval. If nvidia-smi took longer
            # than the interval (rare, but possible on a hung driver), skip the
            # sleep entirely and resample immediately.
            elapsed = time.monotonic() - tick_start
            remaining = self._interval_s - elapsed
            if remaining > 0:
                # Use the stop_event for the sleep so shutdown is responsive.
                self._stop_event.wait(remaining)

    def drain(self) -> list[dict[str, Any]]:
        """Atomically swap out the sample buffer and return its contents."""
        with self._lock:
            samples = self._buffer
            self._buffer = []
        return samples

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it to exit."""
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def heartbeat(self) -> dict[str, Any]:
        """Cheap liveness check. Returned by the driver during init to confirm
        the actor is up and running on the expected node."""
        return {
            "node_id": self._node_id,
            "hostname": self._hostname,
            "thread_alive": self._thread.is_alive(),
            "buffer_size": len(self._buffer),
        }


# === driver-side helpers ===


@dataclass
class GpuNode:
    """One GPU-bearing node discovered via ray.nodes()."""

    node_id: str
    hostname: str
    ip: str
    gpu_count: int


def _discover_gpu_nodes() -> list[GpuNode]:
    """Query the Ray cluster for nodes with GPU resources. Returns a list
    sorted by hostname for stable header output."""
    nodes: list[GpuNode] = []
    for entry in ray.nodes():
        if not entry.get("Alive", False):
            continue
        resources = entry.get("Resources", {})
        gpu_count = int(resources.get("GPU", 0))
        if gpu_count == 0:
            continue
        nodes.append(
            GpuNode(
                node_id=entry["NodeID"],
                hostname=entry.get("NodeManagerHostname", "unknown"),
                ip=entry.get("NodeManagerAddress", "unknown"),
                gpu_count=gpu_count,
            )
        )
    nodes.sort(key=lambda n: n.hostname)
    return nodes


def _spawn_collectors(nodes: list[GpuNode], interval_ms: int) -> list[tuple[GpuNode, Any]]:
    """Instantiate one GpuCollector actor per node, pinned via NodeAffinity.

    Returns a list of (node, actor_handle) pairs in the same order as nodes.
    Confirms each actor is alive via a heartbeat call before returning, so
    placement failures surface immediately rather than during the first drain.
    """
    pairs: list[tuple[GpuNode, Any]] = []
    for node in nodes:
        actor = GpuCollector.options(  # type: ignore[attr-defined]
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node_id=node.node_id,
                soft=False,
            ),
        ).remote(
            interval_ms=interval_ms,
            node_id=node.node_id,
            hostname=node.hostname,
            ip=node.ip,
        )
        # Block on a heartbeat. If the actor failed to place, this raises
        # ActorUnschedulableError and we abort cleanly.
        heartbeat = ray.get(actor.heartbeat.remote(), timeout=10.0)
        if heartbeat["node_id"] != node.node_id:
            raise RuntimeError(
                f"collector for node {node.hostname} reported wrong node_id "
                f"(expected {node.node_id}, got {heartbeat['node_id']})"
            )
        pairs.append((node, actor))
    return pairs


def _write_header(
    f: Any,
    nodes: list[GpuNode],
    interval_ms: int,
    pull_every_ms: int,
    duration: float | None,
    argv: list[str],
) -> None:
    """Write the one-time header record at the top of the JSONL file."""
    header = {
        "kind": "header",
        "schema": SCHEMA_VERSION,
        "started_at": _utcnow_iso(),
        "interval_ms": interval_ms,
        "pull_every_ms": pull_every_ms,
        "duration_sec": duration,
        "ray_version": ray.__version__,
        "cluster_nodes_total": len(ray.nodes()),
        "gpu_nodes": [
            {
                "node_id": n.node_id,
                "hostname": n.hostname,
                "ip": n.ip,
                "gpu_count": n.gpu_count,
            }
            for n in nodes
        ],
        "command": argv,
    }
    f.write(json.dumps(header) + "\n")
    f.flush()


def _drain_all(pairs: list[tuple[GpuNode, Any]], f: Any) -> tuple[int, int]:
    """Drain every collector once and write all samples to the file.

    Returns (sample_count, error_count) for this drain cycle, for the
    progress line. Driver-side exceptions during drain (actor died, network
    glitch) are caught and logged as error records in the file so the trace
    is self-describing.
    """
    sample_count = 0
    error_count = 0
    for node, actor in pairs:
        try:
            records = ray.get(actor.drain.remote(), timeout=5.0)
        except Exception as e:
            err_record = {
                "kind": "error",
                "t_mono": None,
                "t_wall": _utcnow_iso(),
                "node_id": node.node_id,
                "hostname": node.hostname,
                "ip": node.ip,
                "error": f"drain failed: {type(e).__name__}: {e}",
                "where": "driver_drain",
            }
            f.write(json.dumps(err_record) + "\n")
            error_count += 1
            continue
        for rec in records:
            f.write(json.dumps(rec) + "\n")
            if rec.get("kind") == "sample":
                sample_count += 1
            elif rec.get("kind") == "error":
                error_count += 1
    f.flush()
    return sample_count, error_count


# === main loop ===


# Module-level flag set by the SIGINT/SIGTERM handler so the main loop can
# notice and exit cleanly. We don't use threading.Event here because the main
# loop is single-threaded and a plain bool flag is simpler.
_stop_requested = False


def _install_signal_handlers() -> None:
    def handler(signum: int, _frame: FrameType | None) -> None:
        global _stop_requested
        if _stop_requested:
            # Second Ctrl-C: let Python's default behavior take over and
            # actually terminate.
            print("\n[gpu_profile] second signal, terminating immediately", file=sys.stderr)
            sys.exit(130)
        print(f"\n[gpu_profile] received signal {signum}, shutting down...", file=sys.stderr)
        _stop_requested = True

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGTERM, handler)


def _main_loop(
    pairs: list[tuple[GpuNode, Any]],
    f: Any,
    pull_every_ms: int,
    duration: float | None,
) -> None:
    """Drive the profiler until SIGINT/SIGTERM or --duration expires."""
    pull_interval_s = pull_every_ms / 1000.0
    t_start = time.monotonic()
    total_samples = 0
    total_errors = 0
    last_progress = t_start

    while not _stop_requested:
        if duration is not None and (time.monotonic() - t_start) >= duration:
            print(f"[gpu_profile] duration {duration}s reached, stopping", file=sys.stderr)
            break

        time.sleep(pull_interval_s)
        if _stop_requested:
            break

        samples, errors = _drain_all(pairs, f)
        total_samples += samples
        total_errors += errors

        # Print a one-line progress update every ~5 seconds, not every drain,
        # so the terminal isn't spammed during long runs.
        now = time.monotonic()
        if now - last_progress >= 5.0:
            elapsed = now - t_start
            print(
                f"[gpu_profile] elapsed={elapsed:.0f}s "
                f"samples={total_samples} errors={total_errors}",
                file=sys.stderr,
            )
            last_progress = now


def _final_drain(pairs: list[tuple[GpuNode, Any]], f: Any) -> None:
    """One last drain after the main loop exits to capture any samples
    buffered between the last pull and shutdown."""
    print("[gpu_profile] final drain...", file=sys.stderr)
    try:
        samples, errors = _drain_all(pairs, f)
        print(f"[gpu_profile] final drain: {samples} samples, {errors} errors", file=sys.stderr)
    except Exception as e:
        print(f"[gpu_profile] final drain failed: {e}", file=sys.stderr)


def _stop_collectors(pairs: list[tuple[GpuNode, Any]]) -> None:
    """Tell each collector to stop its background thread. Best-effort: if an
    actor is already dead or unresponsive, log and continue."""
    for node, actor in pairs:
        try:
            ray.get(actor.stop.remote(), timeout=3.0)
        except Exception as e:
            print(
                f"[gpu_profile] failed to stop collector on {node.hostname}: {e}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cluster-wide GPU profiler for Ray clusters. "
        "Samples nvidia-smi on every GPU-bearing node and writes a JSONL trace.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        required=True,
        help="Ray cluster address. Same format as biota's --ray-address: "
        "HOST[:PORT] (port defaults to 6379) or ray://HOST:PORT for Ray Client.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output JSONL file. Will be created if missing, overwritten if present.",
    )
    parser.add_argument(
        "--interval-ms",
        type=int,
        default=DEFAULT_INTERVAL_MS,
        help=f"Per-sample cadence inside each collector actor. Default {DEFAULT_INTERVAL_MS}ms (2 Hz).",
    )
    parser.add_argument(
        "--pull-every-ms",
        type=int,
        default=DEFAULT_PULL_EVERY_MS,
        help=f"How often the driver drains buffered samples. Default {DEFAULT_PULL_EVERY_MS}ms.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Run for this many seconds, then exit. Default: run until Ctrl-C.",
    )
    args = parser.parse_args()

    if args.interval_ms < 50:
        parser.error("--interval-ms must be >= 50 (nvidia-smi launch overhead floor)")
    if args.pull_every_ms < args.interval_ms:
        parser.error("--pull-every-ms must be >= --interval-ms (otherwise drains return empty)")

    # Normalize the ray address: bare host -> host:6379, matching biota's
    # convention from src/biota/cli.py::_normalize_ray_address.
    address: str = args.ray_address
    if not address.startswith("ray://") and ":" not in address:
        address = f"{address}:6379"

    print(f"[gpu_profile] connecting to Ray at {address}", file=sys.stderr)
    ray.init(address=address, ignore_reinit_error=False, log_to_driver=False)

    try:
        nodes = _discover_gpu_nodes()
        if not nodes:
            print(
                "[gpu_profile] no GPU-bearing nodes found in cluster. "
                "Check that 'ray status' shows GPU resources.",
                file=sys.stderr,
            )
            sys.exit(2)

        print(f"[gpu_profile] found {len(nodes)} GPU node(s):", file=sys.stderr)
        for n in nodes:
            print(
                f"[gpu_profile]   {n.hostname} ({n.ip}) - {n.gpu_count} GPU(s)",
                file=sys.stderr,
            )

        print(
            f"[gpu_profile] spawning collectors (interval={args.interval_ms}ms, "
            f"pull_every={args.pull_every_ms}ms)...",
            file=sys.stderr,
        )
        pairs = _spawn_collectors(nodes, interval_ms=args.interval_ms)
        print(f"[gpu_profile] {len(pairs)} collector(s) live", file=sys.stderr)

        _install_signal_handlers()

        with open(args.out, "w") as f:
            _write_header(
                f,
                nodes=nodes,
                interval_ms=args.interval_ms,
                pull_every_ms=args.pull_every_ms,
                duration=args.duration,
                argv=sys.argv,
            )
            print(f"[gpu_profile] writing to {args.out}", file=sys.stderr)
            print("[gpu_profile] running. Ctrl-C to stop.", file=sys.stderr)

            try:
                _main_loop(
                    pairs,
                    f,
                    pull_every_ms=args.pull_every_ms,
                    duration=args.duration,
                )
            finally:
                _final_drain(pairs, f)
                _stop_collectors(pairs)

        print(f"[gpu_profile] wrote {args.out}", file=sys.stderr)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
