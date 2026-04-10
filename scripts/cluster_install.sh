#!/usr/bin/env bash
#
# Build the biota wheel from the current source tree and install it into a
# dedicated runtime venv at ~/.biota-runtime. Run this on every cluster node
# (or your laptop, if you want a wheel-installed biota for any reason) before
# starting Ray.
#
# Why a separate runtime venv instead of `uv run` from the source tree:
# `uv run` injects src/ onto sys.path, which makes Ray's runtime_env machinery
# detect biota as a "local development module" and ship+rebuild it on every
# Ray invocation. The rebuild is the dominant cost in Ray startup overhead.
# Installing biota as a regular wheel into a normal site-packages directory
# defeats that detection - Ray sees biota the same way it sees torch or numpy
# (a normal third-party package) and doesn't do anything special.
#
#
# Usage:
#
#     # On every cluster node (and on the laptop if you want):
#     cd ~/biota
#     git pull
#     ./scripts/cluster_install.sh
#     source ~/.biota-runtime/bin/activate
#     ray start ...   # head or worker
#
# Pass --clean to nuke the existing runtime venv and start fresh (slower
# because all deps re-download, but useful when dependencies have shifted in
# a way pip-install-on-top doesn't handle cleanly).
#
# This script must be run from inside the biota repo root (the directory
# containing pyproject.toml). It refuses to run from anywhere else.

set -euo pipefail

RUNTIME_VENV="${HOME}/.biota-runtime"
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=1
            ;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "[cluster_install] unknown argument: $arg" >&2
            echo "[cluster_install] usage: $0 [--clean]" >&2
            exit 2
            ;;
    esac
done

# Sanity: must be in the repo root
if [[ ! -f "pyproject.toml" ]] || ! grep -q '^name = "biota"' pyproject.toml; then
    echo "[cluster_install] error: must be run from the biota repo root" >&2
    echo "[cluster_install] (no pyproject.toml or wrong project)" >&2
    exit 1
fi

# Sanity: uv must be on PATH
if ! command -v uv >/dev/null 2>&1; then
    echo "[cluster_install] error: uv not found on PATH" >&2
    echo "[cluster_install] install from https://docs.astral.sh/uv/" >&2
    exit 1
fi

echo "[cluster_install] biota repo: $(pwd)"
echo "[cluster_install] runtime venv: ${RUNTIME_VENV}"

# Optional clean: remove the existing runtime venv entirely
if [[ $CLEAN -eq 1 ]]; then
    if [[ -d "${RUNTIME_VENV}" ]]; then
        echo "[cluster_install] --clean: removing existing ${RUNTIME_VENV}"
        rm -rf "${RUNTIME_VENV}"
    fi
fi

# Build the wheel. uv build writes to dist/ in the repo root.
echo "[cluster_install] building wheel..."
rm -f dist/biota-*.whl
uv build --wheel >/dev/null

WHEEL=$(find dist -maxdepth 1 -name 'biota-*.whl' -type f | head -n 1)
if [[ -z "${WHEEL}" ]]; then
    echo "[cluster_install] error: uv build did not produce a wheel in dist/" >&2
    exit 1
fi
echo "[cluster_install] wheel: ${WHEEL}"

# Create the runtime venv if it doesn't exist
if [[ ! -d "${RUNTIME_VENV}" ]]; then
    echo "[cluster_install] creating runtime venv with python 3.12..."
    uv venv --python 3.12 "${RUNTIME_VENV}"
fi

# Install the wheel into the runtime venv. --force-reinstall makes this
# idempotent: re-running the script after a code change picks up the new
# wheel cleanly. We use uv pip (much faster than stdlib pip) and target the
# runtime venv explicitly via --python.
echo "[cluster_install] installing wheel into runtime venv..."
uv pip install --python "${RUNTIME_VENV}/bin/python" --force-reinstall "${WHEEL}"

BIOTA_BIN="${RUNTIME_VENV}/bin/biota"
if [[ ! -x "${BIOTA_BIN}" ]]; then
    echo "[cluster_install] error: biota binary not found at ${BIOTA_BIN}" >&2
    echo "[cluster_install] install may have failed silently" >&2
    exit 1
fi

# Sanity-check the install with a doctor invocation
echo "[cluster_install] verifying install..."
"${BIOTA_BIN}" doctor

cat <<EOF

[cluster_install] done.

  binary:    ${BIOTA_BIN}
  activate:  source ${RUNTIME_VENV}/bin/activate

Next steps on this node:

  source ${RUNTIME_VENV}/bin/activate
  ray start ...    # head or worker

After activating, 'biota' on the PATH points at the wheel-installed binary,
not the source tree. ray start launched from this shell will inherit the same
venv, and biota search invocations from this shell will use the same venv,
so the whole chain stays out of source-tree sys.path detection.
EOF
