#!/usr/bin/env bash
# Per-rank NUMA binding launcher for torchrun --no-python.
#
# Usage (from torchrun):
#   torchrun ... --no-python /path/to/numa_bind_launch.sh pretrain_gpt.py [args...]
#
# Environment:
#   LOCAL_RANK          - set by torchrun (required for per-GPU binding)
#   CUDA_VISIBLE_DEVICES - comma-separated physical GPU ids
#   GPU_TO_NUMA         - optional override, e.g. "0 0 0 0 1 1 1 1"
#   NUMA_BIND           - set to 0 to skip numactl (debug / fallback)
#   PYTHON_BIN          - optional python interpreter path

set -euo pipefail

lr="${LOCAL_RANK:-0}"
IFS=',' read -r -a _cvd <<< "${CUDA_VISIBLE_DEVICES:-}"
gpu="${_cvd[$lr]:-$lr}"

if [[ -n "${GPU_TO_NUMA:-}" ]]; then
    read -r -a _map <<< "$GPU_TO_NUMA"
else
    # Default for 2-socket A800: GPU0-3 -> NUMA0, GPU4-7 -> NUMA1
    _map=(0 0 0 0 1 1 1 1)
fi
n="${_map[$gpu]:-0}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
    _py="$PYTHON_BIN"
elif command -v python3 >/dev/null 2>&1; then
    _py=python3
elif command -v python >/dev/null 2>&1; then
    _py=python
else
    echo "[numa_bind] ERROR: no python3/python found in PATH" >&2
    exit 127
fi

if [[ "${NUMA_BIND:-1}" == "1" ]] && command -v numactl >/dev/null 2>&1; then
    exec numactl --cpunodebind="$n" --membind="$n" "$_py" "$@"
fi

if [[ "${NUMA_BIND:-1}" == "1" ]]; then
    echo "[numa_bind] WARN: numactl not installed, skipping NUMA bind" >&2
fi
exec "$_py" "$@"
