#!/bin/bash
# Single-machine smoke test: 2 ranks on one host (torchrun nproc=2).
#
# Usage:
#   # Default GPUs 0 and 1
#   ./run_single_node.sh
#
#   # Specify GPUs
#   ./run_single_node.sh 2,3
#
#   # With NIC binding (same as training scripts)
#   export ECNAIVE_LOCAL_RANK_NIC_0=eth0 ECNAIVE_LOCAL_RANK_NIC_1=eth1
#   ./run_single_node.sh 0,1 --size-mb 32 --iterations 20
#
#   # No GPU (dist only uses gloo; RDMA bench still needs RoCE NIC)
#   ./run_single_node.sh --dist-backend gloo
#
# RDMA on one machine needs distinct IPs/NICs per local_rank (not the same 10.0.0.x):
#   export ECNAIVE_LOCAL_RANK_NIC_0=eth0
#   export ECNAIVE_LOCAL_RANK_NIC_1=eth1
# If both ranks show the same IP in launcher logs, RoCE may hang on first transfer.
#
# Optional env:
#   MASTER_ADDR  (default 127.0.0.1)
#   MASTER_PORT  (default 6000)
#   MICROBENCH_PREFIX, MICROBENCH_PORT

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

GPU_IDS=""
EXTRA_ARGS=()

while [ $# -gt 0 ]; do
    case "$1" in
        --dist-backend|--prefix|--port|--size-mb|--warmup|--iterations|--skip-*|--binary)
            EXTRA_ARGS+=("$1")
            shift
            if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
        --help|-h)
            head -n 22 "$0" | tail -n +2
            exit 0
            ;;
        *)
            if [ -z "$GPU_IDS" ] && [[ "$1" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
                GPU_IDS="$1"
                shift
            else
                EXTRA_ARGS+=("$1")
                shift
            fi
            ;;
    esac
done

export SINGLE_NODE=1
export NNODES=1
export NPROC_PER_NODE=2
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-6000}"

if [ -n "$GPU_IDS" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU_IDS"
    echo "[run_single_node] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

exec "${SCRIPT_DIR}/run_torch_dist_2node.sh" 0 "${EXTRA_ARGS[@]}"
