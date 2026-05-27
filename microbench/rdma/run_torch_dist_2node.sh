#!/bin/bash
# Multi-node or single-node launcher for rdma_ec_bind_bench via bench_launcher.py
#
# Two nodes (1 GPU per node):
#   Node 0: MASTER_ADDR=10.0.0.62 ./run_torch_dist_2node.sh 0
#   Node 1: MASTER_ADDR=10.0.0.62 ./run_torch_dist_2node.sh 1
#
# Single node (2 ranks, 2 GPUs) — prefer run_single_node.sh:
#   ./run_single_node.sh 0,1
#   # or:
#   SINGLE_NODE=1 CUDA_VISIBLE_DEVICES=0,1 ./run_torch_dist_2node.sh 0
#
# GPU: set CUDA_VISIBLE_DEVICES before launch, or use run_single_node.sh <gpu_ids>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
NODE_RANK="${1:-0}"
shift || true

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-6000}"
MICROBENCH_PREFIX="${MICROBENCH_PREFIX:-ECNAIVE}"

if [ "${SINGLE_NODE:-0}" = "1" ] || [ "${NNODES:-2}" = "1" ]; then
    NNODES=1
    NPROC="${NPROC_PER_NODE:-2}"
    NODE_RANK=0
    echo "[launcher] single-node mode: nnodes=1 nproc_per_node=${NPROC} master=${MASTER_ADDR}:${MASTER_PORT}"
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        echo "[launcher] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
    fi
else
    NNODES="${NNODES:-2}"
    NPROC="${NPROC_PER_NODE:-1}"
fi

if [ ! -x "${SCRIPT_DIR}/build/rdma_ec_bind_bench" ]; then
    echo "Building rdma_ec_bind_bench..."
    make -C "${SCRIPT_DIR}" rdma_ec_bind_bench
fi

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export MICROBENCH_PREFIX

cd "${REPO_ROOT}"

torchrun \
    --nnodes="${NNODES}" \
    --nproc_per_node="${NPROC}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    "${SCRIPT_DIR}/bench_launcher.py" \
    --prefix "${MICROBENCH_PREFIX}" \
    "$@"
