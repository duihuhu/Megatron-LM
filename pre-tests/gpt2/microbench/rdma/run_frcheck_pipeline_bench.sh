#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BENCH="$SCRIPT_DIR/bench_frcheck_pipeline.py"
RUNNER="$SCRIPT_DIR/run_frcheck_pipeline_bench.sh"
REMOTE_HOST=${REMOTE_HOST:-node1}
REMOTE_PORT=${REMOTE_PORT:-2222}
REMOTE_IDENTITY_FILE=${REMOTE_IDENTITY_FILE:-/root/.ssh/id_ed25519}
REMOTE_WORKDIR=${REMOTE_WORKDIR:-/workspace/Megatron-LM}
MASTER_ADDR=${MASTER_ADDR:-10.252.129.35}
MASTER_PORT=${MASTER_PORT:-29610}
FRCHECK_BASE_PORT=${FRCHECK_BASE_PORT:-28200}

check_gpu_idle() {
    local pids allowlist unexpected="" pid
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits |
        awk '$1 ~ /^[0-9]+$/ {seen[$1]=1} END {for (pid in seen) print pid}' | sort -n)
    [ -n "$pids" ] || return 0

    allowlist=",${ALLOW_GPU_CONTEXT_PIDS:-},"
    allowlist=${allowlist//:/,}
    while IFS= read -r pid; do
        if [[ "$allowlist" != *",$pid,"* ]]; then
            unexpected="${unexpected:+$unexpected:}$pid"
        fi
    done <<< "$pids"
    if [ -n "$unexpected" ]; then
        echo "ERROR,unallowed GPU compute processes remain,pids=$unexpected" >&2
        return 1
    fi
    echo "WARNING,allowed GPU compute processes remain,pids=$(echo "$pids" | paste -sd: -)" >&2
}

check_node() {
    command -v torchrun >/dev/null || { echo "ERROR,torchrun not found" >&2; return 1; }
    [ -f "$BENCH" ] || { echo "ERROR,missing benchmark $BENCH" >&2; return 1; }
    compgen -G "$ROOT_DIR/megatron/core/dist_checkpointing/strategies/frcheck_native*.so" >/dev/null || {
        echo "ERROR,frcheck_native extension not found" >&2
        return 1
    }
    local hca
    for hca in mlx5_0 mlx5_1 mlx5_4 mlx5_5; do
        [ -d "/sys/class/infiniband/$hca" ] || { echo "ERROR,missing HCA $hca" >&2; return 1; }
    done
    check_gpu_idle
}

export_env() {
    export MASTER_ADDR MASTER_PORT FRCHECK_BASE_PORT
    export FRCHECK_INTERFACE=${FRCHECK_INTERFACE:-bond0}
    export FRCHECK_LOCAL_RANK_NIC_0=mlx5_0 FRCHECK_LOCAL_RANK_NIC_1=mlx5_0
    export FRCHECK_LOCAL_RANK_NIC_2=mlx5_1 FRCHECK_LOCAL_RANK_NIC_3=mlx5_1
    export FRCHECK_LOCAL_RANK_NIC_4=mlx5_4 FRCHECK_LOCAL_RANK_NIC_5=mlx5_4
    export FRCHECK_LOCAL_RANK_NIC_6=mlx5_5 FRCHECK_LOCAL_RANK_NIC_7=mlx5_5
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}
}

run_node() {
    local node_rank=$1
    check_node
    export_env
    exec torchrun --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank" \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --no-python \
        bash -c '
set -euo pipefail
lr=${LOCAL_RANK:?}
if (( lr < 4 )); then base=0; else base=32; fi
slot=$((lr % 4))
first=$((base + slot * 8))
sibling=$((first + 64))
first_cpus=$(seq -s, "$first" "$((first + 7))")
sibling_cpus=$(seq -s, "$sibling" "$((sibling + 7))")
cpus="$first_cpus,$sibling_cpus"
export FRCHECK_RS_CPU_LIST="$cpus"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
exec taskset -c "$cpus" python "$1"
' _ "$BENCH"
}

remote_command() {
    printf 'cd %q && exec env ' "$REMOTE_WORKDIR"
    printf '%q ' \
        "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT" "FRCHECK_BASE_PORT=$FRCHECK_BASE_PORT" \
        "NET_BYTES=${NET_BYTES:-256M}" "NET_SOURCE=${NET_SOURCE:-cpu}" \
        "SHARE_GPU_SOURCE=${SHARE_GPU_SOURCE:-1}" "PCIE_BYTES=${PCIE_BYTES:-256M}" \
        "ENCODE_SOURCE_BYTES=${ENCODE_SOURCE_BYTES:-256M}" "WARMUP=${WARMUP:-2}" "ITERS=${ITERS:-5}" \
        "BENCH_MODES=${BENCH_MODES:-net-only,pcie-only,encode-only,serial,net+pcie,net+encode,pcie+encode,all-overlap}" \
        "ENCODE_JOBS=${ENCODE_JOBS:-8}" "FRCHECK_RDMA_LANES_PER_PEER=${FRCHECK_RDMA_LANES_PER_PEER:-4}" \
        "FRCHECK_LAYER_EXCHANGE_SEG=${FRCHECK_LAYER_EXCHANGE_SEG:-4}" \
        "ALLOW_GPU_CONTEXT_PIDS=${ALLOW_GPU_CONTEXT_PIDS:-}"
    printf 'bash %q node 1' "$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/run_frcheck_pipeline_bench.sh"
}

run_cluster() {
    check_node
    local -a ssh_opts=(-i "$REMOTE_IDENTITY_FILE" -o BatchMode=yes -o ConnectTimeout=10 -p "$REMOTE_PORT")
    scp -i "$REMOTE_IDENTITY_FILE" -o BatchMode=yes -o ConnectTimeout=10 -P "$REMOTE_PORT" \
        "$BENCH" "$RUNNER" "$REMOTE_HOST:$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/"
    local remote_allowlist
    printf -v remote_allowlist '%q' "${ALLOW_GPU_CONTEXT_PIDS:-}"
    ssh "${ssh_opts[@]}" "$REMOTE_HOST" \
        "cd '$REMOTE_WORKDIR' && ALLOW_GPU_CONTEXT_PIDS=$remote_allowlist bash '$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/run_frcheck_pipeline_bench.sh' check"

    local remote_pid="" local_pid="" local_status=0 remote_status=0
    cleanup() {
        trap - EXIT INT TERM
        [ -z "$local_pid" ] || kill -TERM "$local_pid" 2>/dev/null || true
        [ -z "$remote_pid" ] || kill -TERM "$remote_pid" 2>/dev/null || true
        [ -z "$local_pid" ] || wait "$local_pid" 2>/dev/null || true
        [ -z "$remote_pid" ] || wait "$remote_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM
    ssh "${ssh_opts[@]}" "$REMOTE_HOST" "$(remote_command)" & remote_pid=$!
    run_node 0 & local_pid=$!
    wait "$local_pid" || local_status=$?
    local_pid=""
    if [ "$local_status" -ne 0 ]; then
        kill -TERM "$remote_pid" 2>/dev/null || true
    fi
    wait "$remote_pid" || remote_status=$?
    remote_pid=""
    trap - EXIT INT TERM
    if [ "$local_status" -ne 0 ]; then return "$local_status"; fi
    return "$remote_status"
}

case "${1:-run}" in
    node)
        [ "${2:-}" = 0 ] || [ "${2:-}" = 1 ] || { echo "Usage: $0 node {0|1}" >&2; exit 2; }
        run_node "$2"
        ;;
    check) check_node ;;
    run) run_cluster ;;
    *) echo "Usage: $0 {run|node {0|1}|check}" >&2; exit 2 ;;
esac
