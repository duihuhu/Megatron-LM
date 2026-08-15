#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
BENCH="$SCRIPT_DIR/bench_concord_pipeline.py"
RUNNER="$SCRIPT_DIR/run_concord_pipeline_bench.sh"
REMOTE_PORT=${REMOTE_PORT:-2222}
REMOTE_IDENTITY_FILE=${REMOTE_IDENTITY_FILE:-/root/.ssh/id_ed25519}
REMOTE_WORKDIR=${REMOTE_WORKDIR:-/workspace/Megatron-LM}
MASTER_ADDR=${MASTER_ADDR:-10.252.129.35}
MASTER_PORT=${MASTER_PORT:-29610}
CONCORD_BASE_PORT=${CONCORD_BASE_PORT:-28200}
CLUSTER_NODES=${CLUSTER_NODES:-4}
PROFILE=${PROFILE:-10b-concord-save}
DRY_RUN=${DRY_RUN:-0}

apply_profile() {
    case "$PROFILE" in
        10b-concord-save)
            NET_BYTES=${NET_BYTES:-5548000000}
            ENCODE_SOURCE_BYTES=${ENCODE_SOURCE_BYTES:-5548000000}
            EXPECTED_NET_RECV_BYTES=${EXPECTED_NET_RECV_BYTES:-5313000000}
            ENCODE_JOBS=${ENCODE_JOBS:-102}
            CONCORD_RDMA_LANES_PER_PEER=${CONCORD_RDMA_LANES_PER_PEER:-12}
            CONCORD_LAYER_EXCHANGE_SEG=${CONCORD_LAYER_EXCHANGE_SEG:-12}
            LOGICAL_CHUNK_BYTES=${LOGICAL_CHUNK_BYTES:-32M}
            PRODUCTION_BATCH=${PRODUCTION_BATCH:-24}
            NET_SOURCE=${NET_SOURCE:-gpu}
            SHARE_GPU_SOURCE=${SHARE_GPU_SOURCE:-1}
            ;;
        custom) ;;
        *) echo "ERROR,unknown PROFILE=$PROFILE" >&2; return 2 ;;
    esac
    NET_BYTES=${NET_BYTES:-256M}
    ENCODE_SOURCE_BYTES=${ENCODE_SOURCE_BYTES:-256M}
    EXPECTED_NET_RECV_BYTES=${EXPECTED_NET_RECV_BYTES:-0}
    ENCODE_JOBS=${ENCODE_JOBS:-8}
    CONCORD_RDMA_LANES_PER_PEER=${CONCORD_RDMA_LANES_PER_PEER:-4}
    CONCORD_LAYER_EXCHANGE_SEG=${CONCORD_LAYER_EXCHANGE_SEG:-4}
    LOGICAL_CHUNK_BYTES=${LOGICAL_CHUNK_BYTES:-32M}
    PRODUCTION_BATCH=${PRODUCTION_BATCH:-0}
    NET_SOURCE=${NET_SOURCE:-cpu}
    SHARE_GPU_SOURCE=${SHARE_GPU_SOURCE:-1}
    PCIE_BYTES=${PCIE_BYTES:-256M}
    WARMUP=${WARMUP:-2}
    ITERS=${ITERS:-5}
    MODE_ORDER=${MODE_ORDER:-rotate}
    BENCH_MODES=${BENCH_MODES:-net-only,encode-only,serial,net+encode}
}

validate_cluster() {
    [[ "$CLUSTER_NODES" == 2 || "$CLUSTER_NODES" == 4 ]] || {
        echo "ERROR,CLUSTER_NODES must be 2 or 4" >&2
        return 2
    }
}

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
    compgen -G "$ROOT_DIR/megatron/core/dist_checkpointing/strategies/concord_native*.so" >/dev/null || {
        echo "ERROR,concord_native extension not found" >&2
        return 1
    }
    local hca
    for hca in mlx5_0 mlx5_1 mlx5_4 mlx5_5; do
        [ -d "/sys/class/infiniband/$hca" ] || { echo "ERROR,missing HCA $hca" >&2; return 1; }
    done
    check_gpu_idle
}

export_env() {
    export MASTER_ADDR MASTER_PORT CONCORD_BASE_PORT CLUSTER_NODES PROFILE
    export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}
    export NET_BYTES EXPECTED_NET_RECV_BYTES ENCODE_SOURCE_BYTES ENCODE_JOBS NET_SOURCE SHARE_GPU_SOURCE
    export PCIE_BYTES WARMUP ITERS MODE_ORDER BENCH_MODES LOGICAL_CHUNK_BYTES PRODUCTION_BATCH
    export CONCORD_RDMA_LANES_PER_PEER CONCORD_LAYER_EXCHANGE_SEG
    export CONCORD_INTERFACE=${CONCORD_INTERFACE:-bond0}
    export CONCORD_LOCAL_RANK_NIC_0=mlx5_0 CONCORD_LOCAL_RANK_NIC_1=mlx5_0
    export CONCORD_LOCAL_RANK_NIC_2=mlx5_1 CONCORD_LOCAL_RANK_NIC_3=mlx5_1
    export CONCORD_LOCAL_RANK_NIC_4=mlx5_4 CONCORD_LOCAL_RANK_NIC_5=mlx5_4
    export CONCORD_LOCAL_RANK_NIC_6=mlx5_5 CONCORD_LOCAL_RANK_NIC_7=mlx5_5
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}
}

run_node() {
    local node_rank=$1
    if [ "$DRY_RUN" != 1 ]; then check_node; fi
    export_env
    local -a command=(torchrun --nnodes="$CLUSTER_NODES" --nproc-per-node=8 --node-rank="$node_rank"
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" --no-python bash -c '
set -euo pipefail
lr=${LOCAL_RANK:?}
if (( lr < 4 )); then base=0; else base=32; fi
slot=$((lr % 4))
first=$((base + slot * 8))
sibling=$((first + 64))
first_cpus=$(seq -s, "$first" "$((first + 7))")
sibling_cpus=$(seq -s, "$sibling" "$((sibling + 7))")
cpus="$first_cpus,$sibling_cpus"
export CONCORD_RS_CPU_LIST="$cpus"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
echo "CPU_BIND,node_rank=$2,local_rank=$lr,cpu_list=$cpus"
exec taskset -c "$cpus" python "$1"
' _ "$BENCH" "$node_rank")
    if [ "$DRY_RUN" = 1 ]; then
        printf 'NODE_COMMAND,node=node%s,rank=%s,command=' "$node_rank" "$node_rank"
        printf '%q ' "${command[@]}"
        printf '\n'
        return
    fi
    exec "${command[@]}"
}

remote_command() {
    local node_rank=$1 pair
    printf 'cd %q && exec env ' "$REMOTE_WORKDIR"
    for pair in \
        "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT" "CONCORD_BASE_PORT=$CONCORD_BASE_PORT" \
        "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond0}" \
        "CLUSTER_NODES=$CLUSTER_NODES" "PROFILE=$PROFILE" "NET_BYTES=$NET_BYTES" \
        "EXPECTED_NET_RECV_BYTES=$EXPECTED_NET_RECV_BYTES" \
        "NET_SOURCE=$NET_SOURCE" "SHARE_GPU_SOURCE=$SHARE_GPU_SOURCE" "PCIE_BYTES=$PCIE_BYTES" \
        "ENCODE_SOURCE_BYTES=$ENCODE_SOURCE_BYTES" "ENCODE_JOBS=$ENCODE_JOBS" \
        "WARMUP=$WARMUP" "ITERS=$ITERS" "BENCH_MODES=$BENCH_MODES" "MODE_ORDER=$MODE_ORDER" \
        "LOGICAL_CHUNK_BYTES=$LOGICAL_CHUNK_BYTES" "PRODUCTION_BATCH=$PRODUCTION_BATCH" \
        "CONCORD_RDMA_LANES_PER_PEER=$CONCORD_RDMA_LANES_PER_PEER" \
        "CONCORD_LAYER_EXCHANGE_SEG=$CONCORD_LAYER_EXCHANGE_SEG" \
        "ALLOW_GPU_CONTEXT_PIDS=${ALLOW_GPU_CONTEXT_PIDS:-}"; do
        printf '%q ' "$pair"
    done
    printf 'bash %q node %q' \
        "$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/run_concord_pipeline_bench.sh" "$node_rank"
}

run_cluster() {
    validate_cluster
    local -a ssh_opts=(-i "$REMOTE_IDENTITY_FILE" -o BatchMode=yes -o ConnectTimeout=10 -p "$REMOTE_PORT")
    local node command
    if [ "$DRY_RUN" = 1 ]; then
        echo "DRY_RUN,profile=$PROFILE,cluster_nodes=$CLUSTER_NODES,world_size=$((CLUSTER_NODES * 8))"
        for ((node = 1; node < CLUSTER_NODES; node++)); do
            command=$(remote_command "$node")
            printf 'REMOTE_COMMAND,node=node%s,command=ssh ' "$node"
            printf '%q ' "${ssh_opts[@]}" "node$node" "$command"
            printf '\n'
        done
        run_node 0
        return
    fi

    check_node
    for ((node = 1; node < CLUSTER_NODES; node++)); do
        scp -i "$REMOTE_IDENTITY_FILE" -o BatchMode=yes -o ConnectTimeout=10 -P "$REMOTE_PORT" \
            "$BENCH" "$RUNNER" "node$node:$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/"
        ssh "${ssh_opts[@]}" "node$node" \
            "cd '$REMOTE_WORKDIR' && ALLOW_GPU_CONTEXT_PIDS=$(printf %q "${ALLOW_GPU_CONTEXT_PIDS:-}") bash '$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/run_concord_pipeline_bench.sh' check"
    done

    local -a pids=()
    cleanup() {
        local pid
        trap - EXIT INT TERM
        for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
        for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    }
    trap cleanup EXIT INT TERM
    for ((node = 1; node < CLUSTER_NODES; node++)); do
        ssh "${ssh_opts[@]}" "node$node" "$(remote_command "$node")" &
        pids+=("$!")
    done
    run_node 0 &
    pids+=("$!")

    local status=0 finished="" index remaining=${#pids[@]}
    while (( remaining > 0 )); do
        if wait -n -p finished "${pids[@]}"; then
            for index in "${!pids[@]}"; do
                if [ "${pids[$index]}" = "$finished" ]; then unset 'pids[index]'; break; fi
            done
            pids=("${pids[@]}")
            remaining=${#pids[@]}
        else
            status=$?
            cleanup
            return "$status"
        fi
    done
    pids=()
    trap - EXIT INT TERM
}

apply_profile
case "${1:-run}" in
    node)
        validate_cluster
        [[ "${2:-}" =~ ^[0-3]$ ]] && (( 2 < CLUSTER_NODES )) || {
            echo "Usage: $0 node NODE_RANK (0..CLUSTER_NODES-1)" >&2
            exit 2
        }
        run_node "$2"
        ;;
    check) validate_cluster; check_node ;;
    run) run_cluster ;;
    *) echo "Usage: $0 {run|node NODE_RANK|check}" >&2; exit 2 ;;
esac
