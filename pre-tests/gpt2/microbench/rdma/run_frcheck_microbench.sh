#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
ENCODE_SOURCE="$SCRIPT_DIR/isal_encode_bench.cpp"
ENCODE_BINARY="$SCRIPT_DIR/isal_encode_bench"
NETWORK_BENCH="$SCRIPT_DIR/bench_frcheck_network.py"
STRATEGY_DIR="$ROOT_DIR/megatron/core/dist_checkpointing/strategies"

CPU_WORKERS=${CPU_WORKERS:-16}
CPU_LIST=${CPU_LIST:-0-7,64-71}
CPU_TOTAL=${CPU_TOTAL:-2774274048}
CPU_JOBS=${CPU_JOBS:-102}
CPU_K=${CPU_K:-2}
CPU_M=${CPU_M:-2}
CPU_WARMUP=${CPU_WARMUP:-3}
CPU_REPEATS=${CPU_REPEATS:-7}
MASTER_ADDR=${MASTER_ADDR:-10.252.129.35}
MASTER_PORT=${MASTER_PORT:-29600}
FRCHECK_BASE_PORT=${FRCHECK_BASE_PORT:-27200}
NETWORK_BYTES=${NETWORK_BYTES:-256M}
NETWORK_WARMUP=${NETWORK_WARMUP:-5}
NETWORK_ITERS=${NETWORK_ITERS:-20}
NETWORK_LANES=${NETWORK_LANES:-4}
REMOTE_HOST=${REMOTE_HOST:-node1}
REMOTE_PORT=${REMOTE_PORT:-2222}
REMOTE_IDENTITY_FILE=${REMOTE_IDENTITY_FILE:-$HOME/.ssh/id_ed25519}
REMOTE_WORKDIR=${REMOTE_WORKDIR:-/workspace/Megatron-LM}
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}

build_encode() {
    echo "Building ISA-L CPU encode benchmark"
    g++ -O3 -std=c++17 -Wall -Wextra -I/usr/include/isa-l "$ENCODE_SOURCE" -o "$ENCODE_BINARY" -lisal -pthread
}

run_encode() {
    build_encode
    echo "Running ISA-L FRCheck batch encode: workers=$CPU_WORKERS cpus=$CPU_LIST total=$CPU_TOTAL jobs=$CPU_JOBS"
    "$ENCODE_BINARY" --workers "$CPU_WORKERS" --cpus "$CPU_LIST" \
        --k "$CPU_K" --m "$CPU_M" --total "$CPU_TOTAL" --jobs "$CPU_JOBS" \
        --warmup "$CPU_WARMUP" --repeats "$CPU_REPEATS" "$@"
}

check_network_node() {
    command -v torchrun >/dev/null || { echo "ERROR,torchrun not found" >&2; return 1; }
    compgen -G "$STRATEGY_DIR/frcheck_native*.so" >/dev/null || {
        echo "ERROR,no frcheck_native*.so in $STRATEGY_DIR" >&2
        return 1
    }
    local hca
    for hca in mlx5_0 mlx5_1 mlx5_4 mlx5_5; do
        [ -d "/sys/class/infiniband/$hca" ] || { echo "ERROR,missing RDMA HCA $hca" >&2; return 1; }
    done
    [ -f "$NETWORK_BENCH" ] || { echo "ERROR,missing $NETWORK_BENCH" >&2; return 1; }
}

export_network_env() {
    export FRCHECK_INTERFACE=${FRCHECK_INTERFACE:-bond0}
    export FRCHECK_LOCAL_RANK_NIC_0=mlx5_0
    export FRCHECK_LOCAL_RANK_NIC_1=mlx5_0
    export FRCHECK_LOCAL_RANK_NIC_2=mlx5_1
    export FRCHECK_LOCAL_RANK_NIC_3=mlx5_1
    export FRCHECK_LOCAL_RANK_NIC_4=mlx5_4
    export FRCHECK_LOCAL_RANK_NIC_5=mlx5_4
    export FRCHECK_LOCAL_RANK_NIC_6=mlx5_5
    export FRCHECK_LOCAL_RANK_NIC_7=mlx5_5
    export FRCHECK_BASE_PORT
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION
}

run_network_node() {
    local node_rank=$1
    check_network_node
    export_network_env
    echo "Running network node_rank=$node_rank: 8 logical rank channels over 4 physical HCAs"
    exec torchrun \
        --nnodes=2 --nproc-per-node=8 --node-rank="$node_rank" \
        --master-addr="$MASTER_ADDR" --master-port="$MASTER_PORT" \
        -- "$NETWORK_BENCH" --n 2 --simple --cpu-send \
        --bytes "$NETWORK_BYTES" --warmup "$NETWORK_WARMUP" --iters "$NETWORK_ITERS" \
        --lanes-per-peer "$NETWORK_LANES" --base-port "$FRCHECK_BASE_PORT"
}

remote_network_command() {
    printf 'cd %q && exec env ' "$REMOTE_WORKDIR"
    printf '%q ' \
        "MASTER_ADDR=$MASTER_ADDR" "MASTER_PORT=$MASTER_PORT" \
        "FRCHECK_BASE_PORT=$FRCHECK_BASE_PORT" "NETWORK_BYTES=$NETWORK_BYTES" \
        "NETWORK_WARMUP=$NETWORK_WARMUP" "NETWORK_ITERS=$NETWORK_ITERS" \
        "NETWORK_LANES=$NETWORK_LANES" \
        "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=$PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"
    printf 'bash %q network-client' "$REMOTE_WORKDIR/pre-tests/gpt2/microbench/rdma/run_frcheck_microbench.sh"
}

run_network() {
    check_network_node
    export_network_env
    echo "Topology: 8 logical rank channels per node over 4 physical HCAs"
    echo "Mapping: ranks 0/1=mlx5_0, 2/3=mlx5_1, 4/5=mlx5_4, 6/7=mlx5_5"
    echo "RESULT interpretation: e2e_agg_send_gib_s is bidirectional aggregate; divide by 2 for one-direction-equivalent"

    local remote_pid="" local_pid=""
    cleanup() {
        trap - EXIT INT TERM
        [ -z "$local_pid" ] || kill -TERM "$local_pid" 2>/dev/null || true
        [ -z "$remote_pid" ] || kill -TERM "$remote_pid" 2>/dev/null || true
        [ -z "$local_pid" ] || wait "$local_pid" 2>/dev/null || true
        [ -z "$remote_pid" ] || wait "$remote_pid" 2>/dev/null || true
    }
    trap cleanup EXIT INT TERM

    ssh -i "$REMOTE_IDENTITY_FILE" -o BatchMode=yes -o ConnectTimeout=10 \
        -p "$REMOTE_PORT" "$REMOTE_HOST" "$(remote_network_command)" &
    remote_pid=$!
    (run_network_node 0) &
    local_pid=$!

    local status=0
    wait "$local_pid" || status=$?
    local_pid=""
    local remote_status=0
    wait "$remote_pid" || remote_status=$?
    if [ "$status" -eq 0 ] && [ "$remote_status" -ne 0 ]; then
        status=$remote_status
    fi
    remote_pid=""
    trap - EXIT INT TERM
    return "$status"
}

usage() {
    echo "Usage: $0 {encode|network-server|network-client|network|all} [encode arguments]"
}

mode=${1:-}
[ $# -eq 0 ] || shift
case "$mode" in
    encode) run_encode "$@" ;;
    network-server) run_network_node 0 ;;
    network-client) run_network_node 1 ;;
    network) run_network ;;
    all) run_encode "$@" && run_network ;;
    *) usage; exit 2 ;;
esac
