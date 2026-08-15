#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$REPO_ROOT"

MODE="${MODE:-both}"
BYTES="${BENCH_BYTES:-1G}"
ITERS="${BENCH_ITERS:-20}"
WARMUP="${BENCH_WARMUP:-5}"
MASTER_ADDR="${MASTER_ADDR:-172.16.0.224}"
MASTER_PORT="${MASTER_PORT:-6000}"
NNODES="${NNODES:-8}"
BASE_PORT="${NETWORK_BENCH_BASE_PORT:-45000}"
CHANNELS_PER_PEER="${GEMINI2_CHANNELS_PER_PEER:-16}"
GEMINI_GROUP_SIZE="${GEMINI2_GROUP_SIZE:-8}"
GEMINI2_CPU_SEND="${GEMINI2_CPU_SEND:-0}"
CONCORD_N="${CONCORD_N:-8}"
CONCORD_LANES_PER_PEER="${CONCORD_RDMA_LANES_PER_PEER:-16}"
CONCORD_SEND_LANES_PER_PEER="${CONCORD_SEND_LANES_PER_PEER:-$CONCORD_LANES_PER_PEER}"
CONCORD_RECV_LANES_PER_PEER="${CONCORD_RECV_LANES_PER_PEER:-$CONCORD_LANES_PER_PEER}"
CONCORD_SEGMENTS="${CONCORD_LAYER_EXCHANGE_SEG:-4}"
CONCORD_SIMPLE="${CONCORD_SIMPLE:-0}"
CONCORD_CPU_SEND="${CONCORD_CPU_SEND:-0}"
BACKEND="${BENCH_DIST_BACKEND:-gloo}"
NETIFACES_INTERFACE="${NETIFACES_INTERFACE:-eth0}"

export CONCORD_LOCAL_RANK_NIC_0=${CONCORD_LOCAL_RANK_NIC_0:-eth0}
export CONCORD_LOCAL_RANK_NIC_1=${CONCORD_LOCAL_RANK_NIC_1:-eth0}
export CONCORD_LOCAL_RANK_NIC_2=${CONCORD_LOCAL_RANK_NIC_2:-eth0}
export CONCORD_LOCAL_RANK_NIC_3=${CONCORD_LOCAL_RANK_NIC_3:-eth0}
export CONCORD_LOCAL_RANK_NIC_4=${CONCORD_LOCAL_RANK_NIC_4:-eth1}
export CONCORD_LOCAL_RANK_NIC_5=${CONCORD_LOCAL_RANK_NIC_5:-eth1}
export CONCORD_LOCAL_RANK_NIC_6=${CONCORD_LOCAL_RANK_NIC_6:-eth1}
export CONCORD_LOCAL_RANK_NIC_7=${CONCORD_LOCAL_RANK_NIC_7:-eth1}

export GEMINI_REPLICAS_LOCAL_RANK_NIC_0=${GEMINI_REPLICAS_LOCAL_RANK_NIC_0:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_1=${GEMINI_REPLICAS_LOCAL_RANK_NIC_1:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_2=${GEMINI_REPLICAS_LOCAL_RANK_NIC_2:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_3=${GEMINI_REPLICAS_LOCAL_RANK_NIC_3:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_4=${GEMINI_REPLICAS_LOCAL_RANK_NIC_4:-eth1}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_5=${GEMINI_REPLICAS_LOCAL_RANK_NIC_5:-eth1}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_6=${GEMINI_REPLICAS_LOCAL_RANK_NIC_6:-eth1}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_7=${GEMINI_REPLICAS_LOCAL_RANK_NIC_7:-eth1}

usage() {
    cat <<EOF
Usage: $0 <node_rank> <gpu_id_0> [gpu_id_1 ...] [options]

Options:
  --mode gemini2|concord|both
  --bytes SIZE              default: $BYTES
  --iters N                 default: $ITERS
  --warmup N                default: $WARMUP
  --master-addr IP          default: $MASTER_ADDR
  --master-port PORT        default: $MASTER_PORT
  --base-port PORT          default: $BASE_PORT
  --channels-per-peer N     Gemini2 channels, default: $CHANNELS_PER_PEER
  --gemini-group-size N      Gemini2 ring group size, default: $GEMINI_GROUP_SIZE
  --gemini-cpu-send          use pinned CPU send buffers for Gemini2
  --concord-n N              Concord group size, default: $CONCORD_N
  --concord-lanes-per-peer N default shared lanes: $CONCORD_LANES_PER_PEER
  --concord-send-lanes-per-peer N default: $CONCORD_SEND_LANES_PER_PEER
  --concord-recv-lanes-per-peer N default: $CONCORD_RECV_LANES_PER_PEER
  --concord-segments N      default: $CONCORD_SEGMENTS
  --concord-simple          split bytes evenly across n-1 peers
  --concord-cpu-send        use pinned CPU send buffers for Concord
EOF
}

NODE_RANK=0
if [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; then
    NODE_RANK="$1"
    shift
fi

GPU_IDS=()
while [[ $# -gt 0 && "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --bytes) BYTES="$2"; shift 2 ;;
        --iters) ITERS="$2"; shift 2 ;;
        --warmup) WARMUP="$2"; shift 2 ;;
        --master-addr) MASTER_ADDR="$2"; shift 2 ;;
        --master-port) MASTER_PORT="$2"; shift 2 ;;
        --base-port) BASE_PORT="$2"; shift 2 ;;
        --channels-per-peer) CHANNELS_PER_PEER="$2"; shift 2 ;;
        --gemini-group-size) GEMINI_GROUP_SIZE="$2"; shift 2 ;;
        --gemini-cpu-send) GEMINI2_CPU_SEND=1; shift ;;
        --concord-n) CONCORD_N="$2"; shift 2 ;;
        --concord-lanes-per-peer) CONCORD_LANES_PER_PEER="$2"; CONCORD_SEND_LANES_PER_PEER="$2"; CONCORD_RECV_LANES_PER_PEER="$2"; shift 2 ;;
        --concord-send-lanes-per-peer) CONCORD_SEND_LANES_PER_PEER="$2"; shift 2 ;;
        --concord-recv-lanes-per-peer) CONCORD_RECV_LANES_PER_PEER="$2"; shift 2 ;;
        --concord-segments) CONCORD_SEGMENTS="$2"; shift 2 ;;
        --concord-simple) CONCORD_SIMPLE=1; shift ;;
        --concord-cpu-send) CONCORD_CPU_SEND=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
    echo "Error: At least one GPU id must be specified" >&2
    usage
    exit 1
fi
if [[ "$MODE" != "gemini2" && "$MODE" != "concord" && "$MODE" != "both" ]]; then
    echo "Error: invalid mode $MODE" >&2
    exit 1
fi

export MASTER_ADDR MASTER_PORT NETIFACES_INTERFACE
export GEMINI_REPLICAS_INTERFACE="${GEMINI_REPLICAS_INTERFACE:-$NETIFACES_INTERFACE}"
export CONCORD_INTERFACE="${CONCORD_INTERFACE:-$NETIFACES_INTERFACE}"
export CONCORD_RDMA_LANES_PER_PEER=$((CONCORD_SEND_LANES_PER_PEER + CONCORD_RECV_LANES_PER_PEER))
export CONCORD_SEND_LANES_PER_PEER="$CONCORD_SEND_LANES_PER_PEER"
export CONCORD_RECV_LANES_PER_PEER="$CONCORD_RECV_LANES_PER_PEER"
export CONCORD_ALLOW_UNSAFE_LANE_SHARING="${CONCORD_ALLOW_UNSAFE_LANE_SHARING:-1}"

GPUS_PER_NODE=${#GPU_IDS[@]}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
if [[ "$MODE" == "gemini2" || "$MODE" == "both" ]]; then
    if (( WORLD_SIZE % GEMINI_GROUP_SIZE != 0 )); then
        echo "Error: WORLD_SIZE=$WORLD_SIZE must be divisible by GEMINI_GROUP_SIZE=$GEMINI_GROUP_SIZE" >&2
        exit 1
    fi
fi
if [[ "$MODE" == "concord" || "$MODE" == "both" ]]; then
    if (( WORLD_SIZE % CONCORD_N != 0 )); then
        echo "Error: WORLD_SIZE=$WORLD_SIZE must be divisible by CONCORD_N=$CONCORD_N" >&2
        exit 1
    fi
fi
LOG_DIR="${NETWORK_BENCH_LOG_DIR:-$SCRIPT_DIR/logs_network_bench}"
mkdir -p "$LOG_DIR"

run_one_mode() {
    local mode="$1"
    local nccl_port_offset="$2"
    local rdma_port_offset="$3"
    local bench_port=$((MASTER_PORT + nccl_port_offset))
    local rdma_port=$((BASE_PORT + rdma_port_offset))
    local max_rdmabase=$rdma_port
    if [[ "$mode" == "concord" ]]; then
        local num_groups=$((WORLD_SIZE / CONCORD_N))
        max_rdmabase=$((rdma_port + (num_groups - 1) * CONCORD_N * 100))
    elif (( WORLD_SIZE > 0 )); then
        max_rdmabase=$((rdma_port + (WORLD_SIZE - 1) * 100))
    fi
    if (( max_rdmabase > 65000 )); then
        echo "Error: max RDMA base port $max_rdmabase exceeds uint16 limit (65000); lower --base-port" >&2
        return 1
    fi
    local extra_args=()
    if [[ "$mode" == "gemini2" && "$GEMINI2_CPU_SEND" == "1" ]]; then
        extra_args+=(--cpu-send)
    fi
    if [[ "$mode" == "concord" && "$CONCORD_SIMPLE" == "1" ]]; then
        extra_args+=(--simple)
    fi
    if [[ "$mode" == "concord" && "$CONCORD_CPU_SEND" == "1" ]]; then
        extra_args+=(--cpu-send)
    fi
    local script
    if [[ "$mode" == "gemini2" ]]; then
        script="$SCRIPT_DIR/bench_gemini2_network.py"
    else
        script="$SCRIPT_DIR/bench_concord_network.py"
    fi

    echo "========================================="
    echo "Network bench mode=$mode node=$NODE_RANK world=$WORLD_SIZE bytes=$BYTES"
    echo "Gemini group size=$GEMINI_GROUP_SIZE Concord n=$CONCORD_N"
    echo "Concord split lanes send=$CONCORD_SEND_LANES_PER_PEER recv=$CONCORD_RECV_LANES_PER_PEER total=$CONCORD_RDMA_LANES_PER_PEER"
    echo "MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$bench_port BASE_PORT=$rdma_port"
    echo "GPU_IDS=${GPU_IDS[*]} logs=$LOG_DIR"
    echo "========================================="

    local pids=()
    for ((local_rank = 0; local_rank < GPUS_PER_NODE; ++local_rank)); do
        local gpu_id=${GPU_IDS[$local_rank]}
        local rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local log_file="$LOG_DIR/${mode}_node${NODE_RANK}_rank${rank}.log"
        echo "Launching mode=$mode rank=$rank local_rank=$local_rank gpu=$gpu_id log=$log_file"
        if [[ "$mode" == "gemini2" ]]; then
            CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 python3 -X faulthandler "$script" \
                --rank "$rank" \
                --world-size "$WORLD_SIZE" \
                --local-rank "$local_rank" \
                --master-addr "$MASTER_ADDR" \
                --master-port "$bench_port" \
                --base-port "$rdma_port" \
                --bytes "$BYTES" \
                --iters "$ITERS" \
                --warmup "$WARMUP" \
                --channels-per-peer "$CHANNELS_PER_PEER" \
                --group-size "$GEMINI_GROUP_SIZE" \
                "${extra_args[@]}" \
                --backend "$BACKEND" > "$log_file" 2>&1 &
        else
            CUDA_VISIBLE_DEVICES="$gpu_id" PYTHONUNBUFFERED=1 python3 -X faulthandler "$script" \
                --rank "$rank" \
                --world-size "$WORLD_SIZE" \
                --local-rank "$local_rank" \
                --master-addr "$MASTER_ADDR" \
                --master-port "$bench_port" \
                --base-port "$rdma_port" \
                --bytes "$BYTES" \
                --iters "$ITERS" \
                --warmup "$WARMUP" \
                --n "$CONCORD_N" \
                --lanes-per-peer "$CONCORD_LANES_PER_PEER" \
                --send-lanes-per-peer "$CONCORD_SEND_LANES_PER_PEER" \
                --recv-lanes-per-peer "$CONCORD_RECV_LANES_PER_PEER" \
                --segments "$CONCORD_SEGMENTS" \
                "${extra_args[@]}" \
                --backend "$BACKEND" > "$log_file" 2>&1 &
        fi
        pids+=("$!")
    done

    local status=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            status=1
        fi
    done

    echo "========================================="
    echo "Results mode=$mode"
    echo "========================================="
    for ((local_rank = 0; local_rank < GPUS_PER_NODE; ++local_rank)); do
        local rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
        local log_file="$LOG_DIR/${mode}_node${NODE_RANK}_rank${rank}.log"
        if [[ -f "$log_file" ]]; then
            awk '/^(ITER_RESULT|RESULT|SUMMARY),/ { print }' "$log_file" || true
            if [[ "$status" -ne 0 ]]; then
                awk '/(ERROR|Traceback|RuntimeError)/ { print }' "$log_file" || true
            fi
        fi
    done
    return "$status"
}

status=0
if [[ "$MODE" == "gemini2" || "$MODE" == "both" ]]; then
    run_one_mode gemini2 31000 0 || status=1
fi
if [[ "$MODE" == "concord" || "$MODE" == "both" ]]; then
    run_one_mode concord 32000 1000 || status=1
fi
exit "$status"
