#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MASTER_ADDR=${MASTER_ADDR:-172.16.0.224}
BASE_PORT=${GEMINI_GDR_BENCH_BASE_PORT:-36000}
NNODES=${NNODES:-2}
GPUS_PER_NODE_DEFAULT=8

export NETIFACES_INTERFACE=${NETIFACES_INTERFACE:-eth0}
export GEMINI_REPLICAS_INTERFACE=${GEMINI_REPLICAS_INTERFACE:-$NETIFACES_INTERFACE}

export GEMINI_REPLICAS_LOCAL_RANK_NIC_0=${GEMINI_REPLICAS_LOCAL_RANK_NIC_0:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_1=${GEMINI_REPLICAS_LOCAL_RANK_NIC_1:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_2=${GEMINI_REPLICAS_LOCAL_RANK_NIC_2:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_3=${GEMINI_REPLICAS_LOCAL_RANK_NIC_3:-eth0}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_4=${GEMINI_REPLICAS_LOCAL_RANK_NIC_4:-eth1}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_5=${GEMINI_REPLICAS_LOCAL_RANK_NIC_5:-eth1}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_6=${GEMINI_REPLICAS_LOCAL_RANK_NIC_6:-eth1}
export GEMINI_REPLICAS_LOCAL_RANK_NIC_7=${GEMINI_REPLICAS_LOCAL_RANK_NIC_7:-eth1}

NODE_RANK=0
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    NODE_RANK=$1
    shift
fi

GPU_IDS=()
while [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    for ((i = 0; i < GPUS_PER_NODE_DEFAULT; ++i)); do
        GPU_IDS+=("$i")
    done
fi

GPUS_PER_NODE=${#GPU_IDS[@]}
WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
EXE="$SCRIPT_DIR/gemini_gdr_sendrecv_bench"

if [ ! -x "$EXE" ]; then
    echo "Executable not found: $EXE"
    echo "Run: $SCRIPT_DIR/build_gemini_gdr_sendrecv.sh"
    exit 1
fi

DEFAULT_ARGS=(
    --world-size "$WORLD_SIZE"
    --node-rank "$NODE_RANK"
    --master-addr "$MASTER_ADDR"
    --base-port "$BASE_PORT"
    --size-mb 2560
    --iters 20
    --warmup 10
    --chunk-mb 64
    --batch-wr 4
)

echo "========================================="
echo "Gemini GDR+D2H 2-node send/recv bench"
echo "========================================="
echo "MASTER_ADDR=$MASTER_ADDR"
echo "BASE_PORT=$BASE_PORT"
echo "NODE_RANK=$NODE_RANK"
echo "NNODES=$NNODES"
echo "GPUS_PER_NODE=$GPUS_PER_NODE"
echo "WORLD_SIZE=$WORLD_SIZE"
echo "GPU_IDS=${GPU_IDS[*]}"
echo "Extra args: $*"
echo "========================================="

pids=()
for ((local_rank = 0; local_rank < GPUS_PER_NODE; ++local_rank)); do
    gpu_id=${GPU_IDS[$local_rank]}
    rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
    log_file="gemini_gdr_bench_node${NODE_RANK}_rank${rank}.log"
    echo "Launching rank=$rank local_rank=$local_rank gpu=$gpu_id log=$log_file"
    CUDA_VISIBLE_DEVICES="$gpu_id" "$EXE" \
        --rank "$rank" \
        --local-rank "$local_rank" \
        "${DEFAULT_ARGS[@]}" \
        "$@" > "$log_file" 2>&1 &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

echo "========================================="
echo "Results"
echo "========================================="
for ((local_rank = 0; local_rank < GPUS_PER_NODE; ++local_rank)); do
    rank=$((NODE_RANK * GPUS_PER_NODE + local_rank))
    log_file="gemini_gdr_bench_node${NODE_RANK}_rank${rank}.log"
    if [ -f "$log_file" ]; then
        awk '/^(ITER_RESULT|RESULT),/ { print }' "$log_file" || true
        if [ "$status" -ne 0 ]; then
            awk '/ERROR:/ { print }' "$log_file" || true
        fi
    fi
done

exit "$status"
