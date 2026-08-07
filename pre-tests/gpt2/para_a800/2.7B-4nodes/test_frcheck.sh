#!/bin/bash

# ONNX in the container uses legacy generated protobuf descriptors.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}

export FRCHECK_ENABLE_RECOVERY_PARITY_REPAIR=1
export FRCHECK_RECOVERY_SKIP_PADDING=0

# Usage: ./test_frcheck.sh <node_rank> [<gpu_id_0> [gpu_id_1 ...]] [additional_args...]
# Example: ./test_frcheck.sh 0
# Example (2 GPUs per container): ./test_frcheck.sh 0 2 3

cd /workspace/Megatron-LM || exit 1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=${NETIFACES_INTERFACE:-bond0}
export FRCHECK_LAYER_EXCHANGE_SEG=${FRCHECK_LAYER_EXCHANGE_SEG:-12}
export FRCHECK_LAYER_EXCHANGE_CHUNK_MB=${FRCHECK_LAYER_EXCHANGE_CHUNK_MB:-32}
export FRCHECK_LAYER_FRONTIER_ORDER=${FRCHECK_LAYER_FRONTIER_ORDER:-chunk_layer_sid}
export FRCHECK_LAYER_ENCODE_BATCH=${FRCHECK_LAYER_ENCODE_BATCH:-24}
FRCHECK_GDR=${FRCHECK_GDR:-0}
case "$FRCHECK_GDR" in
    0) FRCHECK_GDR_ARGS=() ;;
    1) FRCHECK_GDR_ARGS=(--frcheck-gdr) ;;
    *)
        echo "Error: FRCHECK_GDR must be 0 or 1: $FRCHECK_GDR" >&2
        exit 1
        ;;
esac

FRCHECK_ASYNC_PARITY=${FRCHECK_ASYNC_PARITY:-1}
case "$FRCHECK_ASYNC_PARITY" in
    0) FRCHECK_ASYNC_PARITY_ARGS=() ;;
    1) FRCHECK_ASYNC_PARITY_ARGS=(--frcheck-async-parity) ;;
    *)
        echo "Error: FRCHECK_ASYNC_PARITY must be 0 or 1: $FRCHECK_ASYNC_PARITY" >&2
        exit 1
        ;;
esac

FRCHECK_RECOVERY_ASYNC_PARITY=${FRCHECK_RECOVERY_ASYNC_PARITY:-1}
case "$FRCHECK_RECOVERY_ASYNC_PARITY" in
    0) FRCHECK_RECOVERY_ASYNC_PARITY_ARGS=() ;;
    1) FRCHECK_RECOVERY_ASYNC_PARITY_ARGS=(--frcheck-recovery-async-parity) ;;
    *)
        echo "Error: FRCHECK_RECOVERY_ASYNC_PARITY must be 0 or 1: $FRCHECK_RECOVERY_ASYNC_PARITY" >&2
        exit 1
        ;;
esac

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
MASTER_ADDR=${MASTER_ADDR:-10.252.129.35}

export ECCHECK_USE_ASIO=false
export FRCHECK_INTERFACE=$NETIFACES_INTERFACE
# Resolve each node control-plane IP from FRCHECK_INTERFACE.

# FRCheck native split-lane configuration.  The native layer maps logical
# lane ids to separate forward/reverse RDMA lane pools, so A->B and B->A do
# not contend on the same underlying channel set.
export FRCHECK_SEND_LANES_PER_PEER=${FRCHECK_SEND_LANES_PER_PEER:-12}
export FRCHECK_RECV_LANES_PER_PEER=${FRCHECK_RECV_LANES_PER_PEER:-12}
export FRCHECK_RDMA_LANES_PER_PEER=$((FRCHECK_SEND_LANES_PER_PEER + FRCHECK_RECV_LANES_PER_PEER))
export FRCHECK_LAYER_EXCHANGE_SEG=${FRCHECK_LAYER_EXCHANGE_SEG:-12}
export FRCHECK_ALLOW_UNSAFE_LANE_SHARING=${FRCHECK_ALLOW_UNSAFE_LANE_SHARING:-1}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-4}
export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE

# POA table directory (auto-generates if file not found)
FRCHECK_TABLE_DIR="/workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies"

# If first argument is a numeric node rank use it, otherwise default to 0
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# Next arguments are GPU IDs, collect them until we hit a non-numeric
GPU_IDS=()
while [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    # No GPU IDs provided: use all GPUs in the system
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "Error: nvidia-smi not found and no GPU IDs specified."
        exit 1
    fi
    mapfile -t ALL_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    if [ "${#ALL_IDS[@]}" -eq 0 ]; then
        echo "Error: No GPUs found on this node."
        exit 1
    fi
    GPU_IDS=("${ALL_IDS[@]}")
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

rdma_hca_for_gpu() {
    case "$1" in
        0|1) echo mlx5_0 ;;
        2|3) echo mlx5_1 ;;
        4|5) echo mlx5_4 ;;
        6|7) echo mlx5_5 ;;
        *)
            echo "Unsupported physical GPU id for RDMA binding: $1" >&2
            return 1
            ;;
    esac
}

configure_rdma_local_rank_bindings() {
    local prefix=$1
    local local_rank
    local hca
    for local_rank in "${!GPU_IDS[@]}"; do
        hca=$(rdma_hca_for_gpu "${GPU_IDS[$local_rank]}") || exit 1
        export "${prefix}_LOCAL_RANK_NIC_${local_rank}=${hca}"
    done
}

configure_rdma_local_rank_bindings FRCHECK

export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-"/workspace/Megatron-LM/logs/gpt2-2.7b-4nodes/frcheck"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/dev/shm/models/gpt2-2.7b-4nodes-frcheck"}
# DATA_PATH="/workspace/Megatron-LM/pre-tests/opt/opt_data/wiki_text_sentence"

SHM_PKT="/dev/shm/shm_pkt"


MODE=save
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2|inprocess|inprocess2|inprocess_sw)$ ]]; then
    MODE="$1"
    shift
fi
ARGS_TO_PASS=("$@")
RECOVERY_MODE_ARGS=()
FRCHECK_HW_EARLY_OPTIMIZER=${FRCHECK_HW_EARLY_OPTIMIZER:-0}
case "$FRCHECK_HW_EARLY_OPTIMIZER" in
    0) FRCHECK_HW_EARLY_OPTIMIZER_ARGS=() ;;
    1) FRCHECK_HW_EARLY_OPTIMIZER_ARGS=(--frcheck-hw-early-optimizer) ;;
    *)
        echo "Error: FRCHECK_HW_EARLY_OPTIMIZER must be 0 or 1: $FRCHECK_HW_EARLY_OPTIMIZER" >&2
        exit 1
        ;;
esac
FRCHECK_HW_OPTIMIZER_OVERLAP=${FRCHECK_HW_OPTIMIZER_OVERLAP:-0}
case "$FRCHECK_HW_OPTIMIZER_OVERLAP" in
    0) FRCHECK_HW_OPTIMIZER_OVERLAP_ARGS=() ;;
    1) FRCHECK_HW_OPTIMIZER_OVERLAP_ARGS=(--frcheck-hw-optimizer-overlap) ;;
    *)
        echo "Error: FRCHECK_HW_OPTIMIZER_OVERLAP must be 0 or 1: $FRCHECK_HW_OPTIMIZER_OVERLAP" >&2
        exit 1
        ;;
esac
if [ "$FRCHECK_HW_EARLY_OPTIMIZER" = 1 ] && [ "$FRCHECK_HW_OPTIMIZER_OVERLAP" = 1 ]; then
    echo "Error: FRCHECK_HW_EARLY_OPTIMIZER and FRCHECK_HW_OPTIMIZER_OVERLAP are mutually exclusive" >&2
    exit 1
fi
FT_INPROCESS_RECOVERY_REPEAT=${FT_INPROCESS_RECOVERY_REPEAT:-3}
case "$MODE" in
    save)
        RECOVERY_MODE_ARGS=(
            --save $CHECKPOINT_PATH
            --ec-checkpoint-write-only-penultimate-iter
            "${FRCHECK_ASYNC_PARITY_ARGS[@]}"
            --frcheck-layer-exchange-encode
        )
        ;;
    software)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
        )
        ;;
    hardware)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-frcheck-hardware-failure
            --frcheck-async-recovery-forward
            "${FRCHECK_RECOVERY_ASYNC_PARITY_ARGS[@]}"
            --frcheck-failed-ranks "0,1,2,3,4,5,6,7"
            --frcheck-recovery-safe-point optimizer_step
            --frcheck-recovery-only-teardown
            # Native RDMA stays alive until optimizer_step; forked DataLoader workers segfault.
            --num-workers 0
        )
        ;;
    hardware2)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-frcheck-hardware-failure
            --frcheck-failed-ranks "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
        )
        ;;
    inprocess_sw)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            --ft-inprocess-recovery-software-failure
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "0,1,2,3,4,5,6,7"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
        )
        ;;
    inprocess2)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            "${FRCHECK_HW_EARLY_OPTIMIZER_ARGS[@]}"
            "${FRCHECK_HW_OPTIMIZER_OVERLAP_ARGS[@]}"
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
            --frcheck-async-recovery-forward
            "${FRCHECK_RECOVERY_ASYNC_PARITY_ARGS[@]}"
            --frcheck-recovery-safe-point optimizer_step
            --frcheck-recovery-only-teardown
            --num-workers 0
        )
        ;;
    inprocess)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            "${FRCHECK_HW_EARLY_OPTIMIZER_ARGS[@]}"
            "${FRCHECK_HW_OPTIMIZER_OVERLAP_ARGS[@]}"
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "0,1,2,3,4,5,6,7"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
            --frcheck-async-recovery-forward
            "${FRCHECK_RECOVERY_ASYNC_PARITY_ARGS[@]}"
            --frcheck-recovery-safe-point optimizer_step
            --frcheck-recovery-only-teardown
            --num-workers 0
        )
        ;;
esac

# Model configuration
HIDDEN_SIZE=2560
NUM_ATTENTION_HEADS=32
NUM_LAYERS=32

SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

DATA_ARGS=(
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --mock-data
)

GPT_ARGS=(
    --no-async-tensor-model-parallel-allreduce
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 0.00005
    --train-iters 10
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 1e-2
    --lr-warmup-fraction .05
    --clip-grad 1.0
    --fp16
    --tokenizer-type GPT2BPETokenizer
    --use-mcore-models
    "${FRCHECK_GDR_ARGS[@]}"
    --transformer-impl transformer_engine
    --no-scatter-gather-tensors-in-pipeline
    --num-layers $NUM_LAYERS
    --optimizer adam
    --loss-scale-window 100
    --initial-loss-scale 4096
    --min-loss-scale 1.0
    --hysteresis 2
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 4
    --sequence-parallel
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    #--save $CHECKPOINT_PATH
    --ec-checkpoint-write-only-penultimate-iter
    #--load $CHECKPOINT_PATH
    
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH

    --use-frcheck
    --frcheck-n 4
    --frcheck-table-dir $FRCHECK_TABLE_DIR
    --ckpt-format torch
    --save-embeddings-separately
    # --timing-log-level 2
)

mkdir -p logs
mkdir -p logs/csv

EC_RANK_LAUNCH='
set -e
if [[ ! "${LOCAL_RANK:-}" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid LOCAL_RANK: ${LOCAL_RANK:-<unset>}" >&2
    exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES is empty or unset." >&2
    exit 1
fi
IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( LOCAL_RANK >= ${#visible_gpus[@]} )); then
    echo "Error: LOCAL_RANK $LOCAL_RANK is outside CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
physical_gpu=${visible_gpus[$LOCAL_RANK]//[[:space:]]/}
if [[ ! "$physical_gpu" =~ ^[0-7]$ ]]; then
    echo "Error: Invalid physical GPU id for LOCAL_RANK $LOCAL_RANK: ${physical_gpu:-<empty>}" >&2
    exit 1
fi
if (( physical_gpu < 4 )); then
    numa_node=0
    base=0
else
    numa_node=1
    base=32
fi
slot=$((physical_gpu % 4))
first_start=$((base + slot * 8))
first_end=$((first_start + 7))
sibling_start=$((first_start + 64))
sibling_end=$((sibling_start + 7))
net_physical_cores=${FRCHECK_NET_PHYSICAL_CORES:-0}
if [[ ! "$net_physical_cores" =~ ^[0-9]+$ ]] || (( net_physical_cores > 7 )); then
    echo "Error: FRCHECK_NET_PHYSICAL_CORES must be an integer from 0 to 7: $net_physical_cores" >&2
    exit 1
fi
if (( net_physical_cores == 0 )); then
    first_cpus=$(seq -s, "$first_start" "$first_end")
    sibling_cpus=$(seq -s, "$sibling_start" "$sibling_end")
    rs_cpus="$first_cpus,$sibling_cpus"
    rs_unique_cpus="$rs_cpus"
    process_cpus="$rs_cpus"
else
    net_first_end=$((first_start + net_physical_cores - 1))
    net_sibling_end=$((sibling_start + net_physical_cores - 1))
    rs_first_start=$((net_first_end + 1))
    rs_sibling_start=$((net_sibling_end + 1))
    net_first_cpus=$(seq -s, "$first_start" "$net_first_end")
    net_sibling_cpus=$(seq -s, "$sibling_start" "$net_sibling_end")
    rs_first_cpus=$(seq -s, "$rs_first_start" "$first_end")
    rs_sibling_cpus=$(seq -s, "$rs_sibling_start" "$sibling_end")
    process_cpus="$net_first_cpus,$net_sibling_cpus"
    rs_unique_cpus="$rs_first_cpus,$rs_sibling_cpus"
    IFS=, read -r -a rs_unique_cpu_array <<< "$rs_unique_cpus"
    rs_cpu_array=()
    for ((i = 0; i < 16; i++)); do
        rs_cpu_array+=("${rs_unique_cpu_array[$((i % ${#rs_unique_cpu_array[@]}))]}")
    done
    rs_cpus=$(IFS=,; echo "${rs_cpu_array[*]}")
fi
export FRCHECK_RS_CPU_LIST="$rs_cpus"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
echo "[cpu_bind] local_rank=$LOCAL_RANK physical_gpu=$physical_gpu numa_node=$numa_node net_physical_cores=$net_physical_cores rs_cpus=$rs_cpus rs_unique_cpus=$rs_unique_cpus process_cpus=$process_cpus"
exec taskset -c "$process_cpus" "${PYTHON_BIN:-python}" "$@"
'

# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    printf 'Would run (Node %s, topology-aware per-rank CPU binding): ' "$NODE_RANK"
    printf 'FRCHECK_GDR=%q PYTHONPATH=%q CUDA_VISIBLE_DEVICES=%q ' \
        "$FRCHECK_GDR" "$PYTHONPATH:/workspace/Megatron-LM" "$CUDA_VISIBLE_DEVICES"
    printf '%q ' torchrun "${DISTRIBUTED_ARGS[@]}" --no-python bash -c '<physical-GPU CPU binding>' _ \
        pretrain_gpt.py "${GPT_ARGS[@]}" "${DATA_ARGS[@]}" "${MODEL_PARALLEL_ARGS[@]}" \
        "${RECOVERY_MODE_ARGS[@]}" "${EVAL_AND_LOGGING_ARGS[@]}" --distributed-backend nccl \
        "${ARGS_TO_PASS[@]}"
    printf '\n'
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (FRCheck)"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"
echo "FRCHECK_TABLE_DIR: $FRCHECK_TABLE_DIR"
echo "FRCHECK_INTERFACE: $FRCHECK_INTERFACE"
echo "FRCHECK_HW_EARLY_OPTIMIZER: $FRCHECK_HW_EARLY_OPTIMIZER"
echo "FRCHECK_HW_OPTIMIZER_OVERLAP: $FRCHECK_HW_OPTIMIZER_OVERLAP"
echo "FRCHECK_GDR: $FRCHECK_GDR"
echo "FRCHECK_LAYER_EXCHANGE_CHUNK_MB: $FRCHECK_LAYER_EXCHANGE_CHUNK_MB"
echo "FRCHECK_LAYER_FRONTIER_ORDER: $FRCHECK_LAYER_FRONTIER_ORDER"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun "${DISTRIBUTED_ARGS[@]}" \
    --no-python bash -c "$EC_RANK_LAUNCH" _ \
    pretrain_gpt.py \
    "${GPT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${RECOVERY_MODE_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    --distributed-backend nccl \
    "${ARGS_TO_PASS[@]}"
