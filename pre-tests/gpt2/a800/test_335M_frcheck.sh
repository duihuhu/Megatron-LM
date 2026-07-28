#!/bin/bash

# FRCheck (POA-driven stripe encode with RDMA) multi-node script.
# Usage: ./test_335M_frcheck.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [mode] [additional_args...]
# Example: ./test_335M_frcheck.sh 0 0
# Example (2 GPUs per container): ./test_335M_frcheck.sh 0 2 3 inprocess2
export FRCHECK_ENABLE_RECOVERY_PARITY_REPAIR=1
export FRCHECK_RECOVERY_SKIP_PADDING=0
#export FRCHECK_TRACE_INIT=1
# Experimental: shared-lane RDMA multiplexing (tagged DATA/ACK protocol).
# Reduces channels/rank from peers*num_stripes to peers*lanes. Opt-in for now.
export FRCHECK_RDMA_LANES_PER_PEER=8
export FRCHECK_ALLOW_UNSAFE_LANE_SHARING=1
export FRCHECK_LAYER_EXCHANGE_SEG=4
export FRCHECK_LAYER_ENCODE_BATCH=2
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=bond0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
MASTER_ADDR=${MASTER_ADDR:-10.0.0.62}

export ECCHECK_USE_ASIO=false
export FRCHECK_INTERFACE=$NETIFACES_INTERFACE
export FRCHECK_BASE_IP=$MASTER_ADDR
# FRCheck RDMA listens on FRCHECK_BASE_PORT + rank_in_group.
# Keep this separate from torchrun MASTER_PORT and move it if a port is busy.
export FRCHECK_BASE_PORT=${FRCHECK_BASE_PORT:-27200}
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-8}

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
    echo "Error: At least one GPU id must be specified."
    echo "Usage: ./test_335M_frcheck.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [mode] [additional_args...]"
    exit 1
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

make_rank_range() {
    local count=$1
    local ranks=()
    local rank
    for ((rank=0; rank<count; rank++)); do
        ranks+=("$rank")
    done
    (IFS=,; echo "${ranks[*]}")
}

FRCHECK_SINGLE_NODE_FAILED_RANKS=${FRCHECK_FAILED_RANKS:-$(make_rank_range "$GPUS_PER_NODE")}
FRCHECK_TWO_NODE_FAILED_RANKS=${FRCHECK_HW2_FAILED_RANKS:-$(make_rank_range "$((2 * GPUS_PER_NODE))")}

export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs"
CHECKPOINT_PATH="/dev/shm/data/checkpoint/models/gpt2-345m-0-frcheck"
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document"

SHM_PKT="/dev/shm/shm_pkt"

MODE=save
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2|inprocess|inprocess2|inprocess_sw)$ ]]; then
    MODE="$1"
    shift
fi



ARGS_TO_PASS=("$@")
RECOVERY_MODE_ARGS=()
FT_INPROCESS_RECOVERY_REPEAT=${FT_INPROCESS_RECOVERY_REPEAT:-3}

set_inprocess_recovery_args() {
    local failed_ranks=$1
    RECOVERY_MODE_ARGS=(
        --load $CHECKPOINT_PATH
        --ft-inprocess-recovery-benchmark
        --rerun-mode disabled
        --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
        --ft-inprocess-recovery-failed-ranks "$failed_ranks"
        --ft-inprocess-recovery-after-train-iter 0
        --ft-inprocess-recovery-exit-after-forward
        --frcheck-async-recovery-forward
        --frcheck-recovery-safe-point optimizer_step
        --frcheck-recovery-only-teardown
        --num-workers 0
    )
}

case "$MODE" in
    save)
        RECOVERY_MODE_ARGS=(
            --save $CHECKPOINT_PATH
            --ec-checkpoint-write-only-penultimate-iter
            --frcheck-layer-exchange-encode
        )
        if [ "${FRCHECK_ASYNC_PARITY:-1}" != "0" ]; then
            RECOVERY_MODE_ARGS+=(--frcheck-async-parity)
        fi
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
            --frcheck-failed-ranks "$FRCHECK_SINGLE_NODE_FAILED_RANKS"
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
            --frcheck-async-recovery-forward
            --frcheck-failed-ranks "$FRCHECK_TWO_NODE_FAILED_RANKS"
            --frcheck-recovery-safe-point optimizer_step
            --frcheck-recovery-only-teardown
            # Native RDMA stays alive until optimizer_step; forked DataLoader workers segfault.
            --num-workers 0
        )
        ;;
    inprocess_sw)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            --ft-inprocess-recovery-software-failure
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "$FRCHECK_SINGLE_NODE_FAILED_RANKS"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
        )
        ;;
    inprocess)
        set_inprocess_recovery_args "$FRCHECK_SINGLE_NODE_FAILED_RANKS"
        ;;
    inprocess2)
        set_inprocess_recovery_args "$FRCHECK_TWO_NODE_FAILED_RANKS"
        ;;
esac

if [[ "$MODE" =~ ^(hardware|hardware2|inprocess|inprocess2)$ ]] \
    && [ "${FRCHECK_RECOVERY_ASYNC_PARITY:-1}" != "0" ]; then
    RECOVERY_MODE_ARGS+=(--frcheck-recovery-async-parity)
fi

# Model configuration
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16

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
    --lr 0.00015
    --train-iters 4
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 1e-2
    --lr-warmup-fraction .01
    --clip-grad 1.0
    --fp16
    --tokenizer-type GPT2BPETokenizer
    --use-mcore-models
    --transformer-impl transformer_engine
    --no-scatter-gather-tensors-in-pipeline
    --num-layers 24
    --optimizer adam
    --loss-scale 8192
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    #--save $CHECKPOINT_PATH
    --frcheck-skip-load-teardown-barrier
    #--frcheck-defer-load-teardown
    #--load $CHECKPOINT_PATH
    
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH

    --use-frcheck
    --frcheck-distribute-common
    #--frcheck-debug
    --frcheck-n 8
    #--use-frcheck-software-failure 
    #--frcheck-failed-ranks 0
    --frcheck-table-dir $FRCHECK_TABLE_DIR
    #--use-frcheck-hardware-failure
    --ckpt-format torch
    --save-embeddings-separately
    #--timing-log-level 1
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${RECOVERY_MODE_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (FRCheck)"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"
echo "FRCHECK_TABLE_DIR: $FRCHECK_TABLE_DIR"
echo "FRCHECK_BASE_IP: $FRCHECK_BASE_IP"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${RECOVERY_MODE_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}
