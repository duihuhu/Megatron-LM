#!/bin/bash

# Script to run a single node in 4-node simulation (default 1 GPU per node)
# Usage: ./test_eccheck_4nodes_node_335M_checkcode.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [mode] [additional_args...]
#
# mode (optional, default: save):
#   save      - checkpoint save only (no load / recovery flags)
#   software  - load checkpoint + software failure recovery
#   hardware  - load checkpoint + hardware failure recovery
#
# Example: ./test_eccheck_4nodes_node_335M_checkcode.sh 0 0
# Example (2 GPUs per container): ./test_eccheck_4nodes_node_335M_checkcode.sh 0 2 3 software
# Example (8 GPUs, hardware recovery): ./test_eccheck_4nodes_node_335M_checkcode.sh 0 0 1 2 3 4 5 6 7 hardware

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=eth0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
MASTER_ADDR=172.16.0.224

export ECCHECK_USE_ASIO=true
MASTER_PORT=6000
NNODES=16

export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE
export ECLATIN_INTERFACE=$NETIFACES_INTERFACE
export ECLATIN_LOCAL_RANK_NIC_0=eth0
export ECLATIN_LOCAL_RANK_NIC_1=eth0
export ECLATIN_LOCAL_RANK_NIC_2=eth0
export ECLATIN_LOCAL_RANK_NIC_3=eth0
export ECLATIN_LOCAL_RANK_NIC_4=eth1
export ECLATIN_LOCAL_RANK_NIC_5=eth1
export ECLATIN_LOCAL_RANK_NIC_6=eth1
export ECLATIN_LOCAL_RANK_NIC_7=eth1
# If first argument is a numeric node rank use it, otherwise default to 0
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# Next arguments are GPU IDs, collect them until we hit a non-numeric (additional args start with non-numeric or --)
GPU_IDS=()
while [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "Error: At least one GPU id must be specified."
    echo "Usage: $0 <node_rank> <gpu_id_0> [gpu_id_1 ...] [save|software|hardware|hardware2] [additional_args...]"
    exit 1
fi

# ---- mode parsing (save | software | hardware, after GPU IDs) ----
MODE=save
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2)$ ]]; then
    MODE="$1"
    shift
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

# Set CUDA_VISIBLE_DEVICES by explicitly listing all provided GPU IDs (as comma-separated values)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")

# Set NCCL_DEBUG_FILE after NODE_RANK is determined
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/Megatron-LM/pre-tests/opt/7B/opt-7b-0/logs"
CHECKPOINT_PATH="/dev/shm/models/opt-7b-0-checkcode"
# DATA_PATH="/workspace/Megatron-LM/pre-tests/opt/opt_data/wiki_text_sentence"

SHM_PKT="/dev/shm/shm_pkt"

# Remaining args after node-rank and GPU ids are passed to the training script
ARGS_TO_PASS=("$@")

# Recovery mode args: enabled only for software / hardware load tests
RECOVERY_MODE_ARGS=()
case "$MODE" in
    save)
        ;;
    software)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-checkcode-software-failure
        )
        ;;
    hardware)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
        )
        ;;
    hardware2)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-checkcode-two-failures
        )
        ;;
esac

# Model related configuration here, please do not overlap with json config
HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=32
NUM_LAYERS=32

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
    --lr 0.00005
    --train-iters 4
    --lr-decay-iters 320000
    --lr-decay-style cosine
    --min-lr 1.0e-5
    --weight-decay 1e-2
    --lr-warmup-fraction .05
    --clip-grad 1.0
    --fp16
    --tokenizer-type GPT2BPETokenizer
    --use-mcore-models
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
    --pipeline-model-parallel-size 16
    --sequence-parallel
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    --save $CHECKPOINT_PATH 
    #--load $CHECKPOINT_PATH
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    # --use-eccheck

    # --use-gemini
    # --use-gemini-optimized
    # --use-gemini-software-failure
    # --use-gemini-hardware-failure
    # --use-distributed-optimizer
    # --use-ecnaive-software-failure
    --use-checkcode
    --ckpt-format torch
    # --no-save-optim
    # --no-load-optim
    --save-embeddings-separately
    --use-rdma
    # --timing-log-level 2
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK, mode=$MODE): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} ${RECOVERY_MODE_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (mode=$MODE)"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${RECOVERY_MODE_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}
