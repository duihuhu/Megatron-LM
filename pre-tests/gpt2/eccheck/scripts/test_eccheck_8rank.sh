#!/bin/bash

# ============================================================================
# EC-CHECK 8 Rank Test Script
# ============================================================================
# This script tests EC-CHECK with 8 ranks with 4+4 erasure coding
#   - 8 ranks total (Group 1: ranks 0,1,2,3; Group 2: ranks 4,5,6,7)
#   - EC parameters: k=4, m=4
#   - Each rank has 4 columns (encode threads)
#   - Post-xor pairing: rank0↔rank4, rank1↔rank5, rank2↔rank6, rank3↔rank7
#   - Configuration file: eccheck/configs/eccheck_8rank.json
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECCHECK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="$(cd "$ECCHECK_DIR/.." && pwd)"

# EC-CHECK configuration file path
# Force use the correct path (ignore any pre-existing ECCHECK_CONFIG_PATH)
ECCHECK_CONFIG_PATH="$ECCHECK_DIR/configs/eccheck_8rank.json"
export ECCHECK_CONFIG_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

# ============================================================================
# Rank/Node Configuration
# ============================================================================
# Option 1: Single node with 8 GPUs
GPUS_PER_NODE=8
NNODES=1
NODE_RANK=0

# Option 2: Two nodes with 4 GPUs each (uncomment to use)
# GPUS_PER_NODE=4
# NNODES=2
# NODE_RANK=${1:-0}  # Get from command line argument

# Option 3: Four nodes with 2 GPUs each (uncomment to use)
# GPUS_PER_NODE=2
# NNODES=4
# NODE_RANK=${1:-0}  # Get from command line argument

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "============================================================================"
echo "EC-CHECK 8 Rank Test Configuration"
echo "============================================================================"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "EC-CHECK Groups:"
echo "  Group 1: ranks 0,1,2,3"
echo "  Group 2: ranks 4,5,6,7"
echo "EC-CHECK Post-XOR Pairing:"
echo "  rank0↔rank4, rank1↔rank5, rank2↔rank6, rank3↔rank7"
echo "EC Parameters: k=4, m=4 (4+4 erasure coding)"
echo "Each rank: 4 columns (encode threads)"
echo "Config: $ECCHECK_CONFIG_PATH"
echo "============================================================================"

MASTER_ADDR=127.0.0.1
MASTER_PORT=6000
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0

if [ "$NODE_RANK" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
fi

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs"
CHECKPOINT_PATH="/workspace/data/checkpoint/models/gpt2-345m-0"

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
    --train-iters 50
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
    --num-layers 12 
    --optimizer adam
    --loss-scale 8192
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 2
    --pipeline-model-parallel-size 1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 1
    --save $CHECKPOINT_PATH 
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --use-eccheck
    --ckpt-format torch_dist
)

mkdir -p logs
mkdir -p logs/csv

if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run: PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl"
    exit 0
fi

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

cd "$BASE_DIR"

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl

