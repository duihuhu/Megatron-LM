#!/bin/bash

# ============================================================================
# EC-CHECK 4 Rank Multi-Container Test Script
# ============================================================================
# This script tests EC-CHECK with 4 containers (1 GPU per container)
#   - 4 containers total, each with 1 GPU
#   - EC-CHECK pairing: rank0↔rank2, rank1↔rank3
#   - Configuration file: eccheck/configs/eccheck_4rank.json
#   - True distributed environment with network communication
# ============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ECCHECK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BASE_DIR="$(cd "$ECCHECK_DIR/.." && pwd)"

# EC-CHECK configuration file path
ECCHECK_CONFIG_PATH="$ECCHECK_DIR/configs/eccheck_4rank.json"
export ECCHECK_CONFIG_PATH

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

# ============================================================================
# Multi-Container Shared Storage Configuration
# ============================================================================
# IMPORTANT: For multi-container setups, NCCL ID files must be stored in a shared location
# Set ECCHECK_NCCL_SHARED_PATH to a directory accessible by all containers
# Examples:
#   - Shared volume mount: /workspace/shared
#   - Network file system: /mnt/nfs/shared
#   - Host volume mount: /host/shared
# If not set, defaults to /tmp (which may not be shared across containers)
export ECCHECK_NCCL_SHARED_PATH=${ECCHECK_NCCL_SHARED_PATH:-/workspace/Megatron-LM/tmp}

# Create directory and verify it's writable
if ! mkdir -p "$ECCHECK_NCCL_SHARED_PATH" 2>/dev/null; then
    echo "WARNING: Failed to create shared directory: $ECCHECK_NCCL_SHARED_PATH"
    echo "Trying alternative: /tmp/eccheck_shared"
    export ECCHECK_NCCL_SHARED_PATH=/tmp/eccheck_shared
    mkdir -p "$ECCHECK_NCCL_SHARED_PATH"
fi

# Verify write access
if ! touch "$ECCHECK_NCCL_SHARED_PATH/.test_write" 2>/dev/null; then
    echo "ERROR: Cannot write to shared directory: $ECCHECK_NCCL_SHARED_PATH"
    echo "Please ensure this directory is accessible and writable by all containers"
    exit 1
fi
rm -f "$ECCHECK_NCCL_SHARED_PATH/.test_write"

echo "EC-CHECK: Using shared storage path: $ECCHECK_NCCL_SHARED_PATH"
echo "EC-CHECK: Shared path is writable: ✓"

# ============================================================================
# Multi-Container Configuration
# ============================================================================
# Each container runs with 1 GPU
GPUS_PER_NODE=1
NNODES=4  # 4 containers = 4 nodes
NODE_RANK=${1:-0}  # Get from command line argument (0-3)

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo "============================================================================"
echo "EC-CHECK 4 Rank Multi-Container Test Configuration"
echo "============================================================================"
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "NNODES: $NNODES"
echo "NODE_RANK: $NODE_RANK"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "EC-CHECK Pairing: rank0↔rank2, rank1↔rank3"
echo "Config: $ECCHECK_CONFIG_PATH"
echo "============================================================================"

# Network configuration for multi-container
# IMPORTANT: Set these based on your container network setup
MASTER_ADDR=${MASTER_ADDR:-"172.20.0.5"}  # Host IP or container network gateway
MASTER_PORT=${MASTER_PORT:-6000}

# NCCL network configuration
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}  # Network interface
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}

# Set visible GPU for this container based on NODE_RANK
# When container can see all GPUs (same as host), we need to specify which GPU to use
if [ "$NODE_RANK" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
elif [ "$NODE_RANK" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=1
elif [ "$NODE_RANK" -eq 2 ]; then
    export CUDA_VISIBLE_DEVICES=2
elif [ "$NODE_RANK" -eq 3 ]; then
    export CUDA_VISIBLE_DEVICES=3
else
    # For more than 4 nodes, use NODE_RANK directly
    export CUDA_VISIBLE_DEVICES=$NODE_RANK
fi

echo "NODE_RANK=$NODE_RANK: Using GPU(s): $CUDA_VISIBLE_DEVICES"

# For multi-container, you may need to set:
# export NCCL_IB_DISABLE=0  # Enable InfiniBand if available
# export NCCL_IB_HCA=mlx5  # InfiniBand HCA name
# export NCCL_SOCKET_IFNAME=ib0  # InfiniBand interface

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
    --pipeline-model-parallel-size 2
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

