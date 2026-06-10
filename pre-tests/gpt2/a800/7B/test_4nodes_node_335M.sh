#!/bin/bash

# Script to run a single node in 4-node simulation (default 1 GPU per node)
# Usage: ./test_4nodes_node_335M.sh <node_rank> <global_gpu_rank> [additional_args...]
# Example: ./test_4nodes_node_335M.sh 0 0
# Example: ./test_4nodes_node_335M.sh 1 3

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=bond0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1

MASTER_ADDR=10.0.0.62
export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE
export ECCHECK_USE_ASIO=true
MASTER_PORT=6000
NNODES=1
GPUS_PER_NODE=1

export ECCHECK_INTERFACE=$NETIFACES_INTERFACE

# Parse node_rank and global_gpu_rank
NODE_RANK=0
GLOBAL_GPU_RANK=0

if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        GLOBAL_GPU_RANK=$1
        shift
    fi
fi

export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

# Set CUDA_VISIBLE_DEVICES using global_gpu_rank
export CUDA_VISIBLE_DEVICES=$GLOBAL_GPU_RANK

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs"
CHECKPOINT_PATH="/workspace/Megatron-LM/data/checkpoint/models/gpt2-345m-0"
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document"

SHM_PKT="/dev/shm/shm_pkt"

# Remaining arguments after node_rank and global_gpu_rank are passed to the training script
ARGS_TO_PASS=("$@")

# Fixed model configuration
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
    --train-iters 10
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
    --pipeline-model-parallel-size 1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    --save $CHECKPOINT_PATH 
    #--load $CHECKPOINT_PATH
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    #--use-eccheck
    #--use-eccheck-software-failure
    # --use-gemini
    # --use-gemini-optimized
    # --use-gemini-software-failure
    # --use-gemini-hardware-failure

    # --use-checkcode
    --ckpt-format torch
    # --no-save-optim
    # --no-load-optim
    --save-embeddings-separately
    --timing-log-level 2
    --layer-wise-optimizer-update
    --no-barrier-with-level-1-timing
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK, Global GPU Rank $GLOBAL_GPU_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with visible global GPU rank $CUDA_VISIBLE_DEVICES"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1
export NVTE_SYNC_P2P=1

export ECCHECK_USE_ASIO=true
PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}

