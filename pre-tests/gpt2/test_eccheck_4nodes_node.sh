#!/bin/bash

# Script to run a single node in 4-node simulation (1 GPU per node), allowing specification of both node rank and GPU device.
# Usage: ./test_eccheck_4nodes_node.sh <node_rank> <gpu_id> [additional_args...]
# Example: ./test_eccheck_4nodes_node.sh 0 1

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1

GPUS_PER_NODE=1
MASTER_ADDR=127.0.0.1
export NCCL_SOCKET_IFNAME=eth0
export GLOO_SOCKET_IFNAME=eth0
export ECCHECK_USE_ASIO=true
MASTER_PORT=6000
NNODES=4

# Parse node rank and gpu id
NODE_RANK=0
GPU_ID=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK="$1"
        shift
    fi
fi

if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        GPU_ID="$1"
        shift
    fi
fi

# Set NCCL_DEBUG_FILE after NODE_RANK is determined
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Set CUDA_VISIBLE_DEVICES for selected GPU
export CUDA_VISIBLE_DEVICES=$GPU_ID

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs" #<Specify path>
CHECKPOINT_PATH="/workspace/data/checkpoint/models/gpt2-345m-0" #<Specify path>
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document" #<Specify path and file prefix>_text_document

SHM_PKT="/dev/shm/shm_pkt"

# Remaining args after node-rank and gpu-id are passed to the training script
ARGS_TO_PASS=("$@")

# fixed Model related configuration here, pls not overlap with json config
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

# Model related configuration here, pls not overlap with json config
GPT_ARGS=(
    --no-async-tensor-model-parallel-allreduce 
    --hidden-size $HIDDEN_SIZE 
    --num-attention-heads $NUM_ATTENTION_HEADS 
    --seq-length $SEQ_LENGTH 
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE 
    --lr 0.00015 
    --train-iters 20
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
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    #--use-eccheck
    --use-checkcode

    --use-gemini
    # --use-gemini-software-failure
    # --use-gemini-hardware-failure
    --ckpt-format torch_dist
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK, GPU $GPU_ID): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPU $GPU_ID"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}

