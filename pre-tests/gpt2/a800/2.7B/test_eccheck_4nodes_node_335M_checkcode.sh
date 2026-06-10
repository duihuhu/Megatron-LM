#!/bin/bash

# Script to run a single node in 4-node simulation (default 1 GPU per node)
# Usage: ./test_eccheck_4nodes_node.sh <node_rank> [gpus_per_node] [additional_args...]
# Example: ./test_eccheck_4nodes_node.sh 0
# Example (2 GPUs per container): ./test_eccheck_4nodes_node.sh 0 2

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=eth0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
GPUS_PER_NODE=1
MASTER_ADDR=172.16.0.1

export ECCHECK_USE_ASIO=true
MASTER_PORT=6000
NNODES=4

export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE
export ECLATIN_INTERFACE=$NETIFACES_INTERFACE
# If first argument is a numeric node rank use it, otherwise default to 0
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# Optional second argument: GPUs per node (default 1). E.g. 2 => container 0 uses 0,1; container 1 uses 2,3; ...
if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
    GPUS_PER_NODE=$1
    shift
fi

# Set NCCL_DEBUG_FILE after NODE_RANK is determined
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Set CUDA_VISIBLE_DEVICES: by node rank when 1 GPU/node; when GPUs per node > 1, use contiguous block (e.g. node 0 => 0,1; node 1 => 2,3)
START_GPU=$(($NODE_RANK * $GPUS_PER_NODE))
export CUDA_VISIBLE_DEVICES=$(seq -s, $START_GPU $((START_GPU + GPUS_PER_NODE - 1)))
VOCAB_FILE="/root/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/root/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs" #<Specify path>
CHECKPOINT_PATH="/dev/shm/models/gpt2-345m-0-checkcode" #<Specify path>
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document" #<Specify path and file prefix>_text_document

# Remaining args after optional node-rank are passed to the training script
ARGS_TO_PASS=("$@")

# fixed Model related configuration here, pls not overlap with json config
HIDDEN_SIZE=2560
NUM_ATTENTION_HEADS=32
NUM_LAYERS=32 

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=2
GLOBAL_BATCH_SIZE=4

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
    --lr 0.00005 
    --train-iters 20
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
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 4
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    # --use-eccheck

    # --use-gemini
    # --use-gemini-optimized
    # --use-gemini-software-failure
    # --use-gemini-hardware-failure
    # --use-distributed-optimizer
    --use-checkcode
    # --use-checkcode-software-failure
    --ckpt-format torch_dist
    --save-embeddings-separately
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES"
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

