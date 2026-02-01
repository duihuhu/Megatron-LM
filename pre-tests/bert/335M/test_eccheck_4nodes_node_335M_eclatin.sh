#!/bin/bash

# Script to run a single node in 4-node simulation (1 GPU per node)
# Usage: ./test_eccheck_4nodes_node.sh <node_rank> [additional_args...]
# Example: ./test_eccheck_4nodes_node.sh 0

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=eth0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1

GPUS_PER_NODE=1
MASTER_ADDR=172.16.0.1
export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE
export ECCHECK_USE_ASIO=true
MASTER_PORT=6000
NNODES=4
NUM_LAYERS=24

export GEMINI_INTERFACE=$NETIFACES_INTERFACE

# If first argument is a numeric node rank use it, otherwise default to 0
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# Set NCCL_DEBUG_FILE after NODE_RANK is determined
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

# Set CUDA_VISIBLE_DEVICES for each node
# export CUDA_VISIBLE_DEVICES=$NODE_RANK

export CUDA_VISIBLE_DEVICES=0
VOCAB_FILE="/root/Megatron-LM/pre-tests/bert/bert_data/vocab.txt"

TENSORBOARD_LOGS_PATH="/root/Megatron-LM/pre-tests/bert/335M/bert-335m-0/logs" #<Specify path>
CHECKPOINT_PATH="/dev/shm/models/bert-335m-0-eclatin" #<Specify path>

DATA_PATH="/root/Megatron-LM/pre-tests/bert/bert_data/wiki_text_sentence" #<Specify path and file prefix>_text_document

# Data cache path: For non-shared storage, use local path
# IMPORTANT: For multi-node without shared storage, pre-generate cache on each node
# Run: ./prepare_bert_cache.sh before starting training
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/root/Megatron-LM/pre-tests/bert/bert_data/cache}"


# Remaining args after optional node-rank are passed to the training script
ARGS_TO_PASS=("$@")

# fixed Model related configuration here, pls not overlap with json config
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=16
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
    # --merge-file $MERGE_FILE 
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --num-dataset-builder-threads 32
    --split 949,50,1
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
    --tokenizer-type BertWordPieceCase 
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
    --use-eclatin
    # --use-eclatin-software-failure
    --ckpt-format torch_dist
    --save-embeddings-separately
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPU $NODE_RANK"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_bert.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}

