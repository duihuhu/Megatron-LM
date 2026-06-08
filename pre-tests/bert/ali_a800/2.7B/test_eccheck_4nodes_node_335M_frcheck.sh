#!/bin/bash

# FRCheck (POA-driven stripe encode with RDMA) — single-node script.
# Usage: ./test_eccheck_4nodes_node_335M_frcheck.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [additional_args...]
# Example: ./test_eccheck_4nodes_node_335M_frcheck.sh 0 0
# Example (2 GPUs per container): ./test_eccheck_4nodes_node_335M_frcheck.sh 0 2 3

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=eth0
export FRCHECK_LOCAL_RANK_NIC_0=eth0
export FRCHECK_LOCAL_RANK_NIC_1=eth0
export FRCHECK_LOCAL_RANK_NIC_2=eth0
export FRCHECK_LOCAL_RANK_NIC_3=eth0
export FRCHECK_LOCAL_RANK_NIC_4=eth1
export FRCHECK_LOCAL_RANK_NIC_5=eth1
export FRCHECK_LOCAL_RANK_NIC_6=eth1
export FRCHECK_LOCAL_RANK_NIC_7=eth1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
MASTER_ADDR=172.16.0.224

export ECCHECK_USE_ASIO=false
export FRCHECK_INTERFACE=$NETIFACES_INTERFACE
export FRCHECK_BASE_IP=$MASTER_ADDR
MASTER_PORT=6000
NNODES=4

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
    echo "Usage: ./test_eccheck_4nodes_node_335M_frcheck.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [additional_args...]"
    exit 1
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/bert/bert_data/vocab.txt"

TENSORBOARD_LOGS_PATH="/workspace/Megatron-LM/pre-tests/bert/7B/bert-7b-0/logs"
CHECKPOINT_PATH="/dev/shm/models/bert-7b-0-frcheck"
DATA_PATH="/workspace/Megatron-LM/pre-tests/bert/bert_data/wiki_text_sentence"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/workspace/Megatron-LM/pre-tests/bert/bert_data/cache}"

SHM_PKT="/dev/shm/shm_pkt"

ARGS_TO_PASS=("$@")

# Model configuration
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
    # --merge-file $MERGE_FILE
    --data-path $DATA_PATH
    --data-cache-path $DATA_CACHE_PATH
    --num-dataset-builder-threads 32
    --split 949,50,1
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
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 4
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

    --use-frcheck
    --frcheck-n 4
    --frcheck-table-dir $FRCHECK_TABLE_DIR
    --frcheck-failed-ranks 0,1
    --use-frcheck-hardware-failure
    --use-rdma
    --ckpt-format torch
    --save-embeddings-separately
    # --timing-log-level 2
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
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
    pretrain_bert.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}
