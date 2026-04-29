#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL
export ECCHECK_USE_ASIO=true
GPUS_PER_NODE=4
# Change for multinode config
# Preserve original args so callers can pass: ./test_eccheck.sh <node_rank> a800
ORIG_ARGS=("$@")
ORIG_SECOND="${ORIG_ARGS[1]:-}"
if [ "$ORIG_SECOND" == "a800" ]; then
    MASTER_ADDR=127.0.0.1
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
else
    # 修复了上一个 Connection timed out 的潜在问题，确保使用本地回环和 eth0
    MASTER_ADDR=127.0.0.1
    export NCCL_SOCKET_IFNAME=eth0
    export GLOO_SOCKET_IFNAME=eth0
fi
MASTER_PORT=6000
NNODES=1
# If first argument is a numeric node rank use it, otherwise default to 0
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs" #<Specify path>
CHECKPOINT_PATH="/workspace/data/checkpoint/models/gpt2-345m-0" #<Specify path>
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document" #<Specify path and file prefix>_text_document

SHM_PKT="/dev/shm/shm_pkt"

if [ "$NODE_RANK" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Remaining args after optional node-rank are passed to the training script
ARGS_TO_PASS=("$@")
TEST_NUM=${ORIG_ARGS[1]:-0}

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
    # --replication
    # --replication-jump 1
    # --replication-factor 1
    # --non-persistent-ckpt-type local
    # --non-persistent-local-ckpt-dir $SHM_PKT
)
#use -- save to save gpu replia, and load to reuse gpu replia
EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 1
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    # --load $SHM_PKT 
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --use-eccheck
    --ckpt-format torch_dist
)


mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# 【已修改】将 PRINT_CMD 检查移动到实际执行命令的上方
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run: PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

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