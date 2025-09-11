#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

GPUS_PER_NODE=1
# Change for multinode config
if [ "$2" == "a800" ]; then
    MASTER_ADDR=10.0.0.62
    export NCCL_SOCKET_IFNAME=bond0
    export GLOO_SOCKET_IFNAME=bond0
else 
    MASTER_ADDR=10.156.154.36
    export NCCL_SOCKET_IFNAME=ens37f0
    export GLOO_SOCKET_IFNAME=ens37f0
fi
MASTER_PORT=6000
NNODES=1
NODE_RANK=$1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))


VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs" #<Specify path>
CHECKPOINT_PATH="/dev/shm/models/gpt2-345m-0" #<Specify path>
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document" #<Specify path and file prefix>_text_document

SHM_PKT="/dev/shm/shm_pkt"

if [ "$NODE_RANK" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0
fi

# export NCCL_SOCKET_IFNAME=ens37f0
# export GLOO_SOCKET_IFNAME=ens37f0


TEST_NUM=${2:-0}

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
    --train-iters 3
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
    # --non-persistent-save-interval 7
    --enable-pipeline-checkpoint
    --num-tensor-groups 4
    # --ckpt-format torch
)


mkdir -p logs
mkdir -p logs/csv

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    --distributed-backend nccl \

