#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

CHECKPOINT_PATH="/workspace/models/gpt2-345m" #<Specify path>
TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m/logs" #<Specify path>
VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json" #<Specify path to file>/gpt2-vocab.json
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt" #<Specify path to file>/gpt2-merges.txt
DATA_PATH="/workspace/models/gpt2-345m/codeparrot_content_document" #<Specify path and file prefix>_text_document

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NUM_NODES 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12 
    --hidden-size 512 
    --num-attention-heads 8 
    --seq-length 1024 
    --max-position-embeddings 1024 
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 1 
    --global-batch-size 1536 
    # --rampup-batch-size 16 16 5859375 
    --train-iters 5 
    --weight-decay 0.1 
    --adam-beta1 0.9 
    --adam-beta2 0.95 
    --init-method-std 0.006 
    --clip-grad 1.0 
    --fp16
    --lr 6.0e-5 
    --lr-decay-style cosine 
    --min-lr 6.0e-6
    --lr-warmup-fraction .001 
    --lr-decay-iters 4
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1 
	--pipeline-model-parallel-size 2
)

DATA_ARGS=(
    --vocab-file $VOCAB_FILE 
    --merge-file $MERGE_FILE 
    # --data-path $DATA_PATH 
    # --split 949,50,1
    --mock-data
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1 
    --eval-interval 10 
    --save $CHECKPOINT_PATH 
    # --load $CHECKPOINT_PATH 
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
