#!/bin/bash

# Script to pre-generate dataset cache for multi-node training without shared storage
# Run this script on EACH node before starting distributed training
#
# Usage:
#   ./prepare_bert_cache.sh
#   CLEAR_CACHE=1 ./prepare_bert_cache.sh   # wipe stale cache before building
#
# Data-related args (train-iters, eval-interval, global-batch-size, etc.) must match
# test_eccheck_4nodes_node_335M_eccheck.sh so cache hashes are identical.

set -euo pipefail

echo "=========================================="
echo "Preparing BERT dataset cache..."
echo "=========================================="

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/bert/bert_data/vocab.txt"
DATA_PATH="/workspace/Megatron-LM/pre-tests/bert/bert_data/wiki_text_sentence"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/workspace/Megatron-LM/pre-tests/bert/bert_data/cache}"

HIDDEN_SIZE=4096
NUM_ATTENTION_HEADS=32
NUM_LAYERS=32

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16

# Must match test_eccheck_4nodes_node_335M_eccheck.sh for cache hash consistency
TRAIN_ITERS=1
EVAL_INTERVAL=100
EVAL_ITERS=1

REQUIRED_SAMPLE_INDEX_FILES=3

echo "Cache will be generated at: $DATA_CACHE_PATH"
echo "Data path: $DATA_PATH"
echo ""

if [ ! -f "$VOCAB_FILE" ]; then
    echo "Error: vocab file not found: $VOCAB_FILE"
    exit 1
fi

if [ ! -f "${DATA_PATH}.idx" ] || [ ! -f "${DATA_PATH}.bin" ]; then
    echo "Error: indexed dataset not found: ${DATA_PATH}.idx / ${DATA_PATH}.bin"
    exit 1
fi

if [ "${CLEAR_CACHE:-0}" != "0" ]; then
    echo "Clearing existing cache at: $DATA_CACHE_PATH"
    rm -rf "${DATA_CACHE_PATH:?}"/*
fi

mkdir -p "$DATA_CACHE_PATH"

LOG_FILE="${DATA_CACHE_PATH}/prepare_bert_cache.log"
echo "Running pretrain_bert.py (single GPU) to build cache..."
echo "Full log: $LOG_FILE"
echo ""

set +e
PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM python3 /workspace/Megatron-LM/pretrain_bert.py \
    --vocab-file "$VOCAB_FILE" \
    --data-path "$DATA_PATH" \
    --data-cache-path "$DATA_CACHE_PATH" \
    --num-dataset-builder-threads 32 \
    --split 949,50,1 \
    --hidden-size "$HIDDEN_SIZE" \
    --num-attention-heads "$NUM_ATTENTION_HEADS" \
    --seq-length "$SEQ_LENGTH" \
    --max-position-embeddings "$MAX_POSITION_EMBEDDINGS" \
    --micro-batch-size "$MICRO_BATCH_SIZE" \
    --global-batch-size "$GLOBAL_BATCH_SIZE" \
    --train-iters "$TRAIN_ITERS" \
    --eval-interval "$EVAL_INTERVAL" \
    --eval-iters "$EVAL_ITERS" \
    --lr 0.00005 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .05 \
    --clip-grad 1.0 \
    --fp16 \
    --tokenizer-type BertWordPieceCase \
    --use-mcore-models \
    --transformer-impl transformer_engine \
    --no-scatter-gather-tensors-in-pipeline \
    --num-layers "$NUM_LAYERS" \
    --optimizer adam \
    --loss-scale-window 100 \
    --initial-loss-scale 4096 \
    --min-loss-scale 1.0 \
    --hysteresis 2 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --distributed-backend nccl \
    --log-interval 1 \
    --no-save-optim \
    2>&1 | tee "$LOG_FILE"
PYTHON_EXIT_CODE=${PIPESTATUS[0]}
set -e

SAMPLE_INDEX_COUNT=$(find "$DATA_CACHE_PATH" -maxdepth 1 \
    -name '*-BERTMaskedWordPieceDataset-sample_index.npy' | wc -l)
SAMPLE_INDEX_COUNT=${SAMPLE_INDEX_COUNT// /}

echo ""
echo "=========================================="
if [ "$SAMPLE_INDEX_COUNT" -ge "$REQUIRED_SAMPLE_INDEX_FILES" ]; then
    echo "Cache generation completed successfully!"
    echo "Cache location: $DATA_CACHE_PATH"
    echo "sample_index.npy files: $SAMPLE_INDEX_COUNT (expected >= $REQUIRED_SAMPLE_INDEX_FILES)"
    ls -lh "$DATA_CACHE_PATH"/*-BERTMaskedWordPieceDataset-sample_index.npy
    if [ "$PYTHON_EXIT_CODE" -ne 0 ]; then
        echo ""
        echo "Note: pretrain_bert.py exited with code $PYTHON_EXIT_CODE after cache was built."
        echo "This is usually fine for cache preparation."
    fi
    CACHE_OK=0
else
    echo "Cache generation failed!"
    echo "sample_index.npy files: $SAMPLE_INDEX_COUNT (expected >= $REQUIRED_SAMPLE_INDEX_FILES)"
    echo "pretrain_bert.py exit code: $PYTHON_EXIT_CODE"
    echo "Full log: $LOG_FILE"
    echo ""
    echo "Last 30 lines of log:"
    tail -n 30 "$LOG_FILE"
    CACHE_OK=1
fi
echo "=========================================="

exit "$CACHE_OK"
