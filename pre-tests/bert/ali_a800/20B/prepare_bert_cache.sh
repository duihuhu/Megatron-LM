#!/bin/bash

# Script to pre-generate dataset cache for multi-node training without shared storage
# Run this script on EACH node before starting distributed training
# Usage: ./prepare_bert_cache.sh

echo "=========================================="
echo "Preparing BERT dataset cache..."
echo "=========================================="

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/bert/bert_data/vocab.txt"
DATA_PATH="/workspace/Megatron-LM/pre-tests/bert/bert_data/wiki_text_sentence"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/workspace/Megatron-LM/pre-tests/bert/bert_data/cache}"

HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=40 
NUM_LAYERS=64

SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=16


echo "Cache will be generated at: $DATA_CACHE_PATH"
echo "Data path: $DATA_PATH"
echo ""

# Create cache directory
mkdir -p $DATA_CACHE_PATH

# Run a minimal training step to generate cache
# We use 1 iteration with 1 GPU to build the cache
PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM python3 /workspace/Megatron-LM/pretrain_bert.py \
    --vocab-file $VOCAB_FILE \
    --data-path $DATA_PATH \
    --data-cache-path $DATA_CACHE_PATH \
    --num-dataset-builder-threads 32 \
    --split 949,50,1 \
    --hidden-size $HIDDEN_SIZE \
    --num-attention-heads $NUM_ATTENTION_HEADS \
    --seq-length $SEQ_LENGTH \
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS \
    --micro-batch-size $MICRO_BATCH_SIZE \
    --global-batch-size $GLOBAL_BATCH_SIZE \
    --lr 0.00005 \
    --train-iters 1 \
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
    --num-layers $NUM_LAYERS \
    --optimizer adam \
    --loss-scale-window 100 \
    --initial-loss-scale 4096 \
    --min-loss-scale 1.0 \
    --hysteresis 2 \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --log-interval 1 \
    --no-save-optim \
    --exit-on-missing-checkpoint \
    2>&1 | grep -E "(Build and save|Load the|total number)"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ] || [ -d "$DATA_CACHE_PATH/BERTMaskedWordPieceDataset_indices" ]; then
    echo "✓ Cache generation completed successfully!"
    echo "Cache location: $DATA_CACHE_PATH"
    ls -lh $DATA_CACHE_PATH/BERTMaskedWordPieceDataset_indices/ 2>/dev/null || echo "Cache files created"
else
    echo "✗ Cache generation may have encountered issues"
    echo "Please check the output above"
fi
echo "=========================================="

