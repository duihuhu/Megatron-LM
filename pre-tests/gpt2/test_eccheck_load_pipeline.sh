#!/bin/bash

# Test script for EC-CHECK load pipeline
# Similar to test_eccheck.sh but specifically for testing load pipeline

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl_load_test.log
export NCCL_DEBUG_SUBSYS=ALL
export ECCHECK_USE_ASIO=true

GPUS_PER_NODE=4
# Change for multinode config
# Preserve original args so callers can pass: ./test_eccheck_load_pipeline.sh <node_rank> a800
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

if [ "$NODE_RANK" -eq 0 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
fi

# Remaining args after optional node-rank are passed to the test script
ARGS_TO_PASS=("$@")

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK 
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# 【已修改】将 PRINT_CMD 检查移动到实际执行命令的上方
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run: PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM/pre-tests/gpt2 torchrun ${DISTRIBUTED_ARGS[@]} test_eccheck_load_pipeline.py ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM/pre-tests/gpt2 torchrun ${DISTRIBUTED_ARGS[@]} \
    test_eccheck_load_pipeline.py \
    ${ARGS_TO_PASS[@]}

