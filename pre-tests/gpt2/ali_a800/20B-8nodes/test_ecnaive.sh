#!/bin/bash

# Script to run a single node in 4-node simulation (default 1 GPU per node)
# Usage: ./test_eccheck_4nodes_node_335M_ecnaive.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [mode] [additional_args...]
#
# mode (optional, default: save):
#   save      - checkpoint save only (no load / recovery flags)
#   software  - load checkpoint + software failure recovery
#   hardware  - load checkpoint + hardware failure recovery (failed ranks only)
#
# Example: ./test_eccheck_4nodes_node_335M_ecnaive.sh 0 0
# Example (2 GPUs per container): ./test_eccheck_4nodes_node_335M_ecnaive.sh 0 2 3 software
# Example (8 GPUs, hardware recovery): ./test_eccheck_4nodes_node_335M_ecnaive.sh 0 0 1 2 3 4 5 6 7 hardware

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=eth0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
MASTER_ADDR=172.16.0.224

export ECCHECK_USE_ASIO=true
MASTER_PORT=6000
NNODES=8

export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE
export ECNAIVE_INTERFACE=$NETIFACES_INTERFACE
export ECLATIN_INTERFACE=$NETIFACES_INTERFACE
export MEGATRON_ECNAIVE_LOAD_NET_TRACE=1
export ECNAIVE_LOCAL_RANK_NIC_0=eth0
export ECNAIVE_LOCAL_RANK_NIC_1=eth0
export ECNAIVE_LOCAL_RANK_NIC_2=eth0
export ECNAIVE_LOCAL_RANK_NIC_3=eth0
export ECNAIVE_LOCAL_RANK_NIC_4=eth1
export ECNAIVE_LOCAL_RANK_NIC_5=eth1
export ECNAIVE_LOCAL_RANK_NIC_6=eth1
export ECNAIVE_LOCAL_RANK_NIC_7=eth1
#  priority from
#  ┌──────────────────────────────────────────┬──────────────────────────┐
#  │                 环境变量                 │           用途           │
#  ├──────────────────────────────────────────┼──────────────────────────┤
#  │ ECNAIVE_RANK_IP_0=10.0.0.1               │ 每个 rank 显式指定 IP    │
#  ├──────────────────────────────────────────┼──────────────────────────┤
#  │ ECNAIVE_LOCAL_RANK_NIC_0=mlx5_0          │ 每个 local_rank 绑定 NIC │
#  ├──────────────────────────────────────────┼──────────────────────────┤
#  │ ECNAIVE_NIC_LIST + ECNAIVE_RANKS_PER_NIC │ 批量 NIC 分配            │
#  ├──────────────────────────────────────────┼──────────────────────────┤
#  │ ECNAIVE_BASE_IP=10.0.0.1                 │ 所有 rank 同一 IP        │
#  ├──────────────────────────────────────────┼──────────────────────────┤
#  │ ECNAIVE_INTERFACE=bond0                  │ 从指定接口自动检测 IP    │
#  ├──────────────────────────────────────────┼──────────────────────────┤
#  │ MASTER_ADDR                              │ 最终 fallback            │
#  └──────────────────────────────────────────┴──────────────────────────┘
# If first argument is a numeric node rank use it, otherwise default to 0
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# Next arguments are GPU IDs, collect them until we hit a non-numeric (additional args start with non-numeric or --)
GPU_IDS=()
while [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

# If no GPU IDs are provided, use all GPUs available on the node by default
if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    if command -v nvidia-smi > /dev/null 2>&1; then
        # Try to get all GPU indices using nvidia-smi, fallback to 0 if nvidia-smi fails
        mapfile -t GPU_IDS < <(nvidia-smi --query-gpu=index --format=csv,noheader)
        if [ "${#GPU_IDS[@]}" -eq 0 ]; then
            GPU_IDS=(0)
        fi
    else
        # If nvidia-smi does not exist, fallback to single GPU 0
        GPU_IDS=(0)
    fi
fi

# ---- mode parsing (save | software | hardware, after GPU IDs) ----
MODE=save
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2|inprocess)$ ]]; then
    MODE="$1"
    shift
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

# Set CUDA_VISIBLE_DEVICES by explicitly listing all provided GPU IDs (as comma-separated values)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")

# Set NCCL_DEBUG_FILE after NODE_RANK is determined
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/Megatron-LM/pre-tests/opt/7B/opt-7b-0/logs"
CHECKPOINT_PATH="/dev/shm/models/opt-7b-0-ecnaive"
# DATA_PATH="/workspace/Megatron-LM/pre-tests/opt/opt_data/wiki_text_sentence"

SHM_PKT="/dev/shm/shm_pkt"

# Remaining args after node-rank and GPU ids are passed to the training script
ARGS_TO_PASS=("$@")

# Recovery mode args: enabled only for software / hardware load tests
RECOVERY_MODE_ARGS=()
FT_INPROCESS_RECOVERY_REPEAT=${FT_INPROCESS_RECOVERY_REPEAT:-6}
case "$MODE" in
    save)
        RECOVERY_MODE_ARGS=(
            --save $CHECKPOINT_PATH
            --ec-checkpoint-write-only-penultimate-iter
        )
        ;;
    software)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-ecnaive-software-failure
        )
        ;;
    hardware)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ecnaive-failed-ranks "0,1,2,3,4,5,6,7"
        )
        ;;
    hardware2)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ecnaive-failed-ranks "8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
        )
        ;;
    inprocess)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "0,1,2,3,4,5,6,7"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
        )
        ;;
esac

# Model related configuration here, please do not overlap with json config
HIDDEN_SIZE=5120
NUM_ATTENTION_HEADS=40
NUM_LAYERS=64

SEQ_LENGTH=4096
MAX_POSITION_EMBEDDINGS=$SEQ_LENGTH
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32

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

GPT_ARGS=(
    --no-async-tensor-model-parallel-allreduce 
    --hidden-size $HIDDEN_SIZE 
    --num-attention-heads $NUM_ATTENTION_HEADS 
    --seq-length $SEQ_LENGTH 
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size $GLOBAL_BATCH_SIZE 
    --lr 0.00005
    --train-iters 10
    --ec-checkpoint-write-only-penultimate-iter
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
    --tensor-model-parallel-size 8
    --pipeline-model-parallel-size 8
    --sequence-parallel
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    #--save $CHECKPOINT_PATH 
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    # --use-eccheck

    # --use-gemini
    # --use-gemini-optimized
    # --use-gemini-software-failure
    # --use-gemini-hardware-failure
    # --use-distributed-optimizer
    # --use-ecnaive-software-failure
    --use-ecnaive
    --ckpt-format torch
    # --no-save-optim
    # --no-load-optim
    --save-embeddings-separately
    --use-rdma
    # --timing-log-level 2

    # --- EC-NAIVE generalized parameters ---
     --ecnaive-rs-k 6             # Number of data blocks for RS encoding (default 2 → 2+2 scheme)
    #                                Group size n = k + 2 (e.g. k=6 → 6+2=8 ranks/group)
    # --ecnaive-failed-ranks "0,1,2,3,4,5,6,7"
    #                                Uses ISA-L RS decoding (GF(2^8)) to recover 1-2 lost blocks
    # --no-load-optim
)

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK, mode=$MODE): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} ${RECOVERY_MODE_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (mode=$MODE)"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${RECOVERY_MODE_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}
