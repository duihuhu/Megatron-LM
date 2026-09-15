#!/bin/bash

# =============================================================================
# Gemini Replicas Torch Legacy Checkpoint test script
#
# Use the torch legacy path (torch.save to .pt files) for replicated checkpoints,
# instead of the original distributed checkpoint (FileSystemWriterAsync + torch_dist) path.
#
# This follows the same pattern as basic_ec_legacy.py and concord_legacy.py.
# =============================================================================
#
# Usage:
#   ./test_eccheck_4nodes_node_335M_gemini_2_replicas.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [mode] [additional_args...]
#
# mode (optional, default: save):
#   save      - checkpoint save only (no load / recovery flags)
#   software  - load checkpoint + software failure recovery
#   hardware  - load checkpoint + hardware failure recovery
#
# Example:
#   ./test_eccheck_4nodes_node_335M_gemini_2_replicas.sh 0 0
#   ./test_eccheck_4nodes_node_335M_gemini_2_replicas.sh 0 0 1 2 3 4 5 6 7 software
#   ./test_eccheck_4nodes_node_335M_gemini_2_replicas.sh 0 2 3 hardware --train-iters 50
# =============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=eth0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1

MASTER_ADDR=172.16.0.224
export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE

# ---------------------------------------------------------------------------
# Gemini Replicas network environment variables
# ---------------------------------------------------------------------------
# High-speed network interface used for data transfer
export GEMINI_REPLICAS_INTERFACE=$NETIFACES_INTERFACE
# Base IP on which ranks listen (usually MASTER_ADDR; each rank derives its port as GEMINI_REPLICAS_BASE_PORT + rank*100)
# export GEMINI_REPLICAS_BASE_IP=$MASTER_ADDR
# Base port; each rank reserves a range of 100 ports to avoid conflicts
# export GEMINI_REPLICAS_BASE_PORT=12345

export GEMINI_REPLICAS_LOCAL_RANK_NIC_0=eth0
export GEMINI_REPLICAS_LOCAL_RANK_NIC_1=eth0
export GEMINI_REPLICAS_LOCAL_RANK_NIC_2=eth0
export GEMINI_REPLICAS_LOCAL_RANK_NIC_3=eth0
export GEMINI_REPLICAS_LOCAL_RANK_NIC_4=eth1
export GEMINI_REPLICAS_LOCAL_RANK_NIC_5=eth1
export GEMINI_REPLICAS_LOCAL_RANK_NIC_6=eth1
export GEMINI_REPLICAS_LOCAL_RANK_NIC_7=eth1
MASTER_PORT=6000
NNODES=4

# ---- Parse the node rank (first argument) ----
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# ---- Parse GPU IDs (subsequent consecutive numeric arguments) ----
# As in the Concord script, collect all consecutive numeric arguments as GPU IDs,
# stop at the first nonnumeric argument, and pass the remaining arguments to the training script.
GPU_IDS=()
while [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "Error: At least one GPU id must be specified."
    echo "Usage: $0 <node_rank> <gpu_id_0> [gpu_id_1 ...] [save|software|hardware|hardware2] [additional_args...]"
    exit 1
fi

# ---- mode parsing (save | software | hardware, after GPU IDs) ----
MODE=save
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2)$ ]]; then
    MODE="$1"
    shift
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/bert/bert_data/vocab.txt"

TENSORBOARD_LOGS_PATH="/workspace/Megatron-LM/pre-tests/bert/7B/bert-7b-0/logs"
CHECKPOINT_PATH="/dev/shm/models/bert-7b-0-gemini-2-replicas"
DATA_PATH="/workspace/Megatron-LM/pre-tests/bert/bert_data/wiki_text_sentence"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/workspace/Megatron-LM/pre-tests/bert/bert_data/cache}"

SHM_PKT="/dev/shm/shm_pkt"

ARGS_TO_PASS=("$@")

# Recovery mode args: enabled only for software / hardware load tests
RECOVERY_MODE_ARGS=()
case "$MODE" in
    save)
        ;;
    software)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-gemini-replicas-software-failure
            --gemini-replicas-recovery-rank "0,1,2,3,4,5,6,7"
        )
        ;;
    hardware)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-gemini-replicas-hardware-failure
            --gemini-replicas-recovery-rank "8,9,10,11,12,13,14,15"
        )
        ;;
    hardware2)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-gemini-replicas-two-failures
            --gemini-replicas-recovery-rank "8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23"
        )
        ;;
esac

# Fixed model parameters
HIDDEN_SIZE=2560
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

# =============================================================================
# Gemini Replicas Torch Legacy core parameter reference
# =============================================================================

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10
    --eval-interval 100
    --save $CHECKPOINT_PATH
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH

    # ---------------------------------------------------------------------------
    # Required: enable Gemini Replicas torch legacy checkpoints
    # ---------------------------------------------------------------------------
    # Enable Gemini Replicas (replaces --use-gemini, which uses two-replica EC-style pairing)
    --use-gemini-replicas
    # Enable the optimized path: use contiguous CPU buffers and C++ ASIO/RDMA transfer, bypassing torch.save serialization overhead
    # --use-gemini-replicas-optimized

    # ---------------------------------------------------------------------------
    # Replica count: store N copies of each rank's data within the group (including the local copy)
    # ---------------------------------------------------------------------------
    --gemini-replicas-num 2          # Default: 3. Set to 2 for two replicas or N for N replicas
                                        # Place replicas round-robin within the group
                                        # Fault tolerance = num_replicas - 1 simultaneous rank failures

    # ---------------------------------------------------------------------------
    # Group size: divide the world into independent groups and rotate replicas only within each group
    # ---------------------------------------------------------------------------
    --gemini-replicas-group-size 4   # Default: None (global rotation without grouping)
                                        # Set to 8 for independent groups of eight ranks
                                        # Must divide world_size evenly
                                        # Independent of node count and ranks per node, but mathematically requires
                                        #   num_nodes % group_size == 0 must hold to enable this mode
                                        #   Interleave ranks across nodes (the same layout as Concord)
                                        # Each rank in a group comes from a different physical node

    # ---------------------------------------------------------------------------
    # Transport: RDMA (InfiniBand) or TCP (ASIO)
    # ---------------------------------------------------------------------------
    --use-rdma                       # Enable RDMA (TCP/ASIO is the default)
                                        # When enabled, both send and receive use InfiniBand verbs
                                        # Requires hardware support and preregistered memory

    # ---------------------------------------------------------------------------
    # Recovery modes (used for load; not needed for save)
    # ---------------------------------------------------------------------------
    # Method 1 - automatic detection (missing file = failure):
    #   Delete the main file for the rank whose failure is being simulated, then load.
    #   The system detects the missing file and selects a sender in the group to transfer replica data through torch.distributed.
    #   The main file is regenerated automatically after recovery.
    #   Test procedure: after save completes, delete gemini_replicas_main_rank*.pt for the failed rank,
    #            then load with the same arguments.
    #
    #   --use-gemini-replicas-hardware-failure

    # Method 2 - specify failed ranks (keep files for precise control):
    #   Specify which ranks simulate failure. These ranks take the recovery path even if their main files exist.
    #   Test procedure: load immediately after save and add the argument below.
    #   Example: "2,3" treats rank 2 and rank 3 as failed.
    #
    #   --gemini-replicas-recovery-rank 2,3
    #
    # ---------------------------------------------------------------------------
    # Checkpoint format: torch is required (legacy path)
    # ---------------------------------------------------------------------------
    --ckpt-format torch               # Required: the legacy path requires torch format
                                        # Do not use torch_dist (that is the distributed checkpoint path)
    # Note: do not set --use-dist-ckpt, or the GLOBAL type will be used instead of LEGACY
    # Internal check: ckpt_type=LEGACY + ckpt_format=torch is required to permit gemini_replicas

    --save-embeddings-separately
    # --no-save-optim                 # Uncomment to skip saving the optimizer
    # --no-load-optim                 # Uncomment to skip loading the optimizer
)

# =============================================================================
# Quick reference for argument combinations
# =============================================================================
#
# Scenario 1 - three global replicas (default, small-scale test):
#   --use-gemini-replicas --use-gemini-replicas-optimized --ckpt-format torch
#
# Scenario 2 - two global replicas (equivalent to the original two-replica Gemini with round-robin pairing):
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-num 2 --ckpt-format torch
#
# Scenario 3 - eight nodes with eight GPUs each, group size 8, four replicas per group:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-num 4 --gemini-replicas-group-size 8 --ckpt-format torch
#
# Scenario 4 - hardware failure recovery (automatically detected after deleting files):
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --use-gemini-replicas-hardware-failure --use-rdma --ckpt-format torch
#
# Scenario 5 - mark ranks 2 and 3 as failed (keep files for precise control):
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-recovery-rank 2,3 --ckpt-format torch
#
# =============================================================================

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK, mode=$MODE): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} ${RECOVERY_MODE_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (Gemini Replicas Legacy, mode=$MODE)"
echo "WORLD_SIZE=$WORLD_SIZE  GPUS_PER_NODE=$GPUS_PER_NODE  NNODES=$NNODES"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} \
    pretrain_bert.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${RECOVERY_MODE_ARGS[@]} \
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}
