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
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [additional_args...]
#
# Examples:
#   # 4 nodes with 1 GPU (ID 0)
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh 0 0
#
#   # 4 nodes with 8 GPUs (IDs 0-7)
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh 0 0 1 2 3 4 5 6 7
#
#   # 4 nodes with 2 GPUs (IDs 2, 3), and pass additional training arguments
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh 0 2 3 --train-iters 50
# =============================================================================

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=bond0

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1

MASTER_ADDR=10.0.0.62
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

MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-8}

# Gemini uses one shared port namespace per distributed job.
# Keep the same base ports on all simulated nodes; change these env vars only
# when running another independent job on the same machine.
export GEMINI_REPLICAS_BASE_PORT=${GEMINI_REPLICAS_BASE_PORT:-$((MASTER_PORT + 30000))}
export GEMINI_REPLICAS_RECOVERY_BASE_PORT=${GEMINI_REPLICAS_RECOVERY_BASE_PORT:-$((MASTER_PORT + 40000))}

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
# stop at the first non-numeric argument, and pass the remaining arguments through to the training script.
GPU_IDS=()
while [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; do
    GPU_IDS+=("$1")
    shift
done

if [ "${#GPU_IDS[@]}" -eq 0 ]; then
    echo "Error: At least one GPU id must be specified."
    echo "Usage: $0 <node_rank> <gpu_id_0> [gpu_id_1 ...] [additional_args...]"
    exit 1
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

check_gemini_port_free() {
    local port=$1
    local label=$2
    if ! command -v ss >/dev/null 2>&1; then
        return 0
    fi
    if ss -ltn "sport = :${port}" | grep -q ":${port} "; then
        echo "Error: ${label} port ${port} is already in use." >&2
        echo "Another Gemini job or stale process is probably still listening." >&2
        echo "Use a different MASTER_PORT/GEMINI_REPLICAS_BASE_PORT, or stop the old job." >&2
        exit 1
    fi
}

for ((local_idx=0; local_idx<GPUS_PER_NODE; local_idx++)); do
    global_rank=$((NODE_RANK * GPUS_PER_NODE + local_idx))
    check_gemini_port_free $((GEMINI_REPLICAS_BASE_PORT + global_rank * 100)) "Gemini save"
    check_gemini_port_free $((GEMINI_REPLICAS_RECOVERY_BASE_PORT + global_rank * 100)) "Gemini recovery"
done

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-345m-0/logs"
GEMINI_CHECKPOINT_BASE="/dev/shm/data/checkpoint/models/gpt2-345m-0-gemini-replicas-legacy"
DATA_PATH="/workspace/models/gpt2-345m-0/codeparrot_content_document"

SHM_PKT="/dev/shm/shm_pkt"


MODE=save
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2|inprocess|inprocess2|inprocess_sw)$ ]]; then
    MODE="$1"
    shift
fi

GEMINI_REPLICAS_NUM_WAS_SET=0
if [ -n "${GEMINI_REPLICAS_NUM+x}" ]; then
    GEMINI_REPLICAS_NUM_WAS_SET=1
fi
GEMINI_REPLICAS_NUM=${GEMINI_REPLICAS_NUM:-2}
if [[ "$MODE" == "hardware2" || "$MODE" == "inprocess2" ]]; then
    if [ "$GEMINI_REPLICAS_NUM_WAS_SET" -eq 0 ]; then
        GEMINI_REPLICAS_NUM=3
    elif [ "$GEMINI_REPLICAS_NUM" -lt 3 ]; then
        echo "Error: $MODE requires GEMINI_REPLICAS_NUM>=3 in this benchmark." >&2
        echo "Save and load the HW2 checkpoint with the same GEMINI_REPLICAS_NUM value." >&2
        exit 1
    fi
fi
if [ -n "${CHECKPOINT_PATH+x}" ]; then
    CHECKPOINT_PATH=${CHECKPOINT_PATH}
elif [ "$GEMINI_REPLICAS_NUM" -eq 2 ]; then
    CHECKPOINT_PATH=$GEMINI_CHECKPOINT_BASE
else
    CHECKPOINT_PATH="${GEMINI_CHECKPOINT_BASE}-r${GEMINI_REPLICAS_NUM}"
fi
GEMINI_FAILED_RANKS=${GEMINI_FAILED_RANKS:-0}
GEMINI_HW2_FAILED_RANKS=${GEMINI_HW2_FAILED_RANKS:-0,1}

LATEST_ITER=1
if [[ "$MODE" == "hardware" || "$MODE" == "hardware2" ]]; then
    if [ -n "$1" ] && [[ "$1" =~ ^[0-9]+$ ]]; then
        LATEST_ITER="$1"
        shift
    fi
    # Update the tracker only for a real run; PRINT_CMD must remain read-only.
    if [ "${PRINT_CMD:-0}" = "0" ]; then
        LATEST_ITER_FILE="$CHECKPOINT_PATH/latest_checkpointed_iteration.txt"
        mkdir -p "$CHECKPOINT_PATH"
        echo "$LATEST_ITER" > "$LATEST_ITER_FILE"
    fi
fi

ARGS_TO_PASS=("$@")
GEMINI_REPLICAS_CHANNELS_PER_PEER=${GEMINI_REPLICAS_CHANNELS_PER_PEER:-8}
RECOVERY_MODE_ARGS=(
    --gemini-replicas-channels-per-peer $GEMINI_REPLICAS_CHANNELS_PER_PEER
)
FT_INPROCESS_RECOVERY_REPEAT=${FT_INPROCESS_RECOVERY_REPEAT:-3}
case "$MODE" in
    save)
        RECOVERY_MODE_ARGS+=(
            --save $CHECKPOINT_PATH
            --ec-checkpoint-write-only-penultimate-iter
        )
        ;;
    software)
        RECOVERY_MODE_ARGS+=(
            --load $CHECKPOINT_PATH
            --use-gemini-replicas-software-failure
            --gemini-replicas-recovery-rank "$GEMINI_FAILED_RANKS"
        )
        ;;
    hardware|hardware2)
        FAILED_RANKS=$GEMINI_FAILED_RANKS
        if [ "$MODE" = "hardware2" ]; then
            FAILED_RANKS=$GEMINI_HW2_FAILED_RANKS
        fi
        RECOVERY_MODE_ARGS+=(
            --load $CHECKPOINT_PATH
            --use-gemini-replicas-hardware-failure
            --gemini-replicas-recovery-rank "$FAILED_RANKS"
        )
        ;;
    inprocess_sw)
        RECOVERY_MODE_ARGS+=(
            --load $CHECKPOINT_PATH
            --use-gemini-replicas-software-failure
            --gemini-replicas-recovery-rank "$GEMINI_FAILED_RANKS"
            --ft-inprocess-recovery-benchmark
            --ft-inprocess-recovery-software-failure
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "$GEMINI_FAILED_RANKS"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
        )
        ;;
    inprocess|inprocess2)
        FAILED_RANKS=$GEMINI_FAILED_RANKS
        if [ "$MODE" = "inprocess2" ]; then
            FAILED_RANKS=$GEMINI_HW2_FAILED_RANKS
        fi
        RECOVERY_MODE_ARGS+=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "$FAILED_RANKS"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
        )
        ;;
esac
# Fixed model parameters
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

GPT_ARGS=(
    --no-async-tensor-model-parallel-allreduce
    --hidden-size $HIDDEN_SIZE
    --num-attention-heads $NUM_ATTENTION_HEADS
    --seq-length $SEQ_LENGTH
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS
    --micro-batch-size $MICRO_BATCH_SIZE
    --global-batch-size $GLOBAL_BATCH_SIZE
    --lr 0.00015
    --train-iters 4
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
    --num-layers 24
    --optimizer adam
    --loss-scale 8192
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size 1
    --pipeline-model-parallel-size 8
)

# =============================================================================
# Gemini Replicas Torch Legacy core parameter notes
# =============================================================================

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    #--save $CHECKPOINT_PATH
    #--load $CHECKPOINT_PATH          # Uncomment to test load
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH

    # ---------------------------------------------------------------------------
    # Required: enable Gemini Replicas torch legacy checkpoints
    # ---------------------------------------------------------------------------
    # Enable Gemini Replicas (replaces --use-gemini, which uses two-replica EC-style pairing)
    --use-gemini-replicas
    # Enable the optimized path: use contiguous CPU buffers and C++ ASIO/RDMA transfer, bypassing torch.save serialization overhead
    --use-gemini-replicas-optimized

    # ---------------------------------------------------------------------------
    # Replica count: store N copies of each rank's data within the group (including the local copy)
    # ---------------------------------------------------------------------------
    --gemini-replicas-num $GEMINI_REPLICAS_NUM
                                        # Place replicas round-robin within the group
                                        # Fault tolerance = num_replicas - 1 simultaneous rank failures

    # ---------------------------------------------------------------------------
    # Group size: divide the world into independent groups and rotate replicas only within each group
    # ---------------------------------------------------------------------------
    --gemini-replicas-group-size 8  # Default: None (global rotation without grouping)
                                        # Set to 8 for independent groups of eight ranks
                                        # Must divide world_size evenly
                                        # Independent of node count and ranks per node, but mathematically requires
                                        #   num_nodes % group_size == 0 must hold to enable this mode
                                        #   Interleave ranks across nodes (the same layout as Concord)
                                        # Each rank in a group comes from a different physical node

    # ---------------------------------------------------------------------------
    # Transport: RDMA (InfiniBand) or TCP (ASIO)
    # ---------------------------------------------------------------------------
    --use-rdma                       # Enable RDMA (TCP/ASIO is used by default).
                                        # When enabled, both send and receive use InfiniBand verbs
                                        # Requires hardware support and preregistered memory

    # ---------------------------------------------------------------------------
    # Recovery mode (used for load; not needed for save)
    # ---------------------------------------------------------------------------
    # Method 1 — automatic detection (missing file = failure):
    #   Delete the main file of the rank whose failure is being simulated, then load.
    #   The system detects the missing file and selects a sender in the group to transfer replica data through torch.distributed.
    #   The main file is regenerated automatically after recovery.
    #   Test procedure: after save completes, delete the failed rank's gemini_replicas_main_rank*.pt,
    #            then load with the same arguments.
    #
    #   --use-gemini-replicas-hardware-failure

    # Method 2 — specified failed ranks (without deleting files for precise control):
    #   Specify which ranks simulate failures. These ranks use the recovery path even if their main files exist.
    #   Test procedure: load directly after save completes and add the argument below.
    # Example: "2,3" treats rank 2 and rank 3 as failed.
    #
    #--use-gemini-replicas-software-failure
    #--use-gemini-replicas-hardware-failure
    #--gemini-replicas-recovery-rank "0,2" # "0,1,2,3,4,5,6,7" for ali 8 ranks per node, but untested yet

    # ---------------------------------------------------------------------------
    # Checkpoint format: torch is required (legacy path)
    # ---------------------------------------------------------------------------
    --ckpt-format torch               # Required! Legacy path requires torch format
                                        # Do not use torch_dist (that is the distributed checkpoint path)
    # Note: do not set --use-dist-ckpt; otherwise it uses GLOBAL type instead of LEGACY
    # Internal validation: ckpt_type=LEGACY + ckpt_format=torch is required to allow gemini_replicas

    --save-embeddings-separately
    # --no-save-optim                 # Uncomment to skip saving the optimizer.
    # --no-load-optim                 # Uncomment to skip loading the optimizer.
)

# =============================================================================
# Argument combination quick reference
# =============================================================================
#
# Scenario 1 — three global replicas (default, small-scale test):
#   --use-gemini-replicas --use-gemini-replicas-optimized --ckpt-format torch
#
# Scenario 2 — two global replicas (equivalent to the original two-replica Gemini with round-robin pairing):
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-num 2 --ckpt-format torch
#
# Scenario 3 — eight nodes with eight GPUs each, group size 8, four replicas per group:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-num 4 --gemini-replicas-group-size 8 --ckpt-format torch
#
# Scenario 4 — hardware failure recovery (automatic detection after deleting files):
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --use-gemini-replicas-hardware-failure --use-rdma --ckpt-format torch
#
# Scenario 5 — treat rank 2 and rank 3 as failed without deleting files:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-recovery-rank 2,3 --ckpt-format torch
#
# =============================================================================

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} ${RECOVERY_MODE_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (Gemini Replicas Legacy)"
echo "WORLD_SIZE=$WORLD_SIZE  GPUS_PER_NODE=$GPUS_PER_NODE  NNODES=$NNODES"
echo "MODE=$MODE  GEMINI_REPLICAS_NUM=$GEMINI_REPLICAS_NUM  CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "GEMINI_REPLICAS_BASE_PORT=$GEMINI_REPLICAS_BASE_PORT  GEMINI_REPLICAS_RECOVERY_BASE_PORT=$GEMINI_REPLICAS_RECOVERY_BASE_PORT"
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
