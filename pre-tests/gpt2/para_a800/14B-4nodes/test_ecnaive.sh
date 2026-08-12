#!/bin/bash

# ONNX in the container uses legacy generated protobuf descriptors.
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=${PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION:-python}

# Script to run a single node in 4-node simulation (default 1 GPU per node)
# Usage: ./test_ecnaive.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [mode] [additional_args...]
#
# mode (optional, default: save):
#   save      - checkpoint save only (no load / recovery flags)
#   software  - load checkpoint + software failure recovery
#   hardware  - load checkpoint + hardware failure recovery (failed ranks only)
#
# Example: ./test_ecnaive.sh 0 0
# Example (2 GPUs per container): ./test_ecnaive.sh 0 2 3 software
# Example (8 GPUs, hardware recovery): ./test_ecnaive.sh 0 0 1 2 3 4 5 6 7 hardware

export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1
export NETIFACES_INTERFACE=${NETIFACES_INTERFACE:-bond0}

export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL
export NCCL_IB_DISABLE=1
MASTER_ADDR=${MASTER_ADDR:-10.252.129.35}

export ECCHECK_USE_ASIO=true
MASTER_PORT=${MASTER_PORT:-6000}
NNODES=${NNODES:-4}

export NCCL_SOCKET_IFNAME=$NETIFACES_INTERFACE
export GLOO_SOCKET_IFNAME=$NETIFACES_INTERFACE
export ECNAIVE_INTERFACE=$NETIFACES_INTERFACE
export ECLATIN_INTERFACE=$NETIFACES_INTERFACE
export MEGATRON_ECNAIVE_LOAD_NET_TRACE=1
#  priority from
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
if [ -n "$1" ] && [[ "$1" =~ ^(save|software|hardware|hardware2|inprocess|inprocess2|inprocess_sw)$ ]]; then
    MODE="$1"
    shift
fi

GPUS_PER_NODE=${#GPU_IDS[@]}

export RDMA_HCA_PROFILE=${RDMA_HCA_PROFILE:-full}
case "$RDMA_HCA_PROFILE" in
    full|half|quarter) ;;
    *)
        echo "Error: RDMA_HCA_PROFILE must be full, half, or quarter, got '$RDMA_HCA_PROFILE'." >&2
        exit 2
        ;;
esac

rdma_hca_for_gpu() {
    case "$1" in
        [0-7]) ;;
        *)
            echo "Unsupported physical GPU id for RDMA binding: $1" >&2
            return 1
            ;;
    esac

    case "$RDMA_HCA_PROFILE" in
        full)
            case "$1" in
                0|1) echo mlx5_0 ;;
                2|3) echo mlx5_1 ;;
                4|5) echo mlx5_4 ;;
                6|7) echo mlx5_5 ;;
            esac
            ;;
        half)
            case "$1" in
                0|1|2|3) echo mlx5_0 ;;
                4|5|6|7) echo mlx5_4 ;;
            esac
            ;;
        quarter) echo mlx5_0 ;;
    esac
}

configure_rdma_local_rank_bindings() {
    local prefix=$1
    local local_rank
    local hca
    for local_rank in "${!GPU_IDS[@]}"; do
        hca=$(rdma_hca_for_gpu "${GPU_IDS[$local_rank]}") || exit 1
        export "${prefix}_LOCAL_RANK_NIC_${local_rank}=${hca}"
    done
}

configure_rdma_local_rank_bindings ECNAIVE

# Set CUDA_VISIBLE_DEVICES by explicitly listing all provided GPU IDs (as comma-separated values)
export CUDA_VISIBLE_DEVICES=$(IFS=, ; echo "${GPU_IDS[*]}")

# Set NCCL_DEBUG_FILE after NODE_RANK is determined
export NCCL_DEBUG_FILE=./nccl.log.node${NODE_RANK}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/opt/opt_data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH=${TENSORBOARD_LOGS_PATH:-"/workspace/Megatron-LM/logs/gpt2-14b-4nodes/ecnaive"}
CHECKPOINT_PATH=${CHECKPOINT_PATH:-"/dev/shm/models/gpt2-14b-4nodes-ecnaive"}
# DATA_PATH="/workspace/Megatron-LM/pre-tests/opt/opt_data/wiki_text_sentence"

SHM_PKT="/dev/shm/shm_pkt"

# Remaining args after node-rank and GPU ids are passed to the training script
ARGS_TO_PASS=("$@")

# Recovery mode args: enabled only for software / hardware load tests
RECOVERY_MODE_ARGS=()
FT_INPROCESS_RECOVERY_REPEAT=${FT_INPROCESS_RECOVERY_REPEAT:-3}
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
            --ecnaive-failed-ranks "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
        )
        ;;
    inprocess_sw)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --use-ecnaive-software-failure
            --ft-inprocess-recovery-benchmark
            --ft-inprocess-recovery-software-failure
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
        )
        ;;
    inprocess2)
        RECOVERY_MODE_ARGS=(
            --load $CHECKPOINT_PATH
            --ft-inprocess-recovery-benchmark
            --rerun-mode disabled
            --ft-inprocess-recovery-repeat $FT_INPROCESS_RECOVERY_REPEAT
            --ft-inprocess-recovery-failed-ranks "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
            --ft-inprocess-recovery-after-train-iter 0
            --ft-inprocess-recovery-exit-after-forward
            --ecnaive-require-hw2
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
NUM_LAYERS=40

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
    --pipeline-model-parallel-size 4
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
     --ecnaive-rs-k 2
    # --ecnaive-failed-ranks "0,1,2,3,4,5,6,7"
    #                                Uses ISA-L RS decoding (GF(2^8)) to recover 1-2 lost blocks
    # --no-load-optim
)

mkdir -p logs
mkdir -p logs/csv

EC_RANK_LAUNCH='
set -e
if [[ ! "${LOCAL_RANK:-}" =~ ^[0-9]+$ ]]; then
    echo "Error: Invalid LOCAL_RANK: ${LOCAL_RANK:-<unset>}" >&2
    exit 1
fi
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    echo "Error: CUDA_VISIBLE_DEVICES is empty or unset." >&2
    exit 1
fi
IFS=, read -r -a visible_gpus <<< "$CUDA_VISIBLE_DEVICES"
if (( LOCAL_RANK >= ${#visible_gpus[@]} )); then
    echo "Error: LOCAL_RANK $LOCAL_RANK is outside CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" >&2
    exit 1
fi
physical_gpu=${visible_gpus[$LOCAL_RANK]//[[:space:]]/}
if [[ ! "$physical_gpu" =~ ^[0-7]$ ]]; then
    echo "Error: Invalid physical GPU id for LOCAL_RANK $LOCAL_RANK: ${physical_gpu:-<empty>}" >&2
    exit 1
fi
if (( physical_gpu < 4 )); then
    numa_node=0
    base=0
else
    numa_node=1
    base=32
fi
slot=$((physical_gpu % 4))
first_start=$((base + slot * 8))
first_end=$((first_start + 7))
sibling_start=$((first_start + 64))
sibling_end=$((sibling_start + 7))
first_cpus=$(seq -s, "$first_start" "$first_end")
sibling_cpus=$(seq -s, "$sibling_start" "$sibling_end")
ec_cpus="$first_cpus,$sibling_cpus"
export ECNAIVE_XOR_CPU_LIST="$ec_cpus"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
echo "[cpu_bind] local_rank=$LOCAL_RANK physical_gpu=$physical_gpu numa_node=$numa_node ec_cpus=$ec_cpus"
exec taskset -c "$ec_cpus" "${PYTHON_BIN:-python}" "$@"
'

# -------------------------------------------------------------------------
# Print command if PRINT_CMD is set
# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    printf 'Would run (Node %s, mode=%s, topology-aware per-rank CPU binding): ' "$NODE_RANK" "$MODE"
    printf 'RDMA_HCA_PROFILE=%q PYTHONPATH=%q CUDA_VISIBLE_DEVICES=%q ' "$RDMA_HCA_PROFILE" "$PYTHONPATH:/workspace/Megatron-LM" "$CUDA_VISIBLE_DEVICES"
    printf '%q ' torchrun "${DISTRIBUTED_ARGS[@]}" --no-python bash -c '<physical-GPU CPU binding>' _ \
        pretrain_gpt.py "${GPT_ARGS[@]}" "${DATA_ARGS[@]}" "${MODEL_PARALLEL_ARGS[@]}" \
        "${EVAL_AND_LOGGING_ARGS[@]}" "${RECOVERY_MODE_ARGS[@]}" --distributed-backend nccl \
        "${ARGS_TO_PASS[@]}"
    printf '\n'
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (mode=$MODE)"
echo "RDMA_HCA_PROFILE: $RDMA_HCA_PROFILE"
echo "NCCL_DEBUG_FILE: $NCCL_DEBUG_FILE"

export USE_FLASH_ATTN=1 && \
export NVTE_SYNC_P2P=1 && \

PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun "${DISTRIBUTED_ARGS[@]}" \
    --no-python bash -c "$EC_RANK_LAUNCH" _ \
    pretrain_gpt.py \
    "${GPT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${RECOVERY_MODE_ARGS[@]}" \
    --distributed-backend nccl \
    "${ARGS_TO_PASS[@]}"
