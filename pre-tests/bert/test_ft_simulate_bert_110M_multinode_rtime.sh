#!/bin/bash
# bash test_ft_simulate_multinode_rtime.sh  0 2 1
# bash test_ft_simulate_multinode_rtime.sh 1 2 1

# Generate data
# python preprocess_data.py   --input /workspace/data/text/AA/wiki.jsonl   --output-prefix /workspace/data/text/AA/wiki   --tokenizer-type BertWordPieceCase   --vocab-file /workspace/Megatron-LM/pre-tests/bert/data/vocab.txt   --json-keys text   --workers 32   --append-eod
# =============================================================================
# Use ft_launcher to simulate multi-node training on one multi-GPU host
# Purpose: test multi-node fault-tolerance logic using only one host
# =============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

# =============================================================================
# Simulated multi-node configuration
# =============================================================================
# Arguments:
#   $1: Simulated node rank (0, 1, 2, ...)
#   $2: Total node count (default: 2)
#   $3: GPUs per simulated node (default: 1)

NODE_RANK=${1:-0}           # Current simulated node rank
TOTAL_NODES=${2:-2}         # Total simulated node count
GPUS_PER_NODE=${3:-1}       # GPUs used by each simulated node
export HOSTNAME="simnode${NODE_RANK}"
# export CUDA_VISIBLE_DEVICES=$NODE_RANK
# Validate arguments
if [ -z "$1" ]; then
    echo "Error: a node rank must be specified"
    echo "Usage: $0 <node_rank> [total_nodes] [gpus_per_node]"
    echo "Examples:"
    echo "  Terminal 1: $0 0 2 1  # simulate node 0 of 2, with 1 GPU per node"
    echo "  Terminal 2: $0 1 2 1  # simulate node 1 of 2, with 1 GPU per node"
    exit 1
fi

# =============================================================================
# Network configuration for single-host multi-node simulation
# =============================================================================
MASTER_ADDR=127.0.0.1  # Use localhost on a single host
MASTER_PORT=6000
RDZV_PORT=29500

# =============================================================================
# Explanation of port-binding warnings
# =============================================================================
# The c10d rendezvous mechanism coordinates nodes using a client-server model:
#
# 1. Server mode (node 0):
#    - The first node to arrive (node 0) binds to the rdzv_endpoint port (29500)
#    - It acts as the rendezvous server and waits for other nodes to connect
#    - It coordinates the rendezvous process for all nodes
#
# 2. Client mode (nodes 1, 2, ...):
#    - Subsequent nodes first try to bind the port at startup (to check whether they should become the server)
#    - If the port is occupied (meaning node 0 is already the server), binding fails
#    - The node then switches to client mode automatically and connects to node 0
#    - In client mode, the node connects to the server instead of listening for connections
#
# 3. Why does the warning appear?
#    - This is part of the c10d design: every node first attempts to bind the port
#    - If binding fails because the port is occupied, a server already exists, so the node becomes a client
#    - During single-host multi-GPU simulation, all "nodes" are actually on the same host
#    - Therefore, port-binding attempts by nodes 1 and above fail as expected
#
# 4. Is this related to single-host multi-GPU simulation?
#    - Yes, although similar warnings can also occur in a real multi-node environment
#    - The c10d design has each node attempt to bind and choose its role based on the result
#    - In a real multi-node environment, node 1 becomes the server if it starts before node 0
#    - In single-host simulation, port conflicts are more visible because all nodes share one host, but the mechanism is the same
#
# The warning can be safely ignored and does not affect training.

# Network interface - use local loopback
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

WORLD_SIZE=$((TOTAL_NODES * GPUS_PER_NODE))

# =============================================================================
# GPU assignment - each simulated node must use different GPUs
# =============================================================================
# Assign GPUs according to node rank
# Node 0: GPU 0
# Node 1: GPU 1
# Node 2: GPU 2
# And so on...

if [ $GPUS_PER_NODE -eq 1 ]; then
    # One GPU per node - direct mapping
    GPU_ID=$NODE_RANK
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "Node $NODE_RANK uses GPU: $GPU_ID"
else
    # Multiple GPUs per node
    START_GPU=$((NODE_RANK * GPUS_PER_NODE))
    END_GPU=$((START_GPU + GPUS_PER_NODE - 1))
    GPU_LIST=$(seq -s, $START_GPU $END_GPU)
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    echo "Node $NODE_RANK uses GPU: $GPU_LIST"
fi

# Validate GPU assignment
REQUIRED_GPUS=$((TOTAL_NODES * GPUS_PER_NODE))
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ $AVAILABLE_GPUS -lt $REQUIRED_GPUS ]; then
    echo "Warning: available GPUs ($AVAILABLE_GPUS) are fewer than required GPUs ($REQUIRED_GPUS)"
    echo "Current configuration: $TOTAL_NODES nodes x $GPUS_PER_NODE GPUs/node = $REQUIRED_GPUS GPUs"
    echo "Suggestion: reduce the node count or GPUs per node"
fi

# =============================================================================
# Path configuration
# =============================BERT-L-336M================================================
VOCAB_FILE="/workspace/Megatron-LM/pre-tests/bert/data/vocab.txt"
# MERGE_FILE="/workspace/Megatron-LM/pre-tests/bert2/data/bert2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/bert-110M-ft-simnode/logs"
CHECKPOINT_PATH="/dev/shm/bert-110M-ft-simnode"
DATA_PATH="/workspace/data/text/AA/wiki_text_document"

# Create required directories
mkdir -p $CHECKPOINT_PATH
mkdir -p $TENSORBOARD_LOGS_PATH
mkdir -p logs
mkdir -p logs/csv

# =============================================================================
# Fault-tolerance configuration
# =============================================================================
FT_TIMEOUT_SETUP=600
FT_TIMEOUT_STEP=300
FT_TIMEOUT_CHECKPOINTING=420
FT_TIMEOUT_OUT_OF_SECTION=300

# Failure simulation (optional)
# export FT_SIM_FAULT_DESC="rank_killed;1;60.0"  # 60 seconds before killing rank 1

# =============================================================================
# Model and training configuration
# =============================================================================
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12
SEQ_LENGTH=128
MAX_POSITION_EMBEDDINGS=512
MICRO_BATCH_SIZE=4
# GLOBAL_BATCH_SIZE=16

# =============================================================================
# Megatron training arguments
# =============================================================================
DATA_ARGS=(
    --vocab-file $VOCAB_FILE 
    # --merge-file $MERGE_FILE 
    --data-path $DATA_PATH
    --data-cache-path /dev/shm/bert_cache
    --num-dataset-builder-threads 32
    --split 949,50,1
)

MODEL_ARGS=(
    --no-async-tensor-model-parallel-allreduce 
    --hidden-size $HIDDEN_SIZE 
    --num-attention-heads $NUM_ATTENTION_HEADS 
    --seq-length $SEQ_LENGTH 
    --max-position-embeddings $MAX_POSITION_EMBEDDINGS 
    --micro-batch-size $MICRO_BATCH_SIZE 
    --global-batch-size 8
    --lr 1e-4
    --train-iters 200
    --lr-decay-iters 320000 
    --lr-decay-style linear 
    --min-lr 1e-6
    --weight-decay 1e-2 
    --lr-warmup-fraction 0.01 
    --clip-grad 1.0 
    --fp16 
    --tokenizer-type BertWordPieceCase 
    --use-mcore-models 
    --transformer-impl transformer_engine 
    --no-scatter-gather-tensors-in-pipeline 
    --num-layers 12
    --optimizer adam
    --adam-beta1 0.9
    --adam-beta2 0.999
    --adam-eps 1e-6
    --loss-scale-window 1000
    --initial-loss-scale 4096
    --min-loss-scale 1.0
    --hysteresis 2
    --bert-no-binary-head
)

# Model-parallel configuration based on GPUs per node
if [ $GPUS_PER_NODE -gt 1 ]; then
    # Multiple GPUs per node: use tensor parallelism
    TENSOR_PARALLEL=$GPUS_PER_NODE
    PIPELINE_PARALLEL=1
else
    # One GPU per node: pipeline parallelism can be used
    TENSOR_PARALLEL=1
    PIPELINE_PARALLEL=$TOTAL_NODES  # or set to 1, depending on requirements
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL
    --pipeline-model-parallel-size $PIPELINE_PARALLEL
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 10
    --eval-interval 50
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --ckpt-format torch
    --use-eccheck
    # --rerun-mode disabled
)

# =============================================================================
# Fault-tolerance arguments
# =============================================================================
FT_ARGS=(
    --enable-ft-package
    --calc-ft-timeouts
)

# =============================================================================
# ft_launcher arguments for multi-node simulation
# =============================================================================
FT_LAUNCHER_ARGS=(
    # Rendezvous configuration
    --rdzv_backend=c10d
    --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT}
    --rdzv_id=megatron_bert_simulated_multinode  # Unique job ID
    
    # Multi-node configuration
    --nnodes=${TOTAL_NODES}:${TOTAL_NODES}  # Minimum/maximum node count
    --nproc-per-node=${GPUS_PER_NODE}       # Processes per node
    --node-rank=${NODE_RANK}                # Current node rank
    
    # Fault-tolerance arguments
    --ft-param-rank_section_timeouts=setup:${FT_TIMEOUT_SETUP},step:${FT_TIMEOUT_STEP},checkpointing:${FT_TIMEOUT_CHECKPOINTING}
    --ft-param-rank_out_of_section_timeout=${FT_TIMEOUT_OUT_OF_SECTION}
    --ft-param-rank_heartbeat_timeout=60
    
    # Maximum restart count
    --max-restarts=3
    
    # Logging configuration - separate logs for each node
    --log-dir=./ft_logs/node_${NODE_RANK}
)

# =============================================================================
# Set the Python path
# =============================================================================
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM
export USE_FLASH_ATTN=1
export NVTE_SYNC_P2P=1

# =============================================================================
# Start fault-tolerant training
# =============================================================================
echo "=========================================================================="
echo "Simulating fault-tolerant multi-node distributed training on one host"
echo "=========================================================================="
echo "Configuration summary:"
echo "  Total simulated nodes: ${TOTAL_NODES}"
echo "  Current node rank: ${NODE_RANK}"
echo "  GPUs per node: ${GPUS_PER_NODE}"
echo "  Total processes: ${WORLD_SIZE}"
echo "  Current node GPUs: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "Network configuration:"
echo "  Master address: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Rendezvous: ${MASTER_ADDR}:${RDZV_PORT}"
echo "  Network interface: ${NCCL_SOCKET_IFNAME}"
echo ""
echo "Model parallelism:"
echo "  Tensor parallelism: ${TENSOR_PARALLEL}"
echo "  Pipeline parallelism: ${PIPELINE_PARALLEL}"
echo ""
echo "Storage paths:"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  Log directory: ./ft_logs/node_${NODE_RANK}"
echo ""
echo "Fault-tolerance timeout configuration:"
echo "  Setup: ${FT_TIMEOUT_SETUP}s"
echo "  Step: ${FT_TIMEOUT_STEP}s"
echo "  Checkpointing: ${FT_TIMEOUT_CHECKPOINTING}s"
echo "  Out-of-section: ${FT_TIMEOUT_OUT_OF_SECTION}s"
echo "=========================================================================="
echo ""

# Wait for user confirmation (optional)
# if [ "$NODE_RANK" -eq 0 ]; then
#     echo "Tip: start the other nodes in separate terminals"
#     for i in $(seq 1 $((TOTAL_NODES - 1))); do
#         echo "  Terminal $((i+1)): bash $0 $i $TOTAL_NODES $GPUS_PER_NODE"
#     done
#     echo ""
#     read -p "Press Enter to start node $NODE_RANK..." -r
# fi

echo "Starting node $NODE_RANK ..."
echo ""

# Run ft_launcher
ft_launcher \
    ${FT_LAUNCHER_ARGS[@]} \
    pretrain_bert.py \
    ${MODEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FT_ARGS[@]} \
    --distributed-backend nccl

# =============================================================================
# Training complete
# =============================================================================
echo "=========================================================================="
echo "Training task for node ${NODE_RANK} has finished"
echo "=========================================================================="

