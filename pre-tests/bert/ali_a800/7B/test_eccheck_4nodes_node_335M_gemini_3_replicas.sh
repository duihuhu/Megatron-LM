#!/bin/bash

# =============================================================================
# Gemini Replicas Torch Legacy Checkpoint 测试脚本
#
# 使用 torch legacy 路径（torch.save → .pt 文件）进行多副本 checkpoint，
# 替代原有的 distributed checkpoint（FileSystemWriterAsync + torch_dist）路径。
#
# 与 ecnaive_legacy.py / frcheck_legacy.py 遵循相同模式。
# =============================================================================
#
# 用法:
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh <node_rank> <gpu_id_0> [gpu_id_1 ...] [additional_args...]
#
# 示例:
#   # 4 节点各 1 GPU (id 0)
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh 0 0
#
#   # 4 节点各 8 GPU (id 0-7)
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh 0 0 1 2 3 4 5 6 7
#
#   # 4 节点各 2 GPU (id 2,3)，额外传入训练参数
#   ./test_eccheck_4nodes_node_335M_gemini_replicas_legacy.sh 0 2 3 --train-iters 50
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
# Gemini Replicas 网络环境变量
# ---------------------------------------------------------------------------
# 数据传输走的高速网络接口名
export GEMINI_REPLICAS_INTERFACE=$NETIFACES_INTERFACE
# 各 rank 监听的基础 IP（通常设为 MASTER_ADDR，各 rank 用 GEMINI_REPLICAS_BASE_PORT + rank*100 派生端口）
# export GEMINI_REPLICAS_BASE_IP=$MASTER_ADDR
# 基础端口号，每个 rank 占用 100 个端口范围以避免冲突
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

# ---- 节点 rank 解析（第一个参数） ----
NODE_RANK=0
if [ -n "$1" ]; then
    if [[ "$1" =~ ^[0-9]+$ ]]; then
        NODE_RANK=$1
        shift
    fi
fi

# ---- GPU ID 解析（后续连续数字参数） ----
# 与 FRCheck 脚本相同：收集所有连续数字参数作为 GPU ID，
# 遇到第一个非数字参数停止收集，之后的参数透传给训练脚本。
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

VOCAB_FILE="/workspace/Megatron-LM/pre-tests/bert/bert_data/vocab.txt"

TENSORBOARD_LOGS_PATH="/workspace/Megatron-LM/pre-tests/bert/7B/bert-7b-0/logs"
CHECKPOINT_PATH="/dev/shm/models/bert-7b-0-gemini-3-replicas"
DATA_PATH="/workspace/Megatron-LM/pre-tests/bert/bert_data/wiki_text_sentence"
DATA_CACHE_PATH="${DATA_CACHE_PATH:-/workspace/Megatron-LM/pre-tests/bert/bert_data/cache}"

SHM_PKT="/dev/shm/shm_pkt"

ARGS_TO_PASS=("$@")

# 模型固定参数
HIDDEN_SIZE=4096
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
# Gemini Replicas Torch Legacy 核心参数说明
# =============================================================================

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 100
    --save $CHECKPOINT_PATH
    #--load $CHECKPOINT_PATH          # 取消注释以测试 load
    --eval-iters 1
    --tensorboard-dir $TENSORBOARD_LOGS_PATH

    # ---------------------------------------------------------------------------
    # 必选：启用 Gemini Replicas torch legacy checkpoint
    # ---------------------------------------------------------------------------
    # 启用 Gemini Replicas（替代原有的 --use-gemini，后者是两副本 EC 风格配对）
    --use-gemini-replicas
    # 启用优化路径：使用连续 CPU buffer + C++ ASIO/RDMA 网络传输，跳过 torch.save 序列化开销
    --use-gemini-replicas-optimized

    # ---------------------------------------------------------------------------
    # 副本数：每个 rank 的数据在组内存放 N 份（含本地）
    # ---------------------------------------------------------------------------
    --gemini-replicas-num 3          # 默认 3。设为 2 即两副本，设为 N 即 N 副本
                                        # 在组内 round-robin 轮询放置副本
                                        # 容错能力 = num_replicas - 1 个 rank 同时故障

    # ---------------------------------------------------------------------------
    # 分组大小：将 world 划分为独立组，副本仅在组内轮询
    # ---------------------------------------------------------------------------
    --gemini-replicas-group-size 4   # 默认 None（全局轮询，不做分组）
                                        # 设 8 则每 8 个 rank 一组，每组独立
                                        # 必须能被 world_size 整除
                                        # 独立于节点数和每节点 rank 数，但数学上要求
                                        #   num_nodes % group_size == 0 才能启用
                                        #   跨节点交错排列（FRCheck 同款布局）
                                        # 每组内每个 rank 来自不同物理节点

    # ---------------------------------------------------------------------------
    # 传输方式：RDMA（InfiniBand）或 TCP（ASIO）
    # ---------------------------------------------------------------------------
    --use-rdma                       # 启用 RDMA（默认走 TCP/ASIO）
                                        # 开启后 send/recv 双向走 InfiniBand verbs
                                        # 需要硬件支持 + 提前注册内存

    # ---------------------------------------------------------------------------
    # 恢复模式（load 时使用，save 不需要）
    # ---------------------------------------------------------------------------
    # 方式 1 — 自动检测（文件缺失 = 故障）:
    #   删掉要模拟故障的 rank 的 main 文件，然后 load。
    #   系统自动检测缺失 → 组内选 sender 通过 torch.distributed 发送副本数据。
    #   恢复完成后自动重新生成 main 文件。
    #   测试方法：save 完成后删故障 rank 的 gemini_replicas_main_rank*.pt，
    #            然后带相同参数 load。
    #
    #   --use-gemini-replicas-hardware-failure

    # 方式 2 — 指定故障 rank（不删文件，精确控制）:
    #   指定哪些 rank 模拟故障。这些 rank 即使 main 文件存在也会走恢复路径。
    #   测试方法：save 完成后直接 load，加下面参数。
    #   示例："2,3" 表示 rank2 和 rank3 当作故障处理。
    #
    #   --gemini-replicas-recovery-rank 2,3

    # ---------------------------------------------------------------------------
    # ckpt 格式：必须用 torch（legacy 路径）
    # ---------------------------------------------------------------------------
    --ckpt-format torch               # 必须！legacy 路径要求 torch 格式
                                        # 不能用 torch_dist（那是分布式 checkpoint 路径）
    # 注意：不能设置 --use-dist-ckpt，否则会走 GLOBAL 类型而非 LEGACY
    # 程序内部检测：ckpt_type=LEGACY + ckpt_format=torch 才允许 gemini_replicas

    --save-embeddings-separately
    # --no-save-optim                 # 取消注释以跳过 optimizer 保存
    # --no-load-optim                 # 取消注释以跳过 optimizer 加载
)

# =============================================================================
# 参数组合速查
# =============================================================================
#
# 场景 1 — 全局 3 副本（默认，小规模测试）:
#   --use-gemini-replicas --use-gemini-replicas-optimized --ckpt-format torch
#
# 场景 2 — 全局 2 副本（等同于原 Gemini 两副本，round-robin 配对）:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-num 2 --ckpt-format torch
#
# 场景 3 — 8 节点各 8 GPU，组大小 8，组内 4 副本:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-num 4 --gemini-replicas-group-size 8 --ckpt-format torch
#
# 场景 4 — 硬件故障恢复（删文件后自动检测）:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --use-gemini-replicas-hardware-failure --use-rdma --ckpt-format torch
#
# 场景 5 — 指定 rank2,3 故障（不删文件，精确控制）:
#   --use-gemini-replicas --use-gemini-replicas-optimized
#   --gemini-replicas-recovery-rank 2,3 --ckpt-format torch
#
# =============================================================================

mkdir -p logs
mkdir -p logs/csv

# -------------------------------------------------------------------------
if [ "${PRINT_CMD:-0}" != "0" ]; then
    echo "Would run (Node $NODE_RANK): PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM torchrun ${DISTRIBUTED_ARGS[@]} pretrain_bert.py ${GPT_ARGS[@]} ${DATA_ARGS[@]} ${MODEL_PARALLEL_ARGS[@]} ${EVAL_AND_LOGGING_ARGS[@]} --distributed-backend nccl ${ARGS_TO_PASS[@]}"
    exit 0
fi
# -------------------------------------------------------------------------

echo "Starting Node $NODE_RANK with GPUs $CUDA_VISIBLE_DEVICES (Gemini Replicas Legacy)"
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
    --distributed-backend nccl \
    ${ARGS_TO_PASS[@]}
