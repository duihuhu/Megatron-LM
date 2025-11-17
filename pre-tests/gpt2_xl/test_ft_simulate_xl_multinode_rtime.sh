#!/bin/bash
# bash test_ft_simulate_multinode_rtime.sh  0 2 1
# bash test_ft_simulate_multinode_rtime.sh 1 2 1

# =============================================================================
# 使用 ft_launcher 在单机多卡上模拟多节点训练
# 用途: 测试多节点容错逻辑，但只用一台机器
# =============================================================================
export CUDA_DEVICE_MAX_CONNECTIONS=1
export DEBUG_COMMUNICATE=1
export DEBUG_PARALLEL_STATES=1

export NCCL_DEBUG=INFO
export NCCL_DEBUG_FILE=./nccl.log
export NCCL_DEBUG_SUBSYS=ALL

# =============================================================================
# 模拟多节点配置
# =============================================================================
# 参数说明:
#   $1: 模拟的节点rank (0, 1, 2, ...)
#   $2: 总节点数 (默认2)
#   $3: 每个模拟节点的GPU数 (默认1)

NODE_RANK=${1:-0}           # 当前模拟节点的rank
TOTAL_NODES=${2:-2}         # 模拟的总节点数
GPUS_PER_NODE=${3:-1}       # 每个模拟节点使用的GPU数
export HOSTNAME="simnode${NODE_RANK}"
export CUDA_VISIBLE_DEVICES=$NODE_RANK
# 验证参数
if [ -z "$1" ]; then
    echo "错误: 需要指定节点rank"
    echo "用法: $0 <node_rank> [total_nodes] [gpus_per_node]"
    echo "示例:"
    echo "  终端1: $0 0 2 1  # 模拟节点0，共2个节点，每节点1GPU"
    echo "  终端2: $0 1 2 1  # 模拟节点1，共2个节点，每节点1GPU"
    exit 1
fi

# =============================================================================
# 网络配置 - 单机模拟多节点
# =============================================================================
MASTER_ADDR=127.0.0.1  # 单机使用 localhost
MASTER_PORT=6000
RDZV_PORT=29500

# =============================================================================
# 关于端口绑定警告的说明
# =============================================================================
# 在 c10d rendezvous 机制中，节点协调采用客户端-服务器模式：
#
# 1. 服务器模式（节点0）：
#    - 第一个到达的节点（节点0）会绑定到 rdzv_endpoint 端口（29500）
#    - 充当 rendezvous 服务器，等待其他节点连接
#    - 负责协调所有节点的 rendezvous 过程
#
# 2. 客户端模式（节点1, 2, ...）：
#    - 后续节点在启动时会先尝试绑定端口（检查是否应该成为服务器）
#    - 如果端口已被占用（说明节点0已经是服务器），绑定会失败
#    - 然后自动切换为客户端模式，连接到节点0的服务器
#    - 客户端模式：主动连接到服务器，而不是监听端口等待连接
#
# 3. 为什么会出现警告？
#    - 这是 c10d 的设计机制：每个节点都会先尝试绑定端口
#    - 如果绑定失败（端口已被占用），说明已经有服务器了，就切换为客户端
#    - 在单机多卡模拟多节点时，所有"节点"实际在同一台机器上
#    - 因此节点1+ 的端口绑定尝试会失败，这是正常的
#
# 4. 这与单机多卡模拟有关吗？
#    - 是的，但即使在真实多节点环境中也可能出现类似的警告
#    - 因为 c10d 的设计就是让每个节点都尝试绑定，然后根据结果决定角色
#    - 在真实多节点环境中，如果节点1比节点0先启动，节点1会成为服务器
#    - 在单机模拟中，由于是同一台机器，端口冲突更明显，但机制相同
#
# 警告信息可以安全忽略，不会影响训练。

# 网络接口 - 使用本地回环
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo

WORLD_SIZE=$((TOTAL_NODES * GPUS_PER_NODE))

# =============================================================================
# GPU分配 - 关键！每个模拟节点使用不同的GPU
# =============================================================================
# 根据节点rank分配GPU
# 节点0: GPU 0
# 节点1: GPU 1
# 节点2: GPU 2
# 以此类推...

if [ $GPUS_PER_NODE -eq 1 ]; then
    # 每个节点1个GPU - 简单映射
    GPU_ID=$NODE_RANK
    export CUDA_VISIBLE_DEVICES=$GPU_ID
    echo "节点 $NODE_RANK 使用 GPU: $GPU_ID"
else
    # 每个节点多个GPU
    START_GPU=$((NODE_RANK * GPUS_PER_NODE))
    END_GPU=$((START_GPU + GPUS_PER_NODE - 1))
    GPU_LIST=$(seq -s, $START_GPU $END_GPU)
    export CUDA_VISIBLE_DEVICES=$GPU_LIST
    echo "节点 $NODE_RANK 使用 GPU: $GPU_LIST"
fi

# 验证GPU分配
REQUIRED_GPUS=$((TOTAL_NODES * GPUS_PER_NODE))
AVAILABLE_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)

if [ $AVAILABLE_GPUS -lt $REQUIRED_GPUS ]; then
    echo "警告: 可用GPU ($AVAILABLE_GPUS) 少于需要的GPU ($REQUIRED_GPUS)"
    echo "当前配置: $TOTAL_NODES 个节点 × $GPUS_PER_NODE GPU/节点 = $REQUIRED_GPUS GPU"
    echo "建议: 减少节点数或每节点GPU数"
fi

# =============================================================================
# 路径配置
# =============================================================================
VOCAB_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-vocab.json"
MERGE_FILE="/workspace/Megatron-LM/pre-tests/gpt2/data/gpt2-merges.txt"

TENSORBOARD_LOGS_PATH="/workspace/models/gpt2-xl-ft-simnode/logs"
CHECKPOINT_PATH="/dev/shm/gpt2-xl-ft-simnode"
DATA_PATH="/workspace/models/gpt2-xl-0/codeparrot_content_document"

# 创建必要的目录
mkdir -p $CHECKPOINT_PATH
mkdir -p $TENSORBOARD_LOGS_PATH
mkdir -p logs
mkdir -p logs/csv

# =============================================================================
# 容错配置参数
# =============================================================================
FT_TIMEOUT_SETUP=600
FT_TIMEOUT_STEP=300
FT_TIMEOUT_CHECKPOINTING=420
FT_TIMEOUT_OUT_OF_SECTION=300

# 故障模拟（可选）
# export FT_SIM_FAULT_DESC="rank_killed;1;60.0"  # 60秒后kill rank 1

# =============================================================================
# 模型和训练配置
# =============================================================================
HIDDEN_SIZE=1600
NUM_ATTENTION_HEADS=25
SEQ_LENGTH=1024
MAX_POSITION_EMBEDDINGS=1024
MICRO_BATCH_SIZE=4
# GLOBAL_BATCH_SIZE=16
# =============================================================================
# Megatron 训练参数
# =============================================================================
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
    # --global-batch-size $GLOBAL_BATCH_SIZE 
    --lr 0.00015 
    --train-iters 1000
    --lr-decay-iters 320000 
    --lr-decay-style cosine 
    --min-lr 1.5e-5 
    --weight-decay 1e-2 
    --lr-warmup-fraction 0.01 
    --clip-grad 1.0 
    --fp16 
    --tokenizer-type GPT2BPETokenizer 
    --use-mcore-models 
    --transformer-impl transformer_engine 
    --no-scatter-gather-tensors-in-pipeline 
    --num-layers 48
    --optimizer adam
    --loss-scale-window 1000
    --initial-loss-scale 4096
    --min-loss-scale 1.0
    --hysteresis 2
)

# 模型并行配置 - 根据每节点GPU数量
if [ $GPUS_PER_NODE -gt 1 ]; then
    # 多GPU per node: 使用张量并行
    TENSOR_PARALLEL=$GPUS_PER_NODE
    PIPELINE_PARALLEL=1
else
    # 单GPU per node: 可以使用流水线并行
    TENSOR_PARALLEL=1
    PIPELINE_PARALLEL=$TOTAL_NODES  # 或者设为1，取决于需求
fi

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size $TENSOR_PARALLEL
    --pipeline-model-parallel-size $PIPELINE_PARALLEL
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 1
    --save-interval 1
    --eval-interval 50
    --save $CHECKPOINT_PATH 
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --ckpt-format torch_dist
    --use-eccheck
    # --rerun-mode disabled
)

# =============================================================================
# 容错功能参数
# =============================================================================
FT_ARGS=(
    --enable-ft-package
    --calc-ft-timeouts
)

# =============================================================================
# ft_launcher 参数配置 - 模拟多节点
# =============================================================================
FT_LAUNCHER_ARGS=(
    # Rendezvous 配置
    --rdzv_backend=c10d
    --rdzv_endpoint=${MASTER_ADDR}:${RDZV_PORT}
    --rdzv_id=megatron_gpt_simulated_multinode  # 作业唯一ID
    
    # 多节点配置 - 关键！
    --nnodes=${TOTAL_NODES}:${TOTAL_NODES}  # 最小:最大节点数
    --nproc-per-node=${GPUS_PER_NODE}       # 每节点进程数
    --node-rank=${NODE_RANK}                # 当前节点rank（重要！）
    
    # 容错参数
    --ft-param-rank_section_timeouts=setup:${FT_TIMEOUT_SETUP},step:${FT_TIMEOUT_STEP},checkpointing:${FT_TIMEOUT_CHECKPOINTING}
    --ft-param-rank_out_of_section_timeout=${FT_TIMEOUT_OUT_OF_SECTION}
    --ft-param-rank_heartbeat_timeout=60
    
    # 最大重启次数
    --max-restarts=3
    
    # 日志配置 - 每个节点独立的日志
    --log-dir=./ft_logs/node_${NODE_RANK}
)

# =============================================================================
# 设置Python路径
# =============================================================================
export PYTHONPATH=$PYTHONPATH:/workspace/Megatron-LM
export USE_FLASH_ATTN=1
export NVTE_SYNC_P2P=1

# =============================================================================
# 启动容错训练
# =============================================================================
echo "=========================================================================="
echo "在单机上模拟多节点容错分布式训练"
echo "=========================================================================="
echo "配置摘要:"
echo "  模拟节点总数: ${TOTAL_NODES}"
echo "  当前节点Rank: ${NODE_RANK}"
echo "  每节点GPU数: ${GPUS_PER_NODE}"
echo "  总进程数: ${WORLD_SIZE}"
echo "  当前节点GPU: ${CUDA_VISIBLE_DEVICES}"
echo ""
echo "网络配置:"
echo "  Master地址: ${MASTER_ADDR}:${MASTER_PORT}"
echo "  Rendezvous: ${MASTER_ADDR}:${RDZV_PORT}"
echo "  网络接口: ${NCCL_SOCKET_IFNAME}"
echo ""
echo "模型并行:"
echo "  Tensor并行: ${TENSOR_PARALLEL}"
echo "  Pipeline并行: ${PIPELINE_PARALLEL}"
echo ""
echo "存储路径:"
echo "  Checkpoint: ${CHECKPOINT_PATH}"
echo "  日志目录: ./ft_logs/node_${NODE_RANK}"
echo ""
echo "容错超时配置:"
echo "  Setup: ${FT_TIMEOUT_SETUP}s"
echo "  Step: ${FT_TIMEOUT_STEP}s"
echo "  Checkpointing: ${FT_TIMEOUT_CHECKPOINTING}s"
echo "  Out-of-section: ${FT_TIMEOUT_OUT_OF_SECTION}s"
echo "=========================================================================="
echo ""

# 等待用户确认（可选）
if [ "$NODE_RANK" -eq 0 ]; then
    echo "提示: 请在其他终端启动其他节点"
    for i in $(seq 1 $((TOTAL_NODES - 1))); do
        echo "  终端$((i+1)): bash $0 $i $TOTAL_NODES $GPUS_PER_NODE"
    done
    echo ""
    read -p "按回车键开始启动节点 $NODE_RANK..." -r
fi

echo "启动节点 $NODE_RANK ..."
echo ""

# 执行 ft_launcher
ft_launcher \
    ${FT_LAUNCHER_ARGS[@]} \
    pretrain_gpt.py \
    ${GPT_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${FT_ARGS[@]} \
    --distributed-backend nccl

# =============================================================================
# 训练结束
# =============================================================================
echo "=========================================================================="
echo "节点 ${NODE_RANK} 训练任务结束"
echo "=========================================================================="

