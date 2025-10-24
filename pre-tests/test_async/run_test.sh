#!/bin/bash
# 运行分布式异步 checkpoint 测试
# 两个独立的程序，模拟分布式训练场景

echo "======================================================================"
echo "分布式异步 Checkpoint 测试（三种方案对比）"
echo "======================================================================"
echo ""
echo "方案 1: 原生 torch (torch.to + dist.send)"
echo "方案 2: CPUMemoryPool + dist.send (共享内存 + PyTorch 序列化)"
echo "方案 3: CPUMemoryPool + Raw Bytes (共享内存 + 原始字节) ⭐ 最优"
echo ""
echo "启动两个进程 (rank 0 和 rank 1)..."
echo ""

# 设置环境变量
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

# 测试模式（可以修改为 torch, pool, pool-raw, 或 all）
MODE=${1:-all}

echo "测试模式: $MODE"
echo ""

# 启动 rank 0
echo "启动 Rank 0..."
python test_async_checkpoint.py --rank 0 --world-size 2 --mode $MODE &
PID0=$!

# 等待一下，确保 rank 0 先启动
sleep 1

# 启动 rank 1  
echo "启动 Rank 1..."
python test_async_checkpoint.py --rank 1 --world-size 2 --mode $MODE &
PID1=$!

echo ""
echo "等待两个进程完成..."
echo "  Rank 0 PID: $PID0"
echo "  Rank 1 PID: $PID1"

# 等待两个进程
wait $PID0
wait $PID1

echo ""
echo "======================================================================"
echo "测试完成！"
echo "======================================================================"
echo ""
echo "提示："
echo "  • 查看 Rank 0 的输出获取详细性能对比"
echo "  • 方案 3 (CPUMemoryPool + Raw Bytes) 应该是最快的"
echo ""

