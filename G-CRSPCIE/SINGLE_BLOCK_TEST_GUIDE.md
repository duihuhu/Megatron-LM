# GPU单线程块性能测试指南

## 📋 概述

本测试用于评估GPU在**单线程块（minimal resources）**配置下的EC和XOR性能，模拟GPU在pretrain时匀出来的性能。

## 🎯 使用场景

当GPU正在运行大模型pretrain时，通常只有少量计算资源可用于其他任务。本测试通过限制为**1个线程块，32个线程**（warp大小）来模拟这种场景。

## 🚀 快速开始

### 1. 编译测试程序

```bash
cd /workspace/Megatron-LM/G-CRSPCIE
make test
```

### 2. 运行单个测试

```bash
# 单线程块EC测试
./test_gpu_tensor --test 5 -k 2 -m 1 -w 4 -b 256

# 单线程块XOR测试
./test_gpu_tensor --test 6 -k 2 -b 256
```

### 3. 运行完整测试套件

```bash
# 运行所有测试（包括单线程块测试）
./test_gpu_tensor --test 0 -k 2 -m 1 -w 4 -b 256

# 或使用专门的脚本
./test_single_block.sh
```

## 📊 测试配置

### 单线程块配置
- **线程块数量**: 1（固定）
- **每个块的线程数**: 32（warp大小，GPU的最小高效单位）
- **总计算资源**: 32个线程

### 为什么选择这个配置？

1. **Warp大小**: 32是GPU的warp大小，是最小的高效执行单位
2. **单块**: 模拟GPU只有很少资源可用的情况
3. **真实场景**: 接近GPU在pretrain时剩余的资源量

## 🔍 测试结果解读

### 性能指标

- **Avg Time (ms)**: 平均执行时间
- **Throughput (MB/s)**: 吞吐量
- **资源占用**: 1 block × 32 threads = 32 threads total

### 预期结果

单线程块测试的性能特点：
- **执行时间**: 通常比全资源测试慢很多（因为只有32个线程）
- **吞吐量**: 较低，但可以评估GPU在受限资源下的编码能力
- **适用场景**: 适合评估GPU在pretrain时是否能同时进行编码操作

## 📈 性能对比

### 单线程块 vs 全资源对比

| 配置 | 线程数 | 块数 | 用途 |
|------|--------|------|------|
| **单线程块** | 32 | 1 | 模拟pretrain时的剩余资源 |
| **全资源** | 128-512 | 自动计算 | 评估GPU最大性能 |

### 示例输出

```
Single-Block EC Encoding Test
✓ Single-block config: threads_per_block=32, blocks_per_grid=1
✓ This simulates GPU with minimal available resources
  Average time (10 iterations): X.XXX ms
  Throughput: XXX.XX MB/s
```

## 🔧 自定义测试

### 修改测试参数

编辑 `test_single_block.sh`，修改以下变量：

```bash
# 块大小列表（KB）
BLOCK_SIZES=(64 128 256 512 1024 2048 4096 8192)
```

### 手动测试不同块大小

```bash
# 测试不同块大小的单线程块性能
for size in 64 128 256 512 1024; do
    echo "Testing block size: ${size} KB"
    ./test_gpu_tensor --test 5 -k 2 -m 1 -w 4 -b $size
    ./test_gpu_tensor --test 6 -k 2 -b $size
done
```

## 📝 输出文件

### CSV结果文件
- 文件名格式: `single_block_results_YYYYMMDD_HHMMSS.csv`
- 包含列: Test Type, Block Size, Avg Time, Throughput, Notes

### 日志文件
- 文件名格式: `single_block_log_YYYYMMDD_HHMMSS.log`
- 包含所有测试的详细输出

## 🎯 实际应用场景

### 场景1: Pretrain时的编码性能评估

```bash
# 评估GPU在pretrain时能否同时进行编码
./test_gpu_tensor --test 5 -k 2 -m 1 -w 4 -b 1024
```

### 场景2: 资源受限环境下的性能基准

```bash
# 建立性能基准
./test_single_block.sh
```

### 场景3: XOR vs EC性能对比（受限资源）

```bash
# 对比单线程块下XOR和EC的性能差异
./test_gpu_tensor --test 5 -k 2 -m 1 -w 4 -b 512  # EC
./test_gpu_tensor --test 6 -k 2 -b 512              # XOR
```

## ⚠️ 注意事项

1. **资源限制**: 单线程块测试使用最小的GPU资源，性能会比全资源测试慢很多
2. **真实场景**: 实际pretrain时，GPU的资源分配可能更复杂，这只是一个简化模型
3. **块大小**: 较小的块大小可能无法充分利用单线程块，建议测试多个块大小
4. **对比分析**: 建议同时运行全资源测试和单线程块测试，进行对比分析

## 📚 相关文档

- `TEST_USAGE.md`: 测试程序使用指南
- `BENCHMARK_GUIDE.md`: 性能基准测试指南
- `ARCHITECTURE.md`: 技术架构文档

## 🔗 示例：完整测试流程

```bash
# 1. 编译
make test

# 2. 运行单线程块测试套件
./test_single_block.sh

# 3. 查看结果
cat single_block_results_*.csv | column -t -s,

# 4. 对比全资源测试（可选）
./benchmark_xor_vs_ec.sh
```

这样可以帮助你全面了解GPU在不同资源分配下的编码性能！



