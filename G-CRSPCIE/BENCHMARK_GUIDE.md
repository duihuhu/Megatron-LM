# GPU XOR vs EC 性能测试指南

## 📋 概述

本测试套件用于对比纯GPU计算中XOR和EC编码的性能，编码参数固定为 **k=2, m=1**。

## 🚀 快速开始

### 1. 编译测试程序

```bash
cd /workspace/Megatron-LM/G-CRSPCIE
make test
```

### 2. 运行完整测试套件

```bash
./benchmark_xor_vs_ec.sh
```

这会运行多个测试场景并生成CSV结果文件。

### 3. 分析结果

```bash
python3 analyze_benchmark.py
```

或者直接查看CSV文件：

```bash
cat benchmark_results_*.csv | column -t -s,
```

## 📊 测试场景

脚本包含以下测试场景：

### Test 1: 变化块大小
- 固定线程数: 128
- 固定任务数: 1
- 变化块大小: 64KB, 128KB, 256KB, 512KB, 1MB, 2MB
- 对比XOR和EC性能

### Test 2: 变化线程数
- 固定块大小: 256KB
- 固定任务数: 1
- 变化线程数: 64, 128, 256, 512
- 对比XOR和EC性能

### Test 3: 变化流水线深度（仅EC）
- 固定块大小: 256KB
- 固定线程数: 128
- 变化任务数: 1, 2, 4
- 测试流水线对EC性能的影响

### Test 4: 大块大小性能
- 固定线程数: 256
- 固定任务数: 1
- 变化块大小: 512KB, 1MB, 2MB, 4MB
- 测试大数据量下的性能

### Test 5: 最优配置对比
- 测试几种推荐配置
- 对比XOR和EC在不同配置下的表现

## 📈 输出文件

### CSV结果文件
- 文件名格式: `benchmark_results_YYYYMMDD_HHMMSS.csv`
- 包含列: Test Type, Block Size, Threads Per Block, Blocks Per Grid, Task Num, Avg Time, Throughput

### 日志文件
- 文件名格式: `benchmark_log_YYYYMMDD_HHMMSS.log`
- 包含所有测试的详细输出

## 🔍 手动测试示例

如果需要手动测试特定配置：

### XOR测试
```bash
# 基本XOR测试
./test_gpu_tensor --test 4 -k 2 -b 256 --threads 128

# 自定义块数量
./test_gpu_tensor --test 4 -k 2 -b 512 --threads 256 --blocks 100
```

### EC编码测试
```bash
# 基本EC测试
./test_gpu_tensor --test 2 -k 2 -m 1 -w 4 -b 256 --threads 128

# 带流水线
./test_gpu_tensor --test 2 -k 2 -m 1 -w 4 -b 256 -t 4 --threads 128
```

## 📊 结果解读

### 性能指标
- **Avg Time (ms)**: 平均执行时间，越小越好
- **Throughput (MB/s)**: 吞吐量，越大越好
- **Speedup**: EC相对于XOR的倍数（EC通常比XOR慢）

### 预期结果
- XOR通常比EC快 **2-5倍**（因为EC需要Galois域乘法）
- 大块大小通常性能更好（减少kernel启动开销）
- 线程数对性能影响取决于GPU架构
- 流水线对EC有帮助（可以重叠传输和计算）

## 🔧 自定义测试

### 修改测试参数

编辑 `benchmark_xor_vs_ec.sh`，修改以下变量：

```bash
# 块大小列表（KB）
BLOCK_SIZES=(64 128 256 512 1024 2048)

# 线程数列表
THREADS_PER_BLOCK=(64 128 256 512)

# 流水线深度列表
TASK_NUMS=(1 2 4)
```

### 添加新测试场景

在脚本末尾添加新的测试函数调用：

```bash
# 自定义测试
run_test "XOR" 1024 256 0 1
run_test "EC" 1024 256 0 1
```

## 📝 注意事项

1. **GPU温度**: 长时间运行可能导致GPU过热，脚本中包含sleep来缓解
2. **内存**: 确保GPU有足够内存，特别是大块大小测试
3. **独占使用**: 建议在GPU独占使用时运行测试，避免其他进程干扰
4. **多次运行**: 建议多次运行取平均值，减少随机波动

## 🐛 故障排除

### 问题: 测试程序找不到
```bash
# 确保已编译
make clean && make test
```

### 问题: CUDA错误
- 检查GPU是否可用: `nvidia-smi`
- 检查CUDA版本: `nvcc --version`

### 问题: 内存不足
- 减小块大小
- 减少并发任务数

## 📚 相关文档

- `ARCHITECTURE.md`: 技术架构文档
- `TEST_USAGE.md`: 测试程序使用指南
- `XOR_TEST_GUIDE.md`: XOR测试详细指南

