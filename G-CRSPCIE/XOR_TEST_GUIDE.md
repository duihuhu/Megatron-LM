# XOR 测试功能使用指南

## 📋 概述

G-CRSPCIE 现在支持独立的 XOR 操作测试，可以像 EC 编码一样配置线程块等参数。XOR 操作是 EC 编码的核心步骤之一，但这里提供了独立的测试接口，方便单独测试 XOR 性能。

## 🎯 XOR vs EC 编码

### XOR 操作
- **功能**: 简单地将 k 个输入块进行 XOR 运算，生成 1 个输出块
- **复杂度**: O(k) - 只需要 k-1 次 XOR 操作
- **用途**: 测试纯 XOR 性能，不涉及 Galois 域乘法

### EC 编码
- **功能**: 使用编码矩阵进行编码，包含 XOR 和 Galois 域乘法
- **复杂度**: O(k*m*w) - 更复杂
- **用途**: 完整的纠删码编码

## 🚀 使用方法

### 基本用法

```bash
# 运行XOR测试（使用默认配置）
./test_gpu_tensor --test 4

# 自定义配置
./test_gpu_tensor --test 4 -k 8 -b 256 --threads 256

# 运行所有测试（包括XOR）
./test_gpu_tensor
```

### 配置参数

XOR 测试支持以下参数：

- `-k <num>`: 输入块数量（默认: 4）
- `-b <size>`: 每个块的大小，单位KB（默认: 64）
- `--threads <num>`: 每个线程块的线程数（默认: 128）
- `--blocks <num>`: GPU线程块数量（默认: 0 = 自动计算）

**注意**: XOR 测试不需要 `-m` 和 `-w` 参数（因为不涉及 EC 编码）

## 📊 使用示例

### 示例1: 基本XOR测试
```bash
# 测试4个块的XOR
./test_gpu_tensor --test 4 -k 4 -b 128
```

### 示例2: 测试不同线程配置
```bash
# 测试不同threads_per_block的性能
for t in 64 128 256 512; do
    echo "=== Testing threads=$t ==="
    ./test_gpu_tensor --test 4 -k 8 -b 256 --threads $t
done
```

### 示例3: 测试不同块数量
```bash
# 测试不同k值的性能
for k in 2 4 8 16; do
    echo "=== Testing k=$k ==="
    ./test_gpu_tensor --test 4 -k $k -b 256
done
```

### 示例4: 性能对比（XOR vs EC编码）
```bash
# XOR性能
echo "=== XOR Performance ==="
./test_gpu_tensor --test 4 -k 4 -b 256 --threads 256

# EC编码性能（相同配置）
echo "=== EC Encoding Performance ==="
./test_gpu_tensor --test 2 -k 4 -m 2 -b 256 --threads 256
```

## 🔧 API使用

### C API

```c
#include "GCRSCoding.h"

// 直接调用XOR kernel
int k = 4;
int threads_per_block = 256;
int blocks_per_grid = 100;
int workSizeInLong = block_size / sizeof(long);

char *gpu_input, *gpu_output;
cudaMalloc(&gpu_input, block_size * k);
cudaMalloc(&gpu_output, block_size);

// 执行XOR
gcrs_xor_coding(k, 0, gpu_input, gpu_output, 
                threads_per_block, blocks_per_grid, 
                workSizeInLong, stream);

// 或者使用测量函数
float time_elapsed;
gcrs_xor_measure(k, threads_per_block, blocks_per_grid, 
                 workSizeInLong, gpu_input, gpu_output, 
                 &time_elapsed);
```

## 📈 性能特点

### XOR 操作的优势
1. **更简单**: 只有 XOR 操作，没有 Galois 域乘法
2. **更快**: 通常比 EC 编码快 2-5 倍
3. **内存带宽友好**: 主要是内存操作，计算量小

### 性能调优建议

1. **线程数**: XOR 操作对线程数不敏感，64-512 都可以
2. **块大小**: 较大的块大小（256KB-1MB）通常性能更好
3. **块数量**: 确保有足够的块来充分利用 GPU

## 🎯 典型测试场景

### 场景1: 快速XOR基准测试
```bash
./test_gpu_tensor --test 4 -k 8 -b 512 --threads 256
```

### 场景2: 参数扫描
```bash
# 扫描不同k值
for k in 2 4 8 16 32; do
    ./test_gpu_tensor --test 4 -k $k -b 256 | grep "Throughput"
done

# 扫描不同线程配置
for t in 64 128 256 512; do
    ./test_gpu_tensor --test 4 -k 4 -b 256 --threads $t | grep "Throughput"
done
```

### 场景3: 与EC编码对比
```bash
# 创建对比脚本
cat > compare_xor_ec.sh << 'EOF'
#!/bin/bash
echo "=== XOR Performance ==="
./test_gpu_tensor --test 4 -k 4 -b 256 --threads 256 | grep "Throughput"
echo ""
echo "=== EC Encoding Performance ==="
./test_gpu_tensor --test 2 -k 4 -m 2 -b 256 --threads 256 | grep "Throughput"
EOF
chmod +x compare_xor_ec.sh
./compare_xor_ec.sh
```

## 📝 输出说明

XOR 测试会输出：
- 配置信息（k值、块大小）
- CUDA配置（线程数、块数）
- 平均执行时间（多次迭代）
- 吞吐量（MB/s）
- 结果验证

示例输出：
```
[Test 4] XOR Operation Test
---------------------------
✓ Configuration: k=4, block_size=256 KB
✓ GPU memory allocated (input=1024 KB, output=256 KB)
✓ GPU input data prepared (1048576 bytes)
✓ CUDA config: threads_per_block=256, blocks_per_grid=32
Running XOR operation (10 iterations)...
✓ XOR operation completed successfully
  Average time (10 iterations): 0.015 ms
  Throughput: 65536.00 MB/s
  Note: XOR is simpler than EC encoding, should be faster!
  ✓ XOR result verified (non-zero output)
```

## ⚠️ 注意事项

1. **k值限制**: k 值应该合理（通常 2-32），过大的 k 值可能导致性能下降
2. **内存对齐**: 块大小会自动对齐到 `sizeof(long)` 的倍数
3. **GPU内存**: 确保 GPU 有足够内存，特别是大 k 值和大块大小
4. **线程数**: 建议使用 32 的倍数（warp大小）

## 🔗 相关文档

- **参数配置**: 查看 `PARAMETERS.md`
- **测试使用**: 查看 `TEST_USAGE.md`
- **GPU零拷贝**: 查看 `GPU_TENSOR_GUIDE.md`

