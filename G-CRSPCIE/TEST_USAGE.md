# G-CRSPCIE 测试程序使用指南

## 📋 基本用法

### 查看帮助
```bash
./test_gpu_tensor --help
```

### 使用默认配置运行所有测试
```bash
./test_gpu_tensor
```

### 运行特定测试
```bash
# 只运行CPU编码测试
./test_gpu_tensor --test 1

# 只运行GPU零拷贝测试
./test_gpu_tensor --test 2

# 只运行手动零拷贝测试
./test_gpu_tensor --test 3
```

## 🎛️ 可配置参数

### 编码参数
- `-k <num>`: 数据块数量 (默认: 4)
- `-m <num>`: 校验块数量 (默认: 2)
- `-w <num>`: Galois域宽度 (默认: 4, 范围: 4-8)
- `-b <size>`: 每个块的大小，单位KB (默认: 64)

### 流水线参数
- `-t <num>`: 任务数量，用于流水线 (默认: 1)

### CUDA执行配置
- `--threads <num>`: 每个线程块的线程数 (默认: 128)
- `--blocks <num>`: GPU线程块数量 (默认: 0 = 自动计算)

### 测试选择
- `--test <num>`: 选择运行的测试 (0=全部, 1=CPU, 2=GPU零拷贝, 3=手动零拷贝, 4=XOR)

## 📝 使用示例

### 示例1: 基本配置测试
```bash
# 测试不同的k和m值
./test_gpu_tensor -k 8 -m 4 -b 128

# 测试不同的w值
./test_gpu_tensor -k 4 -m 2 -w 8

# 测试流水线效果
./test_gpu_tensor -k 4 -m 2 -t 4
```

### 示例2: CUDA线程配置测试
```bash
# 测试不同的线程数
./test_gpu_tensor --threads 64 --test 2
./test_gpu_tensor --threads 128 --test 2
./test_gpu_tensor --threads 256 --test 2
./test_gpu_tensor --threads 512 --test 2

# 手动设置线程块数量
./test_gpu_tensor --threads 256 --blocks 50 --test 2
```

### 示例3: 性能对比测试
```bash
# 测试不同配置的性能
echo "=== Testing threads_per_block ==="
for threads in 64 128 256 512; do
    echo "Threads: $threads"
    ./test_gpu_tensor --threads $threads --test 2 | grep "Throughput"
done

echo "=== Testing task_num ==="
for tasks in 1 2 4 8; do
    echo "Tasks: $tasks"
    ./test_gpu_tensor -t $tasks --test 1 | grep "Throughput"
done

echo "=== Testing block_size ==="
for size in 64 128 256 512; do
    echo "Block size: ${size}KB"
    ./test_gpu_tensor -b $size --test 2 | grep "Throughput"
done
```

### 示例4: XOR测试
```bash
# 运行XOR测试
./test_gpu_tensor --test 4 -k 8 -b 256 --threads 256

# 测试不同线程配置的XOR性能
for t in 64 128 256 512; do
    ./test_gpu_tensor --test 4 -k 4 -b 256 --threads $t
done
```

### 示例5: 完整配置测试
```bash
# 大数据量 + 高线程数 + 流水线
./test_gpu_tensor -k 8 -m 4 -w 8 -b 1024 -t 4 --threads 256 --test 0

# 小数据量 + 默认配置
./test_gpu_tensor -k 2 -m 1 -b 32 --test 0
```

## 🔍 输出说明

测试程序会输出：
1. **配置信息**: 显示所有使用的参数
2. **CUDA设备信息**: GPU型号和数量
3. **测试结果**: 每个测试的详细输出
   - Worker初始化状态
   - CUDA配置信息
   - 编码时间和吞吐量
4. **测试总结**: 通过/失败的测试数量

## 📊 性能调优建议

### 1. 测试线程配置
```bash
# 找出最优的threads_per_block
for t in 64 128 256 512; do
    echo "Testing threads=$t"
    ./test_gpu_tensor --threads $t -b 256 --test 2
done
```

### 2. 测试流水线效果
```bash
# 测试不同task_num对性能的影响
for t in 1 2 4 8; do
    echo "Testing task_num=$t"
    ./test_gpu_tensor -t $t -b 512 --test 1
done
```

### 3. 测试数据大小影响
```bash
# 测试不同block_size的性能
for b in 64 128 256 512 1024; do
    echo "Testing block_size=${b}KB"
    ./test_gpu_tensor -b $b --test 2
done
```

## 🎯 典型测试场景

### 场景1: 快速验证功能
```bash
./test_gpu_tensor -k 4 -m 2 -b 64 --test 0
```

### 场景2: 性能基准测试
```bash
./test_gpu_tensor -k 8 -m 4 -w 8 -b 1024 -t 4 --threads 256 --test 0
```

### 场景3: GPU零拷贝性能测试
```bash
./test_gpu_tensor -k 4 -m 2 -b 512 --threads 256 --test 2
```

### 场景4: 参数扫描
```bash
# 测试所有w值
for w in 4 5 6 7 8; do
    echo "=== w=$w ==="
    ./test_gpu_tensor -w $w -b 256 --test 2
done
```

## ⚠️ 注意事项

1. **线程数限制**: `--threads` 必须是32的倍数，如果不是会自动向上取整
2. **块大小对齐**: `-b` 指定的块大小会被自动对齐到 `w * sizeof(long)`
3. **内存限制**: 确保GPU有足够内存，特别是大数据量测试
4. **参数范围**:
   - k: 1-45
   - m: 1-4
   - w: 4-8
   - threads: 1-1024

## 🔧 故障排除

### 问题: "Failed to allocate GPU memory"
**解决**: 减小 `-b` 参数或 `-k` 参数

### 问题: "Invalid k,m,w"
**解决**: 检查参数范围，确保 k>=1, m>=1, w在4-8之间

### 问题: 性能异常低
**解决**: 
- 检查 `--threads` 是否是32的倍数
- 尝试不同的 `-t` 值
- 确保GPU没有被其他程序占用

