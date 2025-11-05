# G-CRSPCIE 可控制参数完整列表

## 📊 当前可控制的参数

### 1. 编码参数（初始化时设置）
- **k**: 数据块数量 (1-45)
- **m**: 校验块数量 (1-4)  
- **w**: Galois域宽度 (4-8)
- **block_size**: 每个数据/校验块的大小（字节）
- **task_num**: 任务数量（用于流水线）

### 2. CUDA执行配置（新增，可配置）
- **threads_per_block**: 每个线程块的线程数
  - 默认值: 128 (MAX_THREAD_NUM)
  - 范围: 1-1024
  - 建议: 32的倍数（warp大小）
  - API: `PErasureWorkerSetThreadsPerBlock(worker, threads)`
  - 获取: `PErasureWorkerGetThreadsPerBlock(worker)`

- **blocks_per_grid**: GPU线程块数量
  - 默认值: 0（自动计算）
  - 范围: >= 0（0表示自动计算）
  - API: `PErasureWorkerSetBlocksPerGrid(worker, blocks)`
  - 获取: `PErasureWorkerGetBlocksPerGrid(worker)`

### 3. 内存传输控制
- **use_external_data_dev_buf**: 是否使用外部输入GPU缓冲区（跳过H2D）
- **use_external_code_dev_buf**: 是否使用外部输出GPU缓冲区（跳过D2H）
- **skip_d2h_transfer**: 是否跳过Device-to-Host传输（零拷贝模式）

## 🎯 使用示例

### 示例1: 设置自定义线程块配置

```c
struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);

// 设置每个线程块的线程数为256（而不是默认的128）
PErasureWorkerSetThreadsPerBlock(worker, 256);

// 设置固定的线程块数量（可选，通常让系统自动计算）
// PErasureWorkerSetBlocksPerGrid(worker, 100);

// 查看当前配置
int threads = PErasureWorkerGetThreadsPerBlock(worker);
int blocks = PErasureWorkerGetBlocksPerGrid(worker);
printf("Threads per block: %d, Blocks per grid: %d\n", threads, blocks);

// 执行编码
fullDuplexRunEncode(worker);
```

### 示例2: 性能调优 - 尝试不同的线程配置

```c
// 测试不同的线程配置
int thread_configs[] = {64, 128, 256, 512};
int num_configs = sizeof(thread_configs) / sizeof(thread_configs[0]);

for (int i = 0; i < num_configs; i++) {
    PErasureWorkerSetThreadsPerBlock(worker, thread_configs[i]);
    
    // 执行编码并测量性能
    fullDuplexRunEncode(worker);
    PErasureWorkerCalculateRecord(worker);
    double time = PErasureWorkerGetEncodeTimeConsume(worker);
    
    printf("Threads: %d, Time: %.3f ms\n", thread_configs[i], time);
}
```

### 示例3: 手动控制线程块数量

```c
// 手动设置线程块数量（通常不推荐，除非有特殊需求）
PErasureWorkerSetBlocksPerGrid(worker, 50);

// 执行编码
fullDuplexRunEncode(worker);
```

## 📝 参数影响分析

### threads_per_block 的影响

| 值 | 优点 | 缺点 | 适用场景 |
|---|---|---|---|
| 64 | 占用寄存器少，适合复杂kernel | GPU利用率可能不足 | 寄存器受限的kernel |
| 128 (默认) | 平衡性能和资源占用 | - | 大多数场景 |
| 256 | 更高的GPU利用率 | 可能受寄存器限制 | 简单kernel，大数据量 |
| 512 | 最大化GPU利用率 | 受限于寄存器数量 | 非常简单的kernel |

### blocks_per_grid 的影响

- **自动计算（默认）**: 根据数据大小和threads_per_block自动计算，通常是最优的
- **手动设置**: 仅在特殊场景下需要，例如：
  - 需要精确控制GPU占用率
  - 调试特定配置
  - 与其他kernel协调资源分配

### task_num 的影响

- **1**: 简单模式，无流水线
- **2-4**: 平衡流水线和资源占用
- **4-8**: 最大化PCIe传输和计算的流水线重叠

## 🔧 参数调优建议

### 性能调优流程

1. **固定其他参数，测试threads_per_block**
   ```c
   // 测试64, 128, 256, 512
   for (int t = 64; t <= 512; t *= 2) {
       PErasureWorkerSetThreadsPerBlock(worker, t);
       // 测量性能
   }
   ```

2. **固定threads_per_block，测试task_num**
   ```c
   // 测试1, 2, 4, 8
   for (int tasks = 1; tasks <= 8; tasks *= 2) {
       // 重新初始化worker
       // 测量性能
   }
   ```

3. **测试不同的w值**（如果编码需求允许）
   ```c
   // w=4通常更快，w=8可能提供更好的编码效率
   ```

### 典型配置推荐

**小数据量 (< 1MB)**:
```c
PErasureWorkerSetThreadsPerBlock(worker, 128);
task_num = 1;
```

**中等数据量 (1-100MB)**:
```c
PErasureWorkerSetThreadsPerBlock(worker, 128);
task_num = 2-4;
```

**大数据量 (> 100MB)**:
```c
PErasureWorkerSetThreadsPerBlock(worker, 256);
task_num = 4-8;
```

**GPU tensor零拷贝模式**:
```c
PErasureWorkerSetThreadsPerBlock(worker, 256);  // 可以尝试更大的值
task_num = 1;  // 零拷贝模式下流水线意义不大
```

## 📚 API参考

### CUDA执行配置API

```c
// 设置每个线程块的线程数
void PErasureWorkerSetThreadsPerBlock(struct PErasureWorker *worker, int threads);

// 设置线程块数量（0表示自动计算）
void PErasureWorkerSetBlocksPerGrid(struct PErasureWorker *worker, int blocks);

// 获取当前线程块线程数
int PErasureWorkerGetThreadsPerBlock(struct PErasureWorker *worker);

// 获取当前线程块数量（如果自动计算，返回计算值）
int PErasureWorkerGetBlocksPerGrid(struct PErasureWorker *worker);
```

### 注意事项

1. **线程数必须是32的倍数**: 如果不是，系统会自动向上取整到最近的32的倍数
2. **线程数范围**: 1-1024（CUDA限制）
3. **blocks_per_grid = 0**: 表示自动计算，这是推荐的方式
4. **参数设置时机**: 可以在初始化后、编码前随时修改
