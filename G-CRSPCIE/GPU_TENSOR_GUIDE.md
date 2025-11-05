# GPU Tensor Zero-Copy Encoding Guide

## 概述

G-CRSPCIE 现在支持两种编码模式：

1. **传统模式**：CPU数据 → GPU编码 → CPU结果（包含H2D和D2H传输）
2. **零拷贝模式**：GPU tensor → GPU编码 → GPU tensor（跳过所有CPU-GPU传输）

## 新增API

### 1. `PErasureWorkerSetSkipD2HTransfer(worker, skip)`
控制是否跳过Device-to-Host传输。

- `skip = 1`: 跳过D2H传输（零拷贝模式）
- `skip = 0`: 执行D2H传输（传统模式，默认）

### 2. `PErasureWorkerSetOutputDevicePtr(worker, device_ptr, data_size)`
设置外部GPU输出缓冲区指针。

- 当设置后，编码结果会直接写入到这个GPU缓冲区
- 需要配合 `PErasureWorkerSetSkipD2HTransfer(worker, 1)` 使用

### 3. `PErasureWorkerEncodeGPUZeroCopy(worker, input_dev_ptr, output_dev_ptr, data_size)`
一键设置GPU tensor零拷贝编码（最便捷的方式）。

- 自动设置输入和输出GPU指针
- 自动启用跳过D2H传输
- 适用于GPU tensor直接编码的场景

## 使用示例

### 示例1: CPU数据编码（传统模式）

```c
// 初始化worker
struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);

// 设置CPU输入数据
char *cpu_input = malloc(total_size);
// ... 填充数据 ...
PErasureWorkerSetInputData(worker, cpu_input, total_size);

// 编码（自动进行H2D和D2H传输）
fullDuplexRunEncode(worker);

// 获取CPU结果
char *cpu_output = malloc(block_size * m);
PErasureWorkerGetOutputData(worker, cpu_output, block_size * m);
```

### 示例2: GPU Tensor零拷贝编码（推荐方式）

```c
// 初始化worker
struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);

// 准备GPU内存（可以是PyTorch tensor的data_ptr）
char *gpu_input = NULL;
char *gpu_output = NULL;
cudaMalloc((void **)&gpu_input, total_size);
cudaMalloc((void **)&gpu_output, block_size * m);

// 方法1: 使用便捷API（推荐）
PErasureWorkerEncodeGPUZeroCopy(worker, gpu_input, gpu_output, total_size);
fullDuplexRunEncode(worker);

// 结果直接在gpu_output中，无需复制回CPU
```

### 示例3: 手动设置零拷贝模式

```c
// 初始化worker
struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);

// 准备GPU内存
char *gpu_input = NULL;
char *gpu_output = NULL;
cudaMalloc((void **)&gpu_input, total_size);
cudaMalloc((void **)&gpu_output, block_size * m);

// 方法2: 手动设置（更灵活）
PErasureWorkerSetInputDevicePtr(worker, gpu_input, total_size);  // 跳过H2D
PErasureWorkerSetOutputDevicePtr(worker, gpu_output, block_size * m);  // 跳过D2H
PErasureWorkerSetSkipD2HTransfer(worker, 1);  // 启用零拷贝

// 编码
fullDuplexRunEncode(worker);

// 获取输出GPU指针（如果需要）
char *output_ptr = PErasureWorkerGetOutputDevicePtr(worker);
```

## 与PyTorch集成

```python
import torch
import ctypes

# 假设你已经有了G-CRSPCIE的Python wrapper
from gcrspcie_wrapper import GCRSPCIEWrapper

# 准备GPU tensors
k, m, block_size = 4, 2, 1024 * 1024
data_tensor = torch.randint(0, 256, (k, block_size), dtype=torch.uint8, device='cuda')
output_tensor = torch.zeros((m, block_size), dtype=torch.uint8, device='cuda')

# 初始化wrapper
wrapper = GCRSPCIEWrapper()

# 使用零拷贝模式编码
# 注意：需要确保wrapper支持use_device_ptr参数
wrapper.encode(
    data_tensor, 
    k=k, m=m, 
    return_gpu=True,  # 返回GPU tensor
    use_device_ptr=True  # 使用设备指针（零拷贝）
)

# 结果直接在GPU上，无需CPU传输
```

## 性能优势

零拷贝模式的优势：
- ✅ **消除PCIe传输开销**：不需要H2D和D2H传输
- ✅ **降低延迟**：直接在GPU上操作，减少内存拷贝时间
- ✅ **提高吞吐量**：特别适合GPU流水线场景
- ✅ **节省内存带宽**：减少不必要的数据移动

## 注意事项

1. **内存对齐**：确保GPU内存按照 `w * sizeof(long)` 对齐（worker内部会自动处理）
2. **缓冲区大小**：确保外部GPU缓冲区足够大（至少 `k * block_size` 输入，`m * block_size` 输出）
3. **CUDA流同步**：在使用结果前，确保GPU操作完成（`cudaDeviceSynchronize()` 或 `cudaStreamSynchronize()`）
4. **任务数量**：如果使用多任务模式（`taskNum > 1`），确保GPU缓冲区按任务分割正确

## API参考

### 输入相关
- `PErasureWorkerSetInputData(worker, cpu_data, size)` - 设置CPU输入数据
- `PErasureWorkerSetInputDevicePtr(worker, gpu_ptr, size)` - 设置GPU输入指针（跳过H2D）

### 输出相关
- `PErasureWorkerGetOutputData(worker, cpu_output, size)` - 获取CPU输出数据
- `PErasureWorkerGetOutputDevicePtr(worker)` - 获取GPU输出指针
- `PErasureWorkerSetOutputDevicePtr(worker, gpu_ptr, size)` - 设置GPU输出指针（跳过D2H）

### 零拷贝控制
- `PErasureWorkerSetSkipD2HTransfer(worker, skip)` - 控制是否跳过D2H传输
- `PErasureWorkerEncodeGPUZeroCopy(worker, input_ptr, output_ptr, size)` - 一键设置零拷贝模式

## 编译

确保CUDA路径正确设置后编译：

```bash
cd G-CRSPCIE
make
```

编译错误通常是因为CUDA路径未配置，请在makefile中设置 `CUDA_INSTALL_PATH`。

