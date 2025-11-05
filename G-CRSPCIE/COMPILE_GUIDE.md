# G-CRSPCIE 编译和使用完整指南

## 📋 目录
1. [环境准备](#环境准备)
2. [编译](#编译)
3. [运行测试](#运行测试)
4. [使用示例](#使用示例)
5. [常见问题](#常见问题)

---

## 🔧 环境准备

### 检查CUDA环境

```bash
# 1. 检查CUDA是否安装
nvcc --version

# 2. 查找CUDA安装路径
ls /usr/local/cuda*
# 或者
find /usr -name "cuda" -type d 2>/dev/null
```

### 设置环境变量（可选）

如果CUDA不在默认位置，需要设置环境变量：

```bash
# 根据你的CUDA版本调整路径
export CUDA_PATH=/usr/local/cuda-12.3
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH
```

### 检查GPU设备

```bash
# 检查GPU是否可用
nvidia-smi
```

---

## 🔨 编译

### 步骤1: 修改CUDA路径（如果需要）

编辑 `makefile`，修改第2行的CUDA路径：

```bash
cd G-CRSPCIE
vim makefile  # 或使用你喜欢的编辑器
```

找到这一行：
```makefile
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.3
```

改为你的CUDA安装路径。

### 步骤2: 编译库和程序

```bash
# 清理之前的编译文件
make clean

# 编译所有内容（库 + 原始测试程序）
make

# 这将生成：
# - libgcrspcie.so: 共享库文件（供其他程序链接使用）
# - PCIEMeasure: 原始PCIe性能测试程序
```

### 步骤3: 编译GPU Tensor测试程序

```bash
# 编译GPU tensor零拷贝测试程序
make test_gpu_tensor

# 这将生成：
# - test_gpu_tensor: 可运行的测试程序
```

---

## 🚀 运行测试

### 运行GPU Tensor零拷贝测试

```bash
# 运行测试程序（会自动测试CPU和GPU两种模式）
./test_gpu_tensor
```

预期输出：
```
==================================================
G-CRSPCIE GPU Tensor Zero-Copy Encoding Tests
==================================================
Found 1 CUDA device(s)
Using device: NVIDIA GeForce RTX 3090

[Test 1] CPU -> GPU -> CPU (Traditional Mode)
------------------------------------------------
✓ Worker initialized: k=4, m=2, w=4, block_size=65536
✓ CPU input data prepared (262144 bytes)
Running encoding...
✓ Encoding completed successfully
  Encoding time: X.XXX ms
  Throughput: XXX.XX MB/s

[Test 2] GPU Tensor -> GPU Encoding -> GPU Tensor (Zero-Copy Mode)
-------------------------------------------------------------------
✓ Worker initialized: k=4, m=2, w=4, block_size=65536
✓ GPU memory allocated
✓ GPU input data prepared (262144 bytes)
✓ Zero-copy mode enabled (no H2D/D2H transfers)
Running encoding (zero-copy mode)...
✓ Zero-copy encoding completed successfully
  Encoding time: X.XXX ms
  Throughput: XXX.XX MB/s
  Note: Zero-copy mode eliminates PCIe transfer overhead!

[Test 3] Manual Zero-Copy Setup
--------------------------------
✓ Worker initialized
✓ Manual zero-copy setup completed
Running encoding...
✓ Output device pointer verified
✓ Manual zero-copy encoding completed

==================================================
Test Summary: 3/3 tests passed
==================================================
```

### 运行原始PCIe测试程序

```bash
# 运行原始测试（需要参数：m workSizePerDataParityBlockInMB numberOfTasks）
./PCIEMeasure 2 10 10
```

---

## 💻 使用示例

### 示例1: C程序中使用GPU Tensor零拷贝编码

```c
#include "PErasureWorker.h"
#include <cuda_runtime.h>

int main() {
    int k = 4, m = 2, w = 4;
    size_t block_size = 1024 * 1024;  // 1MB
    size_t total_size = block_size * k;
    
    // 初始化worker
    struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, 1);
    
    // 分配GPU内存（可以是PyTorch tensor的data_ptr）
    char *gpu_input = NULL;
    char *gpu_output = NULL;
    cudaMalloc((void **)&gpu_input, total_size);
    cudaMalloc((void **)&gpu_output, block_size * m);
    
    // 设置零拷贝模式
    PErasureWorkerEncodeGPUZeroCopy(worker, gpu_input, gpu_output, total_size);
    
    // 执行编码（无H2D/D2H传输）
    fullDuplexRunEncode(worker);
    
    // 结果直接在gpu_output中
    // 可以继续在GPU上使用，无需复制回CPU
    
    // 清理
    cudaFree(gpu_input);
    cudaFree(gpu_output);
    PErasureWorkerDealloc(worker);
    
    return 0;
}
```

编译：
```bash
nvcc -I. -I/usr/local/cuda-12.3/include \
     -L/usr/local/cuda-12.3/lib64 -lcudart -lcuda \
     -o my_program my_program.c \
     PErasureWorker.o GCRSCoding.o GCRSMatrix.o GCRSKernel.o \
     jerasure.o galois.o PErasureUtilities.o
```

### 示例2: 链接共享库使用

```bash
# 编译时链接共享库
gcc -I. -I/usr/local/cuda-12.3/include \
    -L. -L/usr/local/cuda-12.3/lib64 \
    -lgcrspcie -lcudart -lcuda \
    -o my_program my_program.c

# 运行时需要设置库路径
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./my_program
```

### 示例3: Python中使用（需要Python wrapper）

如果你有Python wrapper（如 `gcrspcie_wrapper.py`）：

```python
from gcrspcie_wrapper import GCRSPCIEWrapper
import torch

# 准备GPU tensors
k, m, block_size = 4, 2, 1024 * 1024
data_tensor = torch.randint(0, 256, (k, block_size), dtype=torch.uint8, device='cuda')
output_tensor = torch.zeros((m, block_size), dtype=torch.uint8, device='cuda')

# 初始化wrapper
wrapper = GCRSPCIEWrapper()

# 使用零拷贝模式编码
coding_blocks = wrapper.encode(
    data_tensor, 
    k=k, m=m, 
    return_gpu=True,      # 返回GPU tensor
    use_device_ptr=True   # 使用设备指针（零拷贝）
)

# 结果直接在GPU上
```

---

## ❓ 常见问题

### Q1: 编译错误 "cuda_runtime.h: No such file or directory"

**解决方法：**
```bash
# 方法1: 在makefile中设置正确的CUDA路径
# 编辑makefile，修改 CUDA_INSTALL_PATH

# 方法2: 设置环境变量
export CUDA_INSTALL_PATH=/usr/local/cuda-12.3

# 方法3: 创建符号链接
sudo ln -s /usr/local/cuda-12.3 /usr/local/cuda
```

### Q2: 链接错误 "cannot find -lcudart"

**解决方法：**
```bash
# 检查CUDA库文件是否存在
ls /usr/local/cuda-12.3/lib64/libcudart.so*

# 如果不存在，尝试：
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH
```

### Q3: 运行时错误 "libgcrspcie.so: cannot open shared object file"

**解决方法：**
```bash
# 设置库搜索路径
export LD_LIBRARY_PATH=.:/usr/local/cuda-12.3/lib64:$LD_LIBRARY_PATH

# 或者安装到系统路径
sudo cp libgcrspcie.so /usr/local/lib/
sudo ldconfig
```

### Q4: CUDA设备未找到

**解决方法：**
```bash
# 检查GPU是否可用
nvidia-smi

# 检查CUDA设备
nvidia-smi --query-gpu=name --format=csv
```

### Q5: 测试程序崩溃或输出全零

**可能原因：**
- GPU内存不足
- 数据大小不对齐（需要是 `w * sizeof(long)` 的倍数）
- CUDA设备未正确初始化

**解决方法：**
```bash
# 检查GPU内存
nvidia-smi

# 减小测试数据大小
# 编辑 test_gpu_tensor.c，减小 block_size
```

### Q6: 如何验证零拷贝模式真的跳过了传输？

**方法：**
```bash
# 使用nvprof或nsys分析
nvprof ./test_gpu_tensor

# 或者使用CUDA events记录时间
# 在代码中添加时间测量，对比CPU模式和GPU模式的耗时
```

---

## 📚 更多资源

- **API文档**: 查看 `GPU_TENSOR_GUIDE.md`
- **示例代码**: 查看 `test_gpu_tensor.c` 和 `gpu_tensor_example.c`
- **原始文档**: 查看 `README.md`

---

## 🔗 快速参考

### 编译命令速查

```bash
# 编译库
make clean && make

# 编译测试程序
make test_gpu_tensor

# 运行测试
./test_gpu_tensor

# 清理
make clean
```

### 关键API速查

```c
// 初始化
PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, task_num);

// CPU模式
PErasureWorkerSetInputData(worker, cpu_data, size);
fullDuplexRunEncode(worker);
PErasureWorkerGetOutputData(worker, cpu_output, size);

// GPU零拷贝模式（推荐）
PErasureWorkerEncodeGPUZeroCopy(worker, gpu_input, gpu_output, size);
fullDuplexRunEncode(worker);

// 清理
PErasureWorkerDealloc(worker);
```

---

**编译完成后，你就可以开始使用GPU tensor零拷贝编码功能了！** 🎉

