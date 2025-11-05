# G-CRSPCIE 快速开始指南

## 🚀 快速编译和运行

### 1. 检查CUDA环境
```bash
nvcc --version
nvidia-smi
```

### 2. 修改CUDA路径（如需要）
编辑 `makefile` 第2行，设置你的CUDA路径：
```makefile
CUDA_INSTALL_PATH ?= /usr/local/cuda-12.3
```

### 3. 编译
```bash
cd G-CRSPCIE

# 编译库和原始程序
make

# 编译测试程序
make test

# 或者一次性编译所有
make clean && make && make test
```

### 4. 运行测试
```bash
# 运行GPU tensor零拷贝测试
./test_gpu_tensor

# 运行原始PCIe测试
./PCIEMeasure 2 10 10
```

## 📝 使用示例

### CPU数据编码（传统模式）
```c
struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, 1);
PErasureWorkerSetInputData(worker, cpu_data, size);
fullDuplexRunEncode(worker);
PErasureWorkerGetOutputData(worker, cpu_output, size);
PErasureWorkerDealloc(worker);
```

### GPU Tensor零拷贝编码（推荐）
```c
struct PErasureWorker *worker = PErasureWorkerInit(k, m, w, block_size, 1);
cudaMalloc(&gpu_input, total_size);
cudaMalloc(&gpu_output, output_size);

// 一键设置零拷贝模式
PErasureWorkerEncodeGPUZeroCopy(worker, gpu_input, gpu_output, total_size);
fullDuplexRunEncode(worker);

// 结果直接在gpu_output中，无需复制回CPU
cudaFree(gpu_input);
cudaFree(gpu_output);
PErasureWorkerDealloc(worker);
```

## ❓ 常见问题

**编译错误？**
- 检查CUDA路径是否正确
- 运行 `make clean` 后重新编译

**找不到库？**
```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
```

**详细文档？**
- 完整编译指南：`COMPILE_GUIDE.md`
- API使用指南：`GPU_TENSOR_GUIDE.md`
- 示例代码：`test_gpu_tensor.c`

