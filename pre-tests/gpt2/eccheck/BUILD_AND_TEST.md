# Chunk-Level Post-XOR 编译和测试指南

## ✅ 实现完成状态

所有代码实现已完成，可以编译和测试！

### 完成的功能
1. ✅ Chunk-level post-xor 执行逻辑
2. ✅ 地址和数据大小管理
3. ✅ NCCL send/recv 集成
4. ✅ 配置文件格式更新
5. ✅ Python-C++ 集成

## 📦 编译步骤

### 1. 检查编译依赖
```bash
# 确保有以下依赖：
# - PyTorch
# - NCCL (如果需要)
# - ISA-L (Intel Storage Acceleration Library)
# - pybind11
```

### 2. 编译 C++ 扩展
```bash
cd /workspace/Megatron-LM

# 方式1: 使用项目的编译脚本（如果有）
# 例如：
python setup.py build_ext --inplace

# 方式2: 手动编译（根据项目实际情况调整）
# 查找 eccheck_native.cpp 的编译命令
```

### 3. 验证编译结果
```bash
# 检查 .so 文件是否生成
ls -lh megatron/core/dist_checkpointing/strategies/eccheck_native*.so

# 应该有类似输出：
# eccheck_native.cpython-*.so
```

## 🧪 测试步骤

### 1. 运行 4-Rank 测试
```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck_4rank.sh
```

### 2. 验证日志输出

**期望看到的日志**：
```
EC-CHECK: Setting chunk-level post-xor config: 2 steps, peer_data_base=0x..., capacity=...
EC-CHECK: Chunk 0 parity copied to persistent store at offset 0
EC-CHECK: Chunk 0 post-xor send to rank 2, size: 67108864 bytes
EC-CHECK: Chunk 0 post-xor recv from rank 2, size: 67108864 bytes
EC-CHECK: Chunk 0 post-xor: 1 sends, 1 recvs completed
EC-CHECK: Chunk-level post-xor for chunk 0 completed (size=67108864)
...
```

### 3. 验证点

- ✅ **配置加载**: 日志显示 chunk-level post-xor 配置已设置
- ✅ **Chunk 执行**: 每个 chunk 完成后立即执行 post-xor
- ✅ **地址管理**: 日志显示正确的地址和偏移量
- ✅ **数据交换**: Send/Recv 正确配对执行
- ✅ **同步**: Sync 步骤只在全局执行一次

## 🔍 故障排除

### 问题1: 编译错误 - std::stoi 未定义
**解决**: 已添加 `#include <string>` ✅

### 问题2: 配置未加载
**检查**:
```bash
# 验证环境变量
echo $ECCHECK_CONFIG_PATH

# 应该指向：
# /workspace/Megatron-LM/pre-tests/gpt2/eccheck/configs/eccheck_4rank.json
```

### 问题3: Chunk-level post-xor 未执行
**检查**:
- 配置文件中 `chunk_level: true` 是否正确设置
- `peer_data_buffer` 是否正确分配
- C++ 日志中是否显示 `Chunk-level post-xor config set`

### 问题4: NCCL 通信失败
**检查**:
- NCCL 是否正确初始化
- Send/Recv 的 rank 配对是否正确
- 所有 rank 是否同时到达 post-xor 点

## 📊 性能对比

### 预期改进
- **延迟减少**: ~(num_chunks - 1) × post-xor_time
- **并行度提升**: 多个 chunk 可以同时执行 post-xor
- **流水线效率**: 不需要等待所有 chunk 完成

## 📝 配置文件说明

### chunk_level 字段
- `chunk_level: true`: 每个 chunk 执行一次（用于 send/recv）
- `chunk_level: false`: 全局执行一次（用于 sync）

### 默认行为
- Send/Recv: 默认 `chunk_level: true`（如果未指定）
- Sync: 必须设置 `chunk_level: false`

## ✅ 完成检查清单

- [x] 所有 C++ 代码实现完成
- [x] 所有 Python 代码实现完成
- [x] 配置文件格式更新完成
- [x] 头文件包含完整
- [x] Pybind11 绑定完整
- [x] 地址管理逻辑完整
- [x] NCCL 通信逻辑完整
- [x] 错误处理完整

**准备就绪，可以编译测试！** 🚀

