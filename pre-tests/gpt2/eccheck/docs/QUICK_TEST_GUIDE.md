# EC-CHECK 快速测试指南

## ✅ 测试前检查（已完成）

- ✅ C++ 扩展已编译（`eccheck_native*.so` 存在）
- ✅ 配置文件格式正确（`eccheck_2x2.json`, `eccheck_heterogeneous.json`）
- ✅ Python 代码可以导入

## ⚠️ 重要：需要重新编译 C++ 扩展

由于我们修改了 C++ 代码（添加了 XOR mode、共享 parity buffer 等），**必须重新编译**：

```bash
cd /workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies
bash build_clean.sh
```

这将编译新的代码并生成更新的 `.so` 文件。

## 测试步骤

### 步骤 1: 重新编译 C++ 扩展

```bash
cd /workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies
bash build_clean.sh
```

**预期输出**:
```
Building EC-CHECK native C++ module (clean build)...
✅ EC-CHECK native module built successfully!
✅ EC-CHECK native module imported successfully
```

### 步骤 2: 运行简单模式测试

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2.json
bash test_eccheck.sh
```

**关键检查点**:
- 查看日志中是否有配置解析信息
- 检查是否所有 encoding 线程完成
- 检查是否有缓冲区超时错误

### 步骤 3: 运行异构 pipeline 测试（可选）

```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_heterogeneous.json
bash test_eccheck.sh
```

**关键检查点**:
- 检查 Column 0 是否跳过了 recv
- 检查是否使用了零初始化 parity buffer
- 检查 Column 1 是否正确执行增量更新

## 常见问题

### 编译错误

如果编译失败，检查：
1. C++17 标准支持
2. PyTorch、NCCL、ISA-L 库是否可用
3. 查看编译错误信息

### 运行时错误

如果运行时出错，检查：
1. C++ 扩展是否正确加载：`python3 -c "import eccheck_native"`
2. 配置文件路径是否正确：`echo $ECCHECK_CONFIG_PATH`
3. NCCL 环境变量设置

### 功能验证

**检查日志关键词**:
- `EC-CHECK: Using advanced config mode` - 高级模式配置
- `EC-CHECK: Allocated N zero-initialized parity buffers` - 零初始化 buffer
- `EC-CHECK: Column 0 initializing parity` - Column 0 初始化
- `EC-CHECK: Column 1 incrementally updating shared parity` - Column 1 增量更新
- `EC-CHECK: All encoding threads completed` - 所有线程完成

## 预期行为

### 简单模式（eccheck_2x2.json）

- 所有 column 共享同一个 parity buffer
- Column 0: 初始化 parity（使用零初始化 buffer 或直接写入）
- Column 1: 增量更新 parity（parity XOR recv_encoding）

### 异构模式（eccheck_heterogeneous.json）

- Column 0: 跳过 recv，直接与零 parity XOR
- Column 1: 执行 recv，然后增量更新共享 parity

## 下一步

测试通过后：
1. 可以优化 mutex 内存管理
2. 可以实现多 send/recv 支持（如需要）
3. 可以添加更多错误处理

测试失败时：
1. 查看详细错误日志
2. 检查实现是否有遗漏
3. 验证配置是否正确

