# 2+2 配置文件测试指南

## 配置文件说明

我创建了两个配置文件用于测试共享 parity buffer 模型：

### 1. `eccheck_2x2_shared_parity.json` - 简单模式（推荐用于测试）

**特点**:
- 使用简单模式配置（向后兼容）
- Python 端会自动设置默认行为：
  - Column 0: `xor_mode = WITH_ZERO_PARITY`（初始化 parity）
  - Column 1: `xor_mode = INCREMENTAL`（增量更新）
- 所有 column 共享同一个 parity buffer

**使用方法**:
```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2_shared_parity.json
bash test_eccheck.sh
```

### 2. `eccheck_2x2_advanced.json` - 高级模式

**特点**:
- 显式配置每个 column 的 pipeline
- Column 0: `mode: "with_zero_parity"`（显式指定）
- Column 1: `mode: "incremental"`（显式指定）
- 按 rank 配置（支持多 rank）

**使用方法**:
```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2_advanced.json
bash test_eccheck.sh
```

## 预期行为

### 共享 Parity Buffer 模型

```
每个 Chunk:
  ┌─────────────────────────────────────┐
  │  Shared Parity Buffer (所有 column 共享)  │
  └─────────────────────────────────────┘
         ↑                    ↑
         │                    │
    Column 0 初始化    Column 1 增量更新
         │                    │
  ┌─────────────┐      ┌─────────────┐
  │ Encode →    │      │ Encode →    │
  │ Send        │      │ Send        │
  │ XOR(zero)   │      │ Recv        │
  └─────────────┘      │ XOR(parity) │
                        └─────────────┘
```

### Column 0 流程

1. **Encode**: 编码数据
2. **Send**: 发送编码结果到 paired rank
3. **XOR**: `parity = encoding0 XOR zero_parity`（初始化共享 parity）

**注意**: Column 0 **不需要 recv**，直接使用零初始化 parity buffer

### Column 1 流程

1. **Encode**: 编码数据
2. **Send**: 发送编码结果到 paired rank
3. **Recv**: 接收 paired rank 的编码结果
4. **XOR**: `parity = parity XOR recv_encoding1`（增量更新共享 parity）

## 测试检查点

### Python 端日志检查

查找以下日志信息：

1. **配置解析**:
   ```
   EC-CHECK: Applied simple columns config (2 entries)
   ```
   或
   ```
   EC-CHECK: Using advanced config mode for rank 0
   EC-CHECK: Applied advanced columns config (2 entries)
   ```

2. **零初始化 Buffer** (Column 0):
   ```
   EC-CHECK: Allocated N zero-initialized parity buffers for column 0
   ```

3. **共享 Parity 使用**:
   ```
   EC-CHECK: Column 0 chunk X initializing parity with zero buffer at 0x...
   EC-CHECK: Column 1 chunk X incrementally updating shared parity
   ```

### C++ 端日志检查

查找以下日志信息：

1. **线程启动**:
   ```
   EC-CHECK: Started 8 threads (2 encoding + 2 send + 2 recv + 2 XOR)
   ```

2. **Column 0 行为**:
   - 应该看到 encode 和 send
   - **不应该**看到 recv（因为 `recv_addr=0`）
   - 应该看到 XOR 操作

3. **Column 1 行为**:
   - 应该看到 encode、send、recv
   - 应该看到 XOR 操作（增量更新）

4. **完成标志**:
   ```
   EC-CHECK: All encoding threads completed
   ```

## 快速测试命令

### 测试 1: 简单模式

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2_shared_parity.json
bash test_eccheck.sh
```

### 测试 2: 高级模式

```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2_advanced.json
bash test_eccheck.sh
```

## 配置验证

在运行测试前，可以验证配置：

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
python3 verify_eccheck_config.py eccheck_2x2_shared_parity.json
python3 verify_eccheck_config.py eccheck_2x2_advanced.json
```

## 预期结果

如果一切正常，应该看到：

1. ✅ 配置正确解析
2. ✅ 所有线程正常启动
3. ✅ Column 0 跳过 recv，直接初始化 parity
4. ✅ Column 1 执行 recv 并增量更新 parity
5. ✅ 所有操作完成，无超时错误
6. ✅ 缓冲区正确释放

## 故障排查

如果出现问题，检查：

1. **配置路径**: `echo $ECCHECK_CONFIG_PATH`
2. **C++ 扩展**: 确保已重新编译
3. **日志输出**: 查看详细日志信息
4. **NCCL**: 检查 NCCL 环境变量和通信

