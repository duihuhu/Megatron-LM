# EC-CHECK 测试检查清单

## 测试前准备

### 1. ✅ 编译 C++ 扩展

```bash
cd /workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies
bash build_clean.sh
```

**检查点**:
- [ ] 编译成功，生成 `eccheck_native*.so`
- [ ] Python 可以导入模块：`python3 -c "import eccheck_native; print('OK')"`

### 2. ✅ 配置文件检查

**简单模式测试** (`eccheck_2x2.json`):
```json
{
  "persist": {"recv": true, "parity": true},
  "columns": [
    {"coefficient": 0, "send_peer": -1, "recv_peer": -1},
    {"coefficient": 1, "send_peer": -1, "recv_peer": -1}
  ]
}
```

**检查点**:
- [ ] 配置文件格式正确（JSON 验证）
- [ ] `send_peer` 和 `recv_peer` 为 `-1`（自动计算）
- [ ] `ECCHECK_CONFIG_PATH` 环境变量设置正确

### 3. ✅ 测试环境准备

**检查点**:
- [ ] NCCL 环境变量设置（`test_eccheck.sh` 中已设置）
- [ ] 至少 2 个 GPU 可用（用于 2-rank 测试）
- [ ] 足够的磁盘空间用于 checkpoint

## 测试执行

### 测试 1: 简单模式（2+2 配置）

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash test_eccheck.sh
```

**预期行为**:
1. **Python 端**:
   - 解析配置文件（简单模式）
   - 所有 column 设置默认 pipeline hints（`needs_recv=True`）
   - 所有 column 共享同一个 parity buffer
   - Column 0: `xor_mode=WITH_ZERO_PARITY`（如果配置了零初始化）
   - Column 1: `xor_mode=INCREMENTAL`

2. **C++ 端**:
   - Column 0: encode → send → XOR(encoding0, zero_parity) → parity（初始化）
   - Column 1: encode → send → recv → XOR(parity, recv_encoding) → parity（增量更新）

**检查点**:
- [ ] 所有 encoding 线程完成
- [ ] 所有 XOR 操作完成
- [ ] 没有死锁或超时错误
- [ ] Parity buffer 正确更新
- [ ] 缓冲区正确释放

### 测试 2: 高级模式（异构 pipeline）

**配置文件**: `eccheck_heterogeneous.json`

```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck_heterogeneous.json
bash test_eccheck.sh
```

**预期行为**:
1. **Python 端**:
   - 解析高级模式配置（`ranks` 字段）
   - Column 0: `needs_recv=False`, `needs_zero_parity=True`
   - Column 1: `needs_recv=True`, `needs_zero_parity=False`
   - 分配零初始化 parity buffers

2. **C++ 端**:
   - Column 0: encode → send → XOR（跳过 recv，直接与零 parity XOR）
   - Column 1: encode → send → recv → XOR（增量更新）

**检查点**:
- [ ] Column 0 跳过 recv 步骤
- [ ] Column 0 使用零初始化 parity buffer
- [ ] Column 1 正确执行 recv 和增量 XOR
- [ ] 共享 parity buffer 正确更新

## 常见问题排查

### 问题 1: C++ 模块导入失败

**症状**: `ImportError: No module named 'eccheck_native'`

**解决**:
1. 检查 `.so` 文件是否存在
2. 重新编译：`bash build_clean.sh`
3. 检查 Python 版本兼容性

### 问题 2: 配置解析错误

**症状**: `Failed to apply columns config`

**解决**:
1. 验证 JSON 格式：`python3 -m json.tool eccheck_2x2.json`
2. 检查配置文件路径：`echo $ECCHECK_CONFIG_PATH`
3. 查看日志中的详细错误信息

### 问题 3: 缓冲区超时

**症状**: `TIMEOUT waiting for free parity buffer`

**解决**:
1. 检查缓冲区分配是否正确
2. 检查 C++ 端是否正确释放缓冲区
3. 增加缓冲区数量或超时时间

### 问题 4: NCCL 通信错误

**症状**: `NCCL error` 或通信失败

**解决**:
1. 检查 NCCL 环境变量设置
2. 查看 `nccl.log` 文件
3. 确认 rank 和 world_size 配置正确

## 验证清单

### Python 端验证

- [ ] 配置解析成功
- [ ] Pipeline hints 正确提取
- [ ] 零初始化 parity buffer 分配（如需要）
- [ ] 共享 parity buffer 正确分配
- [ ] 参数正确传递到 C++

### C++ 端验证

- [ ] 所有 worker 线程启动
- [ ] Column 0 正确初始化 parity
- [ ] Column 1+ 正确增量更新 parity
- [ ] 线程安全保护工作正常
- [ ] 缓冲区正确释放

### 端到端验证

- [ ] 数据编码正确
- [ ] NCCL 通信成功
- [ ] XOR 操作正确执行
- [ ] Parity 结果正确
- [ ] 无内存泄漏

## 日志检查

### 关键日志信息

1. **配置解析**:
   ```
   EC-CHECK: Using advanced config mode for rank X
   EC-CHECK: Applied advanced columns config (N entries)
   EC-CHECK: Stored pipeline config for column X: N steps
   ```

2. **缓冲区分配**:
   ```
   EC-CHECK: Allocated N zero-initialized parity buffers for column 0
   ```

3. **XOR 模式**:
   ```
   EC-CHECK: Column 0 chunk X initializing parity with zero buffer
   EC-CHECK: Column 1 chunk X incrementally updating shared parity
   ```

4. **线程启动**:
   ```
   EC-CHECK: Started N threads (encoder, send, recv, xor for each column)
   ```

5. **完成标志**:
   ```
   EC-CHECK: Waiting for N encoding threads to complete
   EC-CHECK: All encoding threads completed
   ```

## 下一步

如果测试通过：
1. 可以继续实现多 send/recv 支持（如果需要）
2. 优化 mutex 内存管理
3. 添加更多错误处理

如果测试失败：
1. 查看详细错误日志
2. 检查配置文件
3. 验证 C++ 扩展编译正确

