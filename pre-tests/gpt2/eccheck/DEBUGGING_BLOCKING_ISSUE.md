# 8-Rank 测试阻塞问题调试指南

## 问题描述

运行 8-rank 测试时，程序在初始化后阻塞，无法继续执行。

## 根本原因

### 问题1: Columns 配置未保存
- `_parse_advanced_columns_config` 解析了 4 个 columns，但没有保存到 `self.eccheck_columns_config`
- 后续代码使用 `getattr(self, 'eccheck_columns_config', None)` 时找不到配置，默认使用 2

### 问题2: 线程启动时机错误
- C++ 构造函数中，`num_columns_` 默认初始化为 2
- 构造函数立即调用 `start_pipeline()`，按照 2 个 columns 启动线程
- 后续调用 `set_columns_config(4)` 时，线程已经启动，无法改变 column 数量
- 导致 NCCL 初始化等待条件永远不满足（等待 4 个 columns，但只启动了 2 个）

## 修复方案

### 修复1: 保存 columns 配置
在 `filesystem_async.py` 的 `_load_eccheck_config` 中，解析后保存配置：

```python
self.eccheck_columns_config = norm_cols
logger.info(f"EC-CHECK: Stored columns config: {len(norm_cols)} columns")
```

### 修复2: 延迟线程启动
修改 C++ 代码，使线程启动延迟到 `set_columns_config` 调用：

1. **构造函数**：不再调用 `start_pipeline()`，只初始化数据结构
2. **set_columns_config**：
   - 如果线程未启动，启动线程并等待 NCCL 初始化
   - 如果线程已启动且 column 数量改变，重启线程

## 验证步骤

重新编译 C++ 扩展后，运行测试：

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck_8rank.sh
```

### 期望看到的日志

1. **配置解析**：
   ```
   EC-CHECK: Using advanced config mode for rank X, columns=4
   EC-CHECK: Applied advanced columns config (4 entries)
   EC-CHECK: Stored columns config: 4 columns
   ```

2. **C++ 初始化**：
   ```
   EC-CHECK: [Rank X] Constructor completed, waiting for columns config before starting threads...
   EC-CHECK: [Rank X] Columns config applied: col0(...) col1(...) col2(...) col3(...)
   EC-CHECK: [Rank X] Starting threads with 4 columns...
   EC-CHECK: [Rank X] Started 16 threads (4 encoding + 4 send + 4 recv + 4 XOR)
   EC-CHECK: [Rank X] Waiting for NCCL initialization (4 columns)...
   EC-CHECK: [Rank X] Pipeline and NCCL initialized successfully
   ```

3. **缓冲区分配**：
   ```
   EC-CHECK: Allocating 4 receive buffers based on peer data size
   ```

## 如果仍然阻塞

### 检查点1: 所有 rank 是否都成功解析配置
在日志中搜索 "Applied advanced columns config"，应该看到 8 条记录，每条都显示 4 entries。

### 检查点2: NCCL 初始化是否完成
在日志中搜索 "Pipeline and NCCL initialized successfully"，应该看到 8 条记录。

### 检查点3: Column 0 的 NCCL ID 文件
检查 `/tmp/eccheck_nccl_column0_id.txt` 到 `/tmp/eccheck_nccl_column3_id.txt` 是否存在。

### 检查点4: 所有 rank 的 column 数量是否一致
如果某些 rank 显示 2 个 columns，而其他 rank 显示 4 个，会导致 NCCL 初始化死锁。

## 常见问题

### Q: 为什么需要等待所有 columns 的 NCCL 初始化？
A: NCCL 是集合通信库，所有参与通信的进程必须同步到达同一个点。如果 rank 0 等待 4 个 columns，但 rank 1 只等待 2 个，它们永远不会同步。

### Q: 如果某些 rank 的配置缺失会怎样？
A: 会回退到简单模式，使用默认的 2 个 columns。这会导致 column 数量不一致，导致死锁。

### Q: 如何验证配置是否正确加载？
A: 查看日志中的 "Applied advanced columns config" 和 "Columns config applied" 消息，确认每个 rank 都有正确的 column 数量。

