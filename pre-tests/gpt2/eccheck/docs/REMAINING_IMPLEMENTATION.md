# 剩余实现清单（不考虑 post-xor）

## ✅ 已完成

1. **Python 端**:
   - ✅ 配置解析（简单和高级模式）
   - ✅ Pipeline hints 提取
   - ✅ 共享 parity buffer 分配（所有 column 共享）
   - ✅ 零初始化 parity buffer 分配
   - ✅ 根据 pipeline 配置决定是否传递 recv_addr
   - ✅ 传递 zero_parity_addr 和 xor_mode_int 到 C++

2. **C++ 端**:
   - ✅ XOR Mode 枚举和数据结构
   - ✅ API 扩展（zero_parity_addr, xor_mode_int）
   - ✅ 多种 XOR 模式实现（WITH_ZERO_PARITY, WITH_RECV_ENCODING, INCREMENTAL）
   - ✅ 共享 parity buffer 增量更新
   - ✅ 线程安全保护（mutex 保护共享 parity buffer）
   - ✅ Column 0 直接 XOR 零初始化 parity（跳过 recv）
   - ✅ Column 1+ 增量更新共享 parity

## ❌ 还缺少的实现

### 1. 多 Send/Recv 支持（count > 1）

**当前状态**: 只支持单次 send/recv

**需要实现**:
- Python 端：从 pipeline 配置中提取 `count` 参数
- C++ 端：在 `send_worker` 和 `recv_worker` 中循环执行多次操作

**配置文件示例**:
```json
{
  "step": "send",
  "type": "nccl_send",
  "target_rank": 1,
  "count": 2  // 需要发送 2 次
}
```

**实现位置**:
- Python: `_parse_pipeline_step()` 已提取 `send_count` 和 `recv_count`，但未使用
- C++: `send_worker()` 和 `recv_worker()` 需要循环执行

### 2. 条件 Send 支持

**当前状态**: 总是执行 send（即使不需要）

**问题**: 
- 某些 pipeline 配置可能不需要 send（但目前没有这样的场景）
- 可以通过检查 `send_count > 0` 来决定

**实现位置**:
- C++: `encoder_worker()` 中检查是否需要 send

### 3. 错误处理和边界情况

**需要检查**:
- 零初始化 parity buffer 未分配时的处理
- 共享 parity buffer 地址不匹配时的处理
- XOR 模式不匹配时的处理

### 4. 代码清理和优化

**可能需要**:
- 清理未使用的代码
- 优化 mutex 创建和销毁（当前为每个 parity buffer 创建 mutex，可能泄漏）
- 添加日志和调试信息

## 优先级

### 🔴 高优先级（核心功能）

1. **多 Send/Recv 支持**
   - 如果配置中使用了 `count > 1`，必须实现
   - 实现难度：⭐⭐
   - 工作量：小到中等

### 🟡 中优先级（优化和健壮性）

2. **条件 Send 支持**
   - 当前总是执行 send，可能浪费资源
   - 实现难度：⭐
   - 工作量：小

3. **错误处理**
   - 确保边界情况的正确处理
   - 实现难度：⭐
   - 工作量：小

### 🟢 低优先级（优化）

4. **代码清理**
   - Mutex 内存管理优化
   - 代码清理和优化
   - 实现难度：⭐
   - 工作量：小

## 总结

**核心功能已基本完成**，剩余主要是：
1. **多 Send/Recv 支持**（如果配置需要）
2. **一些优化和错误处理**

大部分功能已经就绪，可以进行测试了。

