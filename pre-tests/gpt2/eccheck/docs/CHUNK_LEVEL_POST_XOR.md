# Chunk-Level Post-XOR 实现文档

## 概述

Chunk-Level Post-XOR 允许在每个 chunk 的 XOR 操作完成后立即执行 post-xor 数据交换，而不需要等待所有 chunk 完成。这实现了真正的并行化和流水线执行。

## 实现架构

### 执行时机

**当前实现**：
```
Chunk 0: Encode → Send → Recv → XOR → Post-XOR (立即执行) ✅
Chunk 1: Encode → Send → Recv → XOR → Post-XOR (立即执行) ✅
Chunk 2: Encode → Send → Recv → XOR → Post-XOR (立即执行) ✅
...
```

**旧实现（全局）**：
```
所有 Chunks: Encode → Send → Recv → XOR
等待所有完成...
Post-XOR (全局执行)
```

### 地址和数据大小管理

每个 chunk 的地址和大小都正确管理：

1. **Parity 地址**: 每个 chunk 有独立的 parity buffer
2. **接收地址**: 按 chunk 计算偏移量 `chunk_offset = chunk_idx * chunk_size`
3. **发送地址**: 从 persistent parity store 的对应位置发送
4. **数据大小**: 每个 chunk 的大小（通常是 64MB，最后一个可能更小）

**代码位置**:
- C++ XOR Worker: `eccheck_native.cpp:828` - `execute_chunk_level_post_xor()`
- 地址计算: `chunk_offset = chunk_idx * chunk_size`
- 发送地址: `persist_parity_base_ + chunk_offset`
- 接收地址: `peer_data_buffer_base_ + chunk_offset`

## 配置格式

### 基本格式

```json
{
  "post_xor_steps": [
    {
      "step": "sync",
      "type": "barrier",
      "sync_point": "after_all_xor",
      "ranks": [0, 1, 2, 3],
      "chunk_level": false  // 全局执行（在所有 chunk 完成后）
    },
    {
      "step": "send",
      "type": "nccl_send",
      "target_rank": 2,
      "data_type": "parity",
      "data_source": "local_parity",
      "chunk_level": true  // 每个 chunk 执行
    },
    {
      "step": "recv",
      "type": "nccl_recv",
      "source_rank": 2,
      "data_type": "data",
      "data_target": "peer_data_buffer",
      "chunk_level": true  // 每个 chunk 执行
    }
  ]
}
```

### 配置字段说明

#### Sync Step
- `chunk_level: false` (必须): Sync 步骤全局执行
- 在所有 chunk 完成后执行一次

#### Send Step
- `chunk_level: true` (可选，默认 true): 每个 chunk 执行
- `data_source`: 
  - `"local_parity"`: 从 persistent parity store 发送（按 chunk 偏移）
  - `"local_data"`: 从 tensor buffer 发送（待实现）

#### Recv Step
- `chunk_level: true` (可选，默认 true): 每个 chunk 执行
- `data_target`:
  - `"peer_data_buffer"`: 接收到 peer data buffer（按 chunk 偏移）
  - `"peer_parity_buffer"`: 接收到 persistent recv store（按 chunk 偏移）

## 代码实现

### Python 侧

**配置设置**: `filesystem_async.py:1999-2059`
```python
# 在 Phase 2.5 设置 chunk-level post-xor 配置
if post_xor_steps:
    chunk_level_steps = [
        step for step in post_xor_steps 
        if step.get('step') in ['send', 'recv'] 
        and step.get('chunk_level', True)
    ]
    
    # 分配 peer_data_buffer
    # 调用 C++ set_chunk_level_post_xor_config()
```

**任务提交**: `filesystem_async.py:1706-1712`
```python
# 传递 chunk 信息
self._eccheck_native.submit_data_for_encoding(
    col_idx, ..., 
    chunk_idx,      # chunk 索引
    take,           # chunk 大小
    is_last_column  # 是否是最后一个 column
)
```

### C++ 侧

**数据结构**: `eccheck_native.cpp:169-182`
```cpp
struct PostXorStep {
    std::string step_name;
    int target_rank;
    int source_rank;
    std::string data_source;
    std::string data_target;
};
std::vector<PostXorStep> post_xor_steps_config_;
uintptr_t peer_data_buffer_base_;
```

**执行逻辑**: `eccheck_native.cpp:827-938`
```cpp
void execute_chunk_level_post_xor(int chunk_idx, size_t chunk_size, uintptr_t parity_addr) {
    size_t chunk_offset = chunk_idx * chunk_size;
    
    // 复制 parity 到 persistent store
    // 执行 send/recv（按 chunk 偏移）
    // Send: persist_parity_base_ + chunk_offset
    // Recv: peer_data_buffer_base_ + chunk_offset
}
```

**触发点**: `eccheck_native.cpp:795-799`
```cpp
// 在 XOR worker 中
if (is_last_column && chunk_idx >= 0 && chunk_size > 0) {
    execute_chunk_level_post_xor(chunk_idx, chunk_size, task.parity_addr);
}
```

## 数据流

### Chunk 0 的完整流程

```
1. Python 提交任务 (chunk_idx=0, chunk_size=64MB, is_last_column=true for col 1)
   ↓
2. C++ Encode → Send → Recv → XOR
   ↓
3. XOR 完成，检测到 is_last_column=true
   ↓
4. execute_chunk_level_post_xor(0, 64MB, parity_addr)
   ↓
5. 计算偏移: chunk_offset = 0 * 64MB = 0
   ↓
6. 复制 parity: persist_parity_store[0:64MB] = parity_buffer
   ↓
7. 执行 Post-XOR Send:
   - 发送地址: persist_parity_base_ + 0
   - 大小: 64MB
   - 目标: rank 2
   ↓
8. 执行 Post-XOR Recv:
   - 接收地址: peer_data_buffer_base_ + 0
   - 大小: 64MB
   - 源: rank 2
   ↓
9. 完成！Chunk 1 可以立即开始，不需要等待 Chunk 0 完成
```

### Chunk 1 的完整流程

```
1. Python 提交任务 (chunk_idx=1, chunk_size=64MB, is_last_column=true for col 1)
   ↓
2. C++ Encode → Send → Recv → XOR
   ↓
3. XOR 完成，检测到 is_last_column=true
   ↓
4. execute_chunk_level_post_xor(1, 64MB, parity_addr)
   ↓
5. 计算偏移: chunk_offset = 1 * 64MB = 64MB
   ↓
6. 复制 parity: persist_parity_store[64MB:128MB] = parity_buffer
   ↓
7. 执行 Post-XOR Send:
   - 发送地址: persist_parity_base_ + 64MB
   - 大小: 64MB
   ↓
8. 执行 Post-XOR Recv:
   - 接收地址: peer_data_buffer_base_ + 64MB
   - 大小: 64MB
   ↓
9. 完成！
```

## 性能优势

### 时间线对比

**全局 Post-XOR（旧）**:
```
T0: Chunk 0 XOR 完成
T1: Chunk 1 XOR 完成
T2: Chunk 2 XOR 完成
T3: 所有 Chunk 完成 → 执行 Post-XOR → 完成
总时间: T3
```

**Chunk-Level Post-XOR（新）**:
```
T0: Chunk 0 XOR 完成 → 立即 Post-XOR → 完成
T1: Chunk 1 XOR 完成 → 立即 Post-XOR → 完成
T2: Chunk 2 XOR 完成 → 立即 Post-XOR → 完成
总时间: T2 (通常是最后一个 chunk 的完成时间)
```

**优势**: 减少延迟 = `Post-XOR 执行时间 × (num_chunks - 1)`

## 地址映射表

| Chunk | Parity 发送地址 | Data 接收地址 | 偏移量 |
|-------|----------------|--------------|--------|
| 0     | `parity_base + 0` | `peer_data_base + 0` | 0 |
| 1     | `parity_base + 64MB` | `peer_data_base + 64MB` | 64MB |
| 2     | `parity_base + 128MB` | `peer_data_base + 128MB` | 128MB |
| ...   | ... | ... | ... |
| N     | `parity_base + N*64MB` | `peer_data_base + N*64MB` | N*64MB |

**公式**:
- `chunk_offset = chunk_idx × chunk_size`
- `send_addr = persist_parity_base + chunk_offset`
- `recv_addr = peer_data_buffer_base + chunk_offset`

## 注意事项

### 1. Sync 步骤

Sync 步骤**必须全局执行**，因为需要等待所有 chunk 完成：
```json
{
  "step": "sync",
  "chunk_level": false  // 必须为 false
}
```

### 2. 地址边界检查

C++ 代码会检查地址边界：
```cpp
if (chunk_offset + chunk_size <= capacity) {
    // 安全执行
}
```

### 3. NCCL 通信配对

每个 chunk 的 send/recv 必须正确配对：
- Rank 0 发送到 Rank 2 → Rank 2 必须从 Rank 0 接收
- Rank 1 发送到 Rank 3 → Rank 3 必须从 Rank 1 接收

### 4. 线程安全

- Chunk-level post-xor 在 XOR worker 线程中执行
- 使用 Column 0 的 NCCL communicator
- 多个 chunk 的 post-xor 可能并行执行（不同的 chunk 在不同时刻）

## 向后兼容

- **默认行为**: 如果 `chunk_level` 未指定，`send/recv` 默认为 `true`（chunk-level）
- **全局模式**: 设置 `chunk_level: false` 可以强制全局执行
- **混合模式**: 可以同时有 chunk-level 和 global 步骤

## 测试验证

验证点：
1. ✅ 每个 chunk 的 post-xor 在最后一个 column 完成后立即执行
2. ✅ 地址偏移量计算正确
3. ✅ 发送和接收正确配对
4. ✅ 数据正确累积到 persistent store
5. ✅ Sync 步骤只在全局执行

## 总结

Chunk-level post-xor 实现了：
- ✅ **并行化**: 不需要等待所有 chunk 完成
- ✅ **地址管理**: 每个 chunk 的地址和大小正确管理
- ✅ **灵活性**: 支持 chunk-level 和 global 混合配置
- ✅ **向后兼容**: 默认启用 chunk-level，可配置

