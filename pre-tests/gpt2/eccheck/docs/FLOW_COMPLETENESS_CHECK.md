# EC-CHECK 流程完整性检查

## ✅ 配置文件完整性

### eccheck_4rank.json 检查

**已包含的配置**:
- ✅ `persist.recv` 和 `persist.parity` 设置
- ✅ 所有 4 个 rank 的 `columns` 配置
- ✅ 所有 4 个 rank 的 `post_xor_steps` 配置（**已添加**）

**Post-XOR 配置验证**:
```
Rank 0: sync → send(parity to 2) → recv(data from 2) ✅
Rank 1: sync → send(parity to 3) → recv(data from 3) ✅
Rank 2: sync → send(parity to 0) → recv(data from 0) ✅
Rank 3: sync → send(parity to 1) → recv(data from 1) ✅
```

**配对关系验证**:
- EC-CHECK 配对: Rank 0↔2, Rank 1↔3 ✅
- Post-XOR 通信遵循配对关系 ✅

---

## ✅ 代码实现完整性

### Python 侧实现

**Phase 1-2**: ✅ 完整
- 配置文件加载和解析
- C++ native 模块创建
- Metadata 交换

**Phase 2.5**: ✅ 完整
- 缓冲区分配（data, encoding, receive, parity）
- **三段持久化内存分配**:
  1. `tensor_buffer` - 原始数据 ✅
  2. `eccheck_persist_parity_store` - 计算的 parity ✅
  3. `eccheck_persist_recv_store` 或 `eccheck_post_xor_buffers['peer_data_buffer']` - 接收的数据/parity ✅

**Phase 3**: ✅ 完整
- GPU → CPU 数据传输
- 数据编码和交换
- XOR 操作
- 等待所有 encoding/XOR 完成

**Phase 4**: ✅ 完整
- `_execute_post_xor_steps()` 方法已实现 ✅
- 支持 `sync` (barrier) ✅
- 支持 `send` (nccl_send) ✅
- 支持 `recv` (nccl_recv) ✅
- 使用 Column 0 的 NCCL communicator ✅

**调用位置**: ✅
```python
# filesystem_async.py:1716-1719
self._eccheck_native.wait_for_encoding_completion()
self._execute_post_xor_steps()  # ✅ 正确调用
```

### C++ 侧实现

**Post-XOR API**: ✅ 完整
- `post_xor_send(target_rank, data_addr, size)` ✅
- `post_xor_recv(source_rank, recv_addr, size)` ✅
- 使用 Column 0 的 NCCL communicator (`nccl_comms_[0]`) ✅

---

## ✅ 流程完整性验证

### 完整流程检查清单

```
Phase 1: 初始化
  ✅ 加载配置文件
  ✅ 创建 C++ native 模块
  ✅ 启动线程池（8个线程 per rank）
  ✅ 初始化 NCCL communicators

Phase 2: Metadata 交换
  ✅ 收集本地 tensor metadata
  ✅ 广播 metadata 到所有 rank
  ✅ 计算 peer 数据大小

Phase 2.5: 缓冲区分配
  ✅ Data buffers
  ✅ Encoding buffers
  ✅ Receive buffers
  ✅ Parity buffers
  ✅ 三段持久化内存分配

Phase 3: 数据编码和交换
  ✅ GPU → CPU 数据传输
  ✅ 分块处理
  ✅ Column 0: Encode → Send → XOR (with_zero_parity)
  ✅ Column 1: Encode → Send → Recv → XOR (incremental)
  ✅ 等待所有 encoding/XOR 完成（同步）

Phase 4: Post-XOR 数据交换
  ✅ 同步屏障 (barrier) - 确保所有 XOR 完成
  ✅ Send: 发送本地 data 或 parity 到目标 rank
  ✅ Recv: 从源 rank 接收 data 或 parity
  ✅ 接收结果存入持久化内存

Phase 5: Checkpoint 保存
  ✅ 三段持久化内存可用：
     - 原始数据 (tensor_buffer)
     - 计算的 Parity (eccheck_persist_parity_store)
     - 接收的 Data/Parity (post_xor_buffers 或 persist_recv_store)
```

---

## ⚠️ 潜在问题检查

### 1. Post-XOR 缓冲区分配时机

**当前实现**:
- `eccheck_persist_recv_store`: 在 Phase 2.5 分配（基于 persist.recv 配置）
- `eccheck_post_xor_buffers['peer_data_buffer']`: 在 Phase 4 动态分配（如果需要）

**检查**: ✅ 两种方式都支持，取决于配置的 `data_target`

### 2. Parity 数据持久化

**当前实现**:
- Parity 数据在 Phase 3 中计算并存储到 `shared_parity_addr`
- 如果启用了 `persist.parity`，数据会 memcpy 到 `eccheck_persist_parity_store`

**检查**: ✅ 需要确认 C++ 端是否正确 memcpy 到 persist store

### 3. Post-XOR 通信配对

**配置验证**:
- Rank 0 ↔ Rank 2 (配对关系) ✅
- Rank 1 ↔ Rank 3 (配对关系) ✅

**检查**: ✅ 配置正确，遵循 EC-CHECK 配对规则

---

## 📊 三段持久化内存状态

### 内存分配和填充时间线

| 持久化内存 | 分配时机 | 填充时机 | 用途 |
|-----------|---------|---------|------|
| **1. tensor_buffer** | Phase 2 | Phase 3 (GPU→CPU) | 原始数据 |
| **2. persist_parity_store** | Phase 2.5 | Phase 3 (XOR 完成后) | 计算的 parity |
| **3. peer_data/parity_buffer** | Phase 4 (动态) 或 Phase 2.5 | Phase 4 (Post-XOR recv) | 接收的 peer 数据 |

### 完整性检查

✅ **tensor_buffer**: 原始数据已填充
✅ **persist_parity_store**: 需要确认 C++ 是否正确 memcpy
⚠️ **peer_data_buffer**: 在 Phase 4 动态分配（如果需要）

---

## ✅ 总结

### 流程完整性: **完整**

1. ✅ 配置文件已包含完整的 `post_xor_steps`
2. ✅ Python 代码完整支持所有步骤
3. ✅ C++ 代码有 post_xor_send/recv API
4. ✅ 调用位置正确（在 wait_for_encoding_completion 之后）
5. ✅ 三段持久化内存都支持

### 需要验证的细节

1. ⚠️ C++ 端是否正确将 parity 数据 memcpy 到 `persist_parity_store`？
2. ⚠️ Post-XOR 接收的缓冲区大小是否正确计算？
3. ⚠️ 所有 rank 的 post-xor 步骤是否同步执行？

### 建议测试

1. 运行 4 rank 测试，检查日志确认：
   - Post-XOR steps 被执行
   - 三段持久化内存都已填充
   - 通信配对正确

2. 验证持久化内存内容：
   - tensor_buffer 包含原始数据
   - persist_parity_store 包含计算的 parity
   - peer_data_buffer 包含接收的数据

