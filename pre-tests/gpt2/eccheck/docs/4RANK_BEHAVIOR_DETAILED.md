# EC-CHECK 4 Rank 详细行为分析

本文档详细说明 `test_eccheck_4rank` 测试中每个 rank 的行为，包括 high-level 流程和详细实现。

## 📊 测试配置概览

### 基础配置
- **总 Rank 数**: 4 (rank 0, 1, 2, 3)
- **每个 Rank 的 Column 数**: 2 (column 0 和 column 1)
- **EC-CHECK 配对**: 
  - Rank 0 ↔ Rank 2
  - Rank 1 ↔ Rank 3
- **配置规则**: `rank_i ↔ rank_{i + world_size/2}`

### 脚本参数
```bash
GPUS_PER_NODE=4      # 单节点 4 GPU
NNODES=1             # 1 个节点
WORLD_SIZE=4         # 总 rank 数 = 4 × 1 = 4
```

---

## 🔄 High-Level 流程

### 整体流程（所有 Rank 同步执行）

```
Phase 1: 初始化
  ├─ 加载配置文件
  ├─ 创建 C++ native 模块
  ├─ 启动线程池（8个线程 per rank: 2 encoder + 2 send + 2 recv + 2 XOR）
  └─ 初始化 NCCL communicators

Phase 2: Metadata 交换
  ├─ 收集本地 tensor metadata
  ├─ 广播 metadata 到所有 rank
  └─ 计算 peer 数据大小

Phase 2.5: 缓冲区分配
  ├─ 分配 data buffers
  ├─ 分配 encoding buffers
  ├─ 分配 receive buffers（基于 peer 数据大小）
  └─ 分配 parity buffers（共享模型）

Phase 3: 数据编码和交换
  ├─ GPU → CPU 数据传输
  ├─ 分块处理
  ├─ Column 0: Encode → Send → XOR (with_zero_parity)
  ├─ Column 1: Encode → Send → Recv → XOR (incremental)
  └─ 等待所有 encoding/XOR 完成（同步）

Phase 4: Post-XOR 数据交换（可选，根据配置）
  ├─ 同步屏障（barrier）- 确保所有 XOR 完成
  ├─ 根据 post_xor_steps 配置执行：
  │   ├─ Send: 发送本地 data 或 parity 到目标 rank
  │   └─ Recv: 从源 rank 接收 data 或 parity
  └─ 接收结果存入持久化内存

Phase 5: Checkpoint 保存
  └─ 写入文件系统（三段持久化内存）
```

---

## 📋 各 Rank 详细行为

### Rank 0 行为

#### **初始化阶段**

**1. 配置文件解析**
```python
# 从 eccheck_4rank.json 读取 ranks["0"] 配置
rank_config = {
    "columns": [
        {
            "column_idx": 0,
            "coefficient": 0,        # GF 矩阵系数
            "pipeline": [encode, send, xor]  # with_zero_parity
        },
        {
            "column_idx": 1,
            "coefficient": 1,        # GF 矩阵系数
            "pipeline": [encode, send, recv, xor]  # incremental
        }
    ]
}
```

**2. C++ Native 模块创建**
```cpp
// eccheck_native.cpp
ECCHECKNative(rank=0, world_size=4, paired_rank=2)
  ├─ 计算 k_ = 2, rows_ = 2
  ├─ 计算 data_block_index_ = 0  // rank / 2 = 0
  ├─ 生成 RS 编码矩阵 (k × m, m = k + rows = 4)
  ├─ 初始化 ISA-L 编码表 (g_tbls_)
  ├─ 启动 8 个线程：
  │   ├─ encoder_worker[0] (Column 0)
  │   ├─ encoder_worker[1] (Column 1)
  │   ├─ send_worker[0] (Column 0)
  │   ├─ send_worker[1] (Column 1)
  │   ├─ recv_worker[0] (Column 0)
  │   ├─ recv_worker[1] (Column 1)
  │   ├─ xor_worker[0] (Column 0)
  │   └─ xor_worker[1] (Column 1)
  └─ 初始化 NCCL communicators (等待所有 column 完成)
```

**3. NCCL 初始化（Column 0）**
```cpp
// Rank 0 创建 NCCL ID 并写入文件
ncclGetUniqueId(&nccl_id)
write_to_file("/tmp/eccheck_nccl_column0_id.txt")
ncclCommInitRank(&comm[0], world_size=4, nccl_id, rank=0)
// 初始化完成后，其他 rank (1, 2, 3) 才能读取并连接
```

**4. NCCL 初始化（Column 1）**
```cpp
// 同样的过程，但是独立的 communicator
ncclGetUniqueId(&nccl_id)
write_to_file("/tmp/eccheck_nccl_column1_id.txt")
ncclCommInitRank(&comm[1], world_size=4, nccl_id, rank=0)
```

#### **Phase 2: Metadata 交换**

**Rank 0 的行为**:
```python
# filesystem_async.py
1. 收集本地 metadata:
   - tensor_infos: 592 个 tensor 信息
   - non_tensor_data: 49 个非 tensor 项

2. 广播到所有 rank:
   - 使用 torch.distributed.broadcast
   - 接收其他 rank 的 metadata (rank 1: 292 tensors, rank 2, 3: ...)

3. 计算全局 metadata:
   - 总 tensor 数: 884
   - 总数据大小: 2.66 GB

4. 计算配对 rank (Rank 2) 的数据大小:
   - peer_data_size = rank2_tensor_total_size
   - 用于后续分配 receive buffers
```

#### **Phase 2.5: 缓冲区分配**

**Rank 0 分配**:
```python
# 基于 peer (Rank 2) 的数据大小
receive_buffer_size = aligned(rank2_data_size)  # 例如: 1.38 GB
own_total_size = rank0_tensor_total_size        # 例如: 1.34 GB

1. Data buffers: 12 个 (每个 64 MB)
   - tensor_buffer: 1.34 GB (所有 tensor 的原始数据)

2. Encoding buffers: 24 个 (每个 column 12 个)
   - Column 0: 12 buffers
   - Column 1: 12 buffers

3. Receive buffers: 2 个 (每个 1.38 GB)
   - receive_buffer[0]: Column 0 接收 Rank 1 的编码数据
   - receive_buffer[1]: Column 1 接收 Rank 1 的编码数据

4. Parity buffers: 44 个 (共享模型)
   - 所有 column 共享同一个 parity buffer per chunk

5. Persistent stores (三段持久化内存):
   a. 原始数据 (tensor_buffer):
      - 位置: self.tensor_buffer
      - 大小: 1.34 GB (own_total_size)
      - 内容: 所有 tensor 的原始数据
   
   b. 计算的 Parity (eccheck_persist_parity_store):
      - 位置: self.eccheck_persist_parity_store
      - 大小: aligned_own = 1.38 GB
      - 内容: 最终计算的 parity 数据（将在 Phase 3 中填充）
   
   c. 接收的 Data/Parity (eccheck_persist_recv_store 或 post_xor_buffers):
      - 位置: 
        * Phase 3: self.eccheck_persist_recv_store (如果配置了 persist.recv)
        * Phase 4: self.eccheck_post_xor_buffers['peer_data_buffer'] (Post-XOR 接收)
      - 大小: aligned_peer = 1.38 GB (基于 peer 数据大小)
      - 内容: 接收到的 peer rank 的数据或 parity
```

#### **Phase 3: 数据编码和交换**

**数据准备**:
```python
# GPU → CPU 传输
tensor_buffer = 1.34 GB  # Rank 0 的所有 tensor 数据
# 分块处理（假设 22 个 chunks）
```

**Column 0 处理流程**（每 chunk）:

**步骤 1: Encode**
```python
# Python 侧提交任务
_eccheck_native.submit_data_for_encoding(
    column_idx=0,
    data_addr=chunk_addr,
    size=chunk_size,
    encoding_addr=enc_buffer_addr[0],
    recv_addr=0,                    # Column 0 不需要 recv
    recv_chunk_size=0,
    parity_addr=shared_parity_addr, # 共享 parity buffer
    zero_parity_addr=zero_parity_addr,  # 零初始化 buffer
    xor_mode_int=1,                 # WITH_ZERO_PARITY
    send_count=1,
    recv_count=0
)
```

```cpp
// C++ encoder_worker[0] 处理
1. ec_encode_data() 编码:
   - 使用 coefficient=0 (GF 矩阵第 0 行)
   - encoding = encode(data, coefficient=0)

2. 提交 Send 任务:
   - send_queues_[0].push(SendTask(encoding_addr, size, count=1))
   - encoding_ref_counts_[encoding_addr] += 1

3. 提交 XOR 任务（直接，跳过 recv）:
   - xor_queues_[0].push(XorTask(
       local_encoding_addr=encoding_addr,
       recv_encoding_addr=zero_parity_addr,
       parity_addr=shared_parity_addr,
       size,
       mode=WITH_ZERO_PARITY
     ))
```

**步骤 2: Send**
```cpp
// send_worker[0] 执行
for (int i = 0; i < task.count; ++i) {  // count=1
    ncclSend(
        encoding_addr, 
        size, 
        ncclUint8, 
        peer_rank=1,    // 配置中 target_rank=1
        nccl_comms_[0], 
        stream=0
    )
}
// 发送完成后，decrement encoding_ref_counts_[encoding_addr]
```

**步骤 3: XOR**
```cpp
// xor_worker[0] 执行
if (task.mode == WITH_ZERO_PARITY) {
    // parity = encoding XOR zero_parity
    xor_gen(2, size, {
        source1: encoding_addr,
        source2: zero_parity_addr,
        dest: shared_parity_addr
    })
    // 初始化共享 parity buffer
}
// 确保为后续 incremental XOR 创建 mutex
```

**Column 1 处理流程**（每 chunk）:

**步骤 1: Encode**
```python
_eccheck_native.submit_data_for_encoding(
    column_idx=1,
    data_addr=chunk_addr,
    size=chunk_size,
    encoding_addr=enc_buffer_addr[1],
    recv_addr=recv_buffer_addr[1],    # Column 1 需要 recv
    recv_chunk_size=chunk_size,
    parity_addr=shared_parity_addr,   # 共享同一个 parity
    zero_parity_addr=0,               # 不需要
    xor_mode_int=2,                   # INCREMENTAL
    send_count=1,
    recv_count=1
)
```

```cpp
// encoder_worker[1] 处理
1. ec_encode_data() 编码:
   - 使用 coefficient=1 (GF 矩阵第 1 行)
   - encoding = encode(data, coefficient=1)

2. 提交 Send 任务:
   - send_queues_[1].push(SendTask(encoding_addr, size, count=1))

3. 提交 Recv 任务:
   - recv_queues_[1].push(RecvTask(recv_addr, size, count=1))
   - recv_to_xor_mappings_[1][recv_addr] = {
       encoding_addr, shared_parity_addr, size, INCREMENTAL
     }
```

**步骤 2: Send (并行)**
```cpp
// send_worker[1] 执行（与 recv 并行）
ncclSend(
    encoding_addr, 
    size, 
    ncclUint8, 
    peer_rank=1,    // 配置中 target_rank=1
    nccl_comms_[1], 
    stream=0
)
```

**步骤 3: Recv (并行)**
```cpp
// recv_worker[1] 执行（与 send 并行）
for (int i = 0; i < task.count; ++i) {  // count=1
    ncclRecv(
        recv_addr + i * size,  // offset
        size, 
        ncclUint8, 
        peer_rank=1,    // 配置中 source_rank=1
        nccl_comms_[1], 
        stream=0
    )
}
// 接收完成后，查找 mapping 并提交 XOR
xor_queues_[1].push(XorTask(
    local_encoding_addr=shared_parity_addr,  // 当前 parity
    recv_encoding_addr=recv_addr,            // 接收到的编码
    parity_addr=shared_parity_addr,          // 目标（原地更新）
    size,
    mode=INCREMENTAL
))
```

**步骤 4: XOR**
```cpp
// xor_worker[1] 执行（需要 mutex）
if (task.mode == INCREMENTAL) {
    // 获取 shared_parity_addr 的 mutex
    std::lock_guard<std::mutex> lock(*parity_mutexes_[parity_addr])
    
    // parity = parity XOR recv_encoding (原地更新)
    xor_gen(3, size, {
        source1: shared_parity_addr,    // 当前 parity
        source2: recv_addr,              // 接收到的编码
        dest: shared_parity_addr        // 原地更新
    })
}
```

#### **Phase 3 完成：等待所有 Encoding/XOR 完成**

```python
# 等待所有 encoding 和 XOR 操作完成
wait_for_encoding_completion()

# 此时状态：
# - Column 0: 所有 chunks 的共享 parity 已初始化 (WITH_ZERO_PARITY)
# - Column 1: 所有 chunks 的共享 parity 已更新 (INCREMENTAL)
# - parity 数据在各个 shared_parity_addr 中
# - 如果启用了 persist_parity，数据已 memcpy 到 eccheck_persist_parity_store
```

#### **Phase 4: Post-XOR 数据交换**

**配置解析**:
```python
# 从配置文件中读取 post_xor_steps
post_xor_steps = rank_config.get('post_xor_steps', [])
# 例如：
# [
#   {"step": "sync", "type": "barrier", "sync_point": "after_all_xor", "ranks": [0, 1, 2, 3]},
#   {"step": "send", "type": "nccl_send", "target_rank": 1, "data_type": "parity", "data_source": "local_parity"},
#   {"step": "recv", "type": "nccl_recv", "source_rank": 1, "data_type": "data", "data_target": "peer_data_buffer"}
# ]
```

**执行流程**:
```python
# _execute_post_xor_steps() 方法在 _eccheck_phase3_encoding() 中调用
# 位置：wait_for_encoding_completion() 之后

def _execute_post_xor_steps(self):
    """
    执行 Post-XOR 步骤：
    1. 同步屏障确保所有 XOR 完成
    2. 根据配置发送本地 data/parity
    3. 根据配置接收 peer 的 data/parity
    4. 结果存入持久化内存
    """
    
    # 步骤 1: 同步屏障
    if step == "sync" and type == "barrier":
        torch.distributed.barrier()  # 确保所有 rank 的 XOR 都完成
    
    # 步骤 2: Post-XOR Send
    if step == "send" and type == "nccl_send":
        # 根据 data_source 选择缓冲区地址
        if data_source == "local_parity":
            send_addr = int(self.eccheck_persist_parity_store.data_ptr())
            send_size = self.eccheck_persist_parity_store.numel()
        elif data_source == "local_data":
            send_addr = int(self.tensor_buffer.data_ptr())
            send_size = self.tensor_buffer.numel()
        
        # 调用 C++ API (使用 Column 0 的 NCCL communicator)
        self._eccheck_native.post_xor_send(
            target_rank=target_rank,
            data_addr=send_addr,
            size=send_size
        )
    
    # 步骤 3: Post-XOR Recv
    if step == "recv" and type == "nccl_recv":
        # 根据 data_target 选择或分配接收缓冲区
        if data_target == "peer_data_buffer":
            # 分配新缓冲区存储 peer 的原始数据
            if 'peer_data_buffer' not in self.eccheck_post_xor_buffers:
                peer_size = sum(meta.size_bytes for meta in peer_metadata)
                aligned_size = align(peer_size)
                self.eccheck_post_xor_buffers['peer_data_buffer'] = \
                    torch.empty(aligned_size, dtype=torch.uint8, pin_memory=True)
            
            recv_addr = int(self.eccheck_post_xor_buffers['peer_data_buffer'].data_ptr())
            recv_size = self.eccheck_post_xor_buffers['peer_data_buffer'].numel()
            
        elif data_target == "peer_parity_buffer":
            # 使用 persist_recv_store 存储 peer 的 parity
            recv_addr = int(self.eccheck_persist_recv_store.data_ptr())
            recv_size = self.eccheck_persist_recv_store.numel()
        
        # 调用 C++ API (使用 Column 0 的 NCCL communicator)
        self._eccheck_native.post_xor_recv(
            source_rank=source_rank,
            recv_addr=recv_addr,
            size=recv_size
        )
```

**重要说明**:
- **Post-XOR 步骤由 Python 侧执行**，不是 C++ 线程
- **使用 Column 0 的 NCCL communicator** (`nccl_comms_[0]`) 进行通信
- **缓冲区地址已确定**：
  - `local_parity`: `eccheck_persist_parity_store` (已分配)
  - `local_data`: `tensor_buffer` (已分配)
  - `peer_data_buffer`: 动态分配（如果不存在）
  - `peer_parity_buffer`: `eccheck_persist_recv_store` (已分配)
- **配置驱动**：所有行为都从配置文件的 `post_xor_steps` 读取

**C++ 实现**:
```cpp
// eccheck_native.cpp

void post_xor_send(int target_rank, uintptr_t data_addr, size_t size) {
    // 使用 Column 0 的 NCCL communicator
    ncclSend(
        reinterpret_cast<void*>(data_addr),
        size,
        ncclUint8,
        target_rank,
        nccl_comms_[0],  // Column 0
        stream=0
    )
}

void post_xor_recv(int source_rank, uintptr_t recv_addr, size_t size) {
    // 使用 Column 0 的 NCCL communicator
    ncclRecv(
        reinterpret_cast<void*>(recv_addr),
        size,
        ncclUint8,
        source_rank,
        nccl_comms_[0],  // Column 0
        stream=0
    )
}
```

#### **Phase 5: Checkpoint 保存**

**三段持久化内存**:
```python
# 最终有三段持久化内存：

1. 原始数据 (tensor_buffer):
   - 位置: self.tensor_buffer
   - 大小: own_total_size (Rank 自己的数据大小)
   - 内容: 所有 tensor 的原始数据（GPU → CPU 传输后）

2. 计算的 Parity (eccheck_persist_parity_store):
   - 位置: self.eccheck_persist_parity_store
   - 大小: aligned_own (基于 own_total_size)
   - 内容: 所有 chunks 的共享 parity 数据的最终结果
   - 来源: Column 0 初始化和 Column 1 增量更新后

3. 接收的 Data/Parity (eccheck_post_xor_buffers 或 eccheck_persist_recv_store):
   - 位置: 
     * peer_data_buffer: self.eccheck_post_xor_buffers['peer_data_buffer']
     * peer_parity_buffer: self.eccheck_persist_recv_store
   - 大小: 
     * peer_data_buffer: 基于 peer 的原始数据大小
     * peer_parity_buffer: aligned_peer (基于 peer_total_size)
   - 内容: Post-XOR 步骤中接收到的 peer rank 的数据或 parity
   - 来源: Post-XOR recv 操作

# 写入文件系统（如果需要）
# 这三段内存可以用于容错恢复（虽然实际上只需要两段，但保留三段用于完整数据）
```

---

### Rank 1 行为

Rank 1 与 Rank 0 **对称**，但配对关系不同：

#### **关键差异**

**配对关系**:
- Rank 1 的 paired_rank = 1 + 2 = **3** (not 2)
- 但在配置文件中，Column 0 和 Column 1 都配置为与 Rank 0 通信

**配置解析**:
```json
// ranks["1"] 配置
{
  "columns": [
    {
      "column_idx": 0,
      "pipeline": [
        {"step": "encode"},
        {"step": "send", "target_rank": 0},  // 发送给 Rank 0
        {"step": "xor", "mode": "with_zero_parity"}
      ]
    },
    {
      "column_idx": 1,
      "pipeline": [
        {"step": "encode"},
        {"step": "send", "target_rank": 0},
        {"step": "recv", "source_rank": 0},  // 从 Rank 0 接收
        {"step": "xor", "mode": "incremental"}
      ]
    }
  ]
}
```

**数据流**:
- Rank 1 Column 0: Send → Rank 0 Column 0 (接收)
- Rank 1 Column 1: Send → Rank 0 Column 1, Recv ← Rank 0 Column 1

---

### Rank 2 行为

Rank 2 与 Rank 0 **相似**，但配对关系不同：

#### **关键差异**

**配对关系**:
- Rank 2 的 paired_rank = 2 - 2 = **0** (not 1)
- 配置文件：Column 与 Rank 3 通信

**配置解析**:
```json
// ranks["2"] 配置
{
  "columns": [
    {
      "column_idx": 0,
      "pipeline": [
        {"step": "encode"},
        {"step": "send", "target_rank": 3},  // 发送给 Rank 3
        {"step": "xor", "mode": "with_zero_parity"}
      ]
    },
    {
      "column_idx": 1,
      "pipeline": [
        {"step": "encode"},
        {"step": "send", "target_rank": 3},
        {"step": "recv", "source_rank": 3},  // 从 Rank 3 接收
        {"step": "xor", "mode": "incremental"}
      ]
    }
  ]
}
```

**数据流**:
- Rank 2 Column 0: Send → Rank 3 Column 0
- Rank 2 Column 1: Send → Rank 3 Column 1, Recv ← Rank 3 Column 1

---

### Rank 3 行为

Rank 3 与 Rank 2 **对称**：

#### **配置解析**
```json
// ranks["3"] 配置
{
  "columns": [
    {
      "column_idx": 0,
      "pipeline": [
        {"step": "encode"},
        {"step": "send", "target_rank": 2},  // 发送给 Rank 2
        {"step": "xor", "mode": "with_zero_parity"}
      ]
    },
    {
      "column_idx": 1,
      "pipeline": [
        {"step": "encode"},
        {"step": "send", "target_rank": 2},
        {"step": "recv", "source_rank": 2},  // 从 Rank 2 接收
        {"step": "xor", "mode": "incremental"}
      ]
    }
  ]
}
```

---

## 🔗 通信模式可视化

### NCCL 通信图

```
Rank 0                    Rank 1
├─ Column 0              ├─ Column 0
│  └─ Send ──────────────→│  (接收)
│                          │
├─ Column 1              ├─ Column 1
│  ├─ Send ──────────────→│  (接收)
│  └─ Recv ←──────────────│  Send
│                          │
└─ Shared Parity          └─ Shared Parity

Rank 2                    Rank 3
├─ Column 0              ├─ Column 0
│  └─ Send ──────────────→│  (接收)
│                          │
├─ Column 1              ├─ Column 1
│  ├─ Send ──────────────→│  (接收)
│  └─ Recv ←──────────────│  Send
│                          │
└─ Shared Parity          └─ Shared Parity
```

### 配对关系总结

| Rank | Paired Rank (EC-CHECK) | Column 0 Send | Column 1 Send/Recv | 说明 |
|------|------------------------|---------------|-------------------|------|
| 0 | 2 | → Rank 1 | → Rank 1 / ← Rank 1 | 配置中指定与 Rank 1 通信 |
| 1 | 3 | → Rank 0 | → Rank 0 / ← Rank 0 | 配置中指定与 Rank 0 通信 |
| 2 | 0 | → Rank 3 | → Rank 3 / ← Rank 3 | 配置中指定与 Rank 3 通信 |
| 3 | 1 | → Rank 2 | → Rank 2 / ← Rank 2 | 配置中指定与 Rank 2 通信 |

**重要说明**: 
- **EC-CHECK 配对 rank** (0↔2, 1↔3) 是由 `_get_paired_rank()` 自动计算的，用于：
  - 确定配对关系（metadata 交换、缓冲区大小计算）
  - 如果配置文件中 `send_peer`/`recv_peer` 为 `-1`，会自动使用 paired_rank
  
- **实际通信目标**：遵循配置文件中的 `target_rank` 和 `source_rank`，可能与配对 rank 不同
  
- **当前配置**：各 rank 的通信目标是相邻 rank（0↔1, 2↔3），这是合理的配置，但也可以改为配对 rank（0↔2, 1↔3）

---

## 🔄 数据流时序图

### Chunk 处理时序（以 Rank 0 为例）

```
Time →

Rank 0 Column 0:
  Encode ─┐
          ├─→ Send (to Rank 1) ─┐
          └─→ XOR (with_zero_parity) ─┐
                                       └─→ Shared Parity Initialized

Rank 0 Column 1:
  Encode ─┐
          ├─→ Send (to Rank 1) ───┐
          └─→ Recv (from Rank 1) ─┼─┐
                                  │ └─→ XOR (incremental) ─→ Update Shared Parity
                                  │
Rank 1 Column 0:                   │
  Encode ─┐                       │
          └─→ Send (to Rank 0) ───┘ (Column 1 接收)

Rank 1 Column 1:
  Encode ─┐
          ├─→ Send (to Rank 0) ───┐ (Column 1 接收)
          └─→ Recv (from Rank 0) ─┘
```

---

## 🧵 C++ 线程行为详解

### Rank 0 的 8 个线程

#### **Encoder Threads**

**encoder_worker[0]** (Column 0):
```cpp
while (!should_stop) {
    1. 从 encoding_tasks_[0] 队列取出任务
    2. 使用 coefficient=0 编码数据
    3. 提交 Send 任务到 send_queues_[0]
    4. 直接提交 XOR 任务（WITH_ZERO_PARITY，跳过 recv）
    5. 增加 encoding_ref_counts_
}
```

**encoder_worker[1]** (Column 1):
```cpp
while (!should_stop) {
    1. 从 encoding_tasks_[1] 队列取出任务
    2. 使用 coefficient=1 编码数据
    3. 提交 Send 任务到 send_queues_[1]
    4. 提交 Recv 任务到 recv_queues_[1]
    5. 增加 encoding_ref_counts_
}
```

#### **Send Threads**

**send_worker[0]** (Column 0):
```cpp
while (!should_stop) {
    1. 从 send_queues_[0] 取出 SendTask
    2. 执行 ncclSend(encoding_addr, peer=1, comm[0])
    3. 循环 task.count 次
    4. Decrement encoding_ref_counts_[encoding_addr]
}
```

**send_worker[1]** (Column 1):
```cpp
while (!should_stop) {
    1. 从 send_queues_[1] 取出 SendTask
    2. 执行 ncclSend(encoding_addr, peer=1, comm[1])
    3. 循环 task.count 次
    4. Decrement encoding_ref_counts_[encoding_addr]
}
```

#### **Recv Threads**

**recv_worker[0]** (Column 0):
```cpp
// Column 0 不需要 recv，线程可能空闲或处理其他任务
```

**recv_worker[1]** (Column 1):
```cpp
while (!should_stop) {
    1. 从 recv_queues_[1] 取出 RecvTask
    2. 执行 ncclRecv(recv_addr, peer=1, comm[1])
    3. 循环 task.count 次（使用偏移量）
    4. 查找 recv_to_xor_mappings_[1][recv_addr]
    5. 提交 XOR 任务到 xor_queues_[1] (INCREMENTAL 模式)
}
```

#### **XOR Threads**

**xor_worker[0]** (Column 0):
```cpp
while (!should_stop) {
    1. 从 xor_queues_[0] 取出 XorTask
    2. 如果 mode == WITH_ZERO_PARITY:
       - xor_gen(encoding, zero_parity, shared_parity)
       - 初始化共享 parity buffer
    3. 为后续 incremental XOR 创建 mutex
}
```

**xor_worker[1]** (Column 1):
```cpp
while (!should_stop) {
    1. 从 xor_queues_[1] 取出 XorTask
    2. 如果 mode == INCREMENTAL:
       - 获取 shared_parity_addr 的 mutex
       - xor_gen(shared_parity, recv_encoding, shared_parity)
       - 原地更新共享 parity buffer
    3. 释放 mutex
}
```

---

## 📊 内存布局（Rank 0 示例）

```
CPU Memory Layout (Rank 0):

┌─────────────────────────────────────────┐
│ 临时工作缓冲区 (Phase 3 使用)            │
├─────────────────────────────────────────┤
│ Data Buffers (12 × 64 MB)                │
│ └─ tensor_buffer: 1.34 GB              │
│    └─ Phase 2: GPU → CPU 传输后的原始数据│
├─────────────────────────────────────────┤
│ Encoding Buffers (24 × 64 MB)          │
│ ├─ Column 0: 12 buffers                 │
│ │  └─ 存储编码后的数据                   │
│ └─ Column 1: 12 buffers                 │
│    └─ 存储编码后的数据                   │
├─────────────────────────────────────────┤
│ Receive Buffers (Phase 3 编码交换)      │
│ ├─ receive_buffer[0]: 1.38 GB          │
│ │  └─ Column 0 接收 Rank 1 的编码数据   │
│ └─ receive_buffer[1]: 1.38 GB          │
│    └─ Column 1 接收 Rank 1 的编码数据   │
├─────────────────────────────────────────┤
│ Parity Buffers (44 × 64 MB) - 共享模型  │
│ └─ Shared Parity (所有 column 共享)    │
│    ├─ Chunk 0: parity_buffer[0]        │
│    │  └─ Column 0 初始化, Column 1 更新 │
│    ├─ Chunk 1: parity_buffer[1]        │
│    └─ ... (44 chunks)                  │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ 三段持久化内存 (Phase 3/4/5 使用)         │
├─────────────────────────────────────────┤
│ 1. 原始数据 (tensor_buffer)             │
│    └─ 大小: 1.34 GB                    │
│    └─ 内容: Rank 0 的所有 tensor 原始数据 │
│    └─ 用途: 容错恢复的原始数据源          │
├─────────────────────────────────────────┤
│ 2. 计算的 Parity                        │
│    └─ eccheck_persist_parity_store      │
│    └─ 大小: 1.38 GB (aligned)          │
│    └─ 内容: 所有 chunks 的最终 parity   │
│    └─ 来源: Column 0 初始化 + Column 1 增量更新
│    └─ 用途: 容错恢复的 parity 数据      │
├─────────────────────────────────────────┤
│ 3. 接收的 Data/Parity                   │
│    ├─ Phase 3: eccheck_persist_recv_store│
│    │  └─ 大小: 1.38 GB                │
│    │  └─ 内容: Phase 3 接收的编码数据   │
│    └─ Phase 4: peer_data_buffer /       │
│                 peer_parity_buffer      │
│       └─ 大小: 基于配置动态分配          │
│       └─ 内容: Post-XOR 接收的 peer 数据/parity
│       └─ 用途: 容错恢复的 peer 数据      │
└─────────────────────────────────────────┘

注意：
- 临时缓冲区在 Phase 3 中使用，之后可以释放
- 持久化内存用于容错恢复（虽然只需要两段，但保留三段用于完整数据）
- Phase 4 的 post-xor 接收可能使用 persist_recv_store 或新分配的缓冲区
```

---

## 🔍 关键实现细节

### 1. 共享 Parity Buffer 模型

**每个 Chunk**:
- 所有 Column 共享**同一个** `shared_parity_addr`
- Column 0: 使用 `WITH_ZERO_PARITY` 初始化
- Column 1: 使用 `INCREMENTAL` 更新（需要 mutex）

**Mutex 保护**:
```cpp
// C++ 端为每个 parity buffer 创建 mutex
std::lock_guard<std::mutex> lock(*parity_mutexes_[parity_addr]);
xor_gen(shared_parity, recv_encoding, shared_parity);
```

### 2. GF 编码系数选择

**ISA-L EC 编码**:
```cpp
// 每个 column 使用不同的 GF 矩阵行
ec_encode_data(
    k=2,                    // 数据块数
    rows=2,                 // 编码块数（parity）
    g_tbls_,                // GF 表（32 × k × rows）
    data_srcs,              // 数据源（rank 0, 1）
    encoding_dest,          // 编码目标
    coeff_row=column_idx    // 使用 column_idx 作为 GF 矩阵行号
)
```

**系数映射**:
- Column 0 → coefficient=0 → GF 矩阵第 0 行
- Column 1 → coefficient=1 → GF 矩阵第 1 行

### 3. NCCL 通信隔离

**独立的 Communicator**:
- Column 0: `nccl_comms_[0]` (独立 ID 文件)
- Column 1: `nccl_comms_[1]` (独立 ID 文件)
- 每个 Column 可以并行通信，互不干扰

### 4. 引用计数管理

**Encoding Buffer 生命周期**:
```cpp
// 编码完成后，等待所有引用完成
encoding_ref_counts_[encoding_addr] = send_count  // 初始计数

// Send 完成后递减
--encoding_ref_counts_[encoding_addr]

// 计数为 0 时，buffer 可以释放/重用
```

---

## ✅ 验证检查点

### 每个 Rank 应该完成的动作

**Rank 0**:
- ✅ 编码所有数据（Column 0: coeff=0, Column 1: coeff=1）
- ✅ 发送编码数据到 Rank 1
- ✅ 从 Rank 1 接收编码数据（Column 1）
- ✅ 初始化共享 parity (Column 0)
- ✅ 增量更新共享 parity (Column 1)

**Rank 1**:
- ✅ 编码所有数据（Column 0: coeff=0, Column 1: coeff=1）
- ✅ 发送编码数据到 Rank 0
- ✅ 从 Rank 0 接收编码数据（Column 1）
- ✅ 初始化共享 parity (Column 0)
- ✅ 增量更新共享 parity (Column 1)

**Rank 2**:
- ✅ 编码所有数据（Column 0: coeff=0, Column 1: coeff=1）
- ✅ 发送编码数据到 Rank 3
- ✅ 从 Rank 3 接收编码数据（Column 1）
- ✅ 初始化共享 parity (Column 0)
- ✅ 增量更新共享 parity (Column 1)

**Rank 3**:
- ✅ 编码所有数据（Column 0: coeff=0, Column 1: coeff=1）
- ✅ 发送编码数据到 Rank 2
- ✅ 从 Rank 2 接收编码数据（Column 1）
- ✅ 初始化共享 parity (Column 0)
- ✅ 增量更新共享 parity (Column 1)

---

## 📝 总结

### 核心要点

1. **4 个 Rank 独立处理各自数据**，使用相同的配置模式
2. **每个 Rank 有 2 个 Column**，每个 Column 有独立的线程池
3. **共享 Parity Buffer 模型**：所有 Column 共享同一个 parity per chunk
4. **配对关系**：Rank 0↔2, Rank 1↔3，但实际通信遵循配置文件
5. **并行执行**：Encode、Send、Recv、XOR 可以流水线并行执行
6. **线程安全**：Incremental XOR 使用 mutex 保护共享 parity buffer
7. **Post-XOR 步骤**：
   - 在 XOR 同步后执行，由 Python 侧根据配置驱动
   - 使用 Column 0 的 NCCL communicator 进行通信
   - 结果存入持久化内存
8. **三段持久化内存**：
   - 原始数据 (tensor_buffer)
   - 计算的 Parity (eccheck_persist_parity_store)
   - 接收的 Data/Parity (post_xor_buffers 或 persist_recv_store)

### 数据流路径

```
GPU Data → CPU Buffer → Encode (GF) → Send/Recv (NCCL) → XOR (ISA-L) → Parity
```

所有 Rank 遵循相同的流程，但使用不同的配置参数（peer rank、coefficient）和独立的 NCCL communicators。

