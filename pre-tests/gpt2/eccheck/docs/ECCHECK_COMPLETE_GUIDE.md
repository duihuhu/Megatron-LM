# EC-CHECK 完整指南

> **Erasure Coding Checkpointing (EC-CHECK)** 是一个基于纠删码的容错检查点系统，用于 Megatron-LM 分布式训练。

## 📑 目录

1. [High-Level 概述](#high-level-概述)
2. [核心概念](#核心概念)
3. [架构设计](#架构设计)
4. [代码组织](#代码组织)
5. [配置格式](#配置格式)
6. [执行流程](#执行流程)
7. [4-Rank (2+2) 示例详解](#4-rank-22-示例详解)
8. [API 参考](#api-参考)
9. [故障排除](#故障排除)

---

## High-Level 概述

### 什么是 EC-CHECK？

EC-CHECK 是一个**容错的分布式检查点系统**，通过纠删码（Erasure Coding）技术在多个 GPU/rank 之间共享数据，实现：

1. **容错能力**: 即使部分 rank 失败，也能从其他 rank 恢复数据
2. **并行效率**: 多 rank 并行编码和交换数据
3. **存储优化**: 通过编码减少存储开销

### 核心思想

```
传统 Checkpoint:
  Rank 0: [Data 0] → File 0
  Rank 1: [Data 1] → File 1
  Rank 2: [Data 2] → File 2
  Rank 3: [Data 3] → File 3
  如果 Rank 0 失败 → 数据丢失 ❌

EC-CHECK:
  Rank 0: [Data 0] → Encode → Send/Recv → XOR → [Parity]
  Rank 1: [Data 1] → Encode → Send/Recv → XOR → [Parity]
  Rank 2: [Data 2] → Encode → Send/Recv → XOR → [Parity]
  Rank 3: [Data 3] → Encode → Send/Recv → XOR → [Parity]
  如果 Rank 0 失败 → 从 Rank 1,2,3 和 Parity 恢复 ✅
```

### 关键技术

- **纠删码 (EC)**: Reed-Solomon 编码，参数 k=数据节点数, m=校验节点数
- **NCCL 通信**: GPU 间高速数据传输
- **GF(2^8) 运算**: Intel ISA-L 库提供的 Galois Field 运算
- **多线程流水线**: Encode → Send → Recv → XOR 并行执行

---

## 核心概念

### 1. Rank 和配对 (Rank Pairing)

**Rank**: 分布式训练中的一个进程/GPU，编号从 0 开始。

**配对关系**: EC-CHECK 将 rank 配对进行数据交换：
```
规则: rank_i 配对 rank_{i + world_size/2}

例如 4-rank:
  Rank 0 ↔ Rank 2  (0 + 4/2 = 2)
  Rank 1 ↔ Rank 3  (1 + 4/2 = 3)
```

**代码位置**:
```python
# megatron/core/dist_checkpointing/strategies/torch.py:790
def _get_paired_rank(self, my_rank: int, world_size: int) -> int:
    return (my_rank + world_size // 2) % world_size
```

### 2. Column 和多列处理

**Column**: 每个 rank 可以有多个 column（列），每个 column 对应一个编码系数（coefficient）。

**用途**:
- Column 0: 使用系数 0 编码，通常用于初始化共享 parity
- Column 1+: 使用系数 1, 2, ... 编码，用于增量更新 parity

**多列好处**: 允许一个 rank 处理多个数据块并行编码。

**代码位置**:
```cpp
// megatron/core/dist_checkpointing/strategies/eccheck_native.cpp:41
static constexpr int MAX_COLUMNS = 32;  // 最大支持 32 列
```

### 3. 共享 Parity Buffer 模型

**Parity**: 通过 XOR 操作计算的校验数据，用于容错恢复。

**共享模型**: 所有 column 共享同一个 parity buffer（按 chunk 划分），而不是每个 column 独立 parity。

**执行流程**:
1. **Column 0**: 初始化 parity = `encoding_0 XOR zero_parity`
2. **Column 1+**: 增量更新 parity = `parity XOR encoding_i`

**线程安全**: 使用 mutex 保护共享 parity buffer 的并发访问。

**代码位置**:
```cpp
// megatron/core/dist_checkpointing/strategies/eccheck_native.cpp:646
// XOR worker with mutex protection for shared parity
std::lock_guard<std::mutex> parity_lock(parity_mutex_[col]);
```

### 4. Pipeline 执行模型

每个 column 的 pipeline 包含多个步骤：

```
Pipeline = [
  Encode  → GF(2^8) 编码，使用系数 coefficient
  Send    → NCCL 发送编码数据到目标 rank
  Recv    → NCCL 接收配对 rank 的编码数据
  XOR     → 执行 XOR 操作（多种模式）
]
```

**XOR 模式**:
- `with_zero_parity`: `parity = encoding XOR zero_parity` (初始化)
- `with_recv_encoding`: `parity = encoding XOR recv_encoding` (标准模式)
- `incremental`: `parity = parity XOR recv_encoding` (增量更新)

**代码位置**:
```cpp
// megatron/core/dist_checkpointing/strategies/eccheck_native.cpp:44-48
enum class XorMode {
    WITH_RECV_ENCODING = 0,
    WITH_ZERO_PARITY = 1,
    INCREMENTAL = 2
};
```

### 5. Post-XOR 数据交换

**Post-XOR**: 在所有 XOR 操作完成后，额外的数据交换步骤。

**用途**:
- 交换最终的 parity 结果
- 交换 peer rank 的原始数据（用于容错恢复）

**执行时机**: 在所有 pipeline 的 XOR 完成后，通过同步屏障确保一致性。

**代码位置**:
```python
# megatron/core/dist_checkpointing/strategies/filesystem_async.py:1716
self._eccheck_native.wait_for_encoding_completion()
self._execute_post_xor_steps()  # 执行 post-XOR 步骤
```

### 6. 三段持久化内存

EC-CHECK 保存三类数据到 checkpoint：

1. **原始数据** (`tensor_buffer`): rank 的原始 tensor 数据
2. **计算的 Parity** (`eccheck_persist_parity_store`): 通过 XOR 计算的 parity
3. **接收的数据/Parity** (`eccheck_post_xor_buffers`): 从其他 rank 接收的数据或 parity

**容错恢复**: 这三段数据组合可以恢复任意 rank 的数据。

---

## 架构设计

### 分层架构

```
┌─────────────────────────────────────────────┐
│        Python 策略层 (torch.py)              │
│  - TorchDistSaveShardedStrategy             │
│  - 预初始化 C++ native 模块                  │
│  - 生命周期管理                               │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│     Python 执行层 (filesystem_async.py)      │
│  - FileSystemWriterAsync                    │
│  - 配置解析                                   │
│  - Pipeline 调度                              │
│  - Post-XOR 执行                              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│      C++ Native 层 (eccheck_native.cpp)      │
│  - ECCHECKNative 类                          │
│  - 多线程工作池 (encoder, send, recv, xor)   │
│  - NCCL 通信                                  │
│  - GF(2^8) 编码 (ISA-L)                      │
│  - XOR 运算 (ISA-L RAID)                     │
└─────────────────────────────────────────────┘
```

### 线程模型

每个 rank 每个 column 有 4 个工作线程：

```
Column 0:
  ├─ encoder_thread[0]  → 执行 GF(2^8) 编码
  ├─ send_worker[0]     → NCCL 发送编码数据
  ├─ recv_worker[0]     → NCCL 接收编码数据
  └─ xor_worker[0]      → 执行 XOR 操作

Column 1:
  ├─ encoder_thread[1]  → 执行 GF(2^8) 编码
  ├─ send_worker[1]     → NCCL 发送编码数据
  ├─ recv_worker[1]     → NCCL 接收编码数据
  └─ xor_worker[1]      → 执行 XOR 操作（增量更新）
```

**代码位置**:
```cpp
// megatron/core/dist_checkpointing/strategies/eccheck_native.cpp:823-828
for (int i = 0; i < num_columns_; ++i) {
    encoder_threads_[i].emplace(&ECCHECKNative::encoder_worker, this, i);
    send_workers_[i].emplace(&ECCHECKNative::send_worker, this, i);
    recv_workers_[i].emplace(&ECCHECKNative::recv_worker, this, i);
    xor_workers_[i].emplace(&ECCHECKNative::xor_worker, this, i);
}
```

### 数据流

```
GPU Tensor Data
    ↓ (GPU → CPU 传输)
CPU Buffer (data_buffer)
    ↓ (GF(2^8) 编码)
Encoding Buffer (encoding_buffer)
    ↓ (NCCL Send/Recv)
Received Encoding Buffer (recv_encoding_buffer)
    ↓ (XOR 操作)
Parity Buffer (parity_buffer)
    ↓ (Post-XOR 交换)
Persistent Memory
    ├─ tensor_buffer (原始数据)
    ├─ eccheck_persist_parity_store (parity)
    └─ eccheck_post_xor_buffers (接收的数据/parity)
    ↓ (写入文件系统)
Checkpoint Files (.distcp)
```

---

## 代码组织

### 核心文件

#### 1. Python 策略层

**文件**: `megatron/core/dist_checkpointing/strategies/torch.py`

**类**: `TorchDistSaveShardedStrategy`

**职责**:
- 策略级别的 EC-CHECK 初始化
- 预创建 `ECCHECKNative` 实例（生命周期管理）
- 传递给 `FileSystemWriterAsync`

**关键方法**:
```python
# Line 727: 初始化 C++ native 模块
def _init_eccheck_native(self):
    # 加载配置文件读取 k, m
    # 创建 ECCHECKNative(rank, world_size, paired_rank, k, m)
    
# Line 934: 异步保存入口
def async_save(self, sharded_state_dict, checkpoint_dir):
    # 创建 FileSystemWriterAsync，传入预初始化的 _eccheck_native
```

#### 2. Python 执行层

**文件**: `megatron/core/dist_checkpointing/strategies/filesystem_async.py`

**类**: `FileSystemWriterAsync`

**职责**:
- 配置解析（JSON/YAML）
- Pipeline 调度和执行
- Post-XOR 步骤执行
- 数据持久化

**关键方法**:
```python
# Line 223: 加载配置文件
def _load_eccheck_config(self):
    # 读取 ec_params (k, m)
    # 读取 persist 设置
    # 读取 ranks 配置

# Line 196: 初始化 C++ native（如果没有传入）
def _init_eccheck_native(self):
    # 创建 ECCHECKNative 实例

# Line 1962: 解析和应用列配置
def _init_eccheck_buffers(self):
    # 解析 columns 配置
    # 分配缓冲区

# Line 2234: 执行 Post-XOR 步骤
def _execute_post_xor_steps(self):
    # 解析 post_xor_steps 配置
    # 执行 sync, send, recv 步骤

# Line 1458: Phase 3 主流程
def _eccheck_tensor_data_exchange_and_encoding(self):
    # 执行编码和交换
    # 等待完成
    # 执行 post-XOR
```

#### 3. C++ Native 层

**文件**: `megatron/core/dist_checkpointing/strategies/eccheck_native.cpp`

**类**: `ECCHECKNative`

**职责**:
- GF(2^8) 编码（ISA-L）
- NCCL 通信
- XOR 运算（ISA-L RAID）
- 多线程协调

**关键方法**:
```cpp
// Line 836: 构造函数
ECCHECKNative(int rank, int world_size, int paired_rank, int k = -1, int m = -1):
    // 初始化 EC 参数 (k, m)
    // 启动线程池

// Line 462: 编码工作线程
void encoder_worker(int col):
    // 从队列获取任务
    // GF(2^8) 编码 (isa-l erasure_code_encode)

// Line 530: 发送工作线程
void send_worker(int col):
    // NCCL 发送编码数据

// Line 567: 接收工作线程
void recv_worker(int col):
    // NCCL 接收编码数据

// Line 610: XOR 工作线程
void xor_worker(int col):
    // 根据模式执行 XOR (isa-l raid)
    // 线程安全保护共享 parity

// Line 1220: Post-XOR 发送
void post_xor_send(int target_rank, uintptr_t data_addr, size_t size):
    // 使用 Column 0 的 NCCL communicator 发送

// Line 1244: Post-XOR 接收
void post_xor_recv(int source_rank, uintptr_t recv_addr, size_t size):
    // 使用 Column 0 的 NCCL communicator 接收
```

#### 4. 配置文件

**文件**: `pre-tests/gpt2/eccheck/configs/eccheck_4rank.json`

**结构**:
```json
{
  "ec_params": { "k": 2, "m": 2 },
  "persist": { "recv": true, "parity": true },
  "ranks": {
    "0": { "columns": [...], "post_xor_steps": [...] },
    ...
  }
}
```

---

## 配置格式

### 顶层结构

```json
{
  "ec_params": {
    "k": 2,  // 数据节点数
    "m": 2   // 校验节点数（parity rows）
  },
  "persist": {
    "recv": true,   // 是否持久化接收的数据
    "parity": true  // 是否持久化计算的 parity
  },
  "ranks": {
    "0": { ... },
    "1": { ... },
    ...
  }
}
```

### Rank 配置结构

```json
{
  "ranks": {
    "0": {
      "columns": [
        {
          "column_idx": 0,
          "coefficient": 0,
          "pipeline": [
            { "step": "encode", "type": "ec_encode" },
            { "step": "send", "type": "nccl_send", "target_rank": 1, "count": 1 },
            { "step": "xor", "type": "xor_update", "mode": "with_zero_parity" }
          ]
        },
        {
          "column_idx": 1,
          "coefficient": 1,
          "pipeline": [
            { "step": "encode", "type": "ec_encode" },
            { "step": "send", "type": "nccl_send", "target_rank": 1, "count": 1 },
            { "step": "recv", "type": "nccl_recv", "source_rank": 1, "count": 1 },
            { "step": "xor", "type": "xor_update", "mode": "incremental" }
          ]
        }
      ],
      "post_xor_steps": [
        { "step": "sync", "type": "barrier", "sync_point": "after_all_xor" },
        { "step": "send", "type": "nccl_send", "target_rank": 2, "data_type": "parity" },
        { "step": "recv", "type": "nccl_recv", "source_rank": 2, "data_type": "data" }
      ]
    }
  }
}
```

### Pipeline Steps 说明

#### 1. Encode Step
```json
{
  "step": "encode",
  "type": "ec_encode"
}
```
- 使用 GF(2^8) 编码，系数由 `coefficient` 指定

#### 2. Send Step
```json
{
  "step": "send",
  "type": "nccl_send",
  "target_rank": 1,
  "count": 1
}
```
- `target_rank`: 目标 rank
- `count`: 发送次数（支持多次发送同一数据）

#### 3. Recv Step
```json
{
  "step": "recv",
  "type": "nccl_recv",
  "source_rank": 1,
  "count": 1
}
```
- `source_rank`: 源 rank
- `count`: 接收次数

#### 4. XOR Step
```json
{
  "step": "xor",
  "type": "xor_update",
  "mode": "with_zero_parity",  // 或 "with_recv_encoding", "incremental"
  "sources": ["local_encoding"],  // 可选
  "target": "parity"
}
```
- `mode`: XOR 模式
  - `with_zero_parity`: `parity = encoding XOR zero_parity` (初始化)
  - `with_recv_encoding`: `parity = encoding XOR recv_encoding` (标准)
  - `incremental`: `parity = parity XOR recv_encoding` (增量更新)

### Post-XOR Steps 说明

#### 1. Sync Step
```json
{
  "step": "sync",
  "type": "barrier",
  "sync_point": "after_all_xor",
  "ranks": [0, 1, 2, 3]
}
```
- 同步屏障，等待所有 XOR 完成

#### 2. Post-XOR Send
```json
{
  "step": "send",
  "type": "nccl_send",
  "target_rank": 2,
  "data_type": "parity",  // 或 "data"
  "data_source": "local_parity"  // 或 "local_data"
}
```

#### 3. Post-XOR Recv
```json
{
  "step": "recv",
  "type": "nccl_recv",
  "source_rank": 2,
  "data_type": "data",  // 或 "parity"
  "data_target": "peer_data_buffer"  // 或 "peer_parity_buffer"
}
```

---

## 执行流程

### Phase 1: 初始化

**代码位置**: `torch.py:_init_eccheck_native()`

1. 加载配置文件
2. 读取 `ec_params` (k, m)
3. 创建 `ECCHECKNative` 实例
4. 初始化 NCCL communicators（每个 column 一个）
5. 启动线程池（4 线程 × 列数）

### Phase 2: Metadata 交换

**代码位置**: `filesystem_async.py:_eccheck_metadata_exchange()`

1. 收集本地 tensor metadata
2. 使用 `torch.distributed.all_gather` 广播到所有 rank
3. 计算 peer rank 的数据大小
4. 准备缓冲区分配

### Phase 2.5: 缓冲区分配

**代码位置**: `filesystem_async.py:_init_eccheck_buffers()`

1. **Data Buffers**: 存储原始 tensor 数据
2. **Encoding Buffers**: 存储编码结果
3. **Receive Buffers**: 基于 peer 数据大小分配
4. **Parity Buffers**: 共享模型，所有 column 共享
5. **Zero Parity Buffer**: 用于初始化（Column 0）
6. **Persistent Stores**: 三段持久化内存
   - `tensor_buffer`: 原始数据
   - `eccheck_persist_parity_store`: 计算的 parity
   - `eccheck_persist_recv_store`: 接收的数据/parity

### Phase 3: 数据编码和交换

**代码位置**: `filesystem_async.py:_eccheck_tensor_data_exchange_and_encoding()`

**步骤**:
1. **GPU → CPU 传输**: 将 tensor 数据拷贝到 CPU buffer
2. **分块处理**: 按 chunk 大小分割数据
3. **提交编码任务**: 对每个 column 和 chunk 提交任务到 C++ 队列
4. **C++ 并行执行**:
   ```
   For each column:
     encoder_thread → Encode (GF)
     send_worker → Send (NCCL)
     recv_worker → Recv (NCCL)
     xor_worker → XOR (ISA-L RAID)
   ```
5. **等待完成**: 等待所有 encoding 和 XOR 完成
6. **执行 Post-XOR**: 调用 `_execute_post_xor_steps()`

### Phase 4: Post-XOR 数据交换

**代码位置**: `filesystem_async.py:_execute_post_xor_steps()`

1. **同步屏障**: `torch.distributed.barrier()` 确保所有 XOR 完成
2. **执行 Post-XOR Steps**:
   - Send: 发送本地 parity 或 data
   - Recv: 接收 peer parity 或 data
3. **存储结果**: 存入 `eccheck_post_xor_buffers`

### Phase 5: Checkpoint 保存

**代码位置**: `filesystem_async.py:finalize()`

1. **序列化元数据**: 包含三段持久化内存的地址和大小
2. **写入文件**: 写入 `.distcp` 文件，包含：
   - 元数据（JSON）
   - 原始数据 (`tensor_buffer`)
   - Parity (`eccheck_persist_parity_store`)
   - 接收的数据/Parity (`eccheck_post_xor_buffers`)

---

## 4-Rank (2+2) 示例详解

### 配置概览

**场景**: 4 个 rank (0, 1, 2, 3)，使用 k=2, m=2 的纠删码配置。

**配对关系**:
- Rank 0 ↔ Rank 2
- Rank 1 ↔ Rank 3

**Column 配置**: 每个 rank 有 2 个 column

### Rank 0 完整流程

#### 初始化阶段

**代码位置**: `torch.py:763` 和 `filesystem_async.py:196`

1. **加载配置**: `eccheck_4rank.json`
2. **读取 EC 参数**: k=2, m=2
3. **创建 ECCHECKNative**:
   ```python
   ECCHECKNative(rank=0, world_size=4, paired_rank=2, k=2, m=2)
   ```
4. **启动线程池**: 8 个线程（2 columns × 4 threads）

#### Column 0 Pipeline

**配置** (`eccheck_4rank.json:20-44`):
```json
{
  "column_idx": 0,
  "coefficient": 0,
  "pipeline": [
    { "step": "encode", "type": "ec_encode" },
    { "step": "send", "type": "nccl_send", "target_rank": 1, "count": 1 },
    { "step": "xor", "type": "xor_update", "mode": "with_zero_parity" }
  ]
}
```

**执行流程**:

1. **Encode** (`eccheck_native.cpp:462`):
   ```cpp
   // 使用系数 0 进行 GF(2^8) 编码
   erasure_code_encode(data_buffer, encoding_buffer, k=2, m=2, coefficient=0)
   ```
   - 输入: `data_buffer` (原始数据)
   - 输出: `encoding_buffer[0]` (编码结果)

2. **Send** (`eccheck_native.cpp:530`):
   ```cpp
   // NCCL 发送编码数据到 Rank 1
   ncclSend(encoding_buffer, size, target_rank=1, comm=comm[0])
   ```

3. **XOR** (`eccheck_native.cpp:610`):
   ```cpp
   // 初始化共享 parity
   // parity = encoding_0 XOR zero_parity
   raid_xor_gen(encoding_buffer, zero_parity_buffer, parity_buffer)
   ```
   - 模式: `WITH_ZERO_PARITY`
   - 结果: `parity_buffer` 被初始化为 `encoding_0`

#### Column 1 Pipeline

**配置** (`eccheck_4rank.json:46-70`):
```json
{
  "column_idx": 1,
  "coefficient": 1,
  "pipeline": [
    { "step": "encode", "type": "ec_encode" },
    { "step": "send", "type": "nccl_send", "target_rank": 1, "count": 1 },
    { "step": "recv", "type": "nccl_recv", "source_rank": 1, "count": 1 },
    { "step": "xor", "type": "xor_update", "mode": "incremental" }
  ]
}
```

**执行流程**:

1. **Encode**:
   ```cpp
   // 使用系数 1 进行编码
   erasure_code_encode(data_buffer, encoding_buffer, k=2, m=2, coefficient=1)
   ```
   - 输出: `encoding_buffer[1]`

2. **Send**:
   ```cpp
   // 发送编码数据到 Rank 1
   ncclSend(encoding_buffer[1], size, target_rank=1, comm=comm[1])
   ```

3. **Recv**:
   ```cpp
   // 从 Rank 1 接收编码数据
   ncclRecv(recv_encoding_buffer[1], size, source_rank=1, comm=comm[1])
   ```
   - 接收: Rank 1 的 `encoding_buffer[1]` (系数 1 编码)

4. **XOR (增量更新)**:
   ```cpp
   // 增量更新共享 parity（线程安全）
   std::lock_guard<std::mutex> lock(parity_mutex);
   // parity = parity XOR recv_encoding_1
   raid_xor_gen(parity_buffer, recv_encoding_buffer[1], parity_buffer)
   ```
   - 模式: `INCREMENTAL`
   - 结果: `parity = encoding_0 XOR recv_encoding_1`

#### Post-XOR 步骤

**配置** (`eccheck_4rank.json:89-108`):
```json
"post_xor_steps": [
  {
    "step": "sync",
    "type": "barrier",
    "sync_point": "after_all_xor",
    "ranks": [0, 1, 2, 3]
  },
  {
    "step": "send",
    "type": "nccl_send",
    "target_rank": 2,
    "data_type": "parity",
    "data_source": "local_parity"
  },
  {
    "step": "recv",
    "type": "nccl_recv",
    "source_rank": 2,
    "data_type": "data",
    "data_target": "peer_data_buffer"
  }
]
```

**执行流程** (`filesystem_async.py:2234`):

1. **Sync**:
   ```python
   torch.distributed.barrier()  # 等待所有 rank 完成 XOR
   ```

2. **Send Parity**:
   ```python
   # 发送本地计算的 parity 到 Rank 2
   self._eccheck_native.post_xor_send(
       target_rank=2,
       data_addr=parity_buffer_addr,
       size=parity_size
   )
   ```
   - **代码位置**: `eccheck_native.cpp:1220`
   - 使用 Column 0 的 NCCL communicator

3. **Recv Data**:
   ```python
   # 从 Rank 2 接收原始 data
   self._eccheck_native.post_xor_recv(
       source_rank=2,
       recv_addr=peer_data_buffer_addr,
       size=data_size
   )
   ```
   - **代码位置**: `eccheck_native.cpp:1244`
   - 结果存储到 `eccheck_post_xor_buffers['peer_data_buffer']`

### Rank 1, 2, 3 流程

**类似 Rank 0**，但使用不同的配置：

- **Rank 1**: Column 0/1 发送到 Rank 0，接收来自 Rank 0，Post-XOR 发送到 Rank 3
- **Rank 2**: Column 0/1 发送到 Rank 3，接收来自 Rank 3，Post-XOR 发送到 Rank 0
- **Rank 3**: Column 0/1 发送到 Rank 2，接收来自 Rank 2，Post-XOR 发送到 Rank 1

### 完整数据流示例

```
Rank 0:
  Data[0] → Encode(coeff=0) → Send(to 1) → XOR(zero) → Parity[0]
  Data[0] → Encode(coeff=1) → Send(to 1) → Recv(from 1) → XOR(inc) → Parity[0]
  
  最终 Parity = encoding_0(coeff=0) XOR encoding_1(coeff=1)

Rank 1:
  Data[1] → Encode(coeff=0) → Send(to 0) → XOR(zero) → Parity[1]
  Data[1] → Encode(coeff=1) → Send(to 0) → Recv(from 0) → XOR(inc) → Parity[1]
  
  最终 Parity = encoding_0(coeff=0) XOR encoding_1(coeff=1)

Post-XOR:
  Rank 0 → Send(Parity) → Rank 2
  Rank 0 ← Recv(Data) ← Rank 2
  
  结果: Rank 0 拥有自己的 Data[0], Parity[0], 和来自 Rank 2 的 Data[2]
```

### 容错恢复示例

假设 Rank 0 失败：

**恢复数据需要的资源**:
- Rank 1: Data[1], Parity[1]
- Rank 2: Data[2], Parity[2] (从 Post-XOR 收到)
- Rank 3: Data[3], Parity[3]

**恢复算法**:
```
使用 Reed-Solomon 解码:
  - 输入: Data[1], Data[2], Parity[1], Parity[2], Parity[3]
  - 输出: Data[0] (恢复)
  
恢复公式:
  Data[0] = decode(
    [Data[1], Data[2]],  // k=2 个数据节点
    [Parity[1], Parity[2], Parity[3]]  // m=2 个校验节点
  )
```

---

## API 参考

### Python API

#### `TorchDistSaveShardedStrategy`

**类**: `megatron/core/dist_checkpointing/strategies/torch.py:648`

**方法**:
- `__init__()`: 初始化策略，可选启用 EC-CHECK
- `async_save()`: 异步保存 checkpoint
- `_init_eccheck_native()`: 初始化 C++ native 模块

#### `FileSystemWriterAsync`

**类**: `megatron/core/dist_checkpointing/strategies/filesystem_async.py:63`

**方法**:
- `__init__(..., use_eccheck, eccheck_config_path, ...)`: 创建 writer，可选传入预初始化的 `eccheck_native`
- `_load_eccheck_config()`: 加载配置文件
- `_init_eccheck_buffers()`: 分配缓冲区
- `_eccheck_tensor_data_exchange_and_encoding()`: Phase 3 主流程
- `_execute_post_xor_steps()`: 执行 Post-XOR 步骤

### C++ API

#### `ECCHECKNative`

**类**: `megatron/core/dist_checkpointing/strategies/eccheck_native.cpp:28`

**构造函数**:
```cpp
ECCHECKNative(int rank, int world_size, int paired_rank, int k = -1, int m = -1)
```

**方法**:
- `submit_data_for_encoding(...)`: 提交编码任务到队列
- `wait_for_encoding_completion()`: 等待所有编码和 XOR 完成
- `set_columns_config(...)`: 设置列配置
- `set_persist_stores(...)`: 设置持久化内存地址
- `post_xor_send(target_rank, data_addr, size)`: Post-XOR 发送
- `post_xor_recv(source_rank, recv_addr, size)`: Post-XOR 接收
- `stop_pipeline()`: 停止线程池

---

## 故障排除

### 常见问题

#### 1. 配置文件未找到

**症状**: 日志显示 `Config path: ..., exists: False`

**解决方案**:
```bash
# 确保环境变量正确设置
export ECCHECK_CONFIG_PATH="/workspace/Megatron-LM/pre-tests/gpt2/eccheck/configs/eccheck_4rank.json"
```

**代码位置**: `filesystem_async.py:232`

#### 2. NCCL 初始化失败

**症状**: `NCCL not initialized` 错误

**原因**: 
- 分布式环境未初始化
- NCCL communicator 创建失败

**解决方案**:
- 确保在 `torch.distributed.init_process_group()` 之后初始化
- 检查 NCCL 环境变量

**代码位置**: `eccheck_native.cpp:813`

#### 3. Post-XOR 步骤未执行

**症状**: 日志中没有 "Post-XOR" 相关输出

**检查点**:
1. 配置文件中是否有 `post_xor_steps`?
2. `_execute_post_xor_steps()` 是否被调用?

**代码位置**: `filesystem_async.py:1719`

#### 4. 缓冲区不足

**症状**: `Buffer allocation failed` 或内存不足

**解决方案**:
- 检查 `eccheck_data_buffers_count` 参数
- 增加系统内存
- 减少 batch size

#### 5. 线程死锁

**症状**: 程序卡住不继续

**检查点**:
- 检查 `parity_mutex` 是否正确释放
- 检查 NCCL 通信是否配对（send/recv 必须匹配）

**代码位置**: `eccheck_native.cpp:646`

### 调试技巧

#### 启用详细日志

在配置文件中添加调试标志，或在代码中添加：

```python
# filesystem_async.py
logger.setLevel(logging.DEBUG)
```

#### 检查线程状态

```cpp
// eccheck_native.cpp
std::cout << "Thread status: " << encoder_thread_completed << std::endl;
```

#### 验证配置解析

```python
# filesystem_async.py:1962
logger.info(f"EC-CHECK: Loaded config: {columns_config}")
logger.info(f"EC-CHECK: Post-XOR steps: {post_xor_steps}")
```

---

## 总结

EC-CHECK 是一个强大的容错检查点系统，通过纠删码技术实现：

1. **容错能力**: 即使部分 rank 失败也能恢复
2. **并行效率**: 多线程流水线执行
3. **灵活配置**: 支持多种 pipeline 和 post-XOR 步骤

**关键要点**:
- **共享 Parity 模型**: 所有 column 共享同一个 parity buffer
- **多线程架构**: 每个 column 有独立的线程池
- **三段持久化**: 原始数据、parity、接收数据
- **配置驱动**: 通过 JSON 配置文件控制行为

**下一步**:
- 运行测试验证功能
- 根据需求调整配置
- 优化性能参数

---

## 相关文档

- [4-Rank 详细行为分析](./4RANK_BEHAVIOR_DETAILED.md)
- [两个 ECCHECKNative 解释](./TWO_ECCHECK_NATIVE_EXPLAINED.md)
- [配置文件说明](./ECCHECK_CONFIG_EXPLAINED.md)
- [Pipeline 配置](./ECCHECK_PIPELINE_CONFIG.md)

