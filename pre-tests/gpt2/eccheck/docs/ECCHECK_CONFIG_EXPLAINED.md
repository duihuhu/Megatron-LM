# EC-CHECK 配置文件详解

## 配置文件结构

EC-CHECK 使用 JSON（或 YAML）配置文件来指定运行时行为。配置文件分为两个主要部分：

### 1. `persist` 部分 - 持久化设置（Python 处理）

```json
{
  "persist": {
    "recv": true,     // 是否持久化接收到的编码数据
    "parity": true    // 是否持久化 XOR 运算结果（parity）
  }
}
```

**作用：**
- `recv: true`：将接收到的编码数据复制到持久化存储（`eccheck_persist_recv_store`）
  - Python 侧：分配一个大的连续内存缓冲区
  - C++ 侧：`recv_worker` 在接收完成后，会将数据 memcpy 到这个持久化缓冲区
- `parity: true`：将 XOR 运算结果复制到持久化存储（`eccheck_persist_parity_store`）
  - Python 侧：分配一个大的连续内存缓冲区
  - C++ 侧：`xor_worker` 在 XOR 完成后，会将结果 memcpy 到这个持久化缓冲区（仅 column 0 执行，避免重复写入）

**处理流程：**
1. Python 在 `_eccheck_phase2_register_metadata()` 中解析 `persist` 部分
2. 设置 `self.eccheck_persist_recv_enabled` 和 `self.eccheck_persist_parity_enabled`
3. 如果启用，在 Phase 2.5 分配持久化缓冲区（torch.empty）
4. 调用 C++ 的 `set_persist_stores()` 传递缓冲区地址和容量

### 2. `columns` 部分 - 列配置（Python + C++ 共同处理）

```json
{
  "columns": [
    {
      "coefficient": 0,    // GF 编码系数
      "send_peer": 1,      // 发送目标 rank
      "recv_peer": 1       // 接收来源 rank
    },
    {
      "coefficient": 1,
      "send_peer": 1,
      "recv_peer": 1
    }
  ]
}
```

**各字段含义：**

#### `coefficient`（编码系数）
- **类型**：整数
- **作用**：指定该列使用的 GF (Galois Field) 编码系数
- **C++ 使用**：在 `encode_with_parity_index()` 中，用于选择 GF 编码表
  - 通过 `g_tbls_` 数组索引：`parity_idx = column_idx`，`data_block_index = rank % k`
  - 决定了该列如何编码数据块
- **默认值**：如果未指定，使用列索引（0, 1, 2, ...）

#### `send_peer`（发送目标）
- **类型**：整数（rank ID）或 `-1`（自动计算）
- **作用**：指定该列编码后的数据发送到哪个 rank
- **特殊值 `-1`**：表示"自动计算"，会根据当前 rank 计算配对 rank
  - Python 侧：如果为 `-1` 或未指定，会计算 `paired_rank = _get_paired_rank(current_rank, world_size)`
  - C++ 侧：如果为 `-1`，也会回退到 `paired_rank_`
- **多 rank 共享配置**：使用 `-1` 可以让一个配置文件被所有 rank 共享
  - 例如：`send_peer: -1` → Rank 0 发送到 1，Rank 1 发送到 0（自动计算）
  - 例如：`send_peer: 1` → 所有 rank 都发送到 1（不推荐，除非特殊需求）
- **C++ 使用**：在 `send_worker()` 中
  ```cpp
  int peer = (column_configs_[column_idx].send_peer >= 0 
              ? column_configs_[column_idx].send_peer 
              : paired_rank_);
  ncclSend(..., peer, nccl_comms_[column_idx], ...);
  ```

#### `recv_peer`（接收来源）
- **类型**：整数（rank ID）或 `-1`（自动计算）
- **作用**：指定从哪个 rank 接收编码数据
- **特殊值 `-1`**：表示"自动计算"，会根据当前 rank 计算配对 rank
  - Python 侧：如果为 `-1` 或未指定，会计算 `paired_rank = _get_paired_rank(current_rank, world_size)`
  - C++ 侧：如果为 `-1`，也会回退到 `paired_rank_`
- **多 rank 共享配置**：使用 `-1` 可以让一个配置文件被所有 rank 共享
  - 例如：`recv_peer: -1` → Rank 0 从 1 接收，Rank 1 从 0 接收（自动计算）
  - 例如：`recv_peer: 1` → 所有 rank 都从 1 接收（不推荐，除非特殊需求）
- **C++ 使用**：在 `recv_worker()` 中
  ```cpp
  int peer = (column_configs_[column_idx].recv_peer >= 0 
              ? column_configs_[column_idx].recv_peer 
              : paired_rank_);
  ncclRecv(..., peer, nccl_comms_[column_idx], ...);
  ```

**处理流程：**

1. **Python 解析**（`_eccheck_phase2_register_metadata()`）：
   - 读取配置文件
   - 解析 `columns` 数组
   - 规范化每个列的配置：
     ```python
     norm_cols.append({
         'coefficient': int(col.get('coefficient', 0)),
         'send_peer': int(col.get('send_peer', paired_rank)),
         'recv_peer': int(col.get('recv_peer', paired_rank))
     })
     ```

2. **Python 传递到 C++**（Phase 2.5）：
   - 调用 `self._eccheck_native.set_columns_config(norm_cols)`
   - 传递格式：`[{"coefficient": 0, "send_peer": 1, "recv_peer": 1}, ...]`

3. **C++ 应用配置**（`set_columns_config()`）：
   - 更新 `num_columns_`（列数）
   - 存储到 `column_configs_` 数组
   - 各 worker 线程通过 `column_idx` 索引访问配置

## 配置文件的处理时机

### Phase 1: 初始化时（`__init__`）
- 设置 `self.eccheck_config_path`（默认值或环境变量）

### Phase 2: 注册元数据时（`_eccheck_phase2_register_metadata()`）
- **读取配置文件**（支持 JSON 和 YAML）
- **解析 `persist` 部分** → 设置 Python 标志
- **解析 `columns` 部分** → 规范化配置（但不立即传递）

### Phase 2.5: 分配缓冲区后（`_eccheck_phase2_5_allocate_buffers()`）
- **分配持久化缓冲区**（如果 `persist` 启用）
- **传递持久化配置到 C++**：`set_persist_stores()`
- **传递列配置到 C++**：`set_columns_config()`

### Phase 3: 执行编码时（`_copy_tensor_data_to_buffers_pipeline()`）
- 使用 `num_columns` 循环处理所有列
- 每个列使用其配置的 `coefficient`、`send_peer`、`recv_peer`

## 配置文件的决定权

| 配置项 | 决定的内容 | 处理位置 | 作用时机 |
|--------|-----------|---------|---------|
| `persist.recv` | 是否持久化接收数据 | Python 解析，C++ 执行 | 接收完成后 memcpy |
| `persist.parity` | 是否持久化 XOR 结果 | Python 解析，C++ 执行 | XOR 完成后 memcpy |
| `columns[].coefficient` | GF 编码系数 | Python 传递，C++ 使用 | 编码时选择编码表 |
| `columns[].send_peer` | 发送目标 rank | Python 传递，C++ 使用 | 发送时指定 NCCL peer |
| `columns[].recv_peer` | 接收来源 rank | Python 传递，C++ 使用 | 接收时指定 NCCL peer |
| `columns[]` 数组长度 | 列数（线程数） | Python 传递，C++ 使用 | 确定启动多少线程 |

## 多 Rank 共享配置

**重要**：一个配置文件会被所有 rank 读取和使用。为了确保配置对所有 rank 都正确，需要注意：

### 配对 Rank 的计算规则

配对关系由 `_get_paired_rank()` 函数决定：
- **2 ranks**: rank 0 ↔ rank 1
- **4 ranks**: rank 0 ↔ rank 2, rank 1 ↔ rank 3
- **通用规则**: rank_i ↔ rank_{i + world_size/2}

### 使用 `-1` 实现自动计算

**推荐做法**：在配置文件中使用 `-1` 表示"自动计算"，Python 会根据当前 rank 自动计算配对 rank：

```json
{
  "columns": [
    {
      "coefficient": 0,
      "send_peer": -1,    // Rank 0 → 发送到 1, Rank 1 → 发送到 0 (自动计算)
      "recv_peer": -1     // Rank 0 → 从 1 接收, Rank 1 → 从 0 接收 (自动计算)
    }
  ]
}
```

**不推荐做法**：使用固定 rank ID（如 `1`）会导致所有 rank 都使用相同配置：

```json
{
  "columns": [
    {
      "send_peer": 1,    // ⚠️ 所有 rank 都发送到 1（Rank 1 会出错！）
      "recv_peer": 1     // ⚠️ 所有 rank 都从 1 接收（Rank 1 会出错！）
    }
  ]
}
```

### 处理流程示例（2 ranks）

假设配置文件使用 `send_peer: -1`：

1. **Rank 0 读取配置**：
   - 解析到 `send_peer: -1`
   - 计算 `paired_rank = _get_paired_rank(0, 2) = 1`
   - 传递给 C++：`send_peer = 1`

2. **Rank 1 读取配置**：
   - 解析到 `send_peer: -1`
   - 计算 `paired_rank = _get_paired_rank(1, 2) = 0`
   - 传递给 C++：`send_peer = 0`

3. **结果**：
   - Rank 0 → 发送到 Rank 1 ✓
   - Rank 1 → 发送到 Rank 0 ✓

### 何时使用固定 Rank ID

只有在特殊需求时才使用固定的 rank ID，例如：
- 所有 rank 都向某个特定的收集 rank 发送数据
- 复杂的多跳通信模式

## 配置文件的查找顺序

1. **Python 构造函数参数**：`eccheck_config_path="/path/to/config.json"`
2. **环境变量**：`ECCHECK_CONFIG_PATH=/path/to/config.json`
3. **默认路径**：`/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2.json`

## 示例：2+2 配置的工作流程

```json
{
  "persist": {"recv": true, "parity": true},
  "columns": [
    {"coefficient": 0, "send_peer": -1, "recv_peer": -1},
    {"coefficient": 1, "send_peer": -1, "recv_peer": -1}
  ]
}
```

**执行流程（以 Rank 0 为例，world_size=2）：**

1. **Python Phase 2**：
   - 读取配置文件
   - 设置 `persist_recv_enabled = True`，`persist_parity_enabled = True`

2. **Python Phase 2.5**：
   - 分配持久化缓冲区
   - 调用 `set_persist_stores(recv_base, parity_base, ...)`
   - **解析列配置**：
     - Rank 0: `paired_rank = _get_paired_rank(0, 2) = 1`
     - `send_peer: -1` → 计算为 `1`
     - `recv_peer: -1` → 计算为 `1`
     - 调用 `set_columns_config([{coefficient:0, send_peer:1, recv_peer:1}, ...])`

3. **C++ 应用配置**：
   - `num_columns_ = 2`
   - `column_configs_[0] = {coefficient:0, send_peer:1, recv_peer:1}`
   - `column_configs_[1] = {coefficient:1, send_peer:1, recv_peer:1}`

4. **C++ 启动线程**：
   - 启动 8 个线程（2 encoder + 2 send + 2 recv + 2 XOR）
   - 每个线程使用对应的列配置

5. **运行时使用配置**：
   - `encoder_worker(0)`：使用 `column_configs_[0].coefficient = 0`
   - `send_worker(0)`：发送到 `column_configs_[0].send_peer = 1`
   - `recv_worker(0)`：从 `column_configs_[0].recv_peer = 1` 接收
   - 如果 `persist_recv_enabled`，`recv_worker(0)` 会 memcpy 到持久化缓冲区
   - 如果 `persist_parity_enabled`，`xor_worker(0)` 会 memcpy 到持久化缓冲区

**注意**：Rank 1 会执行相同的流程，但 `send_peer` 和 `recv_peer` 会被计算为 `0`（Rank 1 的 paired_rank）。

## 总结

- **Python 侧职责**：
  - 解析配置文件（JSON/YAML）
  - 规范化配置（填充默认值）
  - 分配持久化缓冲区
  - 传递配置到 C++

- **C++ 侧职责**：
  - 接收配置并存储
  - 运行时使用配置（编码系数、NCCL peer、持久化标志）
  - 执行实际的操作（编码、发送、接收、XOR）

- **配置文件完全由 Python 处理**：
  - Python 负责解析和验证
  - Python 负责传递到 C++
  - C++ 只负责接收和使用，不解析配置文件

