# EC-CHECK 异构 Pipeline 配置规范

## 配置格式

配置文件支持两种模式：
1. **简单模式**：向后兼容，所有 rank 使用相同配置
2. **高级模式**：按 rank 配置，支持异构 pipeline

### 简单模式（向后兼容）

```json
{
  "persist": {"recv": true, "parity": true},
  "columns": [
    {"coefficient": 0, "send_peer": -1, "recv_peer": -1}
  ]
}
```

### 高级模式（按 rank 配置）

```json
{
  "persist": {"recv": true, "parity": true},
  "ranks": {
    "0": {
      "columns": [
        {
          "column_idx": 0,
          "coefficient": 0,
          "pipeline": [...]
        }
      ],
      "post_xor_steps": [...]
    }
  }
}
```

## Pipeline Step 类型

### 1. `encode` - 编码步骤
```json
{
  "step": "encode",
  "type": "ec_encode"
}
```

### 2. `send` - 发送步骤（支持多个）
```json
{
  "step": "send",
  "type": "nccl_send",
  "target_rank": 1,      // 目标 rank
  "count": 1             // 发送次数（可选，默认1）
}
```

### 3. `recv` - 接收步骤（支持多个）
```json
{
  "step": "recv",
  "type": "nccl_recv",
  "source_rank": 1,      // 来源 rank
  "count": 1            // 接收次数（可选，默认1）
}
```

### 4. `xor` - XOR 更新步骤

#### 模式 1: 与接收数据 XOR（标准模式）
```json
{
  "step": "xor",
  "type": "xor_update",
  "mode": "with_recv_encoding",
  "sources": ["local_encoding", "recv_encoding"],
  "target": "parity"
}
```

#### 模式 2: 与零初始化 parity XOR（第一个线程）
```json
{
  "step": "xor",
  "type": "xor_update",
  "mode": "with_zero_parity",
  "sources": ["local_encoding"],
  "target": "parity",
  "zero_parity_addr": "initial_parity_buffer"  // 零初始化的 parity buffer
}
```

#### 模式 3: 与现有 parity XOR（增量更新）
```json
{
  "step": "xor",
  "type": "xor_update",
  "mode": "incremental",
  "sources": ["local_encoding", "existing_parity"],
  "target": "parity"
}
```

## 同步和数据交换步骤（post_xor_steps）

`post_xor_steps` 是在所有 column 的 XOR 操作完成后执行的全局步骤。

#### `sync` - 同步屏障
```json
{
  "step": "sync",
  "type": "barrier",
  "sync_point": "after_all_xor",
  "ranks": [0, 1]  // 参与同步的 ranks
}
```

#### `send` - 发送数据/parity（在 post_xor_steps 中）
```json
{
  "step": "send",
  "type": "nccl_send",
  "target_rank": 1,
  "data_type": "parity",           // "parity" 或 "data"
  "data_source": "local_parity"    // 数据源标识（如 "local_parity", "local_data"）
}
```

#### `recv` - 接收数据/parity（在 post_xor_steps 中）
```json
{
  "step": "recv",
  "type": "nccl_recv",
  "source_rank": 1,
  "data_type": "data",              // "parity" 或 "data"
  "data_target": "peer_data_buffer" // 数据目标标识（接收缓冲区）
}
```

**示例场景**：XOR 后发送自己的 parity，接收对方的 data
```json
{
  "post_xor_steps": [
    {"step": "sync", "type": "barrier", "sync_point": "after_all_xor", "ranks": [0, 1]},
    {"step": "send", "type": "nccl_send", "target_rank": 1, "data_type": "parity", "data_source": "local_parity"},
    {"step": "recv", "type": "nccl_recv", "source_rank": 1, "data_type": "data", "data_target": "peer_data_buffer"}
  ]
}
```

## 处理流程

### Python 端处理

1. **读取配置**：根据当前 rank 选择配置
   - 如果存在 `ranks[rank_id]`，使用该配置
   - 否则，使用简单的 `columns` 配置（向后兼容）

2. **解析 Pipeline**：
   - 解析每个 column 的 `pipeline` 数组
   - 构建执行步骤序列

3. **传递给 C++**：
   - 传递 pipeline 配置
   - C++ 根据配置动态构建线程处理流程

### C++ 端处理

1. **接收 Pipeline 配置**：
   - 存储每个 column 的 pipeline steps

2. **动态执行**：
   - encoder_worker 根据 pipeline 执行步骤
   - 不是固定流程（encode → send → recv → xor），而是按配置执行

3. **支持异构流程**：
   - 不同 column 可以有完全不同的 pipeline
   - 支持条件执行、同步点等

## 示例场景

### 场景 1: 第一个线程直接 XOR，其他线程 recv 后 XOR

```json
{
  "ranks": {
    "0": {
      "columns": [
        {
          "column_idx": 0,
          "pipeline": [
            {"step": "encode", "type": "ec_encode"},
            {"step": "send", "type": "nccl_send", "target_rank": 1},
            {"step": "xor", "type": "xor_update", "mode": "with_zero_parity", ...}
          ]
        },
        {
          "column_idx": 1,
          "pipeline": [
            {"step": "encode", "type": "ec_encode"},
            {"step": "send", "type": "nccl_send", "target_rank": 1},
            {"step": "recv", "type": "nccl_recv", "source_rank": 1},
            {"step": "xor", "type": "xor_update", "mode": "with_recv_encoding", ...}
          ]
        }
      ]
    }
  }
}
```

### 场景 2: m-1 个线程需要多次 send/recv

```json
{
  "columns": [
    {
      "column_idx": 1,
      "pipeline": [
        {"step": "encode", "type": "ec_encode"},
        {"step": "send", "type": "nccl_send", "target_rank": 1, "count": 2},
        {"step": "recv", "type": "nccl_recv", "source_rank": 1, "count": 2},
        {"step": "xor", "type": "xor_update", "mode": "with_recv_encoding", ...}
      ]
    }
  ]
}
```

