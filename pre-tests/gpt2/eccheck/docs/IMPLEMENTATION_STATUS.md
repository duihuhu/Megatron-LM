# EC-CHECK 异构 Pipeline 实现状态

## 已完成

### 1. Python 端配置解析（✅ 已完成）

**位置**: `filesystem_async.py`

- ✅ 支持简单模式（向后兼容）
- ✅ 支持高级模式（按 rank 配置）
- ✅ 从 pipeline 中提取 send_peer/recv_peer
- ✅ 存储完整的 pipeline 配置供未来使用

**功能**:
- `_parse_advanced_columns_config()`: 解析高级配置，提取基本信息
- `_store_pipeline_configs()`: 存储 pipeline 配置

### 2. 配置文件格式设计（✅ 已完成）

**示例文件**:
- `eccheck_2x2.json`: 简单模式（向后兼容）
- `eccheck_advanced.json`: 高级模式示例
- `eccheck_heterogeneous.json`: 异构 pipeline 示例

**支持的配置项**:
- 按 rank 配置（`ranks` 字段）
- Pipeline steps（encode, send, recv, xor）
- 不同的 XOR 模式（with_zero_parity, with_recv_encoding, incremental）
- 同步和交换步骤

## 待实现

### 3. C++ 端 Pipeline 执行引擎（🚧 待实现）

需要实现：

1. **Pipeline 配置数据结构**
   ```cpp
   enum class PipelineStepType {
       ENCODE, SEND, RECV, XOR, SYNC, EXCHANGE
   };
   
   struct PipelineStep {
       PipelineStepType type;
       int target_rank;      // for send/recv/exchange
       int source_rank;      // for recv
       int count;            // for multiple send/recv
       std::string mode;     // for xor (with_zero_parity, etc.)
       std::vector<std::string> sources;  // for xor
       std::string target;   // for xor
   };
   
   std::vector<PipelineStep> pipeline_[MAX_COLUMNS];
   ```

2. **动态 Pipeline 执行器**
   - 修改 `encoder_worker()` 使其按 pipeline 执行，而不是固定流程
   - 支持条件执行
   - 支持同步点

3. **不同的 XOR 模式**
   - `with_zero_parity`: 与零初始化 buffer XOR
   - `with_recv_encoding`: 与接收数据 XOR（当前实现）
   - `incremental`: 与现有 parity 增量更新

4. **同步和交换步骤**
   - Barrier 同步
   - Parity/data 交换

### 4. C++ API 扩展

需要添加：

```cpp
// 设置 pipeline 配置
void set_pipeline_config(int column_idx, const std::vector<PipelineStepConfig>& pipeline);

// 支持零初始化 parity buffer
void set_zero_parity_buffer(uintptr_t addr, size_t size);

// 同步点
void wait_for_sync_point(const std::string& sync_point);

// 交换操作
void exchange_parity_data(int target_rank, uintptr_t data_addr, size_t size);
```

## 当前状态

### 当前实现（简单模式）

**C++ 端**: 固定流程
```
encode → send → recv → xor
```

**Python 端**: 
- ✅ 可以解析 pipeline 配置
- ✅ 提取基本信息（send_peer, recv_peer）传递给 C++
- ✅ 存储完整 pipeline 供未来使用

### 下一步实现计划

**Phase 1: 基础 Pipeline 支持**（当前优先级）
1. C++ 端添加 Pipeline 配置数据结构
2. 实现 `set_pipeline_config()` API
3. 修改 `encoder_worker()` 支持按配置执行

**Phase 2: XOR 模式扩展**
1. 实现 `with_zero_parity` 模式
2. 实现 `incremental` 模式
3. 添加零初始化 parity buffer 管理

**Phase 3: 多 send/recv 支持**
1. 支持 `count > 1` 的 send/recv
2. 处理多次通信的状态管理

**Phase 4: 同步和交换**
1. Barrier 同步实现
2. Parity/data 交换实现

## 使用建议

### 当前可用（简单模式）

使用简单配置格式（向后兼容）：
```json
{
  "columns": [
    {"coefficient": 0, "send_peer": -1, "recv_peer": -1}
  ]
}
```

### 未来可用（高级模式）

当 C++ pipeline 引擎实现后，可以使用：
```json
{
  "ranks": {
    "0": {
      "columns": [{
        "pipeline": [
          {"step": "encode", "type": "ec_encode"},
          {"step": "xor", "type": "xor_update", "mode": "with_zero_parity"}
        ]
      }]
    }
  }
}
```

目前 Python 端已经可以解析和存储这些配置，等待 C++ 端实现执行引擎。

