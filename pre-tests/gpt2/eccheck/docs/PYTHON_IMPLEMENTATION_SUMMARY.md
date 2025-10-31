# Python 端 Pipeline 实现总结

## 已完成的功能

### 1. ✅ Pipeline 配置解析和提取

**位置**: `_parse_pipeline_step()` 和 `_parse_advanced_columns_config()`

- 解析每个 column 的 `pipeline` 配置
- 提取执行提示（hints）：
  - `needs_send`: 是否需要执行 send 步骤
  - `needs_recv`: 是否需要执行 recv 步骤
  - `xor_mode`: XOR 模式（`with_zero_parity`, `with_recv_encoding`, `incremental`）
  - `needs_zero_parity`: 是否需要零初始化 parity buffer
  - `send_count`: 发送次数（支持多次 send）
  - `recv_count`: 接收次数（支持多次 recv）

**存储**:
- `self.eccheck_pipeline_configs`: 完整的 pipeline 配置（供 C++ 未来使用）
- `self.eccheck_pipeline_hints`: 执行提示（Python 端决策使用）

### 2. ✅ 零初始化 Parity Buffer 分配

**位置**: `_allocate_zero_parity_buffers_if_needed()`

- 根据 pipeline 配置检测需要零初始化 parity 的 columns
- 为每个需要的 column 分配零初始化的 parity buffers
- 使用 `torch.zeros()` 创建零初始化缓冲区
- 按 chunk 分配（每个 chunk 一个 buffer）

**存储**: `self.eccheck_zero_parity_buffers[column_idx] = [buffers...]`

### 3. ✅ 根据 Pipeline 配置驱动执行

**位置**: `_copy_tensor_data_to_buffers_pipeline()`

- 根据 `eccheck_pipeline_hints` 决定是否传递 `recv_addr`
  - 如果 `needs_recv == False`，设置 `recv_addr = 0` 和 `recv_size = 0`
- 根据 pipeline 配置选择零初始化 parity buffer 地址
- 根据 chunk index 选择对应的零初始化 buffer

**关键逻辑**:
```python
hints = pipeline_hints.get(col_idx, {})
needs_recv = hints.get('needs_recv', True)

if needs_recv:
    recv_addr = recv_buffer_base_addrs[col_idx] + recv_buffer_offsets[col_idx]
    recv_size = recv_chunk_size
else:
    recv_addr = 0
    recv_size = 0
```

### 4. ✅ Post-XOR 步骤框架

**位置**: `_execute_post_xor_steps()`

- 在所有 XOR 操作完成后执行
- 解析 `post_xor_steps` 配置
- 支持步骤类型：
  - `sync`: 同步屏障（已实现使用 `torch.distributed.barrier()`）
  - `send`: 发送数据/parity（框架已就绪，待 C++ 实现）
  - `recv`: 接收数据/parity（框架已就绪，待 C++ 实现）

**当前状态**:
- ✅ `sync` 步骤已实现
- ⏳ `send`/`recv` 步骤框架已就绪，但需要 C++ API 支持才能完全实现

### 5. ✅ 简单模式支持（向后兼容）

- 简单模式下自动设置默认 pipeline hints（`needs_send=True`, `needs_recv=True`）
- 确保所有代码路径都能正常工作

## 实现细节

### Pipeline Hints 结构

```python
{
    'needs_send': True/False,           # 是否需要 send
    'needs_recv': True/False,           # 是否需要 recv
    'xor_mode': 'with_recv_encoding',   # XOR 模式
    'needs_zero_parity': True/False,    # 是否需要零初始化 parity
    'send_count': 1,                    # 发送次数
    'recv_count': 1                     # 接收次数
}
```

### 零初始化 Parity Buffer 分配

```python
# 分配时机：配置解析后（Phase 2.5）
# 分配方式：每个 chunk 一个 buffer
zero_buffer = torch.zeros(buffer_size, dtype=torch.uint8, pin_memory=...)
# 存储：self.eccheck_zero_parity_buffers[column_idx][chunk_idx] = zero_buffer
```

### Recv 地址处理

```python
# 根据 pipeline 配置决定是否传递 recv_addr
if needs_recv:
    recv_addr = recv_buffer_base_addrs[col_idx] + offset
else:
    recv_addr = 0  # C++ 端看到 recv_addr=0 应该跳过 recv 步骤
```

## 待 C++ 端实现的接口

### 1. 零初始化 Parity Buffer 支持

**当前状态**: Python 已准备好地址，但 C++ API 未扩展

**需要扩展**:
```cpp
void submit_data_for_encoding(
    int column_idx,
    uintptr_t data_addr, size_t size,
    uintptr_t encoding_addr,
    uintptr_t recv_addr, size_t recv_chunk_size,
    uintptr_t parity_addr,
    uintptr_t zero_parity_addr = 0,  // 新增参数
    int xor_mode = 0                  // 新增参数：0=with_recv, 1=with_zero, 2=incremental
)
```

### 2. Pipeline 配置传递

**当前状态**: Python 已解析，但未传递给 C++

**需要添加**:
```cpp
void set_pipeline_config(int column_idx, const std::vector<PipelineStepConfig>& pipeline);
```

### 3. Post-XOR Send/Recv API

**当前状态**: Python 框架已就绪，但需要 C++ 实现

**需要添加**:
```cpp
void post_xor_send(int target_rank, uintptr_t data_addr, size_t size, const char* data_type);
void post_xor_recv(int source_rank, uintptr_t recv_addr, size_t size, const char* data_type);
```

## 配置示例

### 异构 Pipeline 配置（第一个 column 与零 XOR）

```json
{
  "ranks": {
    "0": {
      "columns": [
        {
          "column_idx": 0,
          "coefficient": 0,
          "pipeline": [
            {"step": "encode", "type": "ec_encode"},
            {"step": "send", "type": "nccl_send", "target_rank": 1},
            {"step": "xor", "type": "xor_update", "mode": "with_zero_parity", ...}
          ]
        }
      ]
    }
  }
}
```

**Python 端处理**:
1. 解析 pipeline，提取 `needs_zero_parity=True`, `needs_recv=False`
2. 分配零初始化 parity buffers
3. 在提交任务时，设置 `recv_addr=0`, `recv_size=0`
4. 传递零初始化 parity buffer 地址（待 C++ API 扩展）

## 测试状态

- ✅ Python 语法检查通过
- ✅ 配置解析逻辑完整
- ⏳ 待 C++ 端实现后完整测试

## 下一步

1. **C++ 端实现**:
   - 扩展 `submit_data_for_encoding` API 支持零初始化 parity 和 XOR mode
   - 实现 pipeline 配置传递和执行引擎
   - 实现 post-xor send/recv API

2. **端到端测试**:
   - 使用 `eccheck_heterogeneous.json` 配置文件测试
   - 验证第一个 column 跳过 recv 步骤
   - 验证零初始化 parity XOR 功能

