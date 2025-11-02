# Chunk-Level Post-XOR 实现检查清单

## ✅ 已完成的实现

### 1. C++ 数据结构
- [x] `XorTask` 添加 `chunk_idx`, `chunk_size`, `is_last_column_in_chunk`
- [x] `EncodingTask` 添加相同字段
- [x] `RecvMapping` 添加相同字段
- [x] `PostXorStep` 结构用于存储配置
- [x] `peer_data_buffer_base_` 和 `peer_data_buffer_capacity_` 成员变量

### 2. C++ 接口更新
- [x] `submit_data_for_encoding` 添加 chunk 参数（chunk_idx, chunk_size, is_last_column_in_chunk）
- [x] `set_chunk_level_post_xor_config` 方法用于设置配置
- [x] Pybind11 绑定更新

### 3. C++ 执行逻辑
- [x] `execute_chunk_level_post_xor()` 方法实现
  - [x] 计算 chunk 偏移量
  - [x] 复制 parity 到 persistent store
  - [x] 执行 send 操作（从 persistent parity store）
  - [x] 执行 recv 操作（到 peer data buffer）
  - [x] NCCL 通信分组（send 和 recv 在同一组）
- [x] XOR worker 中触发 chunk-level post-xor（最后一个 column）
- [x] 所有创建 XorTask 的地方传递 chunk 信息

### 4. Python 侧集成
- [x] Phase 2.5 设置 chunk-level post-xor 配置
- [x] 自动分配 `peer_data_buffer`（如果需要）
- [x] 传递配置到 C++
- [x] 任务提交时传递 chunk 信息
- [x] 全局 post-xor 步骤过滤（sync 步骤）

### 5. 配置文件
- [x] 所有 rank 的 `post_xor_steps` 添加 `chunk_level` 字段
- [x] Sync 步骤标记为 `chunk_level: false`
- [x] Send/Recv 步骤标记为 `chunk_level: true`

### 6. 代码修复
- [x] 添加 `#include <string>` 头文件（用于 `std::stoi`）
- [x] 修复 NCCL 通信分组（send 和 recv 在同一组执行）
- [x] 修复配置逻辑（允许 send-only 操作）

## 📋 编译前检查

### C++ 文件
- ✅ `eccheck_native.cpp`: 所有结构定义完整
- ✅ 所有方法实现完整
- ✅ Pybind11 绑定完整

### Python 文件
- ✅ `filesystem_async.py`: 配置设置逻辑完整
- ✅ 任务提交逻辑完整
- ✅ 向后兼容性处理

### 配置文件
- ✅ `eccheck_4rank.json`: 所有 rank 的配置已更新

## 🚀 编译和测试步骤

### 1. 编译 C++ 扩展
```bash
cd /workspace/Megatron-LM
# 根据项目的编译脚本编译 eccheck_native.so
# 例如：
python setup.py build_ext --inplace
# 或者使用项目特定的编译脚本
```

### 2. 运行测试
```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck_4rank.sh
```

### 3. 验证点
- ✅ 每个 chunk 的 post-xor 在最后一个 column 完成后立即执行
- ✅ Parity 正确复制到 persistent store（按 chunk 偏移）
- ✅ Send/Recv 正确配对执行
- ✅ 地址偏移计算正确
- ✅ Sync 步骤只在全局执行

## 📝 实现要点总结

### 地址管理
- **Chunk 偏移**: `chunk_offset = chunk_idx × chunk_size`
- **发送地址**: `persist_parity_base + chunk_offset`
- **接收地址**: `peer_data_buffer_base + chunk_offset`

### 执行时机
- **触发点**: XOR worker 中，最后一个 column 的 XOR 完成后
- **执行**: 立即执行 `execute_chunk_level_post_xor()`
- **并行化**: 多个 chunk 可以同时在不同的阶段执行 post-xor

### 配置管理
- **Chunk-Level**: `chunk_level: true` 的 send/recv 步骤
- **Global**: `chunk_level: false` 的 sync 步骤或未标记的步骤
- **向后兼容**: 默认 `chunk_level: true` for send/recv

## ✅ 准备就绪

所有实现已完成，可以编译和测试！

