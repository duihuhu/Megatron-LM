# 共享 Parity Buffer 模型

## 正确理解

所有 encoder 共享**同一个 parity buffer**，采用**增量更新模式**：

### 流程

1. **Encoder 0 (第一个 encoder)**:
   - Encode → Send
   - **Parity = Encoding0 XOR Zero_Parity**（初始化 parity）

2. **Encoder 1, 2, ... (后续 encoder)**:
   - Encode → Send → Recv
   - **Parity = Parity XOR Recv_Encoding**（增量更新同一个 parity）

### 关键点

- **共享资源**：所有 column 操作同一个 parity buffer 地址
- **增量更新**：每个后续 column 在现有 parity 基础上增量 XOR
- **线程安全**：需要确保多个 XOR worker 对同一 parity buffer 的并发更新是安全的

## 修正后的实现

### Python 端

- **每个 chunk 一个 parity buffer**（不是每个 column 一个）
- 所有 column 传递**同一个 parity_addr** 给 C++
- Column 0: 传递 `zero_parity_addr` 和 `xor_mode=WITH_ZERO_PARITY`
- Column 1+: 传递 `recv_addr` 和 `xor_mode=INCREMENTAL`（或 WITH_RECV_ENCODING）

### C++ 端

- **共享 parity buffer**：所有 column 的 XOR 任务使用同一个 `parity_addr`
- **Column 0 XOR**：`parity = encoding0 XOR zero_parity`
- **Column 1+ XOR**：`parity = parity XOR recv_encoding`（增量更新）
- **线程安全**：XOR 操作需要对共享 parity buffer 进行加锁保护

