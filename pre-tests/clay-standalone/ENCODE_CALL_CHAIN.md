# Clay Code 编码调用链分析

## 编码流程概览

当调用 `encoder->encode()` 时，执行的计算流程如下：

## 调用链

```
test_clay.cc
  └─> encoder->encode(want_to_encode, input, &encoded)
      │
      ├─> ErasureCode::encode() [ec/erasure_code.cc:191]
      │   ├─> encode_prepare()  // 准备数据块，分割输入数据
      │   └─> encode_chunks()   // 执行编码
      │
      └─> ErasureCodeClay::encode_chunks() [clay/erasure_code_clay.cc:110]
          │
          ├─> 准备 chunks 和 parity_chunks
          ├─> 创建中间缓冲区（如果需要）
          │
          └─> decode_layered(parity_chunks, &chunks) [clay/erasure_code_clay.cc:649]
              │
              ├─> 初始化 U_buf（uncoupled buffer）
              ├─> 设置子块解码顺序
              │
              └─> 对每个子块（sub_chunk_no 个）循环处理：
                  │
                  ├─> decode_erasures() [clay/erasure_code_clay.cc:718]
                  │   ├─> 处理 coupled/uncoupled chunks 转换
                  │   └─> decode_uncoupled() [clay/erasure_code_clay.cc:750]
                  │       │
                  │       └─> mds.erasure_code->decode_chunks() [clay/erasure_code_clay.cc:766]
                  │           │
                  │           └─> ErasureCodeJerasure::decode_chunks() [jerasure/erasure_code_jerasure.cc]
                  │               │
                  │               └─> jerasure_encode() [jerasure/erasure_code_jerasure.cc:102]
                  │                   │
                  │                   └─> jerasure_matrix_encode() [jerasure库: jerasure.c:308]
                  │                       │
                  │                       └─> jerasure_matrix_dotprod() [jerasure库: jerasure.c]
                  │                           │
                  │                           └─> 最终计算：矩阵乘法 + Galois 域运算
                  │                               (使用 gf-complete 库进行 GF(2^w) 运算)
                  │
                  └─> recover_type1_erasure() / get_coupled_from_uncoupled() 等
                      └─> pft.erasure_code->decode_chunks()
                          └─> 类似的调用链，最终也是 jerasure_matrix_encode()
```

## 最终执行的计算

### 1. **MDS 编码器（主要计算）**

位置：`jerasure_matrix_encode()` → `jerasure_matrix_dotprod()`

**计算内容**：
- **矩阵乘法**：对每个校验块，计算 `coding[i] = Σ(matrix[i][j] * data[j])`
- **Galois 域运算**：在 GF(2^w) 域上进行乘法和加法
  - 对于 k=4, m=2：需要计算 2 个校验块
  - 对于 k=2, m=2：需要计算 2 个校验块
- **子块处理**：对于 k=4, m=2，需要处理 8 个子块；对于 k=2, m=2，需要处理 4 个子块

**代码位置**：
```c
// lib/ec/jerasure/src/jerasure.c:308
void jerasure_matrix_encode(int k, int m, int w, int *matrix,
                            char **data_ptrs, char **coding_ptrs, int size)
{
  for (i = 0; i < m; i++) {
    jerasure_matrix_dotprod(k, w, matrix+(i*k), NULL, k+i, 
                           data_ptrs, coding_ptrs, size);
  }
}
```

### 2. **PFT 编码器（辅助计算）**

位置：`pft.erasure_code->decode_chunks()`

**计算内容**：
- 用于处理 coupled/uncoupled chunks 之间的转换
- 同样使用 Jerasure 库进行矩阵运算
- 配置为 k=2, m=2 的小规模编码

### 3. **内存操作**

- **数据分割**：将输入数据分割成多个子块
- **Coupled/Uncoupled 转换**：在编码过程中进行数据格式转换
- **内存拷贝**：多次内存拷贝操作（coupled ↔ uncoupled）

## 性能瓶颈分析

### 主要计算开销

1. **矩阵乘法计算**（占比最大）
   - 每个子块都需要执行 MDS 编码
   - 对于 k=4, m=2：8 个子块 × 2 个校验块 = 16 次矩阵乘法
   - 对于 k=2, m=2：4 个子块 × 2 个校验块 = 8 次矩阵乘法

2. **Galois 域运算**
   - GF(2^8) 上的乘法和加法
   - 使用 gf-complete 库实现
   - 可能使用 SIMD 优化（如果支持）

3. **内存操作**
   - 多次内存拷贝（coupled/uncoupled 转换）
   - 内存对齐操作（SIMD_ALIGN）

### 小数据时的问题

- **固定开销占比大**：
  - 函数调用开销
  - 内存分配开销
  - 循环控制开销
  - 这些开销在小数据时占比很大

- **子块处理开销**：
  - 即使数据很小，也需要处理所有子块
  - 对于 k=4, m=2，即使只有 1KB 数据，也要处理 8 个子块

## 优化建议

1. **减少子块数量**：使用 k=2, m=2 配置（sub_chunk_no=4）而不是 k=4, m=2（sub_chunk_no=8）

2. **使用更大的数据块**：减少固定开销占比

3. **SIMD 优化**：确保使用 SIMD 指令优化 Galois 域运算

4. **减少内存拷贝**：优化 coupled/uncoupled 转换过程

## 总结

**最终执行的计算是**：
- **矩阵乘法**：在 GF(2^8) 域上计算 `coding = matrix × data`
- **执行位置**：`jerasure_matrix_dotprod()` 函数
- **调用频率**：对于 k=4, m=2，每个子块调用一次，共 8 次；对于 k=2, m=2，每个子块调用一次，共 4 次
- **底层库**：gf-complete（Galois 域运算）+ Jerasure（矩阵编码）

