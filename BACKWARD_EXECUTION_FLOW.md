# Megatron-LM Backward 执行流程详解

本文档详细梳理 Megatron-LM 中 backward pass 的完整执行过程，从触发点到最终的梯度计算和通信。

---

## 目录

1. [Backward 执行概览](#1-backward-执行概览)
2. [入口点：backward_step](#2-入口点backward_step)
3. [PyTorch Autograd 引擎](#3-pytorch-autograd-引擎)
4. [逐层 Backward 详解](#4-逐层-backward-详解)
5. [梯度累积与同步](#5-梯度累积与同步)
6. [完整示例流程](#6-完整示例流程)
7. [优化技术](#7-优化技术)

---

## 1. Backward 执行概览

### 1.1 整体流程图

```
Training Loop (training.py)
    ↓
train_step()
    ↓
forward_backward_func()
    ↓
forward_backward_no_pipelining() / forward_backward_pipelining_with_interleaving()
    ↓
┌─────────────────────────────────────────────────────┐
│  Multiple Microbatches (with no_sync)              │
│                                                      │
│  for i in range(num_microbatches - 1):              │
│      forward_step()  ───────────┐                   │
│      backward_step() ◄──────────┘                   │
└─────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────┐
│  Last Microbatch (triggers gradient sync)          │
│                                                      │
│  forward_step()                                      │
│  backward_step() ◄── 触发 all-reduce/reduce-scatter │
└─────────────────────────────────────────────────────┘
    ↓
finalize_model_grads()
    ↓
optimizer.step()
```

### 1.2 关键概念

**计算图 (Computation Graph):**
- PyTorch 在 forward 时构建动态计算图
- 每个操作（Tensor 运算）创建一个节点
- 节点保存 `grad_fn`，用于 backward 计算梯度

**Autograd Context (`ctx`):**
- 每个自定义 autograd 函数有一个 context
- 通过 `ctx.save_for_backward()` 保存 backward 需要的张量
- 通过 `ctx.saved_tensors` 在 backward 中访问

**梯度流向:**
```
Loss (标量)
  ↓ grad = 1.0
Output Layer
  ↓ grad_input
Layer N
  ↓ grad_input
...
  ↓ grad_input
Layer 1
  ↓ grad_input
Embedding
```

---

## 2. 入口点：backward_step

### 2.1 函数签名和调用

**位置:** `megatron/core/pipeline_parallel/schedules.py:382`

```python
def backward_step(input_tensor, output_tensor, output_tensor_grad, 
                  model_type, config):
    """
    Backward step through passed-in output tensor.
    
    Args:
        input_tensor: 当前 stage 的输入（来自前一个 stage）
        output_tensor: 当前 stage 的输出（传给下一个 stage）
        output_tensor_grad: 从下一个 stage 传回的梯度
        model_type: 模型类型
        config: TransformerConfig
    
    Returns:
        input_tensor_grad: 传给前一个 stage 的梯度
    """
```

### 2.2 执行步骤

#### 步骤 1: 保留输入张量的梯度

```python
# 第 398-405 行
unwrap_input_tensor_grad = False
if not isinstance(input_tensor, list):
    input_tensor = [input_tensor]
    unwrap_input_tensor_grad = True

for x in input_tensor:
    if x is not None:
        x.retain_grad()  # 关键！保留中间变量的梯度
```

**为什么需要 `retain_grad()`？**
- PyTorch 默认只保留叶子节点（parameters）的梯度
- 中间激活不是叶子节点，需要显式调用 `retain_grad()`
- Pipeline Parallelism 需要将梯度传递给前一个 stage

#### 步骤 2: 处理输出张量

```python
# 第 407-414 行
if not isinstance(output_tensor, list):
    output_tensor = [output_tensor]
if not isinstance(output_tensor_grad, list):
    output_tensor_grad = [output_tensor_grad]

# 如果是最后一个 stage，output_tensor_grad 为 None
# 需要应用梯度缩放
if output_tensor_grad[0] is None and config.grad_scale_func is not None:
    output_tensor[0] = config.grad_scale_func(output_tensor[0])
```

#### 步骤 3: 触发 Backward

```python
# 第 421-425 行
if output_tensor[0].requires_grad:
    if config.deallocate_pipeline_outputs:
        # 使用优化的 backward（直接调用 C++ 引擎）
        custom_backward(output_tensor[0], output_tensor_grad[0])
    else:
        # 标准 PyTorch backward
        torch.autograd.backward(output_tensor[0], 
                                grad_tensors=output_tensor_grad[0])
```

**两种 backward 方式的区别:**

| 方式 | 函数 | 特点 |
|------|------|------|
| **标准方式** | `torch.autograd.backward()` | - 有形状检查<br>- 有安全检查<br>- 开销稍大 |
| **优化方式** | `custom_backward()` | - 直接调用 C++ 引擎<br>- 跳过形状检查<br>- 支持 `deallocate_output_tensor` |

#### 步骤 4: 收集输入梯度

```python
# 第 428-435 行
input_tensor_grad = []
for x in input_tensor:
    if x is None:
        input_tensor_grad.append(None)
    else:
        input_tensor_grad.append(x.grad)  # 提取梯度

return input_tensor_grad
```

---

## 3. PyTorch Autograd 引擎

### 3.1 计算图构建（Forward Pass）

在 forward pass 期间，PyTorch 自动构建计算图：

```python
# 示例：单个线性层
# Forward:
hidden = input @ weight.T + bias
# PyTorch 内部创建:
# Node_1: MatmulBackward (保存 input, weight)
# Node_2: AddBackward (保存 bias)
```

**计算图结构:**
```
input (requires_grad=False, 来自前一层)
  │
  ├─→ MatmulBackward
  │     ├─ saved: input, weight
  │     └─ grad_fn: backward()
  │
  └─→ AddBackward
        ├─ saved: bias
        └─ grad_fn: backward()
          │
          └─→ hidden (requires_grad=True)
```

### 3.2 Backward 遍历过程

当调用 `torch.autograd.backward(hidden, grad_tensors=grad_output)` 时：

```
步骤 1: 初始化
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
hidden.grad = grad_output  # 设置初始梯度

步骤 2: 拓扑排序
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
按照计算图的依赖关系，确定 backward 执行顺序
（从 hidden 开始，逆向遍历到 input）

执行顺序:
1. AddBackward.backward()
2. MatmulBackward.backward()

步骤 3: 执行 Backward 函数
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
For each node in reverse topological order:
    node.backward(grad_output)
    # 计算上游梯度
    # 更新参数梯度（如果是叶子节点）
```

### 3.3 Custom Backward

**位置:** `megatron/core/pipeline_parallel/schedules.py:140`

```python
def custom_backward(output, grad_output):
    '''直接调用 C++ autograd 引擎
    
    为了让 'deallocate_output_tensor' 优化生效，必须直接调用 C++
    autograd 引擎，跳过 PyTorch 的 torch.autograd.backward。
    '''
    
    # 验证
    assert output.numel() == 1, "output should be pseudo-'freed'"
    
    # 处理标量输出
    if grad_output is None:
        grad_output = torch.ones_like(output)
    
    # 直接调用 C++ 引擎
    # 见 torch/csrc/autograd/python_engine.cpp
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,        # 不保留计算图
        create_graph=False,      # 不创建二阶导数图
        inputs=tuple(),          # 不指定特定输入
        allow_unreachable=True,  # 允许不可达节点
        accumulate_grad=True,    # 累积梯度（重要！）
    )
```

**`accumulate_grad=True` 的作用:**
- 梯度会累加到现有的 `.grad` 属性中
- 支持梯度累积（多个 microbatch）
- 公式: `param.grad += new_grad`

---

## 4. 逐层 Backward 详解

### 4.1 完整的 Transformer Layer Backward

假设一个标准的 Transformer Layer 结构：

```python
# Forward 伪代码
def forward(hidden_states):
    # 1. Pre-Attention LayerNorm
    residual_1 = hidden_states
    ln1_output = layernorm1(hidden_states)
    
    # 2. Attention
    attn_output = attention(ln1_output)
    hidden_states = residual_1 + attn_output
    
    # 3. Pre-MLP LayerNorm
    residual_2 = hidden_states
    ln2_output = layernorm2(hidden_states)
    
    # 4. MLP
    mlp_output = mlp(ln2_output)
    output = residual_2 + mlp_output
    
    return output
```

**Backward 执行顺序（逆向）:**

```
输入: grad_output (来自下一层或 loss)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

步骤 1: Residual Add Backward (MLP 部分)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grad_mlp_output = grad_output
grad_residual_2 = grad_output
# Residual connection: 梯度直接传递

步骤 2: MLP Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2.1 FC2 Backward
grad_fc2_input = grad_mlp_output @ weight_fc2
weight_fc2.main_grad += activated_output.T @ grad_mlp_output
bias_fc2.grad += grad_mlp_output.sum(dim=(0,1))

# 2.2 Activation (GeLU) Backward
grad_fc1_output = gelu_backward(grad_fc2_input, fc1_output)

# 2.3 FC1 Backward
grad_ln2_output = grad_fc1_output @ weight_fc1
weight_fc1.main_grad += ln2_output.T @ grad_fc1_output
bias_fc1.grad += grad_fc1_output.sum(dim=(0,1))

步骤 3: LayerNorm2 Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grad_hidden_states_2 = layernorm_backward(
    grad_ln2_output, 
    hidden_states_before_ln2,
    mean, variance, weight, bias
)
ln2_weight.grad += (grad_ln2_output * normalized).sum()
ln2_bias.grad += grad_ln2_output.sum()

步骤 4: Residual Add Backward (Attention 部分)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grad_attn_output = grad_residual_2 + grad_hidden_states_2
grad_residual_1 = grad_residual_2 + grad_hidden_states_2

步骤 5: Attention Backward (详细见下文)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grad_ln1_output = attention_backward(grad_attn_output, ...)

步骤 6: LayerNorm1 Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grad_hidden_states_1 = layernorm_backward(...)

步骤 7: 合并梯度
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
grad_input = grad_residual_1 + grad_hidden_states_1

返回: grad_input (传给前一层)
```

### 4.2 Attention Backward 详解

**位置:** `megatron/core/transformer/dot_product_attention.py`

```python
# Forward 保存的数据（标准实现）
ctx.save_for_backward(
    query,              # [sq, b, np, hn]
    key,                # [sk, b, np, hn]
    value,              # [sk, b, np, hn]
    attention_scores,   # [b, np, sq, sk]
    attention_probs,    # [b, np, sq, sk]
)
```

**Backward 计算过程:**

```python
def backward(ctx, grad_output):
    """
    grad_output: 对 context_layer 的梯度 [sq, b, np, hn]
    """
    
    # === 1. 恢复保存的张量 ===
    query, key, value, attention_scores, attention_probs = ctx.saved_tensors
    
    # === 2. 输出投影 Backward ===
    # forward: context = attention_probs @ value
    # backward:
    grad_attention_probs = torch.matmul(
        grad_output,              # [sq, b, np, hn]
        value.transpose(-2, -1)   # [sk, b, np, hn] -> [b, np, hn, sk]
    )  # -> [b, np, sq, sk]
    
    grad_value = torch.matmul(
        attention_probs.transpose(-2, -1),  # [b, np, sk, sq]
        grad_output                          # [sq, b, np, hn]
    )  # -> [sk, b, np, hn]
    
    # === 3. Dropout Backward ===
    # dropout 在 forward 时保存了 mask（隐式）
    # backward: grad *= mask / (1 - dropout_prob)
    grad_attention_probs = dropout_backward(
        grad_attention_probs, 
        dropout_mask
    )
    
    # === 4. Softmax Backward ===
    # forward: attention_probs = softmax(attention_scores)
    # backward 公式:
    # grad_scores = probs * (grad_probs - sum(grad_probs * probs))
    
    sum_grad_probs = (grad_attention_probs * attention_probs).sum(
        dim=-1, keepdim=True
    )
    grad_attention_scores = attention_probs * (
        grad_attention_probs - sum_grad_probs
    )
    
    # === 5. Scale Backward ===
    # forward: attention_scores = scores * scale
    # backward: grad_scores *= scale
    grad_attention_scores = grad_attention_scores * softmax_scale
    
    # === 6. Mask Backward ===
    # Mask 在 forward 时将某些位置设为 -inf
    # backward: 这些位置的梯度应该为 0
    # (softmax 的梯度已经自动处理了这个)
    
    # === 7. QK^T Backward ===
    # forward: attention_scores = query @ key.T
    # backward:
    grad_query = torch.matmul(
        grad_attention_scores,    # [b, np, sq, sk]
        key                       # [sk, b, np, hn]
    )  # -> [sq, b, np, hn]
    
    grad_key = torch.matmul(
        grad_attention_scores.transpose(-2, -1),  # [b, np, sk, sq]
        query                                      # [sq, b, np, hn]
    )  # -> [sk, b, np, hn]
    
    # === 8. QKV 投影 Backward ===
    # 这部分由 Linear layer 的 backward 处理
    
    return grad_query, grad_key, grad_value
```

**内存优化：Flash Attention 的 Backward**

Flash Attention 不保存完整的 `attention_scores` 和 `attention_probs`：

```python
# Forward 只保存:
ctx.save_for_backward(
    query, key, value,
    output,
    softmax_lse,  # log-sum-exp: [b, np, sq] (小！)
    rng_state,    # 随机数状态（用于 dropout）
)

# Backward 通过融合 CUDA kernel 重新计算:
# - 不需要显式读取完整的 attention matrix
# - 利用 block-wise 计算，内存友好
# - 速度反而更快（减少内存访问）
```

### 4.3 Linear Layer Backward (Tensor Parallel)

**位置:** `megatron/core/tensor_parallel/layers.py:487`

这是 Megatron 中最核心的 backward 实现。

```python
@staticmethod
def backward(ctx, grad_output):
    """
    LinearWithGradAccumulationAndAsyncCommunication.backward
    
    参数:
        grad_output: 对输出的梯度 [s, b, h]
    
    返回:
        grad_input, grad_weight, grad_bias, ...
    """
    
    # ========== 恢复保存的数据 ==========
    input, weight = ctx.saved_tensors
    main_grad = ctx.main_grad  # 指向 grad_buffer 的引用
    
    # ========== 配置恢复 ==========
    use_bias = ctx.use_bias
    gradient_accumulation_fusion = ctx.gradient_accumulation_fusion
    allreduce_dgrad = ctx.allreduce_dgrad
    sequence_parallel = ctx.sequence_parallel
    tp_group = ctx.tp_group
    
    # ========== 1. 计算输入梯度 (DGRAD) ==========
    # forward: output = input @ weight.T
    # backward: grad_input = grad_output @ weight
    
    grad_input = grad_output.matmul(weight)  
    # [s, b, h_out] @ [h_out, h_in] = [s, b, h_in]
    
    # ========== 2. 准备权重梯度计算 ==========
    
    # 2.1 Sequence Parallel: all-gather 输入
    if sequence_parallel:
        # 每个 rank 只有部分序列
        # 需要 all-gather 完整输入来计算权重梯度
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * tp_group.size()
        
        all_gather_buffer = get_global_memory_buffer().get_tensor(
            dim_size, input.dtype, "mpu"
        )
        
        # 异步 all-gather
        handle = dist_all_gather_func(
            all_gather_buffer, input, 
            group=tp_group, async_op=True
        )
        total_input = all_gather_buffer
    else:
        total_input = input
    
    # ========== 3. Tensor Parallel 通信（输入梯度）==========
    
    # 3.1 All-reduce input gradient (标准 TP)
    if allreduce_dgrad:
        handle = torch.distributed.all_reduce(
            grad_input, group=tp_group, async_op=True
        )
        # 依赖 CUDA_DEVICE_MAX_CONNECTIONS=1 确保调度顺序
    
    # 3.2 Reduce-scatter input gradient (Sequence Parallel)
    if sequence_parallel:
        sub_grad_input = torch.empty(
            input.size(), dtype=input.dtype, 
            device=torch.cuda.current_device()
        )
        handle = dist_reduce_scatter_func(
            sub_grad_input, grad_input, 
            group=tp_group, async_op=True
        )
        grad_input = sub_grad_input
    
    # ========== 4. 计算权重梯度 (WGRAD) ==========
    
    if sequence_parallel:
        handle.wait()  # 等待 all-gather 完成
    
    # 准备张量（可能需要转置或重塑）
    grad_output, total_input = prepare_input_tensors_for_wgrad_compute(
        grad_output, total_input
    )
    
    if gradient_accumulation_fusion:
        # 使用融合 CUDA kernel 直接累加
        weight.main_grad = main_grad
        
        if weight.main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input,     # [total_s, b, h_in]
                grad_output,     # [s, b, h_out]
                weight.main_grad # [h_out, h_in]
            )
            # 等价于: weight.main_grad += total_input.T @ grad_output
        
        elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                total_input, grad_output, weight.main_grad
            )
        
        grad_weight = None  # 不返回，因为已经累加到 main_grad
    
    else:
        # 标准方式：计算权重梯度
        grad_weight = grad_output.t().matmul(total_input)
        # [h_out, s*b] @ [s*b, h_in] = [h_out, h_in]
    
    # ========== 5. 计算偏置梯度 (BGRAD) ==========
    
    if use_bias:
        grad_bias = grad_output.sum(dim=0)  # 沿 seq 维度求和
        # [s, b, h_out] -> [b, h_out] -> [h_out]
    else:
        grad_bias = None
    
    # ========== 6. 等待通信完成 ==========
    
    if sequence_parallel or allreduce_dgrad:
        handle.wait()
    
    # ========== 7. 返回梯度 ==========
    
    # 返回值顺序必须与 forward 的输入参数对应
    return (
        grad_input,      # 对 input 的梯度
        grad_weight,     # 对 weight 的梯度 (或 None)
        grad_bias,       # 对 bias 的梯度 (或 None)
        None,            # gradient_accumulation_fusion (不需要梯度)
        None,            # allreduce_dgrad
        None,            # sequence_parallel
        None,            # grad_output_buffer
        None,            # wgrad_deferral_limit
        None,            # tp_group
    )
```

**关键优化技术:**

1. **梯度累积融合 (Gradient Accumulation Fusion)**
   ```python
   # 标准方式 (需要额外内存)
   grad_weight = input.T @ grad_output
   weight.main_grad += grad_weight  # 两步
   
   # 融合方式 (直接累加)
   fused_wgrad_gemm_accum(input, grad_output, weight.main_grad)
   # 一步完成，节省内存
   ```

2. **异步通信 (Async Communication)**
   ```python
   # 启动 all-reduce (异步)
   handle = all_reduce(grad_input, async_op=True)
   
   # 继续计算权重梯度（与通信重叠）
   compute_weight_gradient(...)
   
   # 等待通信完成
   handle.wait()
   ```

3. **通信与计算重叠 (Communication-Computation Overlap)**
   ```
   Timeline:
   ──────────────────────────────────────────────
   T0: 启动 all-reduce(grad_input) [异步]
   T1: 计算 weight gradient        [与通信重叠]
   T2: wait() - 通信完成
   T3: 返回结果
   ──────────────────────────────────────────────
   ```

### 4.4 Activation Checkpointing Backward

**位置:** `megatron/core/tensor_parallel/random.py:443`

```python
@staticmethod
def backward(ctx, *args):
    """CheckpointFunction.backward"""
    
    # === 1. 恢复保存的输入 ===
    inputs = ctx.saved_tensors  # 只保存了输入！
    
    # === 2. 恢复 RNG 状态 ===
    with _fork_rng():
        # 恢复到 forward 时的随机数状态
        _set_all_rng_states(*ctx.rng_states)
        
        # === 3. 重新执行 Forward ===
        detached_inputs = detach_variable(inputs)
        with torch.enable_grad():
            # 重新计算所有中间激活
            outputs = ctx.run_function(*detached_inputs)
            # 这次 PyTorch 会保存中间激活用于 backward
    
    # === 4. 执行 Backward ===
    if isinstance(outputs, torch.Tensor):
        outputs = (outputs,)
    
    # 过滤出需要梯度的输出
    outputs, args = zip(*filter(
        lambda x: torch.is_tensor(x[0]) and x[0].requires_grad,
        zip(outputs, args)
    ))
    
    # 标准 backward
    torch.autograd.backward(outputs, args)
    
    # === 5. 收集输入梯度 ===
    grads = tuple(
        inp.grad if isinstance(inp, torch.Tensor) else inp 
        for inp in detached_inputs
    )
    
    return (None, None) + grads
```

**内存-计算权衡:**

| 方案 | Forward 保存 | Backward 计算 | 内存 | 时间 |
|------|-------------|--------------|------|------|
| **无 Checkpointing** | 所有中间激活 | 直接使用保存的激活 | 100% | 100% |
| **Full Checkpointing** | 只保存输入 | 重新执行完整 forward | ~30% | ~133% |
| **Selective Checkpointing** | 部分中间激活 | 部分重计算 | ~60% | ~115% |

---

## 5. 梯度累积与同步

### 5.1 梯度累积机制

**在 grad_buffer 中累积:**

```python
# 初始化 (iteration 开始)
for param in model.parameters():
    param.main_grad.zero_()  # 清零

# Microbatch 0
backward_step()
# weight.main_grad = ∂L₀/∂W

# Microbatch 1
backward_step()
# weight.main_grad += ∂L₁/∂W  (累加！)

# Microbatch 2
backward_step()
# weight.main_grad += ∂L₂/∂W

# Microbatch 3
backward_step()
# weight.main_grad += ∂L₃/∂W

# 最终: weight.main_grad = ∂L₀/∂W + ∂L₁/∂W + ∂L₂/∂W + ∂L₃/∂W
```

### 5.2 跨 GPU 梯度同步

**位置:** `megatron/core/distributed/param_and_grad_buffer.py:319`

```python
def start_grad_sync(self):
    """
    启动梯度同步（all-reduce 或 reduce-scatter）
    """
    
    # === 1. 检查梯度 ===
    if self.ddp_config.check_for_nan_in_grad:
        self.check_grads(check_for_nan_or_inf=True)
    
    # === 2. 缩放梯度（用于平均）===
    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor
            # 例如: grad *= (1.0 / data_parallel_world_size)
    
    # === 3. 决定 reduce 操作类型 ===
    if self.ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG
        # 直接在通信中平均
    else:
        reduce_op = torch.distributed.ReduceOp.SUM
        # 先缩放再求和
    
    # === 4. 执行通信 ===
    async_op = self.ddp_config.overlap_grad_reduce
    
    # 4.1 使用 Distributed Optimizer (ZeRO)
    if self.ddp_config.use_distributed_optimizer:
        with _coalescing_manager(communication_group, async_ops=async_op):
            for bucket in self.buckets:
                local_data_view = shard_buffer(
                    bucket.grad_data, 
                    self.intra_distributed_optimizer_instance_size
                )[self.rank]
                
                # Reduce-scatter: 每个 rank 只保留一部分梯度
                dist_reduce_scatter_func(
                    local_data_view,
                    bucket.grad_data,
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )
    
    # 4.2 标准 DDP (All-reduce)
    else:
        with _coalescing_manager(communication_group, async_ops=async_op):
            for bucket in self.buckets:
                # All-reduce: 所有 rank 得到相同的梯度
                torch.distributed.all_reduce(
                    bucket.grad_data,
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )
```

### 5.3 完整的梯度同步流程

```
训练迭代开始
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. zero_grad_buffer()
   ├─ 清零所有 param.main_grad
   └─ 准备接收新的梯度

Microbatch Loop (with no_sync)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with model.no_sync():
    for i in range(num_microbatches - 1):
        2. forward_step(microbatch[i])
           └─ 计算输出
        
        3. backward_step(microbatch[i])
           ├─ 计算梯度
           ├─ 累加到 param.main_grad
           └─ 不触发通信（no_sync）

Last Microbatch (triggers sync)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
4. forward_step(microbatch[N-1])

5. backward_step(microbatch[N-1])
   ├─ 计算梯度
   ├─ 累加到 param.main_grad
   └─ 自动触发 DDP hooks

6. DDP hooks 被调用
   ├─ start_grad_sync()
   │  ├─ 缩放梯度: grad *= (1/dp_size)
   │  └─ all-reduce 或 reduce-scatter
   └─ 等待通信完成

Finalize Gradients
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
7. finalize_model_grads()
   ├─ finish_grad_sync() (确保通信完成)
   ├─ all-reduce embedding grads (PP)
   ├─ all-reduce layernorm grads (SP)
   └─ scale by num_tokens (如果启用 per-token loss)

Optimizer Step
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
8. optimizer.step()
   └─ 使用 param.main_grad 更新参数

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
迭代结束
```

### 5.4 梯度平均的数学原理

假设有 2 个 GPU，每个处理 2 个 microbatches：

```
GPU 0 处理的数据:
  Microbatch 0: grad₀ = ∂L(batch0)/∂W
  Microbatch 2: grad₂ = ∂L(batch2)/∂W
  本地累积: local_grad_0 = grad₀ + grad₂

GPU 1 处理的数据:
  Microbatch 1: grad₁ = ∂L(batch1)/∂W
  Microbatch 3: grad₃ = ∂L(batch3)/∂W
  本地累积: local_grad_1 = grad₁ + grad₃

方式 1: 预缩放 + SUM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GPU 0: local_grad_0 *= 0.5  →  (grad₀ + grad₂) * 0.5
GPU 1: local_grad_1 *= 0.5  →  (grad₁ + grad₃) * 0.5

All-Reduce (SUM):
final_grad = (grad₀ + grad₂) * 0.5 + (grad₁ + grad₃) * 0.5
           = (grad₀ + grad₁ + grad₂ + grad₃) / 2

方式 2: AVG reduce
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
All-Reduce (AVG):
final_grad = ((grad₀ + grad₂) + (grad₁ + grad₃)) / 2
           = (grad₀ + grad₁ + grad₂ + grad₃) / 2

结果相同！
```

---

## 6. 完整示例流程

### 6.1 单个 Microbatch 的完整 Backward

假设模型：Embedding → Layer0 → Layer1 → Output → Loss

```python
# ========== Forward Pass (已完成) ==========
# 构建了计算图：
# input_ids → embedding → hidden0 → layer0 → hidden1 → layer1 → output → loss

# ========== Backward Pass ==========

# 步骤 1: 触发 backward
backward_step(
    input_tensor=None,  # 第一个 stage 没有输入
    output_tensor=loss,
    output_tensor_grad=None,  # 最后一层，grad 为 1.0
)

# 步骤 2: PyTorch autograd 引擎启动
torch.autograd.backward(loss, grad_tensors=None)
# grad_tensors=None 意味着 loss.grad = 1.0

# 步骤 3: 按拓扑顺序反向遍历计算图
"""
Loss (标量)
  ↓ grad = 1.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Cross Entropy Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  保存数据: exp_logits, target_mask, masked_target
  计算:
    grad_logits = exp_logits / sum_exp_logits
    grad_logits[target] -= 1
    grad_logits *= grad_output  # grad_output = 1.0
  返回: grad_logits [s, b, vocab_size/tp]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output Layer (Linear) Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  保存数据: hidden1, weight_output
  计算:
    grad_hidden1 = grad_logits @ weight_output
    weight_output.main_grad += hidden1.T @ grad_logits
  通信: all-reduce grad_hidden1 (if TP)
  返回: grad_hidden1 [s, b, h]
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 1 Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  ┌──────────────────────────────────┐
  │ Residual Add (MLP)               │
  │   grad_mlp_out = grad_hidden1    │
  │   grad_residual_2 = grad_hidden1 │
  └──────────────────────────────────┘
  ↓
  ┌──────────────────────────────────┐
  │ MLP Backward                     │
  │   FC2: grad_fc2_in = ...         │
  │        weight_fc2.grad += ...    │
  │   GeLU: grad_fc1_out = ...       │
  │   FC1: grad_ln2_out = ...        │
  │        weight_fc1.grad += ...    │
  └──────────────────────────────────┘
  ↓
  ┌──────────────────────────────────┐
  │ LayerNorm 2 Backward             │
  │   grad_hidden_2 = ...            │
  │   ln2.weight.grad += ...         │
  └──────────────────────────────────┘
  ↓
  ┌──────────────────────────────────┐
  │ Residual Add (Attention)         │
  │   grad_attn_out = grad_residual_2│
  │                 + grad_hidden_2  │
  │   grad_residual_1 = ...          │
  └──────────────────────────────────┘
  ↓
  ┌──────────────────────────────────┐
  │ Attention Backward               │
  │   Output Proj: grad_context = ..│
  │   Core Attn: grad_Q, grad_K, ..  │
  │   QKV Proj: grad_ln1_out = ...   │
  └──────────────────────────────────┘
  ↓
  ┌──────────────────────────────────┐
  │ LayerNorm 1 Backward             │
  │   grad_hidden0 = ...             │
  └──────────────────────────────────┘
  ↓
  ┌──────────────────────────────────┐
  │ 合并梯度                          │
  │   grad_layer1_input =            │
  │     grad_residual_1 + grad_hidden0│
  └──────────────────────────────────┘
  返回: grad_layer1_input
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 0 Backward (类似 Layer 1)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  [相同的结构]
  返回: grad_layer0_input
  ↓
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Embedding Backward
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  保存数据: input_ids
  计算:
    embedding.weight.grad[input_ids] += grad_layer0_input
    # 稀疏更新，只更新使用到的 token
  通信: all-reduce embedding.grad (if PP)
  返回: None (embedding 是第一层)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# 步骤 4: 所有梯度已累积到 param.main_grad
# 等待最后一个 microbatch 触发通信
```

### 6.2 多 Microbatch 的梯度累积

```python
# 初始化
model.zero_grad_buffer()

# ========== Microbatch 0 ==========
forward_step(data[0])
# → loss0

with model.no_sync():  # 抑制通信
    backward_step(loss0)
    # weight_fc1.main_grad = grad0_fc1
    # weight_fc2.main_grad = grad0_fc2

# ========== Microbatch 1 ==========
forward_step(data[1])
# → loss1

with model.no_sync():
    backward_step(loss1)
    # weight_fc1.main_grad += grad1_fc1  (累加!)
    # weight_fc2.main_grad += grad2_fc2

# ========== Microbatch 2 (最后一个) ==========
forward_step(data[2])
# → loss2

# 不在 no_sync 上下文中，会触发通信
backward_step(loss2)
# weight_fc1.main_grad += grad2_fc1
# weight_fc2.main_grad += grad2_fc2

# 自动触发 DDP hooks:
#   1. 缩放: grad *= (1/dp_size)
#   2. all-reduce 或 reduce-scatter
#   3. 所有 GPU 的梯度被同步

# ========== Finalize ==========
finalize_model_grads()
# - 确保通信完成
# - 处理特殊梯度 (embedding, layernorm)

# ========== Optimizer Step ==========
optimizer.step()
# 使用 param.main_grad 更新参数

# ========== 清零 ==========
model.zero_grad_buffer()
# 为下一个 iteration 做准备
```

---

## 7. 优化技术

### 7.1 内存优化

#### 7.1.1 Activation Checkpointing

**Trade-off:**
```
无 Checkpointing:
  内存: 保存所有中间激活
  时间: 100% (baseline)

Full Checkpointing:
  内存: 只保存输入 (~70% 节省)
  时间: ~133% (重计算开销)

Selective Checkpointing:
  内存: 保存部分激活 (~40% 节省)
  时间: ~115% (部分重计算)
```

**实现:**
```python
# Full checkpointing
if config.recompute_granularity == 'full':
    hidden_states = checkpoint(layer, hidden_states)

# Selective checkpointing
if config.recompute_granularity == 'selective':
    if 'core_attn' in config.recompute_modules:
        attn_output = checkpoint(core_attention, ...)
```

#### 7.1.2 Flash Attention

**标准 Attention:**
```python
# Forward 保存: [b, h, s, s] attention matrix
# 内存: 4 * batch * heads * seq_len^2 bytes
# 例如: 4 * 8 * 96 * 2048^2 = 12 GB
```

**Flash Attention:**
```python
# Forward 保存: [b, h, s] softmax_lse
# 内存: 4 * batch * heads * seq_len bytes
# 例如: 4 * 8 * 96 * 2048 = 6 MB

# 节省: ~99.95% !
```

#### 7.1.3 Gradient Accumulation Fusion

**标准方式:**
```python
# 计算权重梯度
grad_weight = input.T @ grad_output  # 需要临时内存

# 累加到 main_grad
weight.main_grad += grad_weight  # 额外的内存和操作
```

**融合方式:**
```python
# 直接累加，无临时内存
fused_wgrad_gemm_accum_fp32(
    input, grad_output, weight.main_grad
)
# 节省: grad_weight 的内存
# 加速: ~5-10%
```

### 7.2 通信优化

#### 7.2.1 异步通信

```python
# 标准同步方式
all_reduce(grad)  # 阻塞，等待完成
compute_next()

# 异步方式
handle = all_reduce(grad, async_op=True)  # 不阻塞
compute_next()  # 与通信重叠
handle.wait()  # 只在需要时等待
```

**Timeline 对比:**
```
同步方式:
├──────── all-reduce ────────┤
                              ├── compute ──┤
总时间: T_comm + T_compute

异步方式:
├──────── all-reduce ────────┤
├────── compute ──────┤
总时间: max(T_comm, T_compute)  # 重叠!
```

#### 7.2.2 Gradient Bucketing

```python
# 不分桶: 等所有梯度计算完再通信
backward()  # 计算所有梯度
all_reduce(all_grads)  # 一次性通信

# 分桶: 边计算边通信
for bucket in buckets:
    backward(bucket)  # 计算部分梯度
    all_reduce(bucket.grads, async_op=True)  # 立即启动通信
    # 通信与后续计算重叠
```

**Bucket 策略:**
```python
# 按反向顺序分组（最先计算的梯度最先通信）
buckets = [
    [layer_N_params],      # 最先计算，最先通信
    [layer_N-1_params],
    [layer_N-2_params],
    ...
    [layer_0_params],      # 最后计算，最后通信
]
```

#### 7.2.3 减少通信次数

**No-sync 机制:**
```python
# 不好的做法: 每个 microbatch 都通信
for mb in microbatches:
    forward(mb)
    backward(mb)
    all_reduce()  # 4 次通信 (假设 4 个 microbatches)

# 好的做法: 只在最后通信
with model.no_sync():
    for mb in microbatches[:-1]:
        forward(mb)
        backward(mb)  # 不触发通信

forward(microbatches[-1])
backward(microbatches[-1])  # 触发一次通信

# 结果: 1 次通信 (节省 75%)
```

### 7.3 计算优化

#### 7.3.1 融合算子

**BiasGeLU 融合:**
```python
# 未融合: 3 次内存访问
tmp = input + bias      # 读 input, bias; 写 tmp
output = gelu(tmp)      # 读 tmp; 写 output

# 融合: 2 次内存访问
output = bias_gelu(input, bias)  # 读 input, bias; 写 output
# 节省: ~30% 内存带宽
```

**其他融合:**
- BiasSwiGLU
- BiasGEGLU
- ScaleMaskSoftmax
- LayerNorm + Bias

#### 7.3.2 Kernel 优化

**CUDA Kernels:**
```python
# 标准 PyTorch
grad = input.T @ grad_output  # 调用 cuBLAS

# 自定义融合 kernel
fused_wgrad_gemm_accum_fp32(
    input, grad_output, weight.main_grad
)
# 优势:
# - 直接累加，无临时内存
# - 优化的 CUDA 实现
# - 支持不同精度
```

---

## 总结

### Backward 执行的关键点

1. **触发点:** `backward_step()` → `torch.autograd.backward()`
2. **执行引擎:** PyTorch Autograd Engine (C++)
3. **遍历顺序:** 拓扑排序，从输出到输入
4. **梯度存储:** `param.main_grad` (grad_buffer 的视图)
5. **梯度累积:** 自动累加 (`accumulate_grad=True`)
6. **通信时机:** 最后一个 microbatch 后
7. **优化核心:** 减少内存、重叠通信、融合计算

### 数据流动路径

```
Forward: 
  Input → Layer1 → Layer2 → ... → Output → Loss

Backward:
  Loss.grad(=1) → Output.grad → ... → Layer2.grad → Layer1.grad
                      ↓                    ↓             ↓
                 weight.grad          weight.grad   weight.grad
                      ↓                    ↓             ↓
                 grad_buffer[n]      grad_buffer[2] grad_buffer[1]
                                          ↓
                                    All-Reduce / Reduce-Scatter
                                          ↓
                                   Optimizer.step()
```

### 性能关键

**内存:** Checkpointing + Flash Attention  
**通信:** Async + Overlap + Bucketing  
**计算:** Fusion + Custom Kernels

这样的设计让 Megatron-LM 能够高效地训练超大规模模型！

