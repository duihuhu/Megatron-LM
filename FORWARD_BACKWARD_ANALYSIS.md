# Megatron-LM Core 模块 Forward/Backward 完整分析

本文档详细列出 Megatron-LM `core/` 目录中所有执行 forward 和 backward 的模块，以及它们保存和计算的数据。

---

## 目录
1. [模型层级模块](#1-模型层级模块)
2. [Transformer 核心模块](#2-transformer-核心模块)
3. [Tensor Parallel 操作](#3-tensor-parallel-操作)
4. [融合算子 (Fusions)](#4-融合算子-fusions)
5. [MoE 模块](#5-moe-模块)
6. [总结表格](#6-总结表格)

---

## 1. 模型层级模块

### 1.1 GPTModel (`models/gpt/gpt_model.py`)

**Forward 流程：**
```python
def forward(self, input_ids, position_ids, attention_mask, ...):
    # 1. Embedding
    hidden_states = self.embedding(input_ids, position_ids)
    # 形状: [seq_len, batch, hidden_size]
    
    # 2. Encoder (TransformerBlock)
    hidden_states = self.decoder(hidden_states, attention_mask, ...)
    
    # 3. Final LayerNorm
    if self.post_process:
        hidden_states = self.final_layernorm(hidden_states)
    
    # 4. Output Layer (如果是最后一个 stage)
    if self.post_process and not self.pre_process:
        logits = self.output_layer(hidden_states)
    
    return hidden_states
```

**保存的数据：**
- Embedding 的输出
- 每层 Transformer 的中间激活（通过 PyTorch autograd 自动保存）
- 如果使用 activation checkpointing，只保存每层的输入

**Backward 计算：**
- 自动通过 PyTorch autograd 反向传播
- 从 loss 开始，逐层计算梯度
- 梯度累积在 `param.main_grad` 中

---

### 1.2 LanguageModelEmbedding (`models/common/embeddings/language_model_embedding.py`)

**Forward 流程：**
```python
def forward(self, input_ids, position_ids, tokentype_ids=None):
    # 1. Word Embedding
    words_embeddings = self.word_embeddings(input_ids)
    # 形状: [seq_len, batch, hidden_size]
    
    # 2. Position Embedding (如果启用)
    if self.add_position_embedding:
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings
    
    # 3. Token Type Embedding (如果启用)
    if self.tokentype_embeddings is not None:
        embeddings += self.tokentype_embeddings(tokentype_ids)
    
    # 4. Dropout
    embeddings = self.embedding_dropout(embeddings)
    
    return embeddings
```

**保存的数据：**
- `input_ids`: 输入 token IDs
- `position_ids`: 位置 IDs
- Embedding 权重矩阵引用

**Backward 计算：**
- `word_embeddings.weight.grad[input_ids]`: 只更新使用到的 token 的梯度
- `position_embeddings.weight.grad[position_ids]`: 位置嵌入梯度
- Dropout mask 用于反向传播

---

## 2. Transformer 核心模块

### 2.1 TransformerBlock (`transformer/transformer_block.py`)

**Forward 流程：**
```python
def forward(self, hidden_states, attention_mask, ...):
    # 遍历所有层
    for layer in self.layers:
        hidden_states, context = layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            ...
        )
    
    # Final LayerNorm
    if self.final_layernorm:
        hidden_states = self.final_layernorm(hidden_states)
    
    return hidden_states
```

**保存的数据（每层）：**
- 如果 `recompute_granularity='full'`: 只保存输入 `hidden_states`
- 如果 `recompute_granularity='selective'`: 保存指定模块的输出
- 否则：保存所有中间激活

---

### 2.2 TransformerLayer (`transformer/transformer_layer.py`)

**Forward 流程：**
```python
def forward(self, hidden_states, attention_mask, ...):
    # 1. Attention 部分
    hidden_states, context = self._forward_attention(hidden_states, ...)
    
    # 2. MLP 部分
    output = self._forward_mlp(hidden_states, ...)
    
    return output, context

def _forward_attention(self, hidden_states, ...):
    residual = hidden_states
    
    # LayerNorm
    layernorm_output = self.input_layernorm(hidden_states)
    
    # Self Attention
    attention_output, attention_bias = self.self_attention(
        layernorm_output, attention_mask, ...
    )
    
    # Dropout + Residual
    with self.bias_dropout_add_exec_handler():
        hidden_states = self.self_attn_bda(
            attention_output, attention_bias, residual
        )
    
    return hidden_states, context

def _forward_mlp(self, hidden_states, ...):
    residual = hidden_states
    
    # LayerNorm
    layernorm_output = self.pre_mlp_layernorm(hidden_states)
    
    # MLP
    mlp_output, mlp_bias = self.mlp(layernorm_output)
    
    # Dropout + Residual
    with self.bias_dropout_add_exec_handler():
        output = self.mlp_bda(mlp_output, mlp_bias, residual)
    
    return output
```

**保存的数据：**
- `hidden_states`: 输入
- `residual`: 残差连接的原始值
- `layernorm_output`: LayerNorm 输出
- `attention_output`: Attention 输出
- `mlp_output`: MLP 输出
- LayerNorm 的 `mean` 和 `variance`

**Backward 计算：**
- Dropout backward (使用保存的 mask)
- LayerNorm backward
- Attention backward
- MLP backward
- Residual connection 梯度累加

---

### 2.3 Attention (`transformer/attention.py`)

**Forward 流程：**
```python
def forward(self, hidden_states, attention_mask, ...):
    # 1. QKV 投影
    query, key, value = self.get_query_key_value_tensors(hidden_states)
    # query: [seq_len, batch, num_heads, head_dim]
    
    # 2. Rotary Position Embedding (如果启用)
    if rotary_pos_emb is not None:
        query = apply_rotary_pos_emb(query, rotary_pos_emb)
        key = apply_rotary_pos_emb(key, rotary_pos_emb)
    
    # 3. Core Attention
    context_layer = self.core_attention(
        query, key, value, attention_mask, ...
    )
    
    # 4. Output Projection
    output, bias = self.linear_proj(context_layer)
    
    return output, bias
```

**保存的数据：**
- `hidden_states`: 输入
- `weight_qkv`: QKV 投影权重
- `query, key, value`: QKV 张量
- `weight_proj`: 输出投影权重

**如果使用 checkpointing:**
- 只保存 `hidden_states` 和权重
- 在 backward 时重新计算 QKV

---

### 2.4 DotProductAttention (`transformer/dot_product_attention.py`)

**Forward 流程：**
```python
def forward(self, query, key, value, attention_mask, ...):
    # 1. 计算 attention scores: Q @ K^T
    attention_scores = torch.matmul(query, key.transpose(-2, -1))
    # 形状: [batch, num_heads, seq_len, seq_len]
    
    # 2. 缩放
    attention_scores = attention_scores * self.softmax_scale
    
    # 3. Mask + Softmax
    attention_probs = self.scale_mask_softmax(
        attention_scores, attention_mask
    )
    # 保存: attention_probs
    
    # 4. Dropout
    with tensor_parallel.get_cuda_rng_tracker().fork():
        attention_probs = self.attention_dropout(attention_probs)
    # 保存: dropout mask (隐式)
    
    # 5. 计算 context: attention_probs @ V
    context_layer = torch.matmul(attention_probs, value)
    
    return context_layer
```

**保存的数据（标准实现）：**
```python
# Forward 中需要保存:
ctx.save_for_backward(
    query,              # [sq, b, np, hn]
    key,                # [sk, b, np, hn]
    value,              # [sk, b, np, hn]
    attention_scores,   # [b, np, sq, sk] - 巨大！
    attention_probs,    # [b, np, sq, sk] - 巨大！
)
```

**内存占用（标准实现）：**
- Attention scores: `4 * batch * num_heads * seq_len^2` bytes (fp32)
- 对于 seq_len=2048, batch=8, num_heads=32: ~4GB

**使用 Flash Attention 时：**
```python
# 只保存:
ctx.save_for_backward(
    query, key, value,
    output,
    softmax_lse,  # log-sum-exp，仅 [b, np, sq] 大小
    rng_state,    # 随机数状态
)
# 不保存 attention_scores 和 attention_probs！
```

**Backward 计算：**
```python
def backward(ctx, grad_output):
    query, key, value, attn_probs = ctx.saved_tensors
    
    # 1. Context backward: ∂L/∂attn_probs = grad_output @ V^T
    grad_attn_probs = torch.matmul(grad_output, value.transpose(-2, -1))
    
    # 2. Value backward: ∂L/∂V = attn_probs^T @ grad_output
    grad_value = torch.matmul(attn_probs.transpose(-2, -1), grad_output)
    
    # 3. Dropout backward
    grad_attn_probs = dropout_backward(grad_attn_probs, dropout_mask)
    
    # 4. Softmax backward (最复杂)
    grad_scores = softmax_backward(grad_attn_probs, attn_probs)
    # grad_scores = attn_probs * (grad_attn_probs - 
    #     (grad_attn_probs * attn_probs).sum(dim=-1, keepdim=True))
    
    # 5. Scaled dot-product backward
    grad_scores = grad_scores * softmax_scale
    
    # 6. Query backward: ∂L/∂Q = grad_scores @ K
    grad_query = torch.matmul(grad_scores, key)
    
    # 7. Key backward: ∂L/∂K = grad_scores^T @ Q
    grad_key = torch.matmul(grad_scores.transpose(-2, -1), query)
    
    return grad_query, grad_key, grad_value
```

---

### 2.5 MLP (`transformer/mlp.py`)

**Forward 流程：**
```python
def forward(self, hidden_states, per_token_scale=None):
    # 1. First Linear Layer (FC1)
    intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)
    # 形状: [seq_len, batch, 4*hidden_size]
    
    # 2. Activation Function (with bias fusion)
    if self.config.bias_activation_fusion:
        if self.activation_func == F.gelu:
            intermediate_parallel = bias_gelu_impl(
                intermediate_parallel, bias_parallel
            )
        elif self.activation_func == F.silu:
            intermediate_parallel = bias_swiglu_impl(
                intermediate_parallel, bias_parallel
            )
    else:
        intermediate_parallel = self.activation_func(
            intermediate_parallel + bias_parallel
        )
    
    # 3. Second Linear Layer (FC2)
    output, output_bias = self.linear_fc2(intermediate_parallel)
    
    return output, output_bias
```

**保存的数据：**
- `hidden_states`: 输入 [s, b, h]
- `weight_fc1`: FC1 权重 [4h, h]
- `bias_fc1`: FC1 偏置 [4h]
- `fc1_output`: FC1 输出（激活前）[s, b, 4h]
- `weight_fc2`: FC2 权重 [h, 4h]

**Backward 计算：**
```python
def backward(grad_output):
    # 从 FC2 开始
    # 1. FC2 输入梯度
    grad_fc2_input = grad_output @ weight_fc2  # [s,b,4h]
    
    # 2. FC2 权重梯度
    weight_fc2.grad += activated_output.T @ grad_output
    
    # 3. Activation 梯度 (如 GeLU)
    grad_fc1_output = activation_backward(grad_fc2_input, fc1_output)
    
    # 4. FC1 输入梯度
    grad_input = grad_fc1_output @ weight_fc1  # [s,b,h]
    
    # 5. FC1 权重梯度
    weight_fc1.grad += hidden_states.T @ grad_fc1_output
    
    return grad_input
```

---

## 3. Tensor Parallel 操作

### 3.1 LinearWithGradAccumulationAndAsyncCommunication (`tensor_parallel/layers.py`)

这是 Megatron 中最核心的 Tensor Parallel 线性层实现。

**Forward 保存：**
```python
@staticmethod
def forward(ctx, input, weight, bias, gradient_accumulation_fusion, 
            allreduce_dgrad, sequence_parallel, ...):
    # 保存输入和权重
    ctx.save_for_backward(input, weight)
    
    # 保存主梯度缓冲区引用
    if gradient_accumulation_fusion:
        ctx.main_grad = weight.main_grad
    else:
        ctx.main_grad = None
    
    # 保存配置
    ctx.use_bias = (bias is not None)
    ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
    ctx.allreduce_dgrad = allreduce_dgrad
    ctx.sequence_parallel = sequence_parallel
    ctx.tp_group = tp_group
    
    # Sequence Parallel: all-gather 输入
    if sequence_parallel:
        total_input = all_gather(input, group=tp_group)
    else:
        total_input = input
    
    # 矩阵乘法
    output = torch.matmul(total_input, weight.t())
    
    if bias is not None:
        output = output + bias
    
    return output
```

**Backward 计算：**
```python
@staticmethod
def backward(ctx, grad_output):
    input, weight = ctx.saved_tensors
    main_grad = ctx.main_grad
    
    # ========== 1. 计算输入梯度 (dL/dx) ==========
    grad_input = grad_output.matmul(weight)  # [s,b,h] @ [h,4h] = [s,b,4h]
    
    # Sequence Parallel: all-gather 输入用于权重梯度计算
    if ctx.sequence_parallel:
        total_input = all_gather(input, group=ctx.tp_group)
    else:
        total_input = input
    
    # ========== 2. 计算权重梯度 (dL/dW) ==========
    if ctx.gradient_accumulation_fusion:
        # 使用融合 CUDA kernel 直接累加到 main_grad
        if weight.main_grad.dtype == torch.float32:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                total_input, grad_output, weight.main_grad
            )
        else:
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                total_input, grad_output, weight.main_grad
            )
        # 等价于: weight.main_grad += total_input.T @ grad_output
    else:
        # 标准方式
        grad_weight = total_input.view(-1, total_input.shape[-1]).t() \
                      @ grad_output.view(-1, grad_output.shape[-1])
        weight.grad = grad_weight
    
    # ========== 3. 偏置梯度 (dL/db) ==========
    if ctx.use_bias:
        grad_bias = grad_output.sum(dim=(0, 1))  # 沿 seq 和 batch 维度求和
    else:
        grad_bias = None
    
    # ========== 4. Tensor Parallel 通信 ==========
    if ctx.allreduce_dgrad:
        # All-reduce 输入梯度（异步）
        handle = torch.distributed.all_reduce(
            grad_input, group=ctx.tp_group, async_op=True
        )
    
    if ctx.sequence_parallel:
        # Reduce-scatter 输入梯度
        sub_grad_input = torch.empty(...)
        handle = reduce_scatter(
            sub_grad_input, grad_input, group=ctx.tp_group, async_op=True
        )
        grad_input = sub_grad_input
    
    # 等待通信完成
    if handle is not None:
        handle.wait()
    
    return grad_input, grad_weight, grad_bias, None, None, None, ...
```

**关键点：**
1. **梯度累积融合**: 直接累加到 `main_grad`，避免额外内存分配
2. **异步通信**: all-reduce 和 reduce-scatter 使用 `async_op=True`
3. **通信与计算重叠**: 依赖 `CUDA_DEVICE_MAX_CONNECTIONS=1`

---

### 3.2 Tensor Parallel 映射操作 (`tensor_parallel/mappings.py`)

这些是用于在 Tensor Parallel 中分发和收集数据的自定义 autograd 函数。

#### 3.2.1 `_CopyToModelParallelRegion`

**Forward:**
```python
@staticmethod
def forward(ctx, input_, group):
    ctx.group = group
    return input_  # 直接返回，无操作
```

**Backward:**
```python
@staticmethod
def backward(ctx, grad_output):
    # All-reduce 梯度
    return _reduce(grad_output, ctx.group), None
```

**用途**: 在 forward 时无操作，但在 backward 时 all-reduce 梯度。

---

#### 3.2.2 `_ReduceFromModelParallelRegion`

**Forward:**
```python
@staticmethod
def forward(ctx, input_, group):
    # All-reduce 输入
    return _reduce(input_, group)
```

**Backward:**
```python
@staticmethod
def backward(ctx, grad_output):
    return grad_output, None  # 直接返回，无操作
```

**用途**: 在 forward 时 all-reduce，backward 时无操作。

---

#### 3.2.3 `_ScatterToSequenceParallelRegion`

**Forward:**
```python
@staticmethod
def forward(ctx, input_, group):
    ctx.group = group
    return _split_along_first_dim(input_, group)  # Scatter
```

**Backward:**
```python
@staticmethod
def backward(ctx, grad_output):
    return _gather_along_first_dim(grad_output, ctx.group), None  # All-gather
```

**用途**: Sequence Parallel 的分发和收集。

---

### 3.3 VocabParallelEmbedding (`tensor_parallel/layers.py`)

**Forward:**
```python
def forward(self, input_):
    # Mask 输入（只保留当前 rank 负责的 vocab range）
    input_mask = (input_ < self.vocab_start_index) | \
                 (input_ >= self.vocab_end_index)
    masked_input = input_.clone() - self.vocab_start_index
    masked_input[input_mask] = 0
    
    # Embedding lookup
    output_parallel = F.embedding(masked_input, self.weight, ...)
    
    # Mask 输出
    output_parallel[input_mask, :] = 0.0
    
    # All-reduce across TP ranks
    if self.tensor_model_parallel_size > 1:
        output = reduce_from_tensor_model_parallel_region(output_parallel)
    else:
        output = output_parallel
    
    return output
```

**Backward:**
- PyTorch 的 `F.embedding` 自动处理
- 只更新使用到的 token 的梯度
- Gradient 自动 all-reduce（通过 `reduce_from_tensor_model_parallel_region`）

---

## 4. 融合算子 (Fusions)

### 4.1 BiasGeLU (`fusions/fused_bias_gelu.py`)

**Forward:**
```python
class GeLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        # 融合计算: gelu(input + bias)
        return bias_gelu(bias, input)
```

**Backward:**
```python
    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        # 融合反向: gelu'(input + bias) * grad_output
        tmp = bias_gelu_back(grad_output, bias, input)
        return tmp, tmp  # input 和 bias 的梯度
```

**保存的数据:**
- `input`: 输入张量
- `bias`: 偏置张量

**融合的好处:**
- 减少内存访问
- 避免保存 `input + bias` 的中间结果

---

### 4.2 BiasSwiGLU (`fusions/fused_bias_swiglu.py`)

**Forward:**
```python
@staticmethod
def forward(ctx, input, bias, weights=None, fp8_input_store=False):
    # SwiGLU: swish(x) * x_gate
    # 输入是 [s, b, 2*ffn_hidden_size]
    
    if fp8_input_store:
        input_for_backward = input
    else:
        input_for_backward = input + bias
    
    ctx.save_for_backward(input_for_backward, bias)
    
    # 融合计算 SwiGLU
    return bias_swiglu(input, bias)
```

**Backward:**
```python
@staticmethod
def backward(ctx, grad_output):
    input, bias = ctx.saved_tensors
    
    # SwiGLU backward
    grad_input = swiglu_back(grad_output, input)
    grad_bias = grad_input.sum(dim=(0, 1))
    
    return grad_input, grad_bias
```

---

### 4.3 FusedScaleMaskSoftmax (`fusions/fused_softmax.py`)

**Forward:**
```python
class FusedScaleMaskSoftmax(torch.nn.Module):
    def forward(self, input, mask):
        # 1. Scale
        scaled_input = input * self.scale
        
        # 2. Mask
        if mask is not None:
            scaled_masked_input = scaled_input.masked_fill(mask, -10000.0)
        
        # 3. Softmax
        probs = F.softmax(scaled_masked_input, dim=-1)
        
        return probs
```

**Backward:**
- PyTorch 自动处理 softmax backward
- 公式: `grad_input = probs * (grad_output - (grad_output * probs).sum(dim=-1, keepdim=True))`

---

### 4.4 VocabParallelCrossEntropy (`tensor_parallel/cross_entropy.py`)

**Forward:**
```python
class _VocabParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, label_smoothing):
        # vocab_parallel_logits: [s, b, vocab_size/tp]
        # target: [s, b]
        
        # 1. 计算 softmax
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max, op=torch.distributed.ReduceOp.MAX, group=tp_group
        )
        
        logits = vocab_parallel_logits - logits_max.unsqueeze(dim=-1)
        exp_logits = logits.exp()
        sum_exp_logits = exp_logits.sum(dim=-1)
        torch.distributed.all_reduce(sum_exp_logits, group=tp_group)
        
        # 2. 创建 target mask (只有当前 rank 负责的 token)
        target_mask = (target >= partition_vocab_start_index) & \
                      (target < partition_vocab_end_index)
        masked_target = target - partition_vocab_start_index
        masked_target[~target_mask] = 0
        
        # 3. 计算 loss
        predicted_logits = logits.gather(dim=-1, index=masked_target.unsqueeze(-1))
        predicted_logits = predicted_logits.squeeze(-1)
        predicted_logits[~target_mask] = 0.0
        
        # All-reduce predicted_logits
        torch.distributed.all_reduce(predicted_logits, group=tp_group)
        
        # Loss = -log(softmax(logits)[target])
        loss = torch.log(sum_exp_logits) - predicted_logits
        
        # Label smoothing (如果启用)
        if label_smoothing > 0:
            ...
        
        # 保存用于 backward
        ctx.save_for_backward(exp_logits, target_mask, masked_target)
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_output):
        exp_logits, target_mask, masked_target = ctx.saved_tensors
        
        # Softmax gradient
        grad_input = exp_logits / sum_exp_logits.unsqueeze(-1)
        
        # 减去 target 位置的梯度
        grad_input.scatter_add_(-1, masked_target.unsqueeze(-1), 
                                -target_mask.unsqueeze(-1).float())
        
        # 乘以上游梯度
        grad_input = grad_input * grad_output.unsqueeze(-1)
        
        return grad_input, None, None
```

**保存的数据:**
- `exp_logits`: exp(logits) [s, b, vocab_size/tp]
- `target_mask`: 标识哪些 target 属于当前 rank [s, b]
- `masked_target`: 本地化的 target indices [s, b]

---

## 5. MoE 模块

### 5.1 Router (`transformer/moe/router.py`)

**Forward:**
```python
def forward(self, hidden_states):
    # hidden_states: [s, b, h]
    
    # 1. Router 线性层
    logits = self.router_layer(hidden_states)
    # logits: [s, b, num_experts]
    
    # 2. TopK selection
    if self.routing_type == 'topk':
        scores, indices = torch.topk(logits, k=self.top_k, dim=-1)
        # scores: [s, b, top_k]
        # indices: [s, b, top_k]
    
    # 3. Softmax (归一化权重)
    weights = F.softmax(scores, dim=-1)
    
    # 4. 计算 load balancing loss (辅助损失)
    if self.training:
        aux_loss = self.compute_aux_loss(logits, indices)
    
    return weights, indices, aux_loss
```

**保存的数据:**
- `hidden_states`: 输入
- `router_layer.weight`: Router 权重
- `logits`: Router logits (用于辅助损失计算)

---

### 5.2 MoELayer (`transformer/moe/moe_layer.py`)

**Forward 流程:**
```python
def forward(self, hidden_states):
    # 1. Router
    routing_weights, routing_indices, router_aux_loss = \
        self.router(hidden_states)
    
    # 2. Permute tokens to experts
    # All-to-All 通信：将 token 发送到对应的 expert
    permuted_local_hidden_states, tokens_per_expert = \
        self.token_dispatcher.dispatch(
            hidden_states, routing_indices, routing_weights
        )
    
    # 3. Expert computation
    expert_output = self.experts(
        permuted_local_hidden_states, tokens_per_expert
    )
    
    # 4. Un-permute tokens from experts
    # All-to-All 通信：将 expert 输出发送回原 rank
    output = self.token_dispatcher.combine(
        expert_output, routing_weights, routing_indices
        )
    
    # 5. 添加辅助损失
    if self.training:
        output = MoEAuxLossAutoScaler.apply(output, router_aux_loss)
    
    return output
```

**保存的数据:**
- Router: `routing_weights`, `routing_indices`
- Expert computation: 每个 expert 的输入和权重
- All-to-All 通信的元数据

---

### 5.3 TEGroupedMLP (MoE Experts) (`transformer/moe/experts.py`)

**Forward:**
```python
def forward(self, permuted_local_hidden_states, tokens_per_expert, ...):
    # 使用 Transformer Engine 的 GroupedLinear
    # 一次性计算所有 local experts
    
    # FC1
    fc1_output = self.linear_fc1(
        permuted_local_hidden_states, tokens_per_expert
    )
    
    # Activation
    if self.config.activation_func == F.gelu:
        intermediate = gelu(fc1_output)
    
    # FC2
    output = self.linear_fc2(intermediate, tokens_per_expert)
    
    return output
```

**Backward:**
```python
def backward_dw(self):
    """延迟的权重梯度计算"""
    self.linear_fc2.backward_dw()
    self.linear_fc1.backward_dw()
```

---

## 6. 总结表格

### 6.1 主要模块的内存占用

| 模块 | Forward 保存的数据 | 内存占用（估算） | 是否支持 Checkpointing |
|------|-------------------|----------------|---------------------|
| **Embedding** | input_ids, position_ids | ~O(seq_len × batch) | ❌ (通常很小) |
| **LayerNorm** | input, mean, variance, weight | ~3 × (seq_len × batch × hidden) | ✅ (selective) |
| **Attention (标准)** | Q, K, V, scores, probs | ~2 × (batch × heads × seq_len²) | ✅ (full/selective) |
| **Attention (Flash)** | Q, K, V, softmax_lse | ~O(batch × heads × seq_len) | ✅ |
| **MLP** | input, fc1_output, weights | ~(seq_len × batch × 4×hidden) | ✅ (full/selective) |
| **Router (MoE)** | logits, indices, weights | ~O(seq_len × batch × num_experts) | ✅ |
| **Cross Entropy** | exp_logits, target_mask | ~(seq_len × batch × vocab_size/tp) | ❌ |

**注释:**
- seq_len=2048, batch=8, hidden=12288 (GPT-3 175B)
- Attention scores 内存 = 4 × 8 × 96 × 2048² ≈ 12GB (标准实现)
- Flash Attention 节省 ~99% 的 attention 内存

---

### 6.2 Backward 计算的梯度类型

| 梯度类型 | 存储位置 | 累积方式 | 通信方式 |
|---------|---------|---------|---------|
| **参数梯度 (Weight)** | `param.main_grad` → grad_buffer | 自动累加 (`+=`) | All-reduce (DP) 或 Reduce-scatter (DistOpt) |
| **输入梯度 (Input)** | 临时内存 | 每次计算覆盖 | 传递给前一层 |
| **激活梯度 (Activation)** | 临时内存 | 按需计算 | 无（局部计算） |
| **Embedding 梯度** | `embedding.weight.grad[indices]` | 稀疏累加 | All-reduce (PP stages) |
| **LayerNorm 梯度** | `layernorm.weight.grad` | 累加 | All-reduce (SP) |

---

### 6.3 通信操作汇总

| 操作 | 发生时机 | 数据量 | 通信类型 | 所属并行 |
|------|---------|-------|---------|---------|
| **All-reduce (grad)** | Backward 最后一个 microbatch | 模型参数大小 | 同步/异步 | Data Parallel |
| **Reduce-scatter (grad)** | Backward 每个 microbatch | 模型参数大小 / dp_size | 异步 | Distributed Optimizer |
| **All-gather (param)** | Forward 每层开始 | 分片大小 × dp_size | 异步 | Distributed Optimizer |
| **All-reduce (embedding)** | Backward 结束 | Embedding 大小 | 同步 | Pipeline Parallel |
| **Send/Recv (activation)** | Forward/Backward | seq × batch × hidden | P2P | Pipeline Parallel |
| **All-reduce (input grad)** | Backward 每层 | seq × batch × hidden | 异步 | Tensor Parallel |
| **All-gather (input)** | Forward (SP) | seq × batch × hidden | 异步 | Sequence Parallel |
| **Reduce-scatter (grad)** | Backward (SP) | seq × batch × hidden | 异步 | Sequence Parallel |
| **All-to-All (MoE)** | Forward/Backward | 取决于 token 分布 | 同步 | Expert Parallel |

---

### 6.4 优化技术对比

| 技术 | 内存节省 | 计算开销 | 通信开销 | 适用场景 |
|------|---------|---------|---------|---------|
| **Activation Checkpointing (Full)** | ~70% | +33% | 无 | 所有大模型 |
| **Activation Checkpointing (Selective)** | ~40% | +15% | 无 | 平衡内存和速度 |
| **Flash Attention** | ~99% (attention) | -10% (更快!) | 无 | Attention-heavy 模型 |
| **Gradient Accumulation Fusion** | 轻微 | -5% (减少) | 无 | 所有模型 |
| **Distributed Optimizer** | 50% (optimizer states) | +5% | +通信 | 超大模型 |
| **Sequence Parallel** | ~1/sp_size | 轻微 | +通信 | 长序列 |
| **Tensor Parallel** | ~1/tp_size | 轻微 | +通信 (大) | 超大模型 |

---

## 附录：关键代码位置

### A.1 核心 autograd 函数

| 函数 | 文件 | 行数 | 用途 |
|------|-----|-----|------|
| `LinearWithGradAccumulationAndAsyncCommunication` | `tensor_parallel/layers.py` | 435-580 | TP 线性层 |
| `_VocabParallelCrossEntropy` | `tensor_parallel/cross_entropy.py` | 70-250 | TP Cross Entropy |
| `GeLUFunction` | `fusions/fused_bias_gelu.py` | 36-55 | Bias+GeLU 融合 |
| `SwiGLUFunction` | `fusions/fused_bias_swiglu.py` | 105-180 | Bias+SwiGLU 融合 |
| `CheckpointFunction` | `tensor_parallel/random.py` | 407-475 | Activation Checkpointing |
| `MoEAuxLossAutoScaler` | `transformer/multi_token_prediction.py` | 349-393 | MoE 辅助损失缩放 |

### A.2 主要模块

| 模块 | 文件 | Forward | Backward |
|------|-----|---------|----------|
| **GPTModel** | `models/gpt/gpt_model.py` | ✅ | autograd |
| **TransformerBlock** | `transformer/transformer_block.py` | ✅ | autograd |
| **TransformerLayer** | `transformer/transformer_layer.py` | ✅ | autograd |
| **Attention** | `transformer/attention.py` | ✅ | autograd |
| **DotProductAttention** | `transformer/dot_product_attention.py` | ✅ | autograd |
| **MLP** | `transformer/mlp.py` | ✅ | autograd |
| **MoELayer** | `transformer/moe/moe_layer.py` | ✅ | autograd |
| **Router** | `transformer/moe/router.py` | ✅ | autograd |

---

## 结论

Megatron-LM 的 forward/backward 实现具有以下特点：

1. **高度模块化**: 每个组件都是独立的 `torch.nn.Module` 或 `torch.autograd.Function`
2. **内存优化**: 大量使用 activation checkpointing 和融合算子
3. **通信优化**: 异步通信、通信与计算重叠
4. **灵活性**: 支持多种并行策略的组合（DP+TP+PP+SP+EP）
5. **可扩展性**: 新增自定义层只需实现 forward 和 backward

**关键设计模式:**
- 使用 `ctx.save_for_backward()` 保存必要的中间变量
- 使用融合 CUDA kernel 减少内存访问
- 使用异步通信隐藏通信延迟
- 使用梯度累积减少通信次数

