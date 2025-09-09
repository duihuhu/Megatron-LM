# planner里面的数据来源详细分析

## 1. 数据流向概览

planner里面的数据最终来源于训练过程中的模型状态，通过以下路径传递：

```
训练模型 → generate_state_dict → state_dict → planner.set_up_planner → planner内部数据
```

## 2. 数据来源的完整调用链

### 2.1 训练循环中的checkpoint保存
```python
# megatron/training/training.py:1871-1881
save_checkpoint(
    iteration,
    model,                    # 训练中的模型
    optimizer,               # 优化器
    opt_param_scheduler,     # 学习率调度器
    num_floating_point_operations_so_far,
    checkpointing_context,
    non_persistent_ckpt=non_persistent_ckpt,
    train_data_iterator=train_data_iterator,
    preprocess_common_state_dict_fn=preprocess_common_state_dict,
)
```

### 2.2 save_checkpoint函数中的数据收集
```python
# megatron/training/checkpointing.py:450-460
state_dict = generate_state_dict(
    args,                    # 训练参数
    model,                   # 训练模型
    optimizer,              # 优化器
    opt_param_scheduler,    # 学习率调度器
    rng_state,              # 随机数状态
    iteration=iteration,     # 当前迭代次数
    optim_sd_kwargs=dict(metadata=sharded_sd_metadata),
    model_sd_kwargs=dict(metadata=sharded_sd_metadata),
    rerun_state=rerun_state,
)
```

### 2.3 generate_state_dict函数中的数据提取
```python
# megatron/training/checkpointing.py:729-776
def generate_state_dict(args, model, optimizer, opt_param_scheduler,
                        rng_state, iteration=None,
                        optim_sd_kwargs=None, model_sd_kwargs=None, rerun_state=None):
    """Generate a state dict from given model, optimizer, scheduler, rng state and others."""
    
    state_dict = {}
    state_dict['args'] = args
    state_dict['checkpoint_version'] = 3.0
    if iteration is not None:
        state_dict['iteration'] = iteration

    # 模型状态数据
    for i in range(len(model)):
        key = "model"
        if len(model) > 1:
            key = f"model{i}"

        if args.ckpt_format == "torch_dist":
            model_sd = model[i].sharded_state_dict(**(model_sd_kwargs or {}))
        else:   # torch, torch_dcp
            model_sd = model[i].state_dict_for_save_checkpoint()

        state_dict[key] = model_sd

    # 优化器状态数据
    if not args.no_save_optim:
        if optimizer is not None and not optimizer.is_stub_optimizer:
            if args.ckpt_format == "torch_dist":
                optimizer_sd = optimizer.sharded_state_dict(state_dict, **(optim_sd_kwargs or {}))
            else:
                optimizer_sd = optimizer.state_dict()
            state_dict['optimizer'] = optimizer_sd

        if opt_param_scheduler is not None:
            state_dict['opt_param_scheduler'] = opt_param_scheduler.state_dict()

    # 其他状态数据
    if rerun_state:
        state_dict['rerun_state_machine'] = rerun_state

    if not args.no_save_rng and rng_state:
        state_dict["rng_state"] = rng_state
    
    return state_dict
```

### 2.4 planner.set_up_planner中的数据设置
```python
# megatron/core/dist_checkpointing/strategies/state_dict_saver.py:108
planner.set_up_planner(state_dict, is_coordinator=dist_wrapper.is_coordinator)
```

## 3. planner内部数据的具体内容

### 3.1 模型数据 (model_sd)
- **模型参数**: 权重矩阵、偏置向量等
- **模型状态**: 批归一化统计量、dropout状态等
- **模型配置**: 层数、隐藏维度、注意力头数等

### 3.2 优化器数据 (optimizer_sd)
- **参数组**: 学习率、权重衰减等超参数
- **优化器状态**: 动量、方差、Adam状态等
- **梯度信息**: 梯度累积、梯度裁剪等

### 3.3 学习率调度器数据 (opt_param_scheduler)
- **当前学习率**: 当前步的学习率值
- **调度器状态**: 步数、预热状态等
- **调度器配置**: 调度策略、参数等

### 3.4 训练状态数据
- **迭代次数**: 当前训练步数
- **随机数状态**: 确保训练可重现
- **浮点运算计数**: 性能统计信息
- **训练参数**: 所有训练配置参数

## 4. 数据在planner中的处理

### 4.1 planner.set_up_planner的作用
```python
planner.set_up_planner(state_dict, is_coordinator=dist_wrapper.is_coordinator)
```
- **数据存储**: 将state_dict存储在planner内部
- **权限设置**: 根据is_coordinator设置权限
- **预处理**: 对数据进行预处理和验证

### 4.2 planner.create_local_plan的作用
```python
local_plan = planner.create_local_plan()
```
- **数据分析**: 分析state_dict中的数据结构
- **计划创建**: 创建本地保存计划
- **WriteItem生成**: 为每个数据项创建WriteItem

### 4.3 planner.create_global_plan的作用
```python
_, global_metadata = planner.create_global_plan(all_local_plans)
```
- **全局协调**: 协调所有rank的本地计划
- **元数据生成**: 生成全局元数据
- **数据分布**: 确定数据在分布式环境中的分布

## 5. 数据来源的具体示例

### 5.1 模型参数数据
```python
# 来自 model[i].sharded_state_dict() 或 model[i].state_dict_for_save_checkpoint()
{
    'model.layers.0.attention.weight': tensor([...]),
    'model.layers.0.attention.bias': tensor([...]),
    'model.layers.0.mlp.weight': tensor([...]),
    'model.layers.0.mlp.bias': tensor([...]),
    # ... 更多层参数
}
```

### 5.2 优化器状态数据
```python
# 来自 optimizer.sharded_state_dict() 或 optimizer.state_dict()
{
    'optimizer': {
        'param_groups': [...],
        'state': {
            'param_0': {'momentum_buffer': tensor([...])},
            'param_1': {'exp_avg': tensor([...]), 'exp_avg_sq': tensor([...])},
            # ... 更多参数状态
        }
    }
}
```

### 5.3 训练状态数据
```python
# 来自 generate_state_dict 函数
{
    'args': Namespace(...),                    # 训练参数
    'iteration': 1000,                         # 当前迭代次数
    'checkpoint_version': 3.0,                 # checkpoint版本
    'rng_state': {...},                        # 随机数状态
    'num_floating_point_operations_so_far': 123456789,  # 浮点运算计数
}
```

## 6. 数据在分布式环境中的处理

### 6.1 数据分片
- **模型参数**: 根据张量并行、流水线并行进行分片
- **优化器状态**: 根据数据并行进行分片
- **元数据**: 在coordinator rank上生成

### 6.2 数据协调
- **本地计划**: 每个rank分析自己的数据
- **全局计划**: coordinator协调所有rank的数据
- **元数据**: 描述全局数据分布和结构

## 7. 总结

planner里面的数据最终来源于：

1. **训练模型**: 通过`model[i].sharded_state_dict()`或`model[i].state_dict_for_save_checkpoint()`获取
2. **优化器**: 通过`optimizer.sharded_state_dict()`或`optimizer.state_dict()`获取
3. **学习率调度器**: 通过`opt_param_scheduler.state_dict()`获取
4. **训练状态**: 包括迭代次数、随机数状态、浮点运算计数等
5. **训练参数**: 所有训练配置和超参数

这些数据通过`generate_state_dict`函数收集，然后传递给`planner.set_up_planner`，最终在planner内部被分析和处理，用于创建checkpoint保存计划。
