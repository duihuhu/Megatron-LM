# 函数调用替换指南

## 需要替换的调用点分析

基于代码分析，以下是所有需要替换的 `FileSystemWriterAsync` 调用点：

### 1. 直接实例化调用（需要替换）

#### 文件：`megatron/core/dist_checkpointing/strategies/torch.py`
**位置**：第721行
```python
# 原始代码
writer = FileSystemWriterAsync(
    checkpoint_dir,
    separation_hint=self.separation_hint,
    thread_count=self.thread_count,
    use_msc=MultiStorageClientFeature.is_enabled(),
)

# 替换为
writer = FileSystemWriterAsyncPipeline(
    checkpoint_dir,
    enable_pipeline=True,  # 启用流水线功能
    num_tensor_groups=4,   # 根据需要调整
    separation_hint=self.separation_hint,
    thread_count=self.thread_count,
    use_msc=MultiStorageClientFeature.is_enabled(),
)
```

### 2. 类型注解（需要更新）

#### 文件：`megatron/core/dist_checkpointing/strategies/state_dict_saver.py`
**位置**：第43行、第49行、第214行、第223行
```python
# 原始代码
storage_writer: 'FileSystemWriterAsync'

# 替换为
storage_writer: 'FileSystemWriterAsyncPipeline'
```

### 3. 导入语句（需要更新）

#### 文件：`megatron/core/dist_checkpointing/strategies/torch.py`
**位置**：第60行
```python
# 原始代码
from .filesystem_async import FileSystemWriterAsync

# 替换为
from .filesystem_async_pipeline_corrected import FileSystemWriterAsyncPipeline as FileSystemWriterAsync
```

#### 文件：`megatron/core/dist_checkpointing/strategies/state_dict_saver.py`
**位置**：第18行
```python
# 原始代码
from .filesystem_async import FileSystemWriterAsync

# 替换为
from .filesystem_async_pipeline_corrected import FileSystemWriterAsyncPipeline as FileSystemWriterAsync
```

### 4. 测试文件（需要更新）

#### 文件：`tests/unit_tests/dist_checkpointing/test_async_save.py`
**位置**：第10行
```python
# 原始代码
from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

# 替换为
from megatron.core.dist_checkpointing.strategies.filesystem_async_pipeline_corrected import FileSystemWriterAsyncPipeline as FileSystemWriterAsync
```

## 推荐的替换策略

### 策略1：最小侵入性替换（推荐）
保持原有的 `FileSystemWriterAsync` 类名，但使用流水线实现：

```python
# 在需要流水线功能的地方
from .filesystem_async_pipeline_corrected import FileSystemWriterAsyncPipeline as FileSystemWriterAsync

# 然后正常使用，但添加流水线参数
writer = FileSystemWriterAsync(
    checkpoint_dir,
    enable_pipeline=True,  # 新增参数
    num_tensor_groups=4,   # 新增参数
    # ... 其他原有参数
)
```

### 策略2：完全替换
直接替换所有 `FileSystemWriterAsync` 为 `FileSystemWriterAsyncPipeline`：

```python
# 更新所有导入
from .filesystem_async_pipeline_corrected import FileSystemWriterAsyncPipeline

# 更新所有实例化
writer = FileSystemWriterAsyncPipeline(
    checkpoint_dir,
    enable_pipeline=True,
    num_tensor_groups=4,
    # ... 其他参数
)
```

## 具体替换步骤

### 步骤1：更新导入语句
```bash
# 查找所有需要更新的文件
grep -r "from.*filesystem_async.*import.*FileSystemWriterAsync" megatron/
grep -r "FileSystemWriterAsync(" megatron/
```

### 步骤2：逐个文件替换
1. **torch.py** - 更新导入和实例化
2. **state_dict_saver.py** - 更新类型注解
3. **test_async_save.py** - 更新测试导入

### 步骤3：添加配置参数
在需要流水线功能的地方添加：
```python
enable_pipeline=True,      # 启用流水线
num_tensor_groups=4,       # tensor组数
```

## 向后兼容性

### 保持兼容的方式
```python
# 在 FileSystemWriterAsyncPipeline 的 __init__ 中
def __init__(self, *args, enable_pipeline=False, num_tensor_groups=2, **kwargs):
    # 如果 enable_pipeline=False，行为与原始 FileSystemWriterAsync 完全一致
    super().__init__(*args, **kwargs)
    self.enable_pipeline = enable_pipeline
    self.num_tensor_groups = num_tensor_groups
```

### 渐进式启用
```python
# 可以通过环境变量控制
import os
enable_pipeline = os.getenv('MEGATRON_PIPELINE_CHECKPOINT', '0') == '1'
num_groups = int(os.getenv('MEGATRON_TENSOR_GROUPS', '4'))

writer = FileSystemWriterAsyncPipeline(
    checkpoint_dir,
    enable_pipeline=enable_pipeline,
    num_tensor_groups=num_groups,
    # ... 其他参数
)
```

## 验证替换

### 测试脚本
```python
# 创建测试脚本验证替换
def test_replacement():
    from megatron.core.dist_checkpointing.strategies.filesystem_async_pipeline_corrected import FileSystemWriterAsyncPipeline
    
    # 测试标准模式
    writer_std = FileSystemWriterAsyncPipeline(
        "/tmp/test",
        enable_pipeline=False
    )
    assert not writer_std.enable_pipeline
    
    # 测试流水线模式
    writer_pipe = FileSystemWriterAsyncPipeline(
        "/tmp/test",
        enable_pipeline=True,
        num_tensor_groups=4
    )
    assert writer_pipe.enable_pipeline
    assert writer_pipe.num_tensor_groups == 4
    
    print("替换验证成功！")
```

## 总结

需要替换的主要调用点：
1. **torch.py** - 1个实例化调用
2. **state_dict_saver.py** - 4个类型注解
3. **test_async_save.py** - 1个导入语句
4. **其他文件** - 根据需要更新

建议使用策略1（最小侵入性），通过别名导入保持兼容性。

