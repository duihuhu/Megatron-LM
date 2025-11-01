# EC-CHECK Rank 和 Node 映射说明

## 📊 基本概念

### 关键参数

在 `test_eccheck.sh` 中定义的关键参数：

```bash
GPUS_PER_NODE=2      # 每个节点的 GPU 数量
NNODES=1             # 节点总数
NODE_RANK=0          # 当前节点的 rank（0-based）
WORLD_SIZE=2         # 总进程数 = GPUS_PER_NODE × NNODES
```

### torchrun 参数映射

```bash
torchrun \
    --nproc_per_node $GPUS_PER_NODE \   # 每个节点启动的进程数（= GPU数）
    --nnodes $NNODES \                   # 节点总数
    --node_rank $NODE_RANK \            # 当前节点的 rank
    --master_addr $MASTER_ADDR \         # Master 节点地址
    --master_port $MASTER_PORT          # Master 端口
```

## 🔢 Rank 计算规则

### 全局 Rank 计算

```
全局 rank = NODE_RANK × GPUS_PER_NODE + local_rank
```

其中：
- `NODE_RANK`: 节点的 rank（0, 1, 2, ...）
- `GPUS_PER_NODE`: 每个节点的 GPU 数量
- `local_rank`: 节点内的本地 rank（0, 1, 2, ...）

### 示例映射表

#### 示例 1: 单节点，2 GPU（当前测试配置）

| NODE_RANK | GPUS_PER_NODE | local_rank | 全局 rank | 说明 |
|-----------|---------------|------------|-----------|------|
| 0 | 2 | 0 | 0 | Node 0, GPU 0 |
| 0 | 2 | 1 | 1 | Node 0, GPU 1 |

**总 WORLD_SIZE = 2**

#### 示例 2: 2 节点，每个节点 2 GPU

| NODE_RANK | GPUS_PER_NODE | local_rank | 全局 rank | 说明 |
|-----------|---------------|------------|-----------|------|
| 0 | 2 | 0 | 0 | Node 0, GPU 0 |
| 0 | 2 | 1 | 1 | Node 0, GPU 1 |
| 1 | 2 | 0 | 2 | Node 1, GPU 0 |
| 1 | 2 | 1 | 3 | Node 1, GPU 1 |

**总 WORLD_SIZE = 4**

#### 示例 3: 4 节点，每个节点 2 GPU

| NODE_RANK | GPUS_PER_NODE | local_rank | 全局 rank | 说明 |
|-----------|---------------|------------|-----------|------|
| 0 | 2 | 0 | 0 | Node 0, GPU 0 |
| 0 | 2 | 1 | 1 | Node 0, GPU 1 |
| 1 | 2 | 0 | 2 | Node 1, GPU 0 |
| 1 | 2 | 1 | 3 | Node 1, GPU 1 |
| 2 | 2 | 0 | 4 | Node 2, GPU 0 |
| 2 | 2 | 1 | 5 | Node 2, GPU 1 |
| 3 | 2 | 0 | 6 | Node 3, GPU 0 |
| 3 | 2 | 1 | 7 | Node 3, GPU 1 |

**总 WORLD_SIZE = 8**

## 🎯 EC-CHECK 配对规则

### 当前实现中的配对逻辑

在 `_get_paired_rank()` 函数中：

```python
def _get_paired_rank(self, my_rank: int, world_size: int) -> int:
    """
    配对策略:
    - 2 ranks: rank0 ↔ rank1
    - 4 ranks: rank0 ↔ rank2, rank1 ↔ rank3
    - General: rank_i ↔ rank_{i + world_size/2}
    """
    half_size = world_size // 2
    if my_rank < half_size:
        paired_rank = my_rank + half_size
    else:
        paired_rank = my_rank - half_size
    return paired_rank
```

### EC-CHECK 配对映射

#### 2 Ranks (当前测试)
```
Rank 0 ↔ Rank 1
```

#### 4 Ranks
```
Rank 0 ↔ Rank 2
Rank 1 ↔ Rank 3
```

#### 8 Ranks
```
Rank 0 ↔ Rank 4
Rank 1 ↔ Rank 5
Rank 2 ↔ Rank 6
Rank 3 ↔ Rank 7
```

**规则**: `rank_i` 与 `rank_{i + world_size/2}` 配对

## 📝 测试脚本配置说明

### 当前配置（单节点）

```bash
GPUS_PER_NODE=2     # 2 个 GPU
NNODES=1             # 1 个节点
NODE_RANK=0          # 节点 0
WORLD_SIZE=2         # 总 rank 数 = 2 × 1 = 2
```

**torchrun 会启动**:
- 进程 0: 全局 rank 0 (Node 0, local rank 0)
- 进程 1: 全局 rank 1 (Node 0, local rank 1)

**EC-CHECK 配对**:
- Rank 0 ↔ Rank 1

### 修改为 4 Ranks（2 节点）

修改脚本参数：

```bash
GPUS_PER_NODE=2     # 每个节点 2 个 GPU
NNODES=2             # 2 个节点
NODE_RANK=$1         # 从命令行参数获取（0 或 1）
WORLD_SIZE=4         # 总 rank 数 = 2 × 2 = 4
```

**运行方式**:
```bash
# 在节点 0 运行
bash test_eccheck.sh 0

# 在节点 1 运行（另一台机器）
bash test_eccheck.sh 1
```

**torchrun 会启动**:
- 节点 0: rank 0, 1
- 节点 1: rank 2, 3

**EC-CHECK 配对**:
- Rank 0 ↔ Rank 2
- Rank 1 ↔ Rank 3

### 修改为 4 Ranks（单节点，4 GPU）

修改脚本参数：

```bash
GPUS_PER_NODE=4     # 4 个 GPU
NNODES=1             # 1 个节点
NODE_RANK=0          # 节点 0
WORLD_SIZE=4         # 总 rank 数 = 4 × 1 = 4
```

**torchrun 会启动**:
- 进程 0: rank 0
- 进程 1: rank 1
- 进程 2: rank 2
- 进程 3: rank 3

**EC-CHECK 配对**:
- Rank 0 ↔ Rank 2
- Rank 1 ↔ Rank 3

## 🔍 如何验证当前配置

在 Python 代码中打印 rank 信息：

```python
import torch.distributed as dist

if dist.is_initialized():
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    print(f"Global rank: {rank}, World size: {world_size}, Local rank: {local_rank}")
```

## 📋 常见配置示例

### 配置 1: 单节点，2 GPU（当前）
```bash
GPUS_PER_NODE=2
NNODES=1
WORLD_SIZE=2
```
- Rank 0, 1
- EC-CHECK 配对: 0↔1

### 配置 2: 2 节点，每个节点 2 GPU
```bash
GPUS_PER_NODE=2
NNODES=2
WORLD_SIZE=4
```
- Rank 0, 1, 2, 3
- EC-CHECK 配对: 0↔2, 1↔3

### 配置 3: 单节点，4 GPU
```bash
GPUS_PER_NODE=4
NNODES=1
WORLD_SIZE=4
```
- Rank 0, 1, 2, 3
- EC-CHECK 配对: 0↔2, 1↔3

### 配置 4: 4 节点，每个节点 2 GPU
```bash
GPUS_PER_NODE=2
NNODES=4
WORLD_SIZE=8
```
- Rank 0, 1, 2, 3, 4, 5, 6, 7
- EC-CHECK 配对: 0↔4, 1↔5, 2↔6, 3↔7

## ⚠️ 重要提示

1. **WORLD_SIZE 必须为偶数**：EC-CHECK 配对需要偶数个 rank
2. **配置文件中的 rank**：EC-CHECK 配置文件中的 rank 是指**全局 rank**，不是 node rank
3. **torchrun 会自动分配**：`torchrun` 会根据 `--nproc_per_node` 和 `--nnodes` 自动计算并分配全局 rank

## 🔧 修改测试脚本以支持多节点

如果需要测试 4 ranks，可以修改 `test_eccheck.sh`:

```bash
# 修改这些参数
GPUS_PER_NODE=2
NNODES=2              # 改为 2 个节点
NODE_RANK=$1          # 从命令行参数获取

# 或者单节点 4 GPU
GPUS_PER_NODE=4
NNODES=1
NODE_RANK=0
```

然后在每个节点上运行（多节点模式）:
```bash
# 节点 0
bash test_eccheck.sh 0

# 节点 1（另一台机器）
bash test_eccheck.sh 1
```

