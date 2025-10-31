# EC-CHECK 配置文件索引

## 📋 配置文件列表

### 简单模式（推荐初学者）

| 文件名 | 描述 | 使用场景 |
|--------|------|----------|
| `configs/eccheck_2x2_shared_parity.json` | 2+2 共享 parity（默认推荐） | 最简单的配置，使用共享 parity buffer 模型 |
| `configs/eccheck_2x2.json` | 2+2 基础配置 | 向后兼容的简单配置 |

### 高级模式

| 文件名 | 描述 | 使用场景 |
|--------|------|----------|
| `configs/eccheck_2x2_advanced.json` | 2+2 高级模式 | 显式配置 pipeline 步骤 |
| `configs/eccheck_advanced.json` | 高级模式示例 | 通用高级配置示例 |

### 多 Rank 配置

| 文件名 | 描述 | 使用场景 |
|--------|------|----------|
| `configs/eccheck_4rank.json` | 4 rank 配置 | 4 个 rank 的配置示例 |

### 异构 Pipeline

| 文件名 | 描述 | 使用场景 |
|--------|------|----------|
| `configs/eccheck_heterogeneous.json` | 异构 pipeline | 不同 column 使用不同处理流程 |

## 🚀 快速使用

### 使用默认配置运行测试

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck.sh
```

### 使用特定配置文件

```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck/configs/eccheck_2x2_shared_parity.json
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck.sh
```

### 验证配置文件

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2/eccheck
python3 scripts/verify_eccheck_config.py configs/eccheck_2x2_shared_parity.json
```

## 📖 配置文件详细说明

请参考 `docs/ECCHECK_CONFIG_EXPLAINED.md` 查看配置文件的详细格式说明。

