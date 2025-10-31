# EC-CHECK 测试目录

本目录包含 EC-CHECK (Erasure Coding Checkpoint) 功能的所有测试相关文件。

## 📁 目录结构

```
eccheck/
├── README.md              # 本文件
├── configs/               # 配置文件目录
│   ├── eccheck_2x2.json                    # 简单模式 2+2 配置
│   ├── eccheck_2x2_shared_parity.json      # 共享 parity 模型 2+2 配置（推荐）
│   ├── eccheck_2x2_advanced.json           # 高级模式 2+2 配置
│   ├── eccheck_4rank.json                  # 4 rank 配置示例
│   ├── eccheck_advanced.json               # 高级模式示例
│   └── eccheck_heterogeneous.json          # 异构 pipeline 配置示例
├── scripts/              # 测试脚本目录
│   ├── test_eccheck.sh                     # 主测试脚本
│   └── verify_eccheck_config.py            # 配置文件验证脚本
└── docs/                 # 文档目录
    ├── README_ECCHECK.md                    # EC-CHECK 总体介绍
    ├── ECCHECK_CONFIG_EXPLAINED.md          # 配置文件详解
    ├── ECCHECK_PIPELINE_CONFIG.md           # Pipeline 配置规范
    ├── IMPLEMENTATION_COMPLETE.md            # 实现完成总结
    ├── IMPLEMENTATION_STATUS.md              # 实现状态
    ├── PARITY_SHARED_MODEL.md               # 共享 parity 模型说明
    ├── TEST_2X2_CONFIG.md                   # 2+2 配置测试指南
    ├── TEST_CHECKLIST.md                     # 测试检查清单
    ├── QUICK_TEST_GUIDE.md                  # 快速测试指南
    └── ... (其他实现相关文档)
```

## 🚀 快速开始

### 1. 验证配置文件

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2/eccheck
python3 scripts/verify_eccheck_config.py configs/eccheck_2x2_shared_parity.json
```

### 2. 运行测试

**使用默认配置（2+2 共享 parity）**:
```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck.sh
```

**使用自定义配置文件**:
```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck/configs/eccheck_2x2_advanced.json
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck.sh
```

### 3. 使用 4 rank 配置

```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck/configs/eccheck_4rank.json
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck.sh
```

## 📝 配置文件说明

### 简单模式（推荐初学者）

`configs/eccheck_2x2_shared_parity.json` - 最简单的配置，使用默认行为：
- 2 个 column（column 0 和 column 1）
- 自动计算 paired rank（使用 `-1` 占位符）
- Column 0: 初始化共享 parity
- Column 1: 增量更新共享 parity

### 高级模式

`configs/eccheck_2x2_advanced.json` - 显式配置每个步骤：
- 按 rank 配置
- 显式指定 pipeline 步骤
- 支持 post-xor 步骤

### 异构 Pipeline

`configs/eccheck_heterogeneous.json` - 不同 column 使用不同的处理流程：
- Column 0: encode → send → xor (with zero parity)
- Column 1: encode → send → recv → xor (incremental)
- Post-XOR: sync → send parity → recv data

## 📖 文档索引

- **入门**: `docs/README_ECCHECK.md` - 总体介绍和快速开始
- **配置详解**: `docs/ECCHECK_CONFIG_EXPLAINED.md` - 配置文件格式详解
- **Pipeline 配置**: `docs/ECCHECK_PIPELINE_CONFIG.md` - 高级 pipeline 配置规范
- **共享 Parity 模型**: `docs/PARITY_SHARED_MODEL.md` - 共享 parity buffer 模型说明
- **测试指南**: `docs/QUICK_TEST_GUIDE.md` - 快速测试指南
- **实现状态**: `docs/IMPLEMENTATION_COMPLETE.md` - 完成功能清单

## 🔧 重新编译 C++ 扩展

如果修改了 C++ 代码，需要重新编译：

```bash
cd /workspace/Megatron-LM/megatron/core/dist_checkpointing/strategies
bash build_clean.sh
```

## 📊 支持的配置

- ✅ **2 rank 配置** (rank 0 ↔ rank 1)
- ✅ **4 rank 配置** (rank 0↔1, rank 2↔3)
- ✅ **N rank 配置** (自动配对)
- ✅ **简单模式** (向后兼容)
- ✅ **高级模式** (按 rank 配置)
- ✅ **异构 Pipeline** (不同 column 不同流程)

## 🐛 故障排查

1. **配置路径错误**: 确保 `ECCHECK_CONFIG_PATH` 指向正确的配置文件
2. **C++ 模块未编译**: 运行 `build_clean.sh` 重新编译
3. **NCCL 错误**: 检查 `nccl.log` 文件
4. **缓冲区超时**: 检查日志中的超时错误信息

更多故障排查信息，请参考 `docs/TEST_CHECKLIST.md`。

