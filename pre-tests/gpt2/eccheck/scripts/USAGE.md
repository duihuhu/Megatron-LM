# EC-CHECK 脚本使用说明

## test_eccheck.sh

主测试脚本，用于运行 EC-CHECK 功能测试。

### 使用方法

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash eccheck/scripts/test_eccheck.sh
```

### 环境变量

- `ECCHECK_CONFIG_PATH`: 配置文件路径（可选，默认为 `eccheck/configs/eccheck_2x2_shared_parity.json`）
- `PRINT_CMD`: 设置为非0值将只打印命令而不执行

### 示例

**使用默认配置**:
```bash
bash eccheck/scripts/test_eccheck.sh
```

**使用自定义配置**:
```bash
export ECCHECK_CONFIG_PATH=/workspace/Megatron-LM/pre-tests/gpt2/eccheck/configs/eccheck_4rank.json
bash eccheck/scripts/test_eccheck.sh
```

**仅打印命令**:
```bash
PRINT_CMD=1 bash eccheck/scripts/test_eccheck.sh
```

## verify_eccheck_config.py

配置文件验证脚本，用于检查配置文件格式和内容。

### 使用方法

```bash
cd /workspace/Megatron-LM/pre-tests/gpt2/eccheck
python3 scripts/verify_eccheck_config.py <config_file_path>
```

### 示例

```bash
# 验证默认配置
python3 scripts/verify_eccheck_config.py configs/eccheck_2x2_shared_parity.json

# 验证 4 rank 配置
python3 scripts/verify_eccheck_config.py configs/eccheck_4rank.json
```

### 输出

脚本会显示：
- 持久化设置（persist）
- Column 配置
- send_peer 和 recv_peer 信息（-1 表示自动计算）
- 验证结果

