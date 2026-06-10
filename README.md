# CheckCode

CheckCode is a serialization-free fault-tolerant checkpoint system for large-scale LLM training. It encodes model checkpoints with XOR/stripe pipelines and RDMA-friendly block exchange, supporting fast save/load and software or hardware failure recovery.

This repository is the **official implementation for our paper experiments**, built on [NVIDIA Megatron-LM](https://github.com/NVIDIA/Megatron-LM). Training entry points (`pretrain_gpt.py`, distributed parallelism, etc.) follow Megatron; checkpoint encoding/recovery is implemented in the CheckCode modules under `megatron/training/` and `megatron/core/dist_checkpointing/strategies/`.

## Methods

| Role | CLI flag | Notes |
|------|----------|-------|
| **CheckCode (ours)** | `--use-checkcode` | Main method. Alias `--use-eclatin` is deprecated. |
| Baseline: EC-CHECK | `--use-eccheck` | Early XOR-based EC checkpoint |
| Baseline: EC-NAIVE | `--use-ecnaive` | ISA-L Reed–Solomon erasure coding |
| Baseline: Gemini Replicas | `--use-gemini-replicas` | Multi-replica round-robin placement (`--gemini-replicas-num 2` or `3`) |

Paper scripts use the **legacy torch checkpoint path**: `--ckpt-format torch`, typically with `--use-rdma` and `--save-embeddings-separately`.

`--use-checkcode`, `--use-ecnaive`, and `--use-gemini-replicas` are mutually exclusive.

### Failure recovery modes (CheckCode)

| Mode | Script argument | Extra flags |
|------|-----------------|-------------|
| Save | `save` (default) | — |
| Software failure | `software` | `--use-checkcode-software-failure` |
| Hardware failure (single) | `hardware` | `--load` only |
| Two failures | `hardware2` | `--use-checkcode-two-failures` |

## Build

Compile native extensions under `megatron/core/dist_checkpointing/strategies/`:

```bash
# CheckCode (required for paper experiments)
bash build_clean_checkcode.sh          

# Baselines (only if you run those scripts)
bash build_clean.sh                    # EC-CHECK
bash build_clean_ecnaive.sh            # EC-NAIVE
bash build_gemini_replicas.sh          # Gemini Replicas
```

Dependencies: Python 3, PyTorch, pybind11, Boost; `libisal-dev` for EC-NAIVE; optional `libibverbs-dev` / `librdmacm-dev` for RDMA.

## Paper experiments (`pre-tests/*/ali_a800/`)

Scripts under `pre-tests/{gpt2,opt,bert}/ali_a800/` match the **Ali A800 cluster** setup used in the paper. Each script runs **one node**; launch the same script on every node with a different `node_rank`.

### Naming

```
test_<model>_<method>.sh
```

Examples: `test_7B_checkcode.sh`, `test_2_7B_ecnaive.sh`, `test_7B_8nodes_gemini_2_replicas.sh`, `test_20B_checkcode_layerwise.sh`.

| Directory suffix | Model scale |
|------------------|-------------|
| `335M` (under `gpt2/ali_a800/`) | GPT-2 ~345M |
| `1_5B` | 1.5B |
| `2_7B` | 2.7B |
| `7B` | 7B |
| `20B` | 20B |
| `7B_8nodes` / `7B_16nodes` | 7B scaled to 8 / 16 nodes |

### Layout

```
pre-tests/
├── gpt2/ali_a800/     # GPT-2 family (main paper runs)
│   ├── 1.5B/  2.7B/  7B/  20B/
│   ├── 7B-8nodes/     # NNODES=8
│   └── 7B-16nodes/    # NNODES=16
├── opt/ali_a800/      # OPT (same naming)
└── bert/ali_a800/     # BERT (+ prepare_bert_cache.sh)
```

Per model size, run **CheckCode** and the **same baselines** (`eccheck`, `ecnaive`, `gemini_2_replicas`, `gemini_3_replicas`) for comparison.

### Run

```bash
# Syntax
./test_<model>_<method>.sh <node_rank> <gpu0> [gpu1 ...] [save|software|hardware|hardware2]

# Example: GPT-2 7B, CheckCode save, 4 nodes × 8 GPUs
./pre-tests/gpt2/ali_a800/7B/test_7B_checkcode.sh <node_rank> 0 1 2 3 4 5 6 7 save

# Software failure recovery
./pre-tests/gpt2/ali_a800/7B/test_7B_checkcode.sh <node_rank> 0 1 2 3 4 5 6 7 software

# Print torchrun command without running
PRINT_CMD=1 ./pre-tests/gpt2/ali_a800/7B/test_7B_checkcode.sh 0 0
```

Before running, edit script variables as needed: `MASTER_ADDR`, `MASTER_PORT`, `CHECKPOINT_PATH`, data/vocab paths. Typical cluster settings: `NETIFACES_INTERFACE=eth0`, per-rank NIC via `ECLATIN_LOCAL_RANK_NIC_*` (CheckCode) or `ECNAIVE_LOCAL_RANK_NIC_*` (EC-NAIVE).

Optional NUMA binding on 8-GPU nodes: `numa_bind_launch.sh` (see `gpt2/ali_a800/7B/`).

### Local smoke tests

Smaller dev scripts (non-paper) live under `pre-tests/gpt2/a800/`, e.g. `test_335M_checkcode.sh`.

## Implementation

| Component | Path |
|-----------|------|
| Legacy save/load | `megatron/training/checkcode_legacy.py` |
| Manager / buffers / RDMA | `megatron/core/dist_checkpointing/strategies/checkcode_manager.py` |
| Native encode/decode | `megatron/core/dist_checkpointing/strategies/checkcode_native.cpp` |
| CLI arguments | `megatron/training/arguments.py` (`--use-checkcode`, …) |

> **Partial rename:** user-facing name is CheckCode; on-disk checkpoints still use `eclatin_main_rank*.pt` and magic `ECLT` for backward compatibility. Environment variables remain `ECLATIN_*`.

## License

See [LICENSE](./LICENSE). Megatron-LM components retain their original license terms.
