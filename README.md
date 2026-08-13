# FRCheck

Fault-tolerant checkpointing for large-model training, built on [Megatron-LM](https://github.com/NVIDIA/Megatron-LM).

This repository keeps Megatron's training stack and replaces the NVIDIA-facing documentation with the checkpoint / recovery work in this tree. The current schemes are:

| Scheme | Flag | Placement | Coding |
| --- | --- | --- | --- |
| FRCheck | `--use-frcheck` | POA stripes over node groups of size `n` | Reed–Solomon `(n, n-2)` |
| ECCHECK | `--use-eccheck` | 4-rank XOR groups | XOR parity |
| ECNaive | `--use-ecnaive` | Round-robin RS groups | ISA-L Reed–Solomon `(k+2, k)` |
| Gemini Replicas | `--use-gemini-replicas` | Round-robin replicas | Replication (`--gemini-replicas-num`) |

At most one of `--use-frcheck`, `--use-ecnaive`, and `--use-gemini-replicas` may be enabled. ECCHECK is selected independently with `--use-eccheck`.

## What FRCheck does

FRCheck precomputes a POA table for each supported node-group size and loads the selected table into memory at initialization. Each row represents one stripe and records the nodes that store its data and parity blocks. Checkpoint blocks are assigned by scanning these rows in a fixed order, ensuring that blocks in the same stripe reside on different nodes while original blocks remain local. FRCheck retains the table and block-layout metadata throughout execution and stores the information needed to reproduce the mapping with the checkpoint, allowing saving and recovery to use identical placement decisions without runtime coordination.

Save and recovery both go through RDMA. A layer is flattened into fixed-size blocks, encoded into stripes, and written as per-rank metadata plus per-stripe shards. Hardware recovery reconstructs a failed rank from surviving source and parity blocks in the same POA group (at most two failed node-slots per group). Optional async parity, layer-exchange encode, GPUDirect RDMA, and overlap of recovery with the first training steps are available on the FRCheck path.

## Layout

```
megatron/training/
  frcheck_legacy.py          FRCheck save / load / recovery
  eccheck_legacy.py          ECCHECK save / load / recovery
  ecnaive_legacy.py          ECNaive save / load / recovery
  gemini_replicas_legacy.py  Gemini replica save / load / recovery
  checkpointing.py           scheme dispatch
  arguments.py               CLI flags

megatron/core/dist_checkpointing/strategies/
  frcheck_manager.py         POA grouping, stripe plans, RDMA buffers
  frcheck_native.cpp         RS encode / decode and RDMA
  poa_n4.txt  poa_n8.txt     offline POA tables
  setup_simple_frcheck.py    FRCheck native build
  setup_simple.py            ECCHECK native build
  setup_simple_ecnaive.py    ECNaive native build
  setup_simple_gemini.py     Gemini native build

pre-tests/gpt2/para_a800/
  {2.7B,7B,10B,14B,20B}-4nodes/   per-scheme launch scripts
  run_10b_save_sweep.sh           save sweep driver
  run_10b_inprocess_sweep.sh      in-process recovery sweep driver
```

## Native modules

Each scheme has a pybind11 extension. Build them in `megatron/core/dist_checkpointing/strategies`:

```bash
cd megatron/core/dist_checkpointing/strategies
python3 setup_simple_frcheck.py build_ext --inplace
bash build_clean.sh            # ECCHECK
bash build_clean_ecnaive.sh    # ECNaive
python3 setup_simple_gemini.py build_ext --inplace
```

Dependencies: PyTorch, pybind11, Boost.Asio, ISA-L, libibverbs / rdmacm, and CUDA for GPUDirect paths.

## Running

The four-node A800 scripts under `pre-tests/gpt2/para_a800/` are the current entry points. On each node:

```bash
# save
./pre-tests/gpt2/para_a800/14B-4nodes/test_frcheck.sh <node_rank> save

# software in-process recovery
./pre-tests/gpt2/para_a800/14B-4nodes/test_frcheck.sh <node_rank> inprocess_sw

# hardware in-process recovery (one or two failures)
./pre-tests/gpt2/para_a800/14B-4nodes/test_frcheck.sh <node_rank> inprocess
./pre-tests/gpt2/para_a800/14B-4nodes/test_frcheck.sh <node_rank> inprocess2
```

The same `save | inprocess_sw | inprocess | inprocess2` modes exist for `test_eccheck.sh`, `test_ecnaive.sh`, `test_gemini_2.sh`, and `test_gemini_3.sh`.

Sweep drivers (default schemes: `gemini2 gemini3 frcheck eccheck ecnaive`):

```bash
MODEL_SIZE=14B ./pre-tests/gpt2/para_a800/run_10b_save_sweep.sh
MODEL_SIZE=14B ./pre-tests/gpt2/para_a800/run_10b_inprocess_sweep.sh
```

Useful environment variables:

| Variable | Meaning |
| --- | --- |
| `FRCHECK_INTERFACE` | RDMA / control-plane NIC |
| `FRCHECK_N` / `--frcheck-n` | POA group size |
| `FRCHECK_TABLE_DIR` / `--frcheck-table-dir` | directory of `poa_n{N}.txt` |
| `FRCHECK_GDR` | `1` to enable GPUDirect RDMA on save |
| `FRCHECK_ASYNC_PARITY` | `1` to send P2 in the background after encode |
| `RDMA_HCA_PROFILE` | `full`, `half`, or `quarter` NIC binding |

## FRCheck flags

```
--use-frcheck
--frcheck-n 4
--frcheck-table-dir megatron/core/dist_checkpointing/strategies
--frcheck-layer-exchange-encode
--frcheck-gdr
--frcheck-async-parity
--use-frcheck-hardware-failure
--frcheck-failed-ranks 0,1,2,3,4,5,6,7
--frcheck-recovery-async-parity
--frcheck-async-recovery-forward
--ft-inprocess-recovery-benchmark
```

`--frcheck-n` must divide the number of nodes. The manager loads `poa_n{n}.txt` (or `--frcheck-table-path`) once, compiles per-stripe descriptors, and reuses them for save and recovery.

