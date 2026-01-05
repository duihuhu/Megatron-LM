# Gemini Replicas - Multi-Replica Checkpointing

Gemini Replicas is an extension of Gemini checkpointing that supports multiple replicas (default: 3) with round-robin placement strategy for enhanced fault tolerance.

## Overview

Unlike standard Gemini which creates 2 replicas (local + 1 remote), Gemini Replicas supports configurable number of replicas with round-robin placement:

### Round-Robin Placement Strategy

For 3 replicas and 4 ranks:
- Rank 0 data → stored on ranks [0, 1, 2]
- Rank 1 data → stored on ranks [1, 2, 3]
- Rank 2 data → stored on ranks [2, 3, 0]
- Rank 3 data → stored on ranks [3, 0, 1]

This provides better fault tolerance as each rank's data is distributed across multiple nodes.

## Features

- **Configurable Replicas**: Support 2-N replicas (default: 3)
- **Round-Robin Placement**: Automatic replica distribution across ranks
- **ASIO-Based Communication**: Efficient multi-target broadcast using Boost.ASIO
- **Zero-Copy Transfer**: Direct memory access for efficient data transfer
- **Optimized Mode**: Eliminates torch.save serialization overhead

## Usage

### 1. Build the C++ Module

```bash
cd megatron/core/dist_checkpointing/strategies/
bash build_gemini_replicas.sh
```

### 2. Enable in Training Script

Add the following flags to your training command:

```bash
# Basic usage (3 replicas, default)
--use-gemini-replicas \
--use-gemini-replicas-optimized

# Custom number of replicas
--use-gemini-replicas \
--use-gemini-replicas-optimized \
--gemini-replicas-num 4
```

### 3. Environment Variables (Optional)

```bash
# Set custom IP address (auto-detected by default)
export GEMINI_REPLICAS_BASE_IP=192.168.1.100

# Set custom base port (default: MASTER_PORT + 30000)
export GEMINI_REPLICAS_BASE_PORT=36000
```

## Example

```bash
#!/bin/bash

# 4-node training with 3 replicas per checkpoint
torchrun --nproc_per_node=1 --nnodes=4 \
    pretrain_gpt.py \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 4 \
    --use-gemini-replicas \
    --use-gemini-replicas-optimized \
    --gemini-replicas-num 3 \
    --save /path/to/checkpoints \
    --save-interval 1000 \
    ... # other training args
```

## Architecture

### Components

1. **gemini_replicas_manager.py**: Python manager for replica coordination
2. **gemini_replicas_native.cpp**: C++ module with ASIO for network communication
3. **filesystem_async.py**: Integration with checkpoint saving pipeline
4. **torch.py**: Strategy integration and data preparation

### Data Flow

1. **Preparation Phase** (`_prepare_gemini_replicas_data`):
   - Decompose state_dict into tensors and metadata
   - Allocate continuous CPU buffer
   - Calculate target ranks using round-robin

2. **Transfer Phase** (`_gemini_replicas_preload_to_continuous_buffer`):
   - GPU→CPU: Copy tensors to continuous buffer
   - Broadcast: Send buffer to (num_replicas - 1) target ranks
   - Use ASIO for efficient multi-target communication

3. **Save Phase**:
   - Each rank saves local copy
   - Receiving ranks save remote replicas

## Configuration

### Parameters

- `--use-gemini-replicas`: Enable Gemini Replicas checkpointing
- `--use-gemini-replicas-optimized`: Enable optimized mode (no serialization)
- `--gemini-replicas-num N`: Number of replicas (default: 3, must be ≤ world_size)

### Requirements

- Python 3.x with PyTorch
- pybind11
- Boost libraries (for ASIO)
- C++17 compiler

## Performance

### Benefits

- **Better Fault Tolerance**: Multiple replicas provide redundancy
- **Efficient Communication**: ASIO-based multi-target broadcast
- **Zero-Copy**: Direct memory access eliminates data copying
- **No Serialization**: Continuous buffer approach avoids pickle overhead

### Trade-offs

- **Storage Overhead**: N replicas require N× storage space
- **Network Bandwidth**: More replicas = more network traffic
- **Recommended**: Use 3 replicas for good balance between fault tolerance and overhead

## Troubleshooting

### Build Errors

```bash
# Install Boost if missing
sudo apt-get install libboost-all-dev  # Ubuntu/Debian
brew install boost  # macOS

# Or set BOOST_ROOT manually
export BOOST_ROOT=/path/to/boost
bash build_gemini_replicas.sh
```

### Runtime Errors

1. **Connection Timeout**: Check firewall and network connectivity
2. **Port Conflicts**: Set custom `GEMINI_REPLICAS_BASE_PORT`
3. **Memory Issues**: Reduce `--gemini-replicas-num` or increase system memory

## Comparison with Other Methods

| Method | Replicas | Placement | Serialization | Network |
|--------|----------|-----------|---------------|---------|
| Standard | 1 | Local only | torch.save | None |
| Gemini | 2 | Paired ranks | Optional | ASIO |
| **Gemini Replicas** | **2-N** | **Round-robin** | **Optional** | **ASIO** |
| EC-CHECK | 1+parity | XOR encoding | None | NCCL |
| ECLATIN | 1+parity | Latin square | None | NCCL |

## References

- Gemini: Replica-level checkpointing with paired ranks
- EC-CHECK: Erasure coding with XOR
- ECLATIN: Erasure coding with Latin square


