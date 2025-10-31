# EC-CHECK 2+2 Test Configuration

## Overview
This directory contains the configuration and test script for EC-CHECK 2+2 column testing.

## Files
- `eccheck_2x2.json`: EC-CHECK configuration file for 2+2 setup (2 columns per rank)
- `test_eccheck.sh`: Test script to run EC-CHECK with GPT-2 model

## Configuration File: `eccheck_2x2.json`

This configuration defines:
- **2 columns** (column 0 and column 1) per rank
- **Persist flags**: Both `recv` and `parity` results are persisted to memory stores
- **Column settings**:
  - Column 0: GF coefficient 0, sends/recvs to peer rank 1
  - Column 1: GF coefficient 1, sends/recvs to peer rank 1

### Configuration Structure
```json
{
  "persist": {
    "recv": true,
    "parity": true
  },
  "columns": [
    {
      "coefficient": 0,
      "send_peer": 1,
      "recv_peer": 1
    },
    {
      "coefficient": 1,
      "send_peer": 1,
      "recv_peer": 1
    }
  ]
}
```

### Parameters Explanation
- **coefficient**: GF (Galois Field) encoding coefficient for this column (0 or 1 for 2+2 setup)
- **send_peer**: Rank ID to send encoded data to (1 for paired rank in 2-rank setup)
- **recv_peer**: Rank ID to receive encoded data from (1 for paired rank in 2-rank setup)

## Running the Test

### Prerequisites
1. EC-CHECK native module compiled (`eccheck_native*.so`)
2. Configuration file exists at default path or set `ECCHECK_CONFIG_PATH` environment variable
3. GPT-2 model data prepared

### Basic Usage
```bash
cd /workspace/Megatron-LM/pre-tests/gpt2
bash test_eccheck.sh
```

### Expected Behavior
- Each rank will start **8 threads** (4 threads × 2 columns):
  - 2 encoder threads (one per column)
  - 2 send worker threads
  - 2 recv worker threads
  - 2 XOR worker threads
- Each column uses its own NCCL communicator
- Encoded data is sent to paired rank
- Received data is XORed with local encoding, result stored in parity buffer

### Environment Variables
- `ECCHECK_CONFIG_PATH`: Path to EC-CHECK configuration JSON file (default: `./eccheck_2x2.json`)
- `NCCL_DEBUG`: NCCL debug level (set to INFO in test script)

## Troubleshooting

### Configuration Not Found
If you see warnings about configuration file not found, ensure:
1. File exists at `/workspace/Megatron-LM/pre-tests/gpt2/eccheck_2x2.json`
2. Or set `ECCHECK_CONFIG_PATH` environment variable

### Column Count Mismatch
If you see errors about column count mismatch, check:
1. Configuration file has exactly 2 columns in `columns` array
2. All columns have required fields: `coefficient`, `send_peer`, `recv_peer`

### Thread Count
The log should show:
```
EC-CHECK: [Rank X] Started 8 threads (2 encoding + 2 send + 2 recv + 2 XOR)
```

If you see a different count, verify the configuration is loaded correctly.

## Notes
- For 2-rank setup, `send_peer` and `recv_peer` should both be 1 (paired rank)
- GF coefficients should be sequential (0, 1, 2, ...) for proper encoding
- Each column uses independent NCCL communicators and queues

