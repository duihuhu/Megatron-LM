#!/bin/bash
# Run the distributed asynchronous checkpoint test
# Two independent programs simulating distributed training

echo "======================================================================"
echo "Distributed Asynchronous Checkpoint Test (Three Approaches)"
echo "======================================================================"
echo ""
echo "Approach 1: Native torch (torch.to + dist.send)"
echo "Approach 2: CPUMemoryPool + dist.send (shared memory + PyTorch serialization)"
echo "Approach 3: CPUMemoryPool + Raw Bytes (shared memory + original bytes) ⭐ optimal"
echo ""
echo "Starting two processes (rank 0 and rank 1)..."
echo ""

# Set environment variables
export MASTER_ADDR=localhost
export MASTER_PORT=29500
export WORLD_SIZE=2

# Test mode (may be torch, pool, pool-raw, or all)
MODE=${1:-all}

echo "Test mode: $MODE"
echo ""

# Start rank 0
echo "Starting Rank 0..."
python test_async_checkpoint.py --rank 0 --world-size 2 --mode $MODE &
PID0=$!

# Wait briefly to ensure rank 0 starts first
sleep 1

# Start rank 1
echo "Starting Rank 1..."
python test_async_checkpoint.py --rank 1 --world-size 2 --mode $MODE &
PID1=$!

echo ""
echo "Waiting for both processes to finish..."
echo "  Rank 0 PID: $PID0"
echo "  Rank 1 PID: $PID1"

# Wait for both processes
wait $PID0
wait $PID1

echo ""
echo "======================================================================"
echo "Test completed!"
echo "======================================================================"
echo ""
echo "Tip: "
echo "  • See Rank 0 output for a detailed performance comparison"
echo "  • Approach 3 (CPUMemoryPool + Raw Bytes) should be the fastest"
echo ""

