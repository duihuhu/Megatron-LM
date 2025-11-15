#!/bin/bash

# Script to run 4-node simulation in a single container (1 GPU per node)
# This script will start all 4 nodes in background processes
# Usage: ./test_eccheck_4nodes.sh [additional_args...]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NODE_SCRIPT="$SCRIPT_DIR/test_eccheck_4nodes_node.sh"

# Check if node script exists
if [ ! -f "$NODE_SCRIPT" ]; then
    echo "Error: Node script not found: $NODE_SCRIPT"
    exit 1
fi

# Make sure node script is executable
chmod +x "$NODE_SCRIPT"

# Create logs directory
mkdir -p logs
mkdir -p logs/csv

# Array to store background process PIDs
PIDS=()

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Cleaning up background processes..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing process $pid (Node $((pid_index)))"
            kill "$pid" 2>/dev/null
        fi
    done
    wait
    echo "All processes terminated."
}

# Set trap to cleanup on script exit
trap cleanup EXIT INT TERM

# Start all 4 nodes in background
echo "Starting 4-node simulation (1 GPU per node)..."
echo "Node 0 will use GPU 0"
echo "Node 1 will use GPU 1"
echo "Node 2 will use GPU 2"
echo "Node 3 will use GPU 3"
echo ""

for NODE_RANK in 0 1 2 3; do
    echo "Starting Node $NODE_RANK in background..."
    LOG_FILE="logs/node${NODE_RANK}.log"
    
    # Start node in background and redirect output to log file
    bash "$NODE_SCRIPT" $NODE_RANK "$@" > "$LOG_FILE" 2>&1 &
    PID=$!
    PIDS+=($PID)
    
    echo "  Node $NODE_RANK started with PID $PID (log: $LOG_FILE)"
    sleep 1  # Small delay to avoid port conflicts
done

echo ""
echo "All 4 nodes started. PIDs: ${PIDS[@]}"
echo "Log files:"
echo "  - Node 0: logs/node0.log"
echo "  - Node 1: logs/node1.log"
echo "  - Node 2: logs/node2.log"
echo "  - Node 3: logs/node3.log"
echo "  - NCCL logs: nccl.log.node0, nccl.log.node1, nccl.log.node2, nccl.log.node3"
echo ""
echo "Press Ctrl+C to stop all nodes..."
echo ""

# Wait for all background processes
wait

